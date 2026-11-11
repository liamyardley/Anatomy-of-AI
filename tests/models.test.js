// Run with: node --test
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { createRng } from '../labs/ml/js/rng.js';
import { createDataset, buildPoints } from '../labs/ml/js/datasets.js';
import { confusion, summarise } from '../labs/ml/js/metrics.js';
import { createLogistic, polynomialTerms } from '../labs/ml/js/models/logistic.js';

// Two well-separated clouds: red around (-0.5, -0.5), blue around (0.5, 0.5).
function separatedClouds(n = 200, seed = 1) {
  const rng = createRng(seed, 'test');
  const X = [];
  const y = [];
  for (let i = 0; i < n; i++) {
    const label = i % 2;
    const centre = label === 1 ? 0.5 : -0.5;
    X.push([centre + rng.gaussian() * 0.12, centre + rng.gaussian() * 0.12]);
    y.push(label);
  }
  return { X, y };
}

function accuracy(model, X, y, threshold = 0.5) {
  const probabilities = X.map((x) => model.predictProba(x));
  return summarise(confusion(probabilities, y, threshold)).accuracy;
}

const split = (points) => ({
  X: points.map((p) => [p.x1, p.x2]),
  y: points.map((p) => p.y),
});

test('polynomial terms: 2 at degree 1, 5 at degree 2, 20 at degree 5', () => {
  assert.equal(polynomialTerms(1).length, 2);
  assert.equal(polynomialTerms(2).length, 5);
  assert.equal(polynomialTerms(5).length, 20);
});

test('logistic regression: over 95% training accuracy on well-separated clouds', () => {
  const { X, y } = separatedClouds();
  const model = createLogistic();
  model.train(X, y).finish();
  assert.ok(accuracy(model, X, y) > 0.95);
});

test('logistic regression: stepping to the end matches finish() exactly', () => {
  const { X, y } = split(buildPoints(createDataset('moons', 42)));
  for (const params of [{}, { degree: 3, l2: 0.01 }, { learningRate: 0.3 }]) {
    const stepped = createLogistic(params);
    const trainer = stepped.train(X, y);
    let result = trainer.step();
    let steps = 1;
    while (!result.done) { result = trainer.step(); steps++; }

    const skipped = createLogistic(params);
    const finished = skipped.train(X, y).finish();

    assert.equal(finished.snapshot.iteration, steps);
    assert.deepEqual(stepped.explain().terms, skipped.explain().terms);
    assert.deepEqual(result.snapshot, finished.snapshot);
    for (const x of [[0, 0], [0.3, -0.7], [-1, 1]]) {
      assert.equal(stepped.predictProba(x), skipped.predictProba(x));
    }
  }
});

test('logistic regression: an untrained model is undecided everywhere', () => {
  const model = createLogistic();
  assert.equal(model.predictProba([0.4, -0.2]), 0.5);
  assert.equal(model.explain().trained, false);
});

test('logistic regression: the loss falls during training', () => {
  const { X, y } = separatedClouds();
  const model = createLogistic();
  model.train(X, y).finish();
  const history = model.explain().lossHistory;
  assert.ok(Math.abs(history[0] - Math.log(2)) < 1e-9, 'starts at ln 2');
  assert.ok(history[history.length - 1] < history[0] / 2);
});

test('logistic regression: a straight line fails on the ring, degree 2 succeeds', () => {
  const { X, y } = split(buildPoints(createDataset('ring', 42)));
  const line = createLogistic({ degree: 1 });
  line.train(X, y).finish();
  const curve = createLogistic({ degree: 2 });
  curve.train(X, y).finish();
  assert.ok(accuracy(line, X, y) < 0.75, `degree 1: ${accuracy(line, X, y)}`);
  assert.ok(accuracy(curve, X, y) > 0.9, `degree 2: ${accuracy(curve, X, y)}`);
});

test('logistic regression: regularisation shrinks the weights', () => {
  const { X, y } = separatedClouds();
  const size = (model) => model.explain().terms.slice(1)
    .reduce((sum, t) => sum + t.weight * t.weight, 0);
  const free = createLogistic();
  free.train(X, y).finish();
  const held = createLogistic({ l2: 0.3 });
  held.train(X, y).finish();
  assert.ok(size(held) < size(free));
});

test('logistic regression: the worked example adds up', () => {
  const { X, y } = separatedClouds();
  const model = createLogistic({ degree: 2 });
  model.train(X, y).finish();
  const { example } = model.explain([0.3, -0.2]);
  const z = example.rows.reduce((sum, r) => sum + r.value * r.weight, 0);
  assert.ok(Math.abs(example.z - z) < 1e-12);
  assert.ok(Math.abs(example.p - model.predictProba([0.3, -0.2])) < 1e-12);
});

test('confusion matrix and summary: blue is the positive class', () => {
  const counts = confusion([0.9, 0.6, 0.4, 0.2, 0.7], [1, 1, 1, 0, 0], 0.5);
  assert.deepEqual(counts, { tp: 2, fn: 1, fp: 1, tn: 1 });
  const summary = summarise(counts);
  assert.equal(summary.accuracy, 3 / 5);
  assert.equal(summary.sensitivity, 2 / 3);
  assert.equal(summary.specificity, 1 / 2);
  // A higher threshold predicts blue less often.
  assert.deepEqual(confusion([0.9, 0.6, 0.4, 0.2, 0.7], [1, 1, 1, 0, 0], 0.8),
    { tp: 1, fn: 2, fp: 0, tn: 2 });
  assert.equal(summarise({ tp: 0, fn: 0, fp: 0, tn: 0 }).accuracy, null);
});
