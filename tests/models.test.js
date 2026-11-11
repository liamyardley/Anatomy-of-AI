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
  assert.equal(summary.precision, 2 / 3);
  assert.equal(summary.recall, summary.sensitivity);
  assert.equal(summary.f1, 2 / 3);
  // Nothing predicted blue: precision has nothing to measure, F1 is zero.
  const none = summarise({ tp: 0, fn: 3, fp: 0, tn: 2 });
  assert.equal(none.precision, null);
  assert.equal(none.f1, 0);
  // A higher threshold predicts blue less often.
  assert.deepEqual(confusion([0.9, 0.6, 0.4, 0.2, 0.7], [1, 1, 1, 0, 0], 0.8),
    { tp: 1, fn: 2, fp: 0, tn: 2 });
  assert.equal(summarise({ tp: 0, fn: 0, fp: 0, tn: 0 }).accuracy, null);
});

// ===== k-nearest neighbours, decision tree and random forest =====

import { createKnn } from '../labs/ml/js/models/knn.js';
import { createTree, gini } from '../labs/ml/js/models/tree.js';
import { createForest } from '../labs/ml/js/models/forest.js';

const lab = (type, seed = 7, n = 200) => split(buildPoints(createDataset(type, seed), n, 1, 1));

function fit(model, X, y) {
  model.train(X, y).finish();
  return model;
}

// A grid of probe points covering the plot.
const probes = [];
for (let a = -1; a <= 1; a += 0.1) for (let b = -1; b <= 1; b += 0.1) probes.push([a, b]);

// Noise-free XOR. The quarters hold different numbers of points: with equal
// quarters no single cut makes either side purer, so a tree that picks cuts
// one at a time has nothing to go on.
function xor() {
  const rng = createRng(3, 'xor');
  const X = [];
  const y = [];
  const quarters = [[-1, 1, 60, 0], [1, 1, 10, 1], [-1, -1, 30, 1], [1, -1, 10, 0]];
  for (const [sx, sy, count, label] of quarters) {
    for (let i = 0; i < count; i++) {
      X.push([sx * rng.uniform(0.05, 0.95), sy * rng.uniform(0.05, 0.95)]);
      y.push(label);
    }
  }
  return { X, y };
}

test('k-NN with k = 1 scores 100% on its own training points', () => {
  for (const type of ['moons', 'noise', 'spiral']) {
    const { X, y } = lab(type);
    assert.equal(accuracy(fit(createKnn({ k: 1 }), X, y), X, y), 1);
  }
});

test('k-NN: probability is the share of blue neighbours', () => {
  const X = [[0, 0], [0.1, 0], [0, 0.1], [0.5, 0.5], [0.6, 0.5]];
  const y = [1, 1, 0, 0, 0];
  const model = fit(createKnn({ k: 3 }), X, y);
  assert.equal(model.predictProba([0.01, 0.01]), 2 / 3);
  assert.equal(model.predictProba([0.55, 0.45]), 1 / 3);
  const info = model.explain([0.01, 0.01]);
  assert.equal(info.example.neighbours.length, 3);
  assert.equal(info.example.blueVotes, 2);
  // Weighting by distance gives the closest neighbours a bigger say.
  const weighted = fit(createKnn({ k: 3, weighted: true }), X, y);
  assert.ok(weighted.predictProba([0.09, 0]) > 2 / 3);
});

test('k-NN: stepping to the end equals finish, and nothing is predicted before', () => {
  const { X, y } = lab('moons');
  const stepped = createKnn({ k: 5 });
  assert.equal(stepped.predictProba([0, 0]), 0.5);
  const trainer = stepped.train(X, y);
  let result = trainer.step();
  assert.equal(result.snapshot.phase, 'memorise');
  while (!result.done) result = trainer.step();
  assert.equal(result.snapshot.reveal, 1);
  const whole = fit(createKnn({ k: 5 }), X, y);
  for (const x of probes) assert.equal(stepped.predictProba(x), whole.predictProba(x));
});

test('Gini impurity: 0 for a pure group, 0.5 for an even mix', () => {
  assert.equal(gini(10, 0), 0);
  assert.equal(gini(5, 5), 0.5);
});

test('a depth-2 tree scores 100% on noise-free XOR', () => {
  const { X, y } = xor();
  const model = fit(createTree({ maxDepth: 2 }), X, y);
  assert.equal(accuracy(model, X, y), 1);
  assert.equal(model.explain().splits, 3);
});

test('tree: stepping to the end equals finish, one split per step', () => {
  const { X, y } = lab('islands');
  const stepped = createTree({ maxDepth: 5 });
  const trainer = stepped.train(X, y);
  let result = trainer.step();
  assert.equal(result.snapshot.splits, 1);
  let steps = 1;
  while (!result.done) { result = trainer.step(); steps++; }
  assert.equal(result.snapshot.splits, steps);
  const whole = fit(createTree({ maxDepth: 5 }), X, y);
  for (const x of probes) assert.equal(stepped.predictProba(x), whole.predictProba(x));
});

test('tree: respects maximum depth and minimum samples per leaf', () => {
  const { X, y } = lab('noise');
  const info = fit(createTree({ maxDepth: 4, minLeaf: 8 }), X, y).explain();
  assert.ok(info.depth <= 4);
  for (const node of info.nodes) {
    if (!node.split) assert.ok(node.red + node.blue >= 8);
  }
  // A deep tree with no limits memorises pure noise.
  assert.equal(accuracy(fit(createTree({ maxDepth: 30 }), X, y), X, y), 1);
});

test('tree: the worked example follows the questions to a leaf', () => {
  const { X, y } = xor();
  const model = fit(createTree({ maxDepth: 2 }), X, y);
  const { example } = model.explain([0.5, 0.5]);
  assert.equal(example.path.length, 2);
  for (const q of example.path) assert.equal(q.yes, q.value < q.threshold);
  assert.equal(example.p, model.predictProba([0.5, 0.5]));
});

test('a forest of 1 tree, no bootstrap, both features, matches the single tree', () => {
  const { X, y } = lab('islands');
  const tree = fit(createTree({ maxDepth: 6 }), X, y);
  const forest = fit(createForest({ trees: 1, maxDepth: 6, bootstrap: false, featuresPerSplit: 2 }), X, y);
  for (const x of probes) assert.equal(forest.predictProba(x), tree.predictProba(x));
});

test('forest: stepping to the end equals finish, one tree per step', () => {
  const { X, y } = lab('moons');
  const params = { trees: 12, maxDepth: 5, seed: 9 };
  const stepped = createForest(params);
  const trainer = stepped.train(X, y);
  let result = trainer.step();
  assert.equal(result.snapshot.trees, 1);
  while (!result.done) result = trainer.step();
  assert.equal(result.snapshot.trees, 12);
  const whole = fit(createForest(params), X, y);
  for (const x of probes) assert.equal(stepped.predictProba(x), whole.predictProba(x));
});

test('forest: the same seed grows the same forest, a different seed does not', () => {
  const { X, y } = lab('islands');
  const a = fit(createForest({ trees: 10, seed: 1 }), X, y);
  const b = fit(createForest({ trees: 10, seed: 1 }), X, y);
  const c = fit(createForest({ trees: 10, seed: 2 }), X, y);
  assert.ok(probes.every((x) => a.predictProba(x) === b.predictProba(x)));
  assert.ok(probes.some((x) => a.predictProba(x) !== c.predictProba(x)));
});

test('forest: out-of-bag accuracy, the accuracy curve and the votes add up', () => {
  const { X, y } = lab('moons');
  const model = fit(createForest({ trees: 30, seed: 4 }), X, y);
  const info = model.explain([0.2, 0.1]);
  assert.ok(info.oobAccuracy > 0.7 && info.oobAccuracy <= 1);
  const curve = info.accuracyCurve(X, y);
  assert.equal(curve.length, 30);
  assert.equal(curve[29], accuracy(model, X, y));
  assert.equal(info.example.votes.length, 30);
  assert.ok(Math.abs(info.example.p - model.predictProba([0.2, 0.1])) < 1e-12);
  // Without bootstrap samples there is nothing out of the bag to score.
  assert.equal(fit(createForest({ trees: 3, bootstrap: false }), X, y).explain().oobAccuracy, null);
});
