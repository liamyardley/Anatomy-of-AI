// Run with: node --test
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { createRng } from '../assets/js/rng.js';
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

// ===== ROC curve and the model comparison =====

import { rocCurve } from '../labs/ml/js/metrics.js';
import { trainAll, describeSettings } from '../labs/ml/js/compare.js';

test('ROC curve: perfect scores give area 1, reversed scores give 0, ties give 0.5', () => {
  const labels = [1, 1, 0, 0];
  const perfect = rocCurve([0.9, 0.8, 0.2, 0.1], labels);
  assert.equal(perfect.auc, 1);
  assert.deepEqual(perfect.points[0], { fpr: 0, tpr: 0, threshold: Infinity });
  assert.deepEqual(perfect.points.at(-1), { fpr: 1, tpr: 1, threshold: 0.1 });
  assert.equal(rocCurve([0.1, 0.2, 0.8, 0.9], labels).auc, 0);
  assert.equal(rocCurve([0.5, 0.5, 0.5, 0.5], labels).auc, 0.5);
  // With only one colour there is no curve to draw.
  assert.deepEqual(rocCurve([0.5, 0.6], [1, 1]), { points: [], auc: null });
});

test('ROC curve: every threshold lands on the curve', () => {
  const { X, y } = lab('islands');
  const model = fit(createKnn({ k: 5 }), X, y);
  const probabilities = X.map((x) => model.predictProba(x));
  const { points } = rocCurve(probabilities, y);
  for (const threshold of [0.05, 0.3, 0.5, 0.7, 0.95]) {
    const { tp, fn, fp, tn } = confusion(probabilities, y, threshold);
    const at = { fpr: fp / (fp + tn), tpr: tp / (tp + fn) };
    assert.ok(points.some((p) => p.fpr === at.fpr && p.tpr === at.tpr));
  }
});

test('Train all: one row per model, trained with its own settings', () => {
  const points = buildPoints(createDataset('moons', 5), 200, 1, 0.7);
  const data = { train: split(points.filter((p) => !p.test)), test: split(points.filter((p) => p.test)) };
  // A cut-down stand-in for registry.js (which also loads the page's panels).
  const models = {
    knn: { label: 'k-NN', create: createKnn, params: [{ key: 'k', label: 'Neighbours' }] },
    tree: { label: 'Tree', create: createTree, params: [{ key: 'maxDepth', label: 'Maximum depth' }] },
    forest: {
      label: 'Forest', create: createForest,
      params: [{ key: 'trees', label: 'Trees' }, { key: 'bootstrap', label: 'Bootstrap', format: (v) => (v ? 'on' : 'off') }],
    },
  };
  const params = { knn: { k: 1 }, tree: { maxDepth: 2 }, forest: { trees: 5, bootstrap: true } };
  let clock = 0;
  const rows = trainAll(models, params, 11, data, () => (clock += 2));
  assert.deepEqual(rows.map((r) => r.key), ['knn', 'tree', 'forest']);
  assert.equal(describeSettings(models.forest, params.forest), 'Trees 5, Bootstrap on');
  for (const row of rows) {
    assert.equal(row.ms, 2);
    assert.equal(row.scores.train.probabilities.length, data.train.y.length);
    assert.equal(row.scores.test.probabilities.length, data.test.y.length);
  }
  // k = 1 remembers its own training points perfectly.
  const knn = rows[0].scores.train;
  assert.equal(summarise(confusion(knn.probabilities, knn.labels, 0.5)).accuracy, 1);
  // The forest row matches a forest trained directly with the same seed.
  const direct = fit(createForest({ trees: 5, bootstrap: true, seed: 11 }), data.train.X, data.train.y);
  assert.deepEqual(rows[2].scores.test.probabilities, data.test.X.map((x) => direct.predictProba(x)));
});

// ===== k-NN distance measures =====

import { distanceFor, neighbourhoodOutline } from '../labs/ml/js/models/knn.js';

const close = (a, b) => Math.abs(a - b) < 1e-9;

test('distance measures: each gives the textbook answer', () => {
  const a = [0, 0];
  const b = [0.3, -0.4];
  assert.ok(close(distanceFor('euclidean')(a, b), 0.5));
  assert.ok(close(distanceFor('manhattan')(a, b), 0.7));
  assert.ok(close(distanceFor('chebyshev')(a, b), 0.4));
  // Minkowski joins them up: p = 1 is Manhattan, p = 2 Euclidean, large p nears Chebyshev.
  assert.ok(close(distanceFor('minkowski', 1)(a, b), 0.7));
  assert.ok(close(distanceFor('minkowski', 2)(a, b), 0.5));
  assert.ok(Math.abs(distanceFor('minkowski', 60)(a, b) - 0.4) < 1e-3);
  // Cosine: same direction 0, right angle 1, opposite 2, whatever the lengths.
  const cosine = distanceFor('cosine');
  assert.ok(close(cosine([0.1, 0.1], [0.9, 0.9]), 0));
  assert.ok(close(cosine([0.5, 0], [0, 0.2]), 1));
  assert.ok(close(cosine([0.5, 0], [-0.1, 0]), 2));
  assert.equal(cosine([0, 0], [0.3, 0.3]), 1); // the centre has no direction
});

test('neighbourhood outline: every point on it is the same distance away', () => {
  const centre = [0.2, -0.1];
  for (const [metric, p] of [['euclidean', 2], ['manhattan', 2], ['chebyshev', 2], ['minkowski', 1.5], ['minkowski', 6]]) {
    const distance = distanceFor(metric, p);
    const outline = neighbourhoodOutline(centre, 0.3, metric, p);
    for (const point of outline) assert.ok(Math.abs(distance(centre, point) - 0.3) < 1e-9, metric);
  }
  // Cosine: a wedge from the centre of the plot, whose two straight edges
  // are exactly the cut-off distance from the query direction.
  const wedge = neighbourhoodOutline(centre, 0.2, 'cosine');
  assert.deepEqual(wedge[0], [0, 0]);
  assert.ok(Math.abs(distanceFor('cosine')(centre, wedge[1]) - 0.2) < 1e-9);
  assert.ok(Math.abs(distanceFor('cosine')(centre, wedge.at(-1)) - 0.2) < 1e-9);
  assert.equal(neighbourhoodOutline(centre, 2, 'cosine'), null);
});

test('k-NN: the distance measure changes who the neighbours are', () => {
  // Two stored points: one a short diagonal step away, one a slightly longer straight step.
  const X = [[0.3, 0.3], [0.5, 0]];
  const y = [1, 0];
  const at = [0, 0];
  assert.equal(fit(createKnn({ k: 1, metric: 'euclidean' }), X, y).predictProba(at), 1); // 0.42 against 0.5
  assert.equal(fit(createKnn({ k: 1, metric: 'manhattan' }), X, y).predictProba(at), 0); // 0.6 against 0.5
  assert.equal(fit(createKnn({ k: 1, metric: 'chebyshev' }), X, y).predictProba(at), 1); // 0.3 against 0.5
});

test('k-NN with cosine distance sees only direction', () => {
  const { X, y } = lab('moons');
  const model = fit(createKnn({ k: 5, metric: 'cosine' }), X, y);
  // Two places in the same direction from the centre always get the same answer.
  for (const [a, b] of [[0.2, 0.1], [-0.3, 0.4], [0.1, -0.5]]) {
    assert.equal(model.predictProba([a, b]), model.predictProba([a * 1.8, b * 1.8]));
  }
  // So it cannot tell a disc from the ring around it.
  const ring = lab('ring');
  const euclid = accuracy(fit(createKnn({ k: 5 }), ring.X, ring.y), ring.X, ring.y);
  const cosine = accuracy(fit(createKnn({ k: 5, metric: 'cosine' }), ring.X, ring.y), ring.X, ring.y);
  assert.ok(euclid > 0.95);
  assert.ok(cosine < 0.8);
});

test('k-NN with cosine distance: an extra training phase, and stepping equals finish', () => {
  const { X, y } = lab('moons');
  const stepped = createKnn({ k: 3, metric: 'cosine' });
  const trainer = stepped.train(X, y);
  const phases = new Set();
  let result = trainer.step();
  while (!result.done) { phases.add(result.snapshot.phase); result = trainer.step(); }
  assert.deepEqual([...phases], ['memorise', 'project', 'predict']);
  const whole = fit(createKnn({ k: 3, metric: 'cosine' }), X, y);
  for (const x of probes) assert.equal(stepped.predictProba(x), whole.predictProba(x));
  // The overlay holds a copy of every stored point, all on one circle.
  const marks = whole.explain([0.3, 0.3]).overlay.find((shape) => shape.kind === 'marks');
  assert.equal(marks.points.length, X.length);
  for (const [a, b] of marks.points) assert.ok(Math.abs(Math.hypot(a, b) - 0.92) < 1e-9);
});

// ===== tree and forest options =====

import { entropy } from '../labs/ml/js/models/tree.js';

const treeInfo = (params, data) => fit(createTree(params), data.X, data.y).explain();

test('entropy: 0 for a pure group, 1 for an even mix', () => {
  assert.equal(entropy(10, 0), 0);
  assert.equal(entropy(5, 5), 1);
  assert.ok(entropy(9, 1) < entropy(7, 3));
});

test('tree with entropy: still solves XOR, and records entropy as its impurity', () => {
  const { X, y } = xor();
  const model = fit(createTree({ maxDepth: 2, criterion: 'entropy' }), X, y);
  assert.equal(accuracy(model, X, y), 1);
  const root = model.explain().nodes[0];
  assert.ok(Math.abs(root.impurity - entropy(root.red, root.blue)) < 1e-12);
});

test('tree: minimum samples to split is respected', () => {
  const data = lab('noise');
  const info = treeInfo({ maxDepth: 30, minSplit: 20 }, data);
  for (const node of info.nodes) {
    if (node.split) assert.ok(node.red + node.blue >= 20);
  }
  assert.ok(info.leaves < treeInfo({ maxDepth: 30 }, data).leaves);
});

test('tree: a leaf limit is never exceeded, and is spent on the best splits', () => {
  const data = lab('islands');
  for (const maxLeaves of [2, 3, 5, 8]) {
    const info = treeInfo({ maxDepth: 10, maxLeaves }, data);
    assert.equal(info.leaves, maxLeaves);
    assert.equal(info.splits, maxLeaves - 1);
  }
  // Best first is not level by level: five leaves can reach deeper than
  // a level-by-level tree of five leaves could.
  const limited = treeInfo({ maxDepth: 10, maxLeaves: 5 }, data);
  assert.ok(limited.depth >= 3);
  // With no limit the tree is unchanged from before the option existed.
  const { X, y } = data;
  const plain = fit(createTree({ maxDepth: 4 }), X, y);
  const noLimit = fit(createTree({ maxDepth: 4, maxLeaves: 0 }), X, y);
  for (const x of probes) assert.equal(plain.predictProba(x), noLimit.predictProba(x));
});

test('pruning: stronger pruning never leaves a bigger tree, and enough of it leaves one leaf', () => {
  const data = lab('islands');
  const sizes = [0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 1].map(
    (ccpAlpha) => treeInfo({ maxDepth: 10, ccpAlpha }, data).leaves,
  );
  for (let i = 1; i < sizes.length; i++) assert.ok(sizes[i] <= sizes[i - 1], sizes.join(' '));
  assert.ok(sizes[0] > sizes[4]);
  assert.equal(sizes.at(-1), 1);
});

test('pruning: happens after growth, one split per step, and leaves a tidy tree', () => {
  const { X, y } = lab('islands');
  const model = createTree({ maxDepth: 10, ccpAlpha: 0.01 });
  const trainer = model.train(X, y);
  const phases = [];
  let result = trainer.step();
  let prunedBefore = 0;
  while (!result.done) {
    phases.push(result.snapshot.phase);
    assert.ok(result.snapshot.pruned - prunedBefore <= 1);
    prunedBefore = result.snapshot.pruned;
    result = trainer.step();
  }
  assert.ok(phases.indexOf('prune') > 0);
  assert.ok(!phases.slice(phases.indexOf('prune')).includes('grow'));
  const info = model.explain();
  assert.ok(info.pruned > 0);
  // Ids match positions, every split has two children, and counts agree.
  info.nodes.forEach((node, i) => {
    assert.equal(node.id, i);
    if (node.split) assert.equal(node.left.red + node.left.blue + node.right.red + node.right.blue, node.red + node.blue);
  });
  assert.equal(info.nodes.filter((n) => !n.split).length, info.leaves);
  // Stepping and finishing agree.
  const whole = fit(createTree({ maxDepth: 10, ccpAlpha: 0.01 }), X, y);
  for (const x of probes) assert.equal(model.predictProba(x), whole.predictProba(x));
});

test('pruning a deep tree on noisy labels does not hurt, and usually helps, on new points', () => {
  let better = 0;
  let worse = 0;
  for (let seed = 1; seed <= 12; seed++) {
    const points = buildPoints(createDataset('islands', seed), 300, 1, 0.7);
    const train = split(points.filter((p) => !p.test));
    const test = split(points.filter((p) => p.test));
    const deep = accuracy(fit(createTree({ maxDepth: 10 }), train.X, train.y), test.X, test.y);
    const pruned = accuracy(fit(createTree({ maxDepth: 10, ccpAlpha: 0.01 }), train.X, train.y), test.X, test.y);
    if (pruned > deep) better++;
    if (pruned < deep) worse++;
  }
  assert.ok(better > worse, `better ${better}, worse ${worse}`);
});

test('forest: minimum samples per leaf and sample size are respected', () => {
  const { X, y } = lab('islands');
  const small = fit(createForest({ trees: 8, sampleShare: 0.25, seed: 3 }), X, y);
  const info = small.explain();
  // Small samples leave most points out of each bag, so nearly every point can be scored.
  assert.ok(info.oobCounted >= X.length - 1);
  const full = fit(createForest({ trees: 8, seed: 3 }), X, y);
  assert.ok(probes.some((x) => small.predictProba(x) !== full.predictProba(x)));
  // Big leaves make blunter trees: training accuracy drops.
  const blunt = fit(createForest({ trees: 8, minLeaf: 20, seed: 3 }), X, y);
  assert.ok(accuracy(blunt, X, y) < accuracy(full, X, y));
  // Sample size does nothing without bootstrap samples.
  const a = fit(createForest({ trees: 4, bootstrap: false, sampleShare: 0.25, seed: 3 }), X, y);
  const b = fit(createForest({ trees: 4, bootstrap: false, seed: 3 }), X, y);
  assert.ok(probes.every((x) => a.predictProba(x) === b.predictProba(x)));
});
