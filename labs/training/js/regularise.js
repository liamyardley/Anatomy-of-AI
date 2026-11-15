// Regularisation: adding a penalty for large weights to the loss.
// Pure code: no DOM access, so it can be tested in Node.
//
// The model is a weighted sum with no bias, y = w1 x1 + w2 x2 + ..., and its
// data loss is the mean squared miss. Training lowers
//
//     data loss + strength * penalty
//
// where the penalty depends only on the weights:
//   'none'     no penalty
//   'l2'       half the sum of the squared weights            (ridge)
//   'l1'       the sum of the weights' sizes, ignoring signs  (lasso)
//   'elastic'  half of each                                   (elastic net)
//
//   makeRegression   a seeded dataset where only some features matter
//   dataSurface      the data loss of a two-weight model, for the contour map
//   createProximal   gradient descent with the penalty, one step at a time
//   sweep            the weights it ends at, across a range of strengths
//   penaltyOutline   the shape of all weights with the same penalty

import { createRng } from '../../../assets/js/rng.js';

export const PENALTIES = ['none', 'l2', 'l1', 'elastic'];

// How much of the penalty is L1 and how much is L2.
const shares = (kind) => ({
  none: [0, 0], l2: [0, 1], l1: [1, 0], elastic: [0.5, 0.5],
}[kind]);

// Data where y depends on the features according to `truth` (a weight for
// each feature; zero means the feature is irrelevant). `blend` mixes a
// shared ingredient into every feature, so the features overlap in what
// they say, as real measurements often do.
export function makeRegression(seed, { truth = [1.5, 0.3], count = 60, noise = 0.5, blend = 0.5 } = {}) {
  const rng = createRng(seed, 'regression');
  const X = [];
  const y = [];
  for (let i = 0; i < count; i++) {
    const shared = rng.gaussian();
    const row = truth.map(() => blend * shared + Math.sqrt(1 - blend * blend) * rng.gaussian());
    X.push(row);
    y.push(row.reduce((sum, value, k) => sum + value * truth[k], 0) + rng.gaussian() * noise);
  }
  return { seed, truth, X, y };
}

// The mean squared miss and its gradient, for any number of weights.
export function dataLoss(w, X, y) {
  let total = 0;
  for (let i = 0; i < X.length; i++) {
    let miss = -y[i];
    for (let k = 0; k < w.length; k++) miss += w[k] * X[i][k];
    total += miss * miss;
  }
  return total / X.length;
}
export function dataGradient(w, X, y) {
  const g = w.map(() => 0);
  for (let i = 0; i < X.length; i++) {
    let miss = -y[i];
    for (let k = 0; k < w.length; k++) miss += w[k] * X[i][k];
    for (let k = 0; k < w.length; k++) g[k] += 2 * miss * X[i][k];
  }
  return g.map((value) => value / X.length);
}

// The penalty for a set of weights (before multiplying by the strength).
export function penalty(kind, w) {
  const [l1, l2] = shares(kind);
  let total = 0;
  for (const value of w) total += l1 * Math.abs(value) + (l2 / 2) * value * value;
  return total;
}

// The data loss of a two-weight model as a surface for the contour map.
export function dataSurface(X, y) {
  const surface = {
    loss: (a, b) => dataLoss([a, b], X, y),
    gradient: (a, b) => dataGradient([a, b], X, y),
  };
  // The best weights with no penalty, found by plain gradient descent.
  surface.best = createProximal({ X, y, kind: 'none', strength: 0, start: [0, 0], maxSteps: 5000 }).finish().snapshot.w;
  surface.spots = [{ at: surface.best, kind: 'lowest' }];
  return surface;
}

// Shrink a number towards zero by `amount`, stopping at zero. This is the
// step that lets L1 set a weight to exactly zero.
export const softThreshold = (value, amount) => Math.sign(value) * Math.max(0, Math.abs(value) - amount);

// Gradient descent on the data loss plus the penalty. Each step:
//   1. step downhill on the data loss and the L2 part of the penalty;
//   2. then shrink every weight towards zero by a fixed amount for the L1
//      part (the soft threshold). A weight that would cross zero stops
//      there, and stays there unless the data pulls it away hard enough.
// A snapshot is { step, w, loss, penalty, objective, zeros, outcome } where
// `loss` is the data loss alone and `zeros` counts weights at exactly zero.
// `outcome` is null while going, then 'settled' or 'limit'.
export function createProximal({
  X, y, kind = 'l2', strength = 0, start, learningRate = 0.2, maxSteps = 400, tolerance = 1e-5,
}) {
  const [l1, l2] = shares(kind);
  let w = [...start];
  let outcome = null;
  const history = [];
  const record = () => {
    const loss = dataLoss(w, X, y);
    const cost = penalty(kind, w);
    return {
      step: history.length, w: [...w], loss, penalty: cost, objective: loss + strength * cost, zeros: w.filter((v) => v === 0).length, outcome,
    };
  };
  history.push(record());

  function step() {
    if (outcome) return { done: true, snapshot: history[history.length - 1] };
    const g = dataGradient(w, X, y);
    const next = w.map((value, k) => softThreshold(
      value - learningRate * (g[k] + strength * l2 * value),
      learningRate * strength * l1,
    ));
    const move = Math.hypot(...next.map((value, k) => value - w[k]));
    w = next;
    if (move < tolerance) outcome = 'settled';
    else if (history.length >= maxSteps) outcome = 'limit';
    const snapshot = record();
    history.push(snapshot);
    return { done: outcome !== null, snapshot };
  }

  function finish() {
    let result = step();
    while (!result.done) result = step();
    return result;
  }

  return { step, finish, history };
}

// Train once for every strength in the list, and report where the weights
// end up: [{ strength, w, zeros }]. Each run starts from the previous
// answer, which is close to the next one.
export function sweep({ X, y, kind, strengths, learningRate = 0.1 }) {
  let w = X[0].map(() => 0);
  return strengths.map((strength) => {
    const end = createProximal({
      X, y, kind, strength, start: w, learningRate, maxSteps: 4000, tolerance: 1e-7,
    }).finish().snapshot;
    w = end.w;
    return { strength, w: end.w, zeros: end.zeros };
  });
}

// Every pair of weights with the same penalty as `w`: a circle for L2, a
// diamond for L1, a rounded diamond for the elastic net. Returned as a list
// of [a, b] points going round the shape; empty when there is no penalty or
// the weights are both zero.
export function penaltyOutline(kind, w, count = 120) {
  const target = penalty(kind, w);
  if (kind === 'none' || !(target > 0)) return [];
  const outline = [];
  for (let i = 0; i <= count; i++) {
    const angle = (i / count) * 2 * Math.PI;
    const direction = [Math.cos(angle), Math.sin(angle)];
    // The penalty grows along every ray from the centre, so search outward.
    let low = 0;
    let high = 1;
    while (penalty(kind, [high * direction[0], high * direction[1]]) < target) high *= 2;
    for (let k = 0; k < 40; k++) {
      const mid = (low + high) / 2;
      if (penalty(kind, [mid * direction[0], mid * direction[1]]) < target) low = mid; else high = mid;
    }
    outline.push([high * direction[0], high * direction[1]]);
  }
  return outline;
}
