// The smallest model there is: a straight line through the corner of the
// plot, y = w * x, with one number to find, the weight w.
// Pure code: no DOM access, so it can be tested in Node.
//
// Everything here is about that one weight:
//   makeLineData   a small seeded dataset that roughly follows a line
//   loss           how wrong a given weight is (the mean squared miss)
//   gradient       the slope of the loss at a given weight
//   createDescent  gradient descent, one step at a time

import { createRng } from '../../../assets/js/rng.js';
import { createOptimiser } from '../../../assets/js/optimisers.js';

// The range of weights the page lets you try.
export const WEIGHT_MIN = -0.5;
export const WEIGHT_MAX = 4;

// A few points scattered around a line through the corner. The steepness of
// that line changes with the seed.
//
// The x values are scaled so that the average of x squared is exactly one
// half. That makes the loss curve the same width for every dataset, so the
// learning rate means the same thing each time: 1 reaches the bottom in a
// single step, and anything above 2 gets worse with every step.
export function makeLineData(seed, count = 10) {
  const settings = createRng(seed, 'line-settings');
  const rng = createRng(seed, 'line-points');
  const slope = settings.uniform(0.9, 2.4);
  const raw = Array.from({ length: count }, () => rng.uniform(0.15, 1));
  const meanSquare = raw.reduce((sum, x) => sum + x * x, 0) / count;
  const scale = Math.sqrt(0.5 / meanSquare);
  const points = raw.map((value) => {
    const x = value * scale;
    return { x, y: Math.max(0.03, slope * x + rng.gaussian() * 0.15) };
  });
  return { seed, slope, points };
}

// The line's prediction for one x.
export const predict = (w, x) => w * x;

// The loss: square each miss (prediction minus the true y), then average.
export function loss(w, points) {
  let total = 0;
  for (const p of points) total += (w * p.x - p.y) ** 2;
  return total / points.length;
}

// The gradient: the slope of the loss curve at w. Positive means the loss
// rises as w grows, so w should shrink; negative means w should grow.
export function gradient(w, points) {
  let total = 0;
  for (const p of points) total += 2 * (w * p.x - p.y) * p.x;
  return total / points.length;
}

// The weight with the lowest possible loss, worked out directly. (Gradient
// descent has to find it without this shortcut.)
export function bestWeight(points) {
  let top = 0;
  let bottom = 0;
  for (const p of points) { top += p.x * p.y; bottom += p.x * p.x; }
  return top / bottom;
}

// Gradient descent from a starting weight. Each step: measure the gradient,
// then let the optimiser move the weight downhill.
//
// It has the same shape as a trainer in the Machine Learning Lab, so the
// shared training engine can run it:
//   step()    one step: { done, snapshot }
//   finish()  all remaining steps at once
// A snapshot is { step, w, loss, gradient, move, outcome }. `outcome` is
// null while it is still going, then one of:
//   'settled'   the steps became too small to matter
//   'diverged'  the loss was still growing when it stopped: steps too big
//   'bouncing'  out of steps while jumping from side to side, getting
//               closer each time: steps too big to settle quickly
//   'limit'     out of steps while still creeping towards the bottom:
//               steps too small
export function createDescent({
  points, start = 0, learningRate = 0.3, optimiser = 'gd', maxSteps = 60, tolerance = 1e-4,
}) {
  const rule = createOptimiser(optimiser, { learningRate });
  let w = start;
  let outcome = null;
  const record = (move) => ({
    step: history.length, w, loss: loss(w, points), gradient: gradient(w, points), move, outcome,
  });
  const history = [];
  history.push(record(null));

  function step() {
    if (outcome) return { done: true, snapshot: history[history.length - 1] };
    const [next] = rule.step([w], [gradient(w, points)]);
    const move = next - w;
    w = next;
    if (!Number.isFinite(w) || loss(w, points) > 1e6) outcome = 'diverged';
    else if (Math.abs(move) < tolerance) outcome = 'settled';
    else if (history.length >= maxSteps) {
      // Out of steps. Why? Look at the last two steps.
      const before = history[history.length - 1];
      if (loss(w, points) > before.loss) outcome = 'diverged';
      else if (before.move !== null && move * before.move < 0) outcome = 'bouncing';
      else outcome = 'limit';
    }
    const snapshot = record(move);
    history.push(snapshot);
    return { done: outcome !== null, snapshot };
  }

  function finish() {
    let result = step();
    while (!result.done) result = step();
    return result;
  }

  return {
    step,
    finish,
    history,
    get outcome() { return outcome; },
  };
}
