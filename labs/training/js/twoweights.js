// Two numbers to find: a loss that depends on two weights, a and b.
// Pure code: no DOM access, so it can be tested in Node.
//
// With two weights the loss is no longer a curve but a landscape: every
// point on a map (a across, b up) has a height, the loss. Each surface here
// is a made-up landscape chosen to show one thing:
//   bowl    one round dip: the easy case
//   valley  a long narrow dip: steep sides, a nearly flat floor
//   saddle  flat ground between two dips that is not itself a dip
//   dips    three dips of different depths: the start decides which one
//
// Every surface is lifted so that its lowest point has a loss of zero.
//
// A surface is { label, loss(a, b), gradient(a, b), spots }. The gradient is
// now two numbers, [slope along a, slope along b]. `spots` lists the places
// where the ground is flat: { at: [a, b], kind } with kind 'lowest', 'dip'
// (a dip that is not the lowest) or 'saddle'.

import { createOptimiser } from '../../../assets/js/optimisers.js';

// The map shows both weights from -RANGE to RANGE.
export const RANGE = 2;

// The three dips: where each is centred and how deep it is.
const DIPS = [
  { a: -1.0, b: -0.8, depth: 1.0 },
  { a: 1.1, b: 0.6, depth: 1.4 },
  { a: -0.6, b: 1.2, depth: 0.7 },
];
const DIP_WIDTH = 0.6;

// The dips before the whole surface is lifted so its lowest point is at zero.
const rawDips = (a, b) => {
  let total = 0.15 * (a * a + b * b);
  for (const d of DIPS) {
    total -= d.depth * Math.exp(-((a - d.a) ** 2 + (b - d.b) ** 2) / (2 * DIP_WIDTH ** 2));
  }
  return total;
};
const dipsGradient = (a, b) => {
  let ga = 0.3 * a;
  let gb = 0.3 * b;
  for (const d of DIPS) {
    const pull = d.depth * Math.exp(-((a - d.a) ** 2 + (b - d.b) ** 2) / (2 * DIP_WIDTH ** 2)) / DIP_WIDTH ** 2;
    ga += pull * (a - d.a);
    gb += pull * (b - d.b);
  }
  return [ga, gb];
};

// The bottom of each dip is not exactly at its centre (the dips lean on each
// other), so find each one by walking downhill from the centre.
function dipSpots() {
  const bottoms = DIPS.map((d) => {
    let at = [d.a, d.b];
    for (let i = 0; i < 2000; i++) {
      const g = dipsGradient(at[0], at[1]);
      at = [at[0] - 0.05 * g[0], at[1] - 0.05 * g[1]];
    }
    return at;
  });
  const lowest = Math.min(...bottoms.map((at) => rawDips(at[0], at[1])));
  return { lowest, spots: bottoms.map((at) => ({ at, kind: rawDips(at[0], at[1]) === lowest ? 'lowest' : 'dip' })) };
}
const DIP_SPOTS = dipSpots();
const dipsLoss = (a, b) => rawDips(a, b) - DIP_SPOTS.lowest;

export const SURFACES = {
  bowl: {
    label: 'Bowl',
    loss: (a, b) => 0.5 * a * a + 0.5 * b * b,
    gradient: (a, b) => [a, b],
    spots: [{ at: [0, 0], kind: 'lowest' }],
  },
  valley: {
    label: 'Narrow valley',
    loss: (a, b) => 0.05 * a * a + b * b,
    gradient: (a, b) => [0.1 * a, 2 * b],
    spots: [{ at: [0, 0], kind: 'lowest' }],
  },
  saddle: {
    label: 'Saddle',
    loss: (a, b) => 0.5 * a * a + 0.25 * b ** 4 - 0.5 * b * b + 0.25,
    gradient: (a, b) => [a, b ** 3 - b],
    spots: [
      { at: [0, 1], kind: 'lowest' },
      { at: [0, -1], kind: 'lowest' },
      { at: [0, 0], kind: 'saddle' },
    ],
  },
  dips: {
    label: 'Several dips',
    loss: dipsLoss,
    gradient: dipsGradient,
    spots: DIP_SPOTS.spots,
  },
};

// Which flat spot, if any, a point is sitting on.
export function spotAt(surface, [a, b], within = 0.08) {
  for (const spot of surface.spots) {
    if (Math.hypot(a - spot.at[0], b - spot.at[1]) <= within) return spot;
  }
  return null;
}

// Gradient descent from a starting point on a surface. The same shape as
// the one-weight version, so the shared training engine can run it:
//   step()    one step: { done, snapshot }
//   finish()  all remaining steps at once
// A snapshot is { step, w: [a, b], loss, gradient: [ga, gb], move, outcome,
// spot }. `move` is the distance covered by the step that got here.
// `outcome` is null while it is still going, then one of:
//   'settled'   the steps became too small to matter (see `spot` for where)
//   'diverged'  the loss was still growing when it stopped: steps too big
//   'bouncing'  out of steps while doubling back on itself: steps too big
//   'limit'     out of steps while still creeping along: steps too small
export function createDescent2({
  surface, start = [0, 0], learningRate = 0.3, optimiser = 'gd', maxSteps = 200, tolerance = 1e-4,
}) {
  const rule = createOptimiser(optimiser, { learningRate });
  let w = [...start];
  let outcome = null;
  let lastStep = null; // the previous step as a direction, to spot doubling back
  const record = (move) => ({
    step: history.length,
    w: [...w],
    loss: surface.loss(w[0], w[1]),
    gradient: surface.gradient(w[0], w[1]),
    move,
    outcome,
    spot: outcome === 'settled' ? spotAt(surface, w) : null,
  });
  const history = [];
  history.push(record(null));

  function step() {
    if (outcome) return { done: true, snapshot: history[history.length - 1] };
    const next = rule.step(w, surface.gradient(w[0], w[1]));
    const delta = [next[0] - w[0], next[1] - w[1]];
    const move = Math.hypot(delta[0], delta[1]);
    const before = history[history.length - 1];
    w = next;
    const here = surface.loss(w[0], w[1]);
    if (!Number.isFinite(here) || Math.abs(w[0]) > 1e3 || Math.abs(w[1]) > 1e3) outcome = 'diverged';
    else if (move < tolerance) outcome = 'settled';
    else if (history.length >= maxSteps) {
      if (here > before.loss) outcome = 'diverged';
      else if (lastStep && delta[0] * lastStep[0] + delta[1] * lastStep[1] < 0) outcome = 'bouncing';
      else outcome = 'limit';
    }
    lastStep = delta;
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
