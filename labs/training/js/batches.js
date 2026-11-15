// Batch size: how many data points are looked at before each step.
// Pure code: no DOM access, so it can be tested in Node.
//
// The model is a straight line with two weights, y = a * x + b (a is the
// steepness, b is where it crosses the left edge). Its loss is the mean
// squared miss over the data, as in Part 1.
//
// So far every step has used the whole dataset to measure the gradient
// ("full batch"). The alternative is to measure it on a few points at a
// time ("mini-batch"), or on a single point. That gradient is only an
// estimate, so each step is a little off, but it costs far less work, so
// many more steps can be taken for the same effort.
//
//   makeScatter      a seeded dataset that roughly follows a line
//   lineSurface      the loss and gradient of that data, as a map surface
//   createBatchRace  the same start, the same learning rate, different
//                    batch sizes, all given the same amount of data to look at

import { createRng } from '../../../assets/js/rng.js';

// The ranges of the two weights shown on the map.
export const SLOPE_RANGE = [-1, 3];
export const INTERCEPT_RANGE = [-1.5, 2.5];

export function makeScatter(seed, count = 40) {
  const settings = createRng(seed, 'scatter-settings');
  const rng = createRng(seed, 'scatter-points');
  const slope = settings.uniform(0.8, 1.9);
  const intercept = settings.uniform(0.3, 1.1);
  const points = Array.from({ length: count }, () => {
    const x = rng.uniform(0, 2);
    return { x, y: slope * x + intercept + rng.gaussian() * 0.45 };
  });
  return { seed, slope, intercept, points };
}

// The gradient of the mean squared miss, measured on some of the points.
export function gradientOn(a, b, points) {
  let ga = 0;
  let gb = 0;
  for (const p of points) {
    const miss = a * p.x + b - p.y;
    ga += 2 * miss * p.x;
    gb += 2 * miss;
  }
  return [ga / points.length, gb / points.length];
}

// The whole dataset's loss as a surface for the contour map:
// { loss(a, b), gradient(a, b), best: [a, b], spots }.
export function lineSurface(points) {
  const n = points.length;
  let sx = 0; let sy = 0; let sxx = 0; let sxy = 0;
  for (const p of points) { sx += p.x; sy += p.y; sxx += p.x * p.x; sxy += p.x * p.y; }
  // The best line, worked out directly (least squares).
  const a = (n * sxy - sx * sy) / (n * sxx - sx * sx);
  const b = (sy - a * sx) / n;
  const loss = (wa, wb) => {
    let total = 0;
    for (const p of points) total += (wa * p.x + wb - p.y) ** 2;
    return total / n;
  };
  return {
    loss,
    gradient: (wa, wb) => gradientOn(wa, wb, points),
    best: [a, b],
    spots: [{ at: [a, b], kind: 'lowest' }],
  };
}

// Several runs of plain gradient descent from the same start, differing only
// in batch size. Each call to step() hands every runner the same number of
// new data points to look at (`chunk`); a runner takes a step whenever it has
// been handed a whole batch. So after any number of steps they have all done
// the same amount of work, but the small batches have moved many more times.
//
//   runners   [{ key, batchSize }]
//   passes    how many times through the whole dataset before stopping
//
// A snapshot is { seen, pass, runners } where each runner is
// { key, batchSize, w, loss, updates, outcome, history }. `loss` is always
// measured on the whole dataset. `history` has one entry per update:
// { seen, w, loss }. `outcome` is null while going, then 'done' or 'diverged'.
export function createBatchRace({
  points, start = [0, 0], learningRate = 0.1, runners, passes = 20, seed = 0, chunk = 4,
}) {
  const n = points.length;
  const surface = lineSurface(points);
  const total = passes * n;
  let seen = 0;
  const state = runners.map(({ key, batchSize }) => {
    const size = Math.min(batchSize, n);
    return {
      key,
      batchSize: size,
      w: [...start],
      used: 0,
      updates: 0,
      outcome: null,
      order: points.map((_, i) => i),
      rng: createRng(seed, `batches-${key}`),
      history: [{ seen: 0, w: [...start], loss: surface.loss(start[0], start[1]) }],
    };
  });

  function update(r) {
    // A fresh shuffle at the start of every pass, so each point is used
    // exactly once per pass, in a new order.
    const offset = r.used % n;
    if (offset === 0 && r.batchSize < n) r.rng.shuffle(r.order);
    const batch = [];
    for (let i = 0; i < r.batchSize; i++) batch.push(points[r.order[(offset + i) % n]]);
    const [ga, gb] = gradientOn(r.w[0], r.w[1], batch);
    r.w = [r.w[0] - learningRate * ga, r.w[1] - learningRate * gb];
    r.used += r.batchSize;
    r.updates += 1;
    const loss = surface.loss(r.w[0], r.w[1]);
    r.history.push({ seen: r.used, w: [...r.w], loss });
    if (!Number.isFinite(loss) || loss > 1e6) r.outcome = 'diverged';
  }

  const snapshot = () => ({
    seen,
    pass: seen / n,
    runners: state.map((r) => ({
      key: r.key,
      batchSize: r.batchSize,
      w: [...r.w],
      loss: r.history[r.history.length - 1].loss,
      updates: r.updates,
      outcome: r.outcome,
      history: r.history,
    })),
  });

  function step() {
    if (seen >= total) return { done: true, snapshot: snapshot() };
    seen = Math.min(total, seen + chunk);
    for (const r of state) {
      while (!r.outcome && r.used + r.batchSize <= seen) update(r);
    }
    const done = seen >= total || state.every((r) => r.outcome);
    if (done) {
      seen = total;
      for (const r of state) if (!r.outcome) r.outcome = 'done';
    }
    return { done, snapshot: snapshot() };
  }

  function finish() {
    let result = step();
    while (!result.done) result = step();
    return result;
  }

  return { step, finish, surface, snapshot };
}
