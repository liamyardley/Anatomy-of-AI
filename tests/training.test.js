// Run with: node --test
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { createOptimiser, OPTIMISERS } from '../assets/js/optimisers.js';
import {
  makeLineData, loss, gradient, bestWeight, createDescent,
} from '../labs/training/js/oneweight.js';
import { SURFACES, RANGE, spotAt, createDescent2, createRace } from '../labs/training/js/twoweights.js';
import { makeScatter, lineSurface, gradientOn, createBatchRace } from '../labs/training/js/batches.js';

const close = (a, b, tolerance = 1e-9) => Math.abs(a - b) < tolerance;

// ===== optimisers =====

test('gradient descent: a step is the learning rate times the gradient, downhill', () => {
  const rule = createOptimiser('gd', { learningRate: 0.1 });
  assert.deepEqual(rule.step([1, -2], [4, -10]), [0.6, -1]);
});

test('momentum: the second step carries speed over from the first', () => {
  const rule = createOptimiser('momentum', { learningRate: 0.1, beta: 0.9 });
  const first = rule.step([0], [1]);
  assert.ok(close(first[0], -0.1));
  // speed is now 0.9 * 1 + 1 = 1.9
  const second = rule.step(first, [1]);
  assert.ok(close(second[0], -0.1 - 0.19));
});

test('RMSProp and Adam: the first step is about one learning rate, whatever the slope', () => {
  for (const size of [0.001, 1, 1000]) {
    const adam = createOptimiser('adam', { learningRate: 0.1 }).step([0], [size]);
    assert.ok(close(adam[0], -0.1, 1e-4), `adam ${size}`);
    // RMSProp's first step is the learning rate divided by the square root of (1 - beta).
    const rms = createOptimiser('rmsprop', { learningRate: 0.1, beta: 0.9 }).step([0], [size]);
    assert.ok(close(rms[0], -0.1 / Math.sqrt(0.1), 1e-3), `rmsprop ${size}`);
  }
});

test('every optimiser finds the bottom of a simple bowl', () => {
  // f(a, b) = (a - 3)^2 + 2 (b + 1)^2, lowest at (3, -1).
  const slope = ([a, b]) => [2 * (a - 3), 4 * (b + 1)];
  for (const name of Object.keys(OPTIMISERS)) {
    const rule = createOptimiser(name, { learningRate: name === 'gd' ? 0.1 : 0.05 });
    let at = [0, 0];
    for (let i = 0; i < 2000; i++) at = rule.step(at, slope(at));
    assert.ok(Math.abs(at[0] - 3) < 0.05 && Math.abs(at[1] + 1) < 0.05, `${name}: ${at}`);
  }
});

test('an unknown optimiser is an error, and each run starts with a clean memory', () => {
  assert.throws(() => createOptimiser('nope'));
  const a = createOptimiser('momentum', { learningRate: 0.1 });
  a.step([0], [5]);
  const b = createOptimiser('momentum', { learningRate: 0.1 });
  assert.ok(close(b.step([0], [1])[0], -0.1));
});

// ===== one weight =====

test('line data: same seed, same points; the loss curve is the same width every time', () => {
  assert.deepEqual(makeLineData(42), makeLineData(42));
  assert.notDeepEqual(makeLineData(42).points, makeLineData(43).points);
  for (const seed of [1, 2, 3, 42, 999]) {
    const { points } = makeLineData(seed);
    assert.equal(points.length, 10);
    const meanSquare = points.reduce((sum, p) => sum + p.x * p.x, 0) / points.length;
    assert.ok(close(meanSquare, 0.5));
    for (const p of points) assert.ok(p.x > 0 && p.x < 1.5 && p.y > 0 && p.y < 4.2);
  }
});

test('loss and gradient: the gradient is the slope of the loss, and is zero at the best weight', () => {
  const { points } = makeLineData(7);
  const best = bestWeight(points);
  assert.ok(Math.abs(gradient(best, points)) < 1e-12);
  assert.ok(loss(best, points) < loss(best + 0.1, points));
  assert.ok(loss(best, points) < loss(best - 0.1, points));
  // Compare with the slope measured directly.
  for (const w of [-0.3, 0.5, 2, 3.5]) {
    const measured = (loss(w + 1e-6, points) - loss(w - 1e-6, points)) / 2e-6;
    assert.ok(close(gradient(w, points), measured, 1e-6));
  }
  // Left of the best weight the loss slopes down (negative); right of it, up.
  assert.ok(gradient(best - 1, points) < 0);
  assert.ok(gradient(best + 1, points) > 0);
});

test('gradient descent: a learning rate of 1 lands on the best weight in one step', () => {
  const { points } = makeLineData(5);
  const run = createDescent({ points, start: 0, learningRate: 1 });
  const first = run.step();
  assert.ok(close(first.snapshot.w, bestWeight(points)));
  assert.equal(run.step().snapshot.outcome, 'settled');
});

test('gradient descent: small rates crawl, rates between 1 and 2 overshoot but settle, above 2 blows up', () => {
  const { points } = makeLineData(5);
  const best = bestWeight(points);
  const finish = (learningRate) => {
    const run = createDescent({ points, start: 0, learningRate });
    return { end: run.finish().snapshot, history: run.history };
  };
  const crawl = finish(0.01);
  assert.equal(crawl.end.outcome, 'limit');
  assert.ok(crawl.end.w < best - 0.1);

  const steady = finish(0.3);
  assert.equal(steady.end.outcome, 'settled');
  assert.ok(Math.abs(steady.end.w - best) < 1e-3);
  // Never overshoots: every weight stays on the starting side.
  assert.ok(steady.history.every((h) => h.w <= best + 1e-9));

  const bouncing = finish(1.5);
  assert.equal(bouncing.end.outcome, 'settled');
  assert.ok(bouncing.history[1].w > best && bouncing.history[2].w < best); // jumps from side to side
  assert.ok(Math.abs(bouncing.end.w - best) < 1e-3);

  const blown = finish(2.3);
  assert.equal(blown.end.outcome, 'diverged');
  assert.ok(blown.history[3].loss > blown.history[1].loss);
});

test('gradient descent: running out of steps says why', () => {
  const { points } = makeLineData(5);
  const end = (learningRate, start = 0) => createDescent({ points, start, learningRate }).finish().snapshot;
  // Just above 2 the loss grows too slowly to blow up within the limit,
  // but it is still growing: that is a rate that is too large, not too small.
  assert.equal(end(2.05).outcome, 'diverged');
  assert.equal(end(2.05).step, 60);
  // Just below 2 it is closing in, jumping from side to side.
  assert.equal(end(1.9).outcome, 'bouncing');
  // A tiny rate is still creeping from one side.
  assert.equal(end(0.01).outcome, 'limit');
});

test('gradient descent: a start further from the bottom needs more steps', () => {
  const { points } = makeLineData(5);
  const best = bestWeight(points);
  const steps = (start) => createDescent({ points, start, learningRate: 0.3 }).finish().snapshot.step;
  assert.ok(steps(best + 2) > steps(best + 0.2));
  assert.ok(steps(best + 0.2) > steps(best + 0.001));
});

test('gradient descent: stepping to the end equals finish, and the history adds up', () => {
  const { points } = makeLineData(11);
  const stepped = createDescent({ points, start: 3.5, learningRate: 0.6 });
  let result = stepped.step();
  while (!result.done) result = stepped.step();
  const whole = createDescent({ points, start: 3.5, learningRate: 0.6 });
  assert.deepEqual(whole.finish().snapshot, result.snapshot);
  stepped.history.forEach((entry, i) => {
    assert.equal(entry.step, i);
    if (i > 0) assert.ok(close(entry.w - stepped.history[i - 1].w, entry.move));
    assert.ok(close(entry.loss, loss(entry.w, points)));
  });
});

// ===== two weights =====

const run2 = (name, start, learningRate) => {
  const run = createDescent2({ surface: SURFACES[name], start, learningRate });
  return { end: run.finish().snapshot, history: run.history };
};

test('surfaces: each gradient is the slope of its loss, and is zero at every flat spot', () => {
  for (const [name, surface] of Object.entries(SURFACES)) {
    for (const [a, b] of [[-1.5, 0.3], [0.4, -1.2], [1.7, 1.1], [0, 0.5]]) {
      const [ga, gb] = surface.gradient(a, b);
      const ma = (surface.loss(a + 1e-6, b) - surface.loss(a - 1e-6, b)) / 2e-6;
      const mb = (surface.loss(a, b + 1e-6) - surface.loss(a, b - 1e-6)) / 2e-6;
      assert.ok(close(ga, ma, 1e-5) && close(gb, mb, 1e-5), `${name} at ${a}, ${b}`);
    }
    for (const spot of surface.spots) {
      const [ga, gb] = surface.gradient(spot.at[0], spot.at[1]);
      assert.ok(Math.hypot(ga, gb) < 1e-6, `${name} spot ${spot.at}`);
      assert.ok(Math.abs(spot.at[0]) < RANGE && Math.abs(spot.at[1]) < RANGE);
    }
  }
});

test('several dips: three separate dips, one of them the lowest', () => {
  const { spots, loss: f } = SURFACES.dips;
  assert.equal(spots.length, 3);
  assert.equal(spots.filter((s) => s.kind === 'lowest').length, 1);
  for (let i = 0; i < 3; i++) {
    for (let j = i + 1; j < 3; j++) {
      assert.ok(Math.hypot(spots[i].at[0] - spots[j].at[0], spots[i].at[1] - spots[j].at[1]) > 1);
    }
    // A real dip: lower than the ground just around it.
    const [a, b] = spots[i].at;
    for (const [da, db] of [[0.1, 0], [-0.1, 0], [0, 0.1], [0, -0.1]]) assert.ok(f(a, b) < f(a + da, b + db));
  }
});

test('bowl: a learning rate of 1 lands on the bottom in one step, from anywhere', () => {
  const { end, history } = run2('bowl', [1.6, -1.2], 1);
  assert.ok(Math.hypot(...history[1].w) < 1e-12);
  assert.equal(end.outcome, 'settled');
  assert.equal(end.spot.kind, 'lowest');
});

test('narrow valley: the steep sides limit the rate, the flat floor makes it slow', () => {
  // Above 1 the steep direction blows up.
  assert.equal(run2('valley', [1.5, 1], 1.5).end.outcome, 'diverged');
  // A safe small rate is still creeping along the floor when the steps run out.
  const slow = run2('valley', [1.8, 1], 0.1).end;
  assert.equal(slow.outcome, 'limit');
  assert.ok(Math.abs(slow.w[1]) < 1e-6 && slow.w[0] > 0.1);
  // Close to the limit it zigzags across the valley but gets there.
  const zigzag = run2('valley', [1.8, 1], 0.9);
  assert.equal(zigzag.end.outcome, 'settled');
  assert.ok(zigzag.history[1].w[1] < 0 && zigzag.history[2].w[1] > 0);
});

test('saddle: a start on the ridge stops on the saddle; a start just off it reaches a dip', () => {
  const stuck = run2('saddle', [1.5, 0], 0.3).end;
  assert.equal(stuck.outcome, 'settled');
  assert.equal(stuck.spot.kind, 'saddle');
  const up = run2('saddle', [1.5, 0.05], 0.3).end;
  assert.equal(up.spot.kind, 'lowest');
  assert.ok(close(up.w[1], 1, 1e-2));
  const down = run2('saddle', [1.5, -0.05], 0.3).end;
  assert.ok(close(down.w[1], -1, 1e-2));
});

test('several dips: the start decides which dip gradient descent finds', () => {
  const kinds = new Set();
  const ends = new Set();
  for (const start of [[-1.5, -1.5], [1.5, 1], [-1, 1.7], [1.8, -1.5], [-1.8, 0.4]]) {
    const { end } = run2('dips', start, 0.2);
    assert.equal(end.outcome, 'settled', `${start}`);
    assert.ok(end.spot, `${start} ended at ${end.w}`);
    kinds.add(end.spot.kind);
    ends.add(end.spot.at.join());
  }
  assert.ok(kinds.has('lowest') && kinds.has('dip'));
  assert.equal(ends.size, 3);
});

test('two weights: stepping to the end equals finish, and spotAt only fires near a spot', () => {
  const surface = SURFACES.dips;
  const stepped = createDescent2({ surface, start: [0.2, -1.6], learningRate: 0.2 });
  let result = stepped.step();
  while (!result.done) result = stepped.step();
  assert.deepEqual(createDescent2({ surface, start: [0.2, -1.6], learningRate: 0.2 }).finish().snapshot, result.snapshot);
  assert.equal(spotAt(SURFACES.bowl, [0.5, 0.5]), null);
  assert.equal(spotAt(SURFACES.bowl, [0.01, -0.02]).kind, 'lowest');
});

// ===== the optimiser race =====

const RACE_RATES = { gd: 0.3, momentum: 0.1, rmsprop: 0.05, adam: 0.1 };
const race = (name, start, rates = RACE_RATES) => {
  const runners = Object.entries(rates).map(([key, learningRate]) => ({ name: key, learningRate }));
  const end = createRace({ surface: SURFACES[name], start, runners }).finish().snapshot;
  return Object.fromEntries(end.runners.map((r) => [r.name, r]));
};

test('race: on the bowl everyone reaches the bottom, and plain gradient descent is quickest', () => {
  const end = race('bowl', [1.5, 1.2]);
  for (const r of Object.values(end)) {
    assert.equal(r.outcome, 'arrived', r.name);
    assert.equal(r.spot.kind, 'lowest', r.name);
    assert.equal(r.history.length >= r.steps + 1, true);
  }
  assert.ok(end.gd.steps < Math.min(end.momentum.steps, end.rmsprop.steps, end.adam.steps));
});

test('race: on the narrow valley the other three all beat plain gradient descent', () => {
  const end = race('valley', [1.5, 1.2]);
  for (const name of ['momentum', 'rmsprop', 'adam']) {
    assert.equal(end[name].outcome, 'arrived', name);
    assert.ok(end[name].steps < end.gd.steps, `${name} ${end[name].steps} against ${end.gd.steps}`);
  }
});

test('race: momentum can roll through a shallow dip that stops gradient descent', () => {
  const end = race('dips', [-1.6, -1.5]);
  assert.equal(end.gd.spot.kind, 'dip');
  assert.equal(end.momentum.spot.kind, 'lowest');
});

test('race: a rate that is too large is reported, and the others carry on', () => {
  const end = race('valley', [1.5, 1.2], { gd: 1.5, adam: 0.1 });
  assert.equal(end.gd.outcome, 'diverged');
  assert.equal(end.adam.outcome, 'arrived');
});

test('race: stepping to the end equals finish', () => {
  const make = () => createRace({ surface: SURFACES.saddle, start: [1.5, 0.4], runners: [{ name: 'gd', learningRate: 0.3 }, { name: 'adam', learningRate: 0.1 }] });
  const stepped = make();
  let result = stepped.step();
  while (!result.done) result = stepped.step();
  assert.deepEqual(make().finish().snapshot, result.snapshot);
});

// ===== batch size =====

const BATCH_RUNNERS = [{ key: 'full', batchSize: 40 }, { key: 'mini', batchSize: 8 }, { key: 'single', batchSize: 1 }];
const batches = (seed, learningRate = 0.1, passes = 20) => {
  const { points } = makeScatter(seed);
  const run = createBatchRace({ points, start: [-0.5, 2], learningRate, seed, passes, runners: BATCH_RUNNERS });
  const end = run.finish().snapshot;
  return { points, surface: run.surface, end, by: Object.fromEntries(end.runners.map((r) => [r.key, r])) };
};

test('scatter data: same seed, same points; the best line is where the gradient is zero', () => {
  assert.deepEqual(makeScatter(42), makeScatter(42));
  assert.notDeepEqual(makeScatter(42).points, makeScatter(43).points);
  const { points } = makeScatter(42);
  assert.equal(points.length, 40);
  const surface = lineSurface(points);
  const [ga, gb] = surface.gradient(...surface.best);
  assert.ok(Math.hypot(ga, gb) < 1e-9);
  assert.ok(surface.loss(...surface.best) < surface.loss(surface.best[0] + 0.1, surface.best[1]));
  // Measured on every point, the batch gradient is the full gradient.
  assert.deepEqual(gradientOn(0.3, -0.2, points), surface.gradient(0.3, -0.2));
});

test('batch race: everyone sees the same amount of data, so small batches take more steps', () => {
  const { by, end } = batches(42);
  assert.equal(end.pass, 20);
  assert.equal(by.full.updates, 20);
  assert.equal(by.mini.updates, 100);
  assert.equal(by.single.updates, 800);
  for (const r of end.runners) {
    assert.equal(r.outcome, 'done');
    assert.equal(r.history.length, r.updates + 1);
    assert.equal(r.history[r.history.length - 1].seen, 800);
  }
});

test('batch race: full batch is plain gradient descent, and its loss only ever falls', () => {
  const { points, by, surface } = batches(42);
  let w = [-0.5, 2];
  for (let i = 0; i < 20; i++) {
    const [ga, gb] = surface.gradient(w[0], w[1]);
    w = [w[0] - 0.1 * ga, w[1] - 0.1 * gb];
  }
  assert.ok(close(by.full.w[0], w[0]) && close(by.full.w[1], w[1]));
  for (let i = 1; i < by.full.history.length; i++) assert.ok(by.full.history[i].loss < by.full.history[i - 1].loss);
  assert.equal(points.length, 40);
});

test('batch race: single points are noisy but get much further on the same work', () => {
  for (const seed of [42, 7, 3]) {
    const { by, surface } = batches(seed);
    const off = (r) => Math.hypot(r.w[0] - surface.best[0], r.w[1] - surface.best[1]);
    // Noisy: the loss goes up as well as down.
    const rises = by.single.history.filter((h, i) => i > 0 && h.loss > by.single.history[i - 1].loss).length;
    assert.ok(rises > 50, `seed ${seed}: ${rises} rises`);
    // Further: closer to the best line than full batch after the same passes.
    assert.ok(off(by.single) < off(by.full), `seed ${seed}`);
    assert.ok(off(by.mini) < off(by.full), `seed ${seed}`);
  }
});

test('batch race: the shuffles come from the seed, and stepping equals finish', () => {
  assert.deepEqual(batches(42).end, batches(42).end);
  assert.notDeepEqual(batches(42).by.single.w, batches(43).by.single.w);
  const { points } = makeScatter(5);
  const make = () => createBatchRace({ points, start: [0, 0], learningRate: 0.1, seed: 5, passes: 3, runners: BATCH_RUNNERS });
  const stepped = make();
  let result = stepped.step();
  while (!result.done) result = stepped.step();
  assert.deepEqual(make().finish().snapshot, result.snapshot);
});
