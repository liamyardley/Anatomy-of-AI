// Run with: node --test
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { createOptimiser, OPTIMISERS } from '../assets/js/optimisers.js';
import {
  makeLineData, loss, gradient, bestWeight, createDescent,
} from '../labs/training/js/oneweight.js';

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
