// Run with: node --test
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { createRng } from '../assets/js/rng.js';
import {
  DATASET_TYPES, createDataset, buildPoints, morePoints, samplePoints,
  jitterPoints, ADD_MORE_COUNT,
} from '../labs/ml/js/datasets.js';
import { assignSplit, countPoints } from '../labs/ml/js/metrics.js';

const PREMADE = DATASET_TYPES.map((t) => t.key).filter((k) => k !== 'custom');

test('the RNG gives the same sequence for the same seed and stream', () => {
  const a = createRng(42, 'points');
  const b = createRng(42, 'points');
  for (let i = 0; i < 100; i++) assert.equal(a.next(), b.next());
  assert.equal(a.gaussian(), b.gaussian());
  assert.deepEqual(a.shuffle([1, 2, 3, 4, 5, 6]), b.shuffle([1, 2, 3, 4, 5, 6]));
});

test('different seeds and different streams give different sequences', () => {
  assert.notEqual(createRng(1, 'points').next(), createRng(2, 'points').next());
  assert.notEqual(createRng(1, 'points').next(), createRng(1, 'split').next());
});

test('gaussian draws have roughly mean 0 and standard deviation 1', () => {
  const rng = createRng(7);
  const n = 20000;
  let sum = 0;
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    const z = rng.gaussian();
    sum += z;
    sumSq += z * z;
  }
  const mean = sum / n;
  const sd = Math.sqrt(sumSq / n - mean * mean);
  assert.ok(Math.abs(mean) < 0.03, `mean ${mean}`);
  assert.ok(Math.abs(sd - 1) < 0.03, `sd ${sd}`);
});

for (const type of PREMADE) {
  test(`${type}: the same seed reproduces the same dataset`, () => {
    const first = buildPoints(createDataset(type, 42));
    const second = buildPoints(createDataset(type, 42));
    assert.equal(first.length, 200);
    assert.deepEqual(first, second);
  });

  test(`${type}: a different seed gives a different dataset`, () => {
    const first = buildPoints(createDataset(type, 42));
    const second = buildPoints(createDataset(type, 43));
    assert.notDeepEqual(first, second);
  });

  test(`${type}: points stay inside the plot and both classes appear`, () => {
    for (const seed of [1, 2, 3, 42, 999]) {
      const points = buildPoints(createDataset(type, seed), 200, 2);
      for (const p of points) {
        assert.ok(Math.abs(p.x1) <= 1 && Math.abs(p.x2) <= 1);
        assert.ok(p.y === 0 || p.y === 1);
      }
      const counts = countPoints(points);
      assert.ok(counts.red >= 20 && counts.blue >= 20,
        `seed ${seed}: ${counts.red} red, ${counts.blue} blue`);
    }
  });

  test(`${type}: "Add more points" uses the same generator settings`, () => {
    const dataset = createDataset(type, 42);
    const settingsBefore = JSON.stringify(dataset.settings);
    const existing = buildPoints(dataset);
    const added = morePoints(dataset, existing, 1);

    assert.equal(added.length, ADD_MORE_COUNT);
    // The settings are untouched ...
    assert.equal(JSON.stringify(dataset.settings), settingsBefore);
    // ... and the new points are exactly what the same generator, with the
    // same settings, gives for that batch's random stream.
    const expected = samplePoints(createDataset(type, 42), ADD_MORE_COUNT, 1,
      createRng(42, 'extra-1'));
    assert.deepEqual(added.map(({ x1, x2, y }) => ({ x1, x2, y })), expected);
    // A second press gives different points.
    assert.notDeepEqual(morePoints(dataset, existing, 2), added);
  });
}

test('changing the number of points keeps the first points the same', () => {
  const dataset = createDataset('moons', 5);
  const small = buildPoints(dataset, 50);
  const large = buildPoints(dataset, 300);
  const strip = ({ x1, x2, y }) => ({ x1, x2, y });
  assert.deepEqual(small.map(strip), large.slice(0, 50).map(strip));
});

test('mystery picks a real pattern and hides nothing from the code', () => {
  const kinds = new Set();
  for (let seed = 0; seed < 200; seed++) kinds.add(createDataset('mystery', seed).kind);
  assert.ok(kinds.has('blobs'));
  assert.ok(kinds.size >= 5, `only saw ${[...kinds]}`);
  assert.ok(!kinds.has('noise') && !kinds.has('custom'));
});

test('the split marks 30% of points as test by default', () => {
  const points = buildPoints(createDataset('clouds', 42));
  assert.equal(countPoints(points).test, 60);
  const batch = assignSplit(Array.from({ length: 20 }, () => ({})), 0.7, createRng(1));
  assert.equal(batch.filter((p) => p.test).length, 6);
});

test('custom datasets start empty and grow by copying nearby points', () => {
  const dataset = createDataset('custom', 42);
  assert.deepEqual(buildPoints(dataset), []);
  assert.deepEqual(morePoints(dataset, [], 1), []);

  const existing = [{ x1: 0.5, x2: 0.5, y: 1 }, { x1: -0.5, x2: -0.5, y: 0 }];
  const added = jitterPoints(existing, 50, createRng(3));
  for (const p of added) {
    const source = existing[p.y === 1 ? 0 : 1];
    assert.ok(Math.hypot(p.x1 - source.x1, p.x2 - source.x2) < 0.4);
  }
});
