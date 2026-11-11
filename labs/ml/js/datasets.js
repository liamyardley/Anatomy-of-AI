// Dataset generators. Pure code: no DOM access, so it can be tested in Node.
//
// A point is { x1, x2, y, test } with both features in [-1, 1],
// y = 0 for red and 1 for blue, and test = true for held-back test points.
//
// Each generator has two parts:
//   settings(rng)                 draws the random shape of this variation
//                                 (angles, centres, radii and so on);
//   sample(settings, noise, rng)  draws ONE point from that shape.
// Keeping the settings separate is what lets "Add more points" draw new
// points from exactly the same pattern.
//
// `noise` is a multiplier: 1 is the default amount, 0 is none, 2 is double.

import { createRng } from './rng.js';
import { assignSplit, DEFAULT_TRAIN_SHARE } from './metrics.js';

const TAU = Math.PI * 2;

// Place `count` centres at random, keeping them at least minDist apart
// where possible.
function placeCentres(rng, count, range, minDist, taken = []) {
  const centres = [];
  for (let i = 0; i < count; i++) {
    let best = null;
    for (let attempt = 0; attempt < 40; attempt++) {
      const c = { x: rng.uniform(-range, range), y: rng.uniform(-range, range) };
      best = c;
      const clear = [...taken, ...centres].every(
        (o) => Math.hypot(o.x - c.x, o.y - c.y) >= minDist);
      if (clear) break;
    }
    centres.push(best);
  }
  return centres;
}

const GENERATORS = {
  // Two Gaussian clouds separated along a diagonal.
  clouds: {
    label: 'Two clouds',
    settings(rng) {
      return {
        angle: Math.PI / 4 + rng.uniform(-0.3, 0.3) + rng.pick([0, Math.PI / 2]),
        separation: rng.uniform(0.28, 0.36),
        spread: rng.uniform(0.18, 0.22),
        stretch: rng.uniform(1, 1.3),
        cx: rng.uniform(-0.08, 0.08),
        cy: rng.uniform(-0.08, 0.08),
        blueSide: rng.pick([-1, 1]),
        blueShare: rng.uniform(0.42, 0.58),
      };
    },
    sample(s, noise, rng) {
      const y = rng.bool(s.blueShare) ? 1 : 0;
      const side = (y === 1 ? 1 : -1) * s.blueSide;
      const dx = Math.cos(s.angle);
      const dy = Math.sin(s.angle);
      const along = rng.gaussian() * s.spread * noise;
      const across = rng.gaussian() * s.spread * s.stretch * noise;
      return {
        x1: s.cx + side * s.separation * dx + along * dx - across * dy,
        x2: s.cy + side * s.separation * dy + along * dy + across * dx,
        y,
      };
    },
  },

  // One class in a central disc, the other in a ring around it.
  ring: {
    label: 'Ring',
    settings(rng) {
      return {
        inner: rng.uniform(0.28, 0.38),
        outer: rng.uniform(0.62, 0.74),
        cx: rng.uniform(-0.08, 0.08),
        cy: rng.uniform(-0.08, 0.08),
        innerClass: rng.pick([0, 1]),
        innerShare: rng.uniform(0.4, 0.55),
      };
    },
    sample(s, noise, rng) {
      const isInner = rng.bool(s.innerShare);
      const angle = rng.uniform(0, TAU);
      const radius = isInner
        ? s.inner * Math.sqrt(rng.next())      // even spread over the disc
        : s.outer + rng.gaussian() * 0.05;
      return {
        x1: s.cx + radius * Math.cos(angle) + rng.gaussian() * 0.07 * noise,
        x2: s.cy + radius * Math.sin(angle) + rng.gaussian() * 0.07 * noise,
        y: isInner ? s.innerClass : 1 - s.innerClass,
      };
    },
  },

  // Four quadrants in alternating classes (the XOR pattern), always
  // axis-aligned, with blurred edges.
  checker: {
    label: 'Checkerboard',
    settings(rng) {
      return {
        cx: rng.uniform(-0.15, 0.15),
        cy: rng.uniform(-0.15, 0.15),
        flip: rng.pick([0, 1]),
      };
    },
    sample(s, noise, rng) {
      const x1 = rng.uniform(-0.92, 0.92);
      const x2 = rng.uniform(-0.92, 0.92);
      // The label is decided from a slightly shaken copy of the position,
      // which blurs the edges between quadrants.
      const right = x1 + rng.gaussian() * 0.07 * noise > s.cx;
      const top = x2 + rng.gaussian() * 0.07 * noise > s.cy;
      const y = (right !== top ? 1 : 0) ^ s.flip;
      return { x1, x2, y };
    },
  },

  // Two interlocking crescents.
  moons: {
    label: 'Moons',
    settings(rng) {
      return {
        rotation: rng.uniform(0, TAU),
        gap: rng.uniform(0.3, 0.55),
        scale: rng.uniform(0.46, 0.52),
        upperClass: rng.pick([0, 1]),
      };
    },
    sample(s, noise, rng) {
      const upper = rng.bool(0.5);
      const t = rng.uniform(0, Math.PI);
      // Unit-sized moons, centred on the origin.
      const mx = (upper ? Math.cos(t) : 1 - Math.cos(t)) - 0.5;
      const my = (upper ? Math.sin(t) : s.gap - Math.sin(t)) - s.gap / 2;
      const cos = Math.cos(s.rotation);
      const sin = Math.sin(s.rotation);
      return {
        x1: s.scale * (mx * cos - my * sin) + rng.gaussian() * 0.085 * noise,
        x2: s.scale * (mx * sin + my * cos) + rng.gaussian() * 0.085 * noise,
        y: upper ? s.upperClass : 1 - s.upperClass,
      };
    },
  },

  // Several small clusters per class, with about 10% of labels flipped.
  islands: {
    label: 'Noisy islands',
    settings(rng) {
      const perClass = rng.int(3, 4);
      const centres = placeCentres(rng, perClass * 2, 0.7, 0.32);
      return {
        islands: centres.map((c, i) => ({
          ...c, y: i % 2, spread: rng.uniform(0.09, 0.13),
        })),
        flipShare: 0.1,
      };
    },
    sample(s, noise, rng) {
      const island = rng.pick(s.islands);
      const spread = island.spread * (0.7 + 0.3 * noise);
      const flipped = rng.bool(Math.min(0.35, s.flipShare * noise));
      return {
        x1: island.x + rng.gaussian() * spread,
        x2: island.y + rng.gaussian() * spread,
        y: flipped ? 1 - island.y : island.y,
      };
    },
  },

  // Two interleaved spirals.
  spiral: {
    label: 'Spiral',
    settings(rng) {
      return {
        turns: rng.uniform(1.2, 1.8),
        phase: rng.uniform(0, TAU),
        direction: rng.pick([-1, 1]),
      };
    },
    sample(s, noise, rng) {
      const y = rng.bool(0.5) ? 1 : 0;
      const t = Math.sqrt(rng.next()); // more points further out, for even density
      const radius = 0.1 + 0.78 * t;
      const angle = s.direction * t * s.turns * TAU + s.phase + y * Math.PI;
      return {
        x1: radius * Math.cos(angle) + rng.gaussian() * 0.045 * noise,
        x2: radius * Math.sin(angle) + rng.gaussian() * 0.045 * noise,
        y,
      };
    },
  },

  // Points scattered evenly, labels assigned by coin toss. Nothing to learn.
  noise: {
    label: 'Pure noise',
    settings() {
      return {};
    },
    sample(s, noise, rng) {
      return {
        x1: rng.uniform(-0.95, 0.95),
        x2: rng.uniform(-0.95, 0.95),
        y: rng.bool(0.5) ? 1 : 0,
      };
    },
  },

  // A few blobs of each colour at random positions and sizes.
  // Only used by the Mystery dataset.
  blobs: {
    label: 'Random blobs',
    settings(rng) {
      const red = placeCentres(rng, rng.int(1, 3), 0.65, 0.3);
      const blue = placeCentres(rng, rng.int(1, 3), 0.65, 0.3, red);
      const blob = (c, y) => ({ ...c, y, spread: rng.uniform(0.08, 0.18) });
      return { blobs: [...red.map((c) => blob(c, 0)), ...blue.map((c) => blob(c, 1))] };
    },
    sample(s, noise, rng) {
      const blob = rng.pick(s.blobs);
      const spread = blob.spread * (0.6 + 0.4 * noise);
      return {
        x1: blob.x + rng.gaussian() * spread,
        x2: blob.y + rng.gaussian() * spread,
        y: blob.y,
      };
    },
  },
};

// The premade types a Mystery dataset can turn out to be.
const MYSTERY_POOL = ['clouds', 'ring', 'checker', 'moons', 'islands', 'spiral'];

// What the dataset picker lists, in order.
export const DATASET_TYPES = [
  ...MYSTERY_POOL.map((key) => ({ key, label: GENERATORS[key].label })),
  { key: 'noise', label: 'Pure noise' },
  { key: 'mystery', label: 'Mystery' },
  { key: 'custom', label: 'Custom' },
];

export const DEFAULT_POINT_COUNT = 200;
export const ADD_MORE_COUNT = 20;

// Describe one variation of a dataset type: its seed and its random settings.
// `type` is what the user picked; `kind` is the generator actually used
// (they differ only for Mystery).
export function createDataset(type, seed) {
  if (type === 'custom') return { type, kind: 'custom', seed, settings: null };
  const rng = createRng(seed, 'settings');
  let kind = type;
  if (type === 'mystery') {
    kind = rng.bool(1 / 3) ? 'blobs' : rng.pick(MYSTERY_POOL);
  }
  if (!GENERATORS[kind]) throw new Error(`Unknown dataset type: ${type}`);
  return { type, kind, seed, settings: GENERATORS[kind].settings(rng) };
}

// The plain name of the pattern behind a dataset (used by Reveal).
export function kindLabel(dataset) {
  return dataset.kind === 'custom' ? 'Custom' : GENERATORS[dataset.kind].label;
}

// Draw n points from a dataset's pattern. A point that lands outside the
// plot is drawn again, so nothing piles up along the edges.
export function samplePoints(dataset, n, noise, rng) {
  const generator = GENERATORS[dataset.kind];
  const points = [];
  for (let i = 0; i < n; i++) {
    let p = null;
    for (let attempt = 0; attempt < 30; attempt++) {
      p = generator.sample(dataset.settings, noise, rng);
      if (Math.abs(p.x1) <= 1 && Math.abs(p.x2) <= 1) break;
    }
    p.x1 = Math.max(-1, Math.min(1, p.x1));
    p.x2 = Math.max(-1, Math.min(1, p.x2));
    points.push(p);
  }
  return points;
}

// "More points like these", for custom and edited datasets: copy a random
// existing point, nudge it a little, and keep its colour.
export function jitterPoints(existing, n, rng) {
  if (existing.length === 0) return [];
  const points = [];
  for (let i = 0; i < n; i++) {
    const source = rng.pick(existing);
    points.push({
      x1: Math.max(-1, Math.min(1, source.x1 + rng.gaussian() * 0.06)),
      x2: Math.max(-1, Math.min(1, source.x2 + rng.gaussian() * 0.06)),
      y: source.y,
    });
  }
  return points;
}

// The full starting set of points for a dataset, split into train and test.
export function buildPoints(dataset, n = DEFAULT_POINT_COUNT, noise = 1,
  trainShare = DEFAULT_TRAIN_SHARE) {
  if (dataset.kind === 'custom') return [];
  const points = samplePoints(dataset, n, noise, createRng(dataset.seed, 'points'));
  return assignSplit(points, trainShare, createRng(dataset.seed, 'split'));
}

// One "Add more points" batch. `batch` counts presses (1, 2, 3 ...), so each
// press gives different points but the whole sequence is still repeatable.
export function morePoints(dataset, existing, batch, count = ADD_MORE_COUNT,
  noise = 1, trainShare = DEFAULT_TRAIN_SHARE) {
  const rng = createRng(dataset.seed, `extra-${batch}`);
  const points = dataset.kind === 'custom'
    ? jitterPoints(existing, count, rng)
    : samplePoints(dataset, count, noise, rng);
  return assignSplit(points, trainShare, rng);
}
