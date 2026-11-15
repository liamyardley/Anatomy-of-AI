// Seeded random numbers.
//
// Everything random in the lab (dataset shapes, noise, the train/test split,
// and later bootstrap samples and feature sampling) comes from here, so the
// same seed always gives the same result.

// mulberry32: a tiny, fast generator with a 32-bit state.
// Returns a function that gives a number in [0, 1) each time it is called.
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// FNV-1a hash: turns a string into a 32-bit number.
export function hashString(str) {
  let h = 0x811C9DC5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

// A random number generator with helper methods.
//
// `name` selects an independent stream for the same seed. The lab uses one
// stream for a dataset's settings, another for its points, another for the
// train/test split, and so on. That way, changing the number of points does
// not change the shape of the dataset, and adding points never disturbs the
// ones already there.
export function createRng(seed, name = '') {
  const next = mulberry32(hashString(`${name}:${Math.floor(seed)}`));
  let spare = null; // Box-Muller makes two values at a time; keep the second.

  const rng = {
    // A number in [0, 1).
    next,

    // A number in [min, max).
    uniform(min = 0, max = 1) {
      return min + (max - min) * next();
    },

    // A whole number from min to max, both included.
    int(min, max) {
      return min + Math.floor(next() * (max - min + 1));
    },

    // True with probability p.
    bool(p = 0.5) {
      return next() < p;
    },

    // A draw from a normal (bell curve) distribution.
    gaussian(mean = 0, sd = 1) {
      if (spare !== null) {
        const z = spare;
        spare = null;
        return mean + sd * z;
      }
      let u = 0;
      while (u === 0) u = next(); // avoid log(0)
      const v = next();
      const r = Math.sqrt(-2 * Math.log(u));
      spare = r * Math.sin(2 * Math.PI * v);
      return mean + sd * r * Math.cos(2 * Math.PI * v);
    },

    // One item from an array.
    pick(items) {
      return items[Math.floor(next() * items.length)];
    },

    // Shuffle an array in place (Fisher-Yates) and return it.
    shuffle(items) {
      for (let i = items.length - 1; i > 0; i--) {
        const j = Math.floor(next() * (i + 1));
        [items[i], items[j]] = [items[j], items[i]];
      }
      return items;
    },
  };
  return rng;
}
