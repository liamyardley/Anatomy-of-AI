// Optimisers: the rules for turning a gradient into a step.
// Pure code: no DOM access, so it can be tested in Node.
//
// Training a model with weights means lowering its loss, one step at a time.
// The gradient says which way is uphill for each weight. An optimiser decides
// how far to move each weight, in the downhill direction, given that
// gradient. They all share one shape:
//
//   const optimiser = createOptimiser('adam', { learningRate: 0.1 });
//   weights = optimiser.step(weights, gradient);   // returns new weights
//
// `weights` and `gradient` are arrays of the same length. An optimiser keeps
// its own memory between steps (speed, running averages), so make a new one
// for each training run.

// Plain gradient descent: step straight downhill, in proportion to the slope.
function gradientDescent({ learningRate }) {
  return {
    step: (weights, gradient) => weights.map((w, i) => w - learningRate * gradient[i]),
  };
}

// Momentum: keep some of the previous step's speed, like a ball rolling
// downhill. It speeds up along a steady slope and smooths out zigzags.
function momentum({ learningRate, beta = 0.9 }) {
  let velocity = null;
  return {
    step(weights, gradient) {
      velocity = (velocity || weights.map(() => 0)).map((v, i) => beta * v + gradient[i]);
      return weights.map((w, i) => w - learningRate * velocity[i]);
    },
  };
}

// RMSProp: give each weight its own step size. A weight whose slope has been
// steep lately takes smaller steps; one whose slope has been gentle takes
// larger ones.
function rmsprop({ learningRate, beta = 0.9, epsilon = 1e-8 }) {
  let average = null; // running average of each gradient squared
  return {
    step(weights, gradient) {
      average = (average || weights.map(() => 0))
        .map((a, i) => beta * a + (1 - beta) * gradient[i] ** 2);
      return weights.map((w, i) => w - (learningRate * gradient[i]) / (Math.sqrt(average[i]) + epsilon));
    },
  };
}

// Adam: momentum and RMSProp together. It keeps a running average of the
// gradient (for direction and speed) and of its square (for step size), and
// corrects both for having started at zero.
function adam({ learningRate, beta = 0.9, beta2 = 0.999, epsilon = 1e-8 }) {
  let mean = null;
  let square = null;
  let steps = 0;
  return {
    step(weights, gradient) {
      steps += 1;
      mean = (mean || weights.map(() => 0)).map((m, i) => beta * m + (1 - beta) * gradient[i]);
      square = (square || weights.map(() => 0)).map((s, i) => beta2 * s + (1 - beta2) * gradient[i] ** 2);
      return weights.map((w, i) => {
        const meanNow = mean[i] / (1 - beta ** steps);
        const squareNow = square[i] / (1 - beta2 ** steps);
        return w - (learningRate * meanNow) / (Math.sqrt(squareNow) + epsilon);
      });
    },
  };
}

// The optimisers on offer, in the order a picker should list them.
export const OPTIMISERS = {
  gd: { label: 'Gradient descent', create: gradientDescent },
  momentum: { label: 'Momentum', create: momentum },
  rmsprop: { label: 'RMSProp', create: rmsprop },
  adam: { label: 'Adam', create: adam },
};

// Options: learningRate (needed), and for the others beta, beta2 and epsilon
// (which have the usual defaults).
export function createOptimiser(name = 'gd', options = {}) {
  const entry = OPTIMISERS[name];
  if (!entry) throw new Error(`Unknown optimiser: ${name}`);
  return entry.create({ learningRate: 0.1, ...options });
}
