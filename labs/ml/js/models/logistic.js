// Logistic regression, trained by batch gradient descent.
// Pure code: no DOM access, so it can be tested in Node.
//
// Every model in the lab has the same shape:
//
//   const model = createModel(params);
//   const trainer = model.train(X, y);  // X: array of [x1, x2]; y: 0 (red) or 1 (blue)
//   trainer.step();                     // one training step: { done, snapshot }
//   trainer.finish();                   // all remaining steps at once: { done, snapshot }
//   model.predictProba([x1, x2]);       // probability of blue, 0 to 1
//   model.explain(point);               // the model's internal numbers, for the maths panel
//
// The model can be asked for predictions at any moment during training,
// which is how the plot animates.

const sigmoid = (z) => 1 / (1 + Math.exp(-z));

// All polynomial terms x1^a * x2^b with 1 <= a + b <= degree.
// Degree 1 gives x1 and x2; degree 2 adds x1^2, x1*x2 and x2^2; and so on.
export function polynomialTerms(degree) {
  const terms = [];
  for (let total = 1; total <= degree; total++) {
    for (let b = 0; b <= total; b++) terms.push({ a: total - b, b });
  }
  return terms;
}

const termValue = (term, x) => (x[0] ** term.a) * (x[1] ** term.b);

// The term written in TeX, for example x_1^{2} x_2.
function termTex({ a, b }) {
  const part = (name, power) => {
    if (power === 0) return '';
    return power === 1 ? name : `${name}^{${power}}`;
  };
  return [part('x_1', a), part('x_2', b)].filter(Boolean).join(' ');
}

export function createLogistic({
  degree = 1,
  learningRate = 1,
  l2 = 0,                 // regularisation strength; 0 switches it off
  maxIterations = 1000,
  tolerance = 1e-5,       // stop when a step improves the loss by less than this
} = {}) {
  const terms = polynomialTerms(degree);
  let bias = 0;
  let weights = terms.map(() => 0);
  let iteration = 0;
  let loss = null;        // log-loss on the training data
  let history = [];       // the loss after every iteration
  let finished = false;
  let data = null;        // the training data, kept for explain()

  // The score z: the weighted sum before it is squashed into a probability.
  function score(x) {
    let z = bias;
    for (let k = 0; k < terms.length; k++) z += weights[k] * termValue(terms[k], x);
    return z;
  }

  function train(X, y) {
    const n = X.length;
    const features = X.map((x) => terms.map((t) => termValue(t, x)));
    data = { X, y };
    bias = 0;
    weights = terms.map(() => 0);
    iteration = 0;
    finished = n === 0;

    // Probability of blue for training row i, with the current weights.
    const probability = (i) => {
      let z = bias;
      for (let k = 0; k < terms.length; k++) z += weights[k] * features[i][k];
      return sigmoid(z);
    };

    // Log-loss: small when the right colour gets a high probability, large
    // when the model is confidently wrong. What gradient descent actually
    // minimises is the log-loss plus the regularisation penalty.
    let objective = 0;
    function measure() {
      let total = 0;
      for (let i = 0; i < n; i++) {
        const p = Math.min(1 - 1e-12, Math.max(1e-12, probability(i)));
        total -= y[i] === 1 ? Math.log(p) : Math.log(1 - p);
      }
      loss = n === 0 ? 0 : total / n;
      objective = loss + (l2 / 2) * weights.reduce((sum, w) => sum + w * w, 0);
    }
    measure();
    history = [loss];

    const snapshot = () => ({ iteration, loss, maxIterations, done: finished });

    // One iteration of gradient descent: work out which way to nudge each
    // weight to lower the loss, then nudge them all by a small amount.
    function step() {
      if (finished) return { done: true, snapshot: snapshot() };

      let biasGradient = 0;
      const gradient = terms.map(() => 0);
      for (let i = 0; i < n; i++) {
        const error = probability(i) - y[i];
        biasGradient += error;
        for (let k = 0; k < terms.length; k++) gradient[k] += error * features[i][k];
      }
      bias -= learningRate * (biasGradient / n);
      for (let k = 0; k < terms.length; k++) {
        // The bias is not regularised; every other weight is pulled towards 0.
        weights[k] -= learningRate * (gradient[k] / n + l2 * weights[k]);
      }

      iteration += 1;
      const before = objective;
      measure();
      history.push(loss);
      const gain = before - objective;
      const settled = iteration >= 10 && gain >= 0 && gain < tolerance;
      if (settled || iteration >= maxIterations) finished = true;
      return { done: finished, snapshot: snapshot() };
    }

    function finish() {
      let result = step();
      while (!result.done) result = step();
      return result;
    }

    return { step, finish };
  }

  function predictProba(x) {
    return sigmoid(score(x));
  }

  // The model's internal numbers, plus the full calculation for one point.
  function explain(point = null) {
    const all = [
      { tex: '', weight: bias },                               // the bias, w0
      ...terms.map((t, k) => ({ tex: termTex(t), weight: weights[k] })),
    ];
    let example = null;
    if (point) {
      const rows = all.map((t, k) => {
        const value = k === 0 ? 1 : termValue(terms[k - 1], point);
        return { tex: t.tex, value, weight: t.weight, product: value * t.weight };
      });
      const z = rows.reduce((sum, r) => sum + r.product, 0);
      example = { x: point, rows, z, p: sigmoid(z) };
    }
    return {
      model: 'logistic',
      degree,
      l2,
      trained: iteration > 0,
      finished,
      iteration,
      maxIterations,
      loss,
      lossHistory: history,
      terms: all,
      // Where every training point sits on the sigmoid curve.
      trainingScores: data ? data.X.map((x, i) => ({ z: score(x), y: data.y[i] })) : [],
      example,
    };
  }

  return { train, predictProba, explain };
}
