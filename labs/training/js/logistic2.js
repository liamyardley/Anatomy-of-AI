// The real thing: the loss surface of a logistic regression on a dataset
// from the Machine Learning Lab.
// Pure code: no DOM access, so it can be tested in Node.
//
// The Machine Learning Lab's logistic regression has three numbers to find
// for two features: a weight for each feature and a bias. Three numbers
// cannot be drawn on a flat map, so this version drops the bias: its
// boundary is a straight line that always passes through the centre of the
// data, and only the two weights are learned. Everything else is the same
// model, the same loss (log-loss) and the same gradient descent.

const sigmoid = (z) => 1 / (1 + Math.exp(-z));

// Move the points so that the middle of the two colours sits at (0, 0).
// Returns { X, y, centre } with X as [x1, x2] pairs and y as 0 (red) or
// 1 (blue).
export function centreData(points) {
  const mean = (label, key) => {
    const group = points.filter((p) => p.y === label);
    return group.length ? group.reduce((sum, p) => sum + p[key], 0) / group.length : 0;
  };
  const centre = [(mean(0, 'x1') + mean(1, 'x1')) / 2, (mean(0, 'x2') + mean(1, 'x2')) / 2];
  return {
    X: points.map((p) => [p.x1 - centre[0], p.x2 - centre[1]]),
    y: points.map((p) => p.y),
    centre,
  };
}

// The probability of blue for one (centred) point.
export const probability = (w, x) => sigmoid(w[0] * x[0] + w[1] * x[1]);

// The share of points on the right side of the line.
export function accuracy(w, X, y) {
  if (X.length === 0) return 0;
  let right = 0;
  for (let i = 0; i < X.length; i++) {
    if ((probability(w, X[i]) >= 0.5 ? 1 : 0) === y[i]) right += 1;
  }
  return right / X.length;
}

// The log-loss of the data as a surface over the two weights:
// { loss(a, b), gradient(a, b), spots, best }.
// `best` is the lowest point, or null when there is none within reach: if a
// straight line can separate the colours perfectly, the loss keeps falling
// for ever as the weights grow.
export function logisticSurface(X, y, reach = 30) {
  const n = X.length;
  const loss = (a, b) => {
    let total = 0;
    for (let i = 0; i < n; i++) {
      const z = a * X[i][0] + b * X[i][1];
      // log(1 + e^z) - y z, written so that large z cannot overflow.
      total += Math.max(z, 0) + Math.log(1 + Math.exp(-Math.abs(z))) - y[i] * z;
    }
    return n === 0 ? 0 : total / n;
  };
  const gradient = (a, b) => {
    let ga = 0;
    let gb = 0;
    for (let i = 0; i < n; i++) {
      const error = sigmoid(a * X[i][0] + b * X[i][1]) - y[i];
      ga += error * X[i][0];
      gb += error * X[i][1];
    }
    return n === 0 ? [0, 0] : [ga / n, gb / n];
  };

  // Find the lowest point by Newton's method: a few steps, each solving for
  // where a bowl fitted to the surface at this spot would be lowest.
  let w = [0, 0];
  let best = null;
  for (let i = 0; i < 60 && n > 0; i++) {
    const [ga, gb] = gradient(w[0], w[1]);
    let haa = 1e-9; let hab = 0; let hbb = 1e-9;
    for (let k = 0; k < n; k++) {
      const p = sigmoid(w[0] * X[k][0] + w[1] * X[k][1]);
      const s = (p * (1 - p)) / n;
      haa += s * X[k][0] * X[k][0];
      hab += s * X[k][0] * X[k][1];
      hbb += s * X[k][1] * X[k][1];
    }
    const det = haa * hbb - hab * hab;
    if (!(det > 0)) break;
    const step = [(hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det];
    w = [w[0] - step[0], w[1] - step[1]];
    if (!Number.isFinite(w[0]) || Math.hypot(w[0], w[1]) > reach) break;
    if (Math.hypot(step[0], step[1]) < 1e-10) { best = w; break; }
  }
  return { loss, gradient, best, spots: best ? [{ at: best, kind: 'lowest' }] : [] };
}
