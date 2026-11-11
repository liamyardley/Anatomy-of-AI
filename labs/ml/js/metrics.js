// Train/test split. Accuracy and the confusion matrix join this file in the
// next phase, once there is a model to measure.
//
// A model learns from the training points only. The test points are held
// back, so they show how the model copes with data it has not seen.

export const DEFAULT_TRAIN_SHARE = 0.7;

// Mark a batch of points as training or test. Exactly round(n * (1 - share))
// of them become test points, chosen at random.
export function assignSplit(points, trainShare, rng) {
  const order = rng.shuffle(points.map((_, i) => i));
  const testCount = Math.round(points.length * (1 - trainShare));
  order.forEach((index, position) => {
    points[index].test = position < testCount;
  });
  return points;
}

// For a single point added by hand: test with probability (1 - share).
export function drawIsTest(trainShare, rng) {
  return rng.bool(1 - trainShare);
}

// Counts used by the status line under the plot.
export function countPoints(points) {
  const counts = { total: points.length, train: 0, test: 0, red: 0, blue: 0 };
  for (const p of points) {
    if (p.test) counts.test++; else counts.train++;
    if (p.y === 1) counts.blue++; else counts.red++;
  }
  return counts;
}

// --- measuring a trained model ---
//
// Blue is the positive class. A point is predicted blue when its probability
// of blue is at least the decision threshold.

// Count the four outcomes: tp (blue, predicted blue), fn (blue, predicted
// red), fp (red, predicted blue) and tn (red, predicted red).
export function confusion(probabilities, labels, threshold) {
  const counts = { tp: 0, fn: 0, fp: 0, tn: 0 };
  for (let i = 0; i < labels.length; i++) {
    const predictedBlue = probabilities[i] >= threshold;
    if (labels[i] === 1) counts[predictedBlue ? 'tp' : 'fn']++;
    else counts[predictedBlue ? 'fp' : 'tn']++;
  }
  return counts;
}

// Accuracy: share of all points predicted correctly.
// Precision: share of the points predicted blue that really are blue.
// Recall: share of blue points correctly caught. Sensitivity is another
//   name for exactly the same thing, so both are returned.
// Specificity: share of red points correctly left alone.
// F1: one figure that balances precision and recall (their harmonic mean).
// A value is null when there is nothing to measure it on.
export function summarise({ tp, fn, fp, tn }) {
  const n = tp + fn + fp + tn;
  const recall = tp + fn ? tp / (tp + fn) : null;
  return {
    n,
    accuracy: n ? (tp + tn) / n : null,
    precision: tp + fp ? tp / (tp + fp) : null,
    recall,
    sensitivity: recall,
    specificity: tn + fp ? tn / (tn + fp) : null,
    f1: 2 * tp + fp + fn ? (2 * tp) / (2 * tp + fp + fn) : null,
  };
}
