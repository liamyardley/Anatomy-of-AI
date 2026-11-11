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
