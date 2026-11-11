// A decision tree (CART). Pure code: no DOM access, so it can be tested in Node.
//
// The tree asks yes/no questions such as "is x1 < 0.3?". Each question splits
// a group of training points into two groups that are purer (closer to being
// all one colour). A group that is not split any further is a leaf, and a
// leaf's share of blue points is the probability it predicts.
//
// growTree() does the work and is also used by the random forest. createTree()
// wraps it in the lab's common model interface (see logistic.js), where one
// training step is one split.

// Gini impurity: 0 when a group is all one colour, 0.5 when it is half and half.
export function gini(red, blue) {
  const n = red + blue;
  if (n === 0) return 0;
  const p = blue / n;
  return 1 - p * p - (1 - p) * (1 - p);
}

// The probability of blue predicted by a leaf.
export const leafProbability = (leaf) => {
  const n = leaf.red + leaf.blue;
  return n === 0 ? 0.5 : leaf.blue / n;
};

const FEATURE_NAMES = ['x1', 'x2'];

// Grow a tree on the rows of X (features) and y (0 red, 1 blue).
//   maxDepth          how many questions deep the tree may go
//   minLeaf           the fewest training points a leaf may hold
//   featuresPerSplit  2 looks at both features at every split; 1 looks at
//                     one feature chosen at random (used by the forest)
//   rng               needed only when featuresPerSplit is 1
//   rows              which rows to train on (default: all of them; the
//                     forest passes a bootstrap sample, which repeats rows)
// Splits are made one at a time, level by level, by calling step().
export function growTree(X, y, {
  maxDepth = 3, minLeaf = 1, featuresPerSplit = 2, rng = null, rows = null,
} = {}) {
  const nodes = [];
  let splits = 0;
  let lastSplit = null;

  function makeNode(indices, depth, region, parent) {
    let blue = 0;
    for (const i of indices) blue += y[i];
    const red = indices.length - blue;
    const node = {
      id: nodes.length,
      depth,
      indices,              // the training rows that reached this group
      red,
      blue,
      gini: gini(red, blue),
      region,               // the rectangle of the plot this group covers
      parent,
      split: null,          // { feature, threshold, gain } once it has been split
      candidates: null,     // the best split found for each feature
      leafReason: null,     // why it was left as a leaf
      left: null,           // the "yes" group
      right: null,          // the "no" group
    };
    nodes.push(node);
    return node;
  }

  // The best place to cut this group along one feature: try a cut halfway
  // between every pair of neighbouring values and keep the one that lowers
  // the impurity most.
  function bestSplit(node, feature) {
    const order = node.indices.slice().sort((a, b) => X[a][feature] - X[b][feature]);
    const n = order.length;
    let best = null;
    let leftBlue = 0;
    for (let j = 1; j < n; j++) {
      leftBlue += y[order[j - 1]];
      const before = X[order[j - 1]][feature];
      const after = X[order[j]][feature];
      if (before === after) continue; // cannot cut between equal values
      const nLeft = j;
      const nRight = n - j;
      if (nLeft < minLeaf || nRight < minLeaf) continue;
      const left = { red: nLeft - leftBlue, blue: leftBlue };
      const right = { red: node.red - left.red, blue: node.blue - leftBlue };
      const gain = node.gini
        - (nLeft / n) * gini(left.red, left.blue)
        - (nRight / n) * gini(right.red, right.blue);
      if (!best || gain > best.gain) {
        best = { feature, threshold: (before + after) / 2, gain, left, right };
      }
    }
    return best;
  }

  // Decide whether a group should be split, and how. Returns the chosen
  // split, or null if the group stays a leaf.
  function examine(node) {
    if (node.depth >= maxDepth) { node.leafReason = 'depth'; return null; }
    if (node.gini === 0) { node.leafReason = 'pure'; return null; }
    if (node.indices.length < 2 * minLeaf) { node.leafReason = 'small'; return null; }
    const allowed = featuresPerSplit >= 2 ? [0, 1] : [rng.next() < 0.5 ? 0 : 1];
    node.candidates = [0, 1].map((feature) => {
      if (!allowed.includes(feature)) return { feature, considered: false };
      const found = bestSplit(node, feature);
      return found ? { ...found, considered: true } : { feature, considered: true, gain: null };
    });
    let chosen = null;
    for (const c of node.candidates) {
      if (c.considered && c.gain !== null && c.gain > 1e-12 && (!chosen || c.gain > chosen.gain)) chosen = c;
    }
    if (!chosen) node.leafReason = 'nogain';
    return chosen;
  }

  const root = makeNode(rows || X.map((_, i) => i), 0, { x1: [-1, 1], x2: [-1, 1] }, null);
  const queue = [root];

  // Work through the waiting groups until one can be split.
  function findNext() {
    while (queue.length) {
      const node = queue.shift();
      const choice = examine(node);
      if (choice) return { node, choice };
    }
    return null;
  }
  let pending = findNext();

  // Make one split. Returns true when no more splits are possible.
  function step() {
    if (!pending) return true;
    const { node, choice } = pending;
    const { feature, threshold } = choice;
    node.split = { feature, threshold, gain: choice.gain };
    const key = FEATURE_NAMES[feature];
    const [low, high] = node.region[key];
    const yes = node.indices.filter((i) => X[i][feature] < threshold);
    const no = node.indices.filter((i) => X[i][feature] >= threshold);
    node.left = makeNode(yes, node.depth + 1, { ...node.region, [key]: [low, threshold] }, node);
    node.right = makeNode(no, node.depth + 1, { ...node.region, [key]: [threshold, high] }, node);
    queue.push(node.left, node.right);
    splits += 1;
    lastSplit = node;
    pending = findNext();
    return pending === null;
  }

  function finish() {
    while (!step()) { /* keep splitting */ }
  }

  // The leaf a point ends up in, and the questions asked on the way.
  function leafFor(x, path = null) {
    let node = root;
    while (node.split) {
      const yes = x[node.split.feature] < node.split.threshold;
      if (path) path.push({ node, yes });
      node = yes ? node.left : node.right;
    }
    return node;
  }

  return {
    root,
    nodes,
    step,
    finish,
    leafFor,
    get done() { return pending === null; },
    get splits() { return splits; },
    get lastSplit() { return lastSplit; },
  };
}

export function createTree({ maxDepth = 3, minLeaf = 1 } = {}) {
  let grower = null;

  function train(X, y) {
    grower = growTree(X, y, { maxDepth, minLeaf });
    const snapshot = () => {
      const leaves = grower.nodes.filter((n) => !n.split);
      return {
        splits: grower.splits,
        leaves: leaves.length,
        depth: Math.max(...leaves.map((n) => n.depth)),
        done: grower.done,
      };
    };
    function step() {
      const done = grower.step();
      return { done, snapshot: snapshot() };
    }
    function finish() {
      grower.finish();
      return { done: true, snapshot: snapshot() };
    }
    return { step, finish };
  }

  function predictProba(x) {
    return grower ? leafProbability(grower.leafFor(x)) : 0.5;
  }

  // The tree itself, plus the route one point takes through it.
  function explain(point = null) {
    if (!grower) {
      return {
        model: 'tree', maxDepth, minLeaf, trained: false, finished: false,
        splits: 0, leaves: 0, depth: 0, nodes: [], lastSplit: null, example: null, overlay: [],
      };
    }
    const leaves = grower.nodes.filter((n) => !n.split);
    const overlay = [];
    // Every split is a straight cut across its own rectangle of the plot.
    for (const node of grower.nodes) {
      if (!node.split) continue;
      const t = node.split.threshold;
      const r = node.region;
      overlay.push(node.split.feature === 0
        ? { kind: 'segment', from: [t, r.x2[0]], to: [t, r.x2[1]], style: 'split' }
        : { kind: 'segment', from: [r.x1[0], t], to: [r.x1[1], t], style: 'split' });
    }
    let example = null;
    if (point) {
      const path = [];
      const leaf = grower.leafFor(point, path);
      example = {
        x: point,
        path: path.map(({ node, yes }) => ({
          id: node.id,
          feature: node.split.feature,
          threshold: node.split.threshold,
          value: point[node.split.feature],
          yes,
        })),
        leaf: { id: leaf.id, red: leaf.red, blue: leaf.blue },
        p: leafProbability(leaf),
      };
      overlay.push({ kind: 'box', x1: leaf.region.x1, x2: leaf.region.x2, style: 'leaf' });
    }
    return {
      model: 'tree',
      maxDepth,
      minLeaf,
      trained: true,
      finished: grower.done,
      splits: grower.splits,
      leaves: leaves.length,
      depth: Math.max(...leaves.map((n) => n.depth)),
      nodes: grower.nodes,
      lastSplit: grower.lastSplit ? grower.lastSplit.id : null,
      example,
      overlay,
    };
  }

  return { train, predictProba, explain };
}
