// A decision tree (CART). Pure code: no DOM access, so it can be tested in Node.
//
// The tree asks yes/no questions such as "is x1 < 0.3?". Each question splits
// a group of training points into two groups that are purer (closer to being
// all one colour). A group that is not split any further is a leaf, and a
// leaf's share of blue points is the probability it predicts.
//
// growTree() does the work and is also used by the random forest. createTree()
// wraps it in the lab's common model interface (see logistic.js), where one
// training step is one split, or, once the tree has grown, one split pruned
// away again.

// Gini impurity: 0 when a group is all one colour, 0.5 when it is half and half.
export function gini(red, blue) {
  const n = red + blue;
  if (n === 0) return 0;
  const p = blue / n;
  return 1 - p * p - (1 - p) * (1 - p);
}

// Entropy, a second way to measure impurity: 0 when a group is all one
// colour, 1 when it is half and half.
export function entropy(red, blue) {
  const n = red + blue;
  if (red === 0 || blue === 0) return 0;
  const p = blue / n;
  return -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p));
}

const IMPURITY = { gini, entropy };

// The probability of blue predicted by a leaf.
export const leafProbability = (leaf) => {
  const n = leaf.red + leaf.blue;
  return n === 0 ? 0.5 : leaf.blue / n;
};

const FEATURE_NAMES = ['x1', 'x2'];

// Grow a tree on the rows of X (features) and y (0 red, 1 blue).
//   maxDepth          how many questions deep the tree may go
//   minLeaf           the fewest training points a leaf may hold
//   minSplit          the fewest training points a group needs to be split
//   maxLeaves         stop at this many leaves (0 means no limit)
//   criterion         how impurity is measured: 'gini' or 'entropy'
//   ccpAlpha          pruning strength (0 means no pruning), see below
//   featuresPerSplit  2 looks at both features at every split; 1 looks at
//                     one feature chosen at random (used by the forest)
//   rng               needed only when featuresPerSplit is 1
//   rows              which rows to train on (default: all of them; the
//                     forest passes a bootstrap sample, which repeats rows)
//
// Splits are made one at a time by calling step(). Normally the tree grows
// level by level. With a limit on the number of leaves it instead always
// splits the group whose split helps most, so the limit is spent on the most
// useful questions.
//
// Pruning (cost-complexity pruning) runs after the tree has grown. Every
// split is scored by how much impurity it removes per extra leaf it costs.
// The weakest split is removed, then the next weakest, for as long as the
// weakest scores no more than ccpAlpha. Each removal is also one step().
export function growTree(X, y, {
  maxDepth = 3, minLeaf = 1, minSplit = 2, maxLeaves = 0, criterion = 'gini', ccpAlpha = 0,
  featuresPerSplit = 2, rng = null, rows = null,
} = {}) {
  const impurity = IMPURITY[criterion] || gini;
  const nodes = [];
  let leaves = 1;
  let pruned = 0;
  let version = 0;        // goes up every time the tree changes shape
  let lastChanged = null; // the group most recently split or pruned

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
      impurity: impurity(red, blue),
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

  const total = Math.max(1, (rows || X).length);

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
      const gain = node.impurity
        - (nLeft / n) * impurity(left.red, left.blue)
        - (nRight / n) * impurity(right.red, right.blue);
      if (!best || gain > best.gain) {
        best = { feature, threshold: (before + after) / 2, gain, left, right };
      }
    }
    return best;
  }

  // Decide whether a group could be split, and how. Returns the chosen
  // split, or null if the group has to stay a leaf.
  function examine(node) {
    if (node.depth >= maxDepth) { node.leafReason = 'depth'; return null; }
    if (node.impurity === 0) { node.leafReason = 'pure'; return null; }
    if (node.indices.length < minSplit || node.indices.length < 2 * minLeaf) {
      node.leafReason = 'small';
      return null;
    }
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

  // Groups that could be split and are waiting their turn, oldest first.
  const waiting = [];
  function add(node) {
    const choice = examine(node);
    if (choice) waiting.push({ node, choice });
    return node;
  }
  const root = add(makeNode(rows || X.map((_, i) => i), 0, { x1: [-1, 1], x2: [-1, 1] }, null));

  // Which waiting group to split next.
  function pickNext() {
    if (waiting.length === 0) return null;
    if (maxLeaves <= 0) return waiting.shift(); // level by level
    if (leaves >= maxLeaves) {
      for (const { node } of waiting) node.leafReason = 'leaves';
      waiting.length = 0;
      return null;
    }
    // Best first: the split that removes most impurity from the whole tree.
    const worth = ({ node, choice }) => (node.indices.length / total) * choice.gain;
    let best = 0;
    for (let i = 1; i < waiting.length; i++) {
      if (worth(waiting[i]) > worth(waiting[best])) best = i;
    }
    return waiting.splice(best, 1)[0];
  }

  // --- pruning ---

  // The weakest split in the tree, if it is weak enough to prune. A split's
  // strength is the impurity its whole branch removes, divided by the number
  // of extra leaves the branch costs.
  function findPrune() {
    if (ccpAlpha <= 0) return null;
    const risk = (node) => (node.indices.length / total) * node.impurity;
    let weakest = null;
    let weakestStrength = Infinity;
    (function visit(node) {
      if (!node.split) return { leaves: 1, risk: risk(node) };
      const yes = visit(node.left);
      const no = visit(node.right);
      const branch = { leaves: yes.leaves + no.leaves, risk: yes.risk + no.risk };
      const strength = (risk(node) - branch.risk) / (branch.leaves - 1);
      if (strength < weakestStrength) { weakest = node; weakestStrength = strength; }
      return branch;
    }(root));
    return weakest && weakestStrength <= ccpAlpha + 1e-12 ? weakest : null;
  }

  // Turn a split back into a leaf, dropping everything beneath it.
  function collapse(node) {
    node.split = null;
    node.left = null;
    node.right = null;
    node.leafReason = 'pruned';
    // Rebuild the list of groups, level by level, and renumber them.
    const kept = [root];
    for (let i = 0; i < kept.length; i++) {
      if (kept[i].split) kept.push(kept[i].left, kept[i].right);
    }
    nodes.length = 0;
    for (const item of kept) { item.id = nodes.length; nodes.push(item); }
    leaves = nodes.filter((n) => !n.split).length;
    pruned += 1;
  }

  let pending = pickNext();
  let pendingPrune = pending ? null : findPrune();

  // Make one change to the tree: a split, or (once it has grown) one split
  // pruned away. Returns true when there is nothing left to do.
  function step() {
    if (pending) {
      const { node, choice } = pending;
      const { feature, threshold } = choice;
      node.split = { feature, threshold, gain: choice.gain };
      const key = FEATURE_NAMES[feature];
      const [low, high] = node.region[key];
      const yes = node.indices.filter((i) => X[i][feature] < threshold);
      const no = node.indices.filter((i) => X[i][feature] >= threshold);
      node.left = add(makeNode(yes, node.depth + 1, { ...node.region, [key]: [low, threshold] }, node));
      node.right = add(makeNode(no, node.depth + 1, { ...node.region, [key]: [threshold, high] }, node));
      leaves += 1;
      lastChanged = node;
      version += 1;
      pending = pickNext();
      if (!pending) pendingPrune = findPrune();
    } else if (pendingPrune) {
      collapse(pendingPrune);
      lastChanged = pendingPrune;
      version += 1;
      pendingPrune = findPrune();
    }
    return !pending && !pendingPrune;
  }

  function finish() {
    while (!step()) { /* keep going */ }
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
    get done() { return !pending && !pendingPrune; },
    get phase() { return pending ? 'grow' : (pendingPrune ? 'prune' : 'done'); },
    get splits() { return leaves - 1; },
    get leaves() { return leaves; },
    get pruned() { return pruned; },
    get version() { return version; },
    get lastSplit() { return lastChanged; },
  };
}

export function createTree({
  maxDepth = 3, minLeaf = 1, minSplit = 2, maxLeaves = 0, criterion = 'gini', ccpAlpha = 0,
} = {}) {
  let grower = null;

  const depthOf = () => Math.max(...grower.nodes.filter((n) => !n.split).map((n) => n.depth));

  function train(X, y) {
    grower = growTree(X, y, { maxDepth, minLeaf, minSplit, maxLeaves, criterion, ccpAlpha });
    const snapshot = () => ({
      phase: grower.phase,
      splits: grower.splits,
      leaves: grower.leaves,
      depth: depthOf(),
      pruned: grower.pruned,
      done: grower.done,
    });
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
    const settings = { model: 'tree', maxDepth, minLeaf, minSplit, maxLeaves, criterion, ccpAlpha };
    if (!grower) {
      return {
        ...settings, trained: false, finished: false, phase: 'idle', version: -1,
        splits: 0, leaves: 0, depth: 0, pruned: 0, nodes: [], lastSplit: null, example: null, overlay: [],
      };
    }
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
      ...settings,
      trained: true,
      finished: grower.done,
      phase: grower.phase,
      version: grower.version,
      splits: grower.splits,
      leaves: grower.leaves,
      depth: depthOf(),
      pruned: grower.pruned,
      nodes: grower.nodes,
      lastSplit: grower.lastSplit ? grower.lastSplit.id : null,
      example,
      overlay,
    };
  }

  return { train, predictProba, explain };
}
