// A random forest. Pure code: no DOM access, so it can be tested in Node.
//
// Grow many decision trees, each a little different, and average their
// answers. Two things make the trees differ:
//   - each tree learns from its own bootstrap sample: points drawn at random
//     with replacement (by default as many as the training set holds), so
//     some points appear several times and about a third are left out;
//   - at each split a tree may be allowed to look at only one feature,
//     chosen at random.
// All of that randomness comes from the lab's seeded generator, so the same
// seed always grows the same forest.
//
// The model has the lab's common shape (see logistic.js). One training step
// grows one whole tree.

import { createRng } from '../rng.js';
import { growTree, leafProbability } from './tree.js';

const THUMBNAILS = 9; // how many of the first trees explain() hands over to be drawn

export function createForest({
  trees = 50, maxDepth = 6, bootstrap = true, featuresPerSplit = 1, seed = 0,
  minLeaf = 1,       // the fewest training points a leaf may hold
  sampleShare = 1,   // size of each bootstrap sample, as a share of the training set
} = {}) {
  let grown = [];       // one entry per tree: { tree, sample, view }
  let data = null;
  let finished = false;
  // Out-of-bag record: for each training point, the summed answers of the
  // trees that never saw it.
  let oobSum = null;
  let oobCount = null;

  const treeProba = (tree, x) => leafProbability(tree.leafFor(x));

  function train(X, y) {
    const n = X.length;
    data = { X, y };
    grown = [];
    oobSum = new Float64Array(n);
    oobCount = new Uint16Array(n);
    finished = n === 0;

    const snapshot = () => ({ trees: grown.length, total: trees, done: finished });

    // Grow one tree.
    function step() {
      if (finished) return { done: true, snapshot: snapshot() };
      const rng = createRng(seed, `forest-tree-${grown.length}`);
      let rows = null;
      let inBag = null;
      if (bootstrap) {
        rows = [];
        inBag = new Uint8Array(n);
        const draws = Math.max(1, Math.round(sampleShare * n));
        for (let i = 0; i < draws; i++) {
          const pick = Math.floor(rng.next() * n);
          rows.push(pick);
          inBag[pick] = 1;
        }
      }
      const tree = growTree(X, y, { maxDepth, minLeaf, featuresPerSplit, rng, rows });
      tree.finish();
      if (inBag) {
        for (let i = 0; i < n; i++) {
          if (inBag[i]) continue;
          oobSum[i] += treeProba(tree, X[i]);
          oobCount[i] += 1;
        }
      }
      grown.push({
        tree,
        sample: inBag ? X.filter((_, i) => inBag[i]) : null,
        view: { predictProba: (x) => treeProba(tree, x) },
      });
      if (grown.length >= trees) finished = true;
      return { done: finished, snapshot: snapshot() };
    }

    function finish() {
      let result = step();
      while (!result.done) result = step();
      return result;
    }

    return { step, finish };
  }

  // The forest's answer: the average of its trees' answers.
  function predictProba(x) {
    if (grown.length === 0) return 0.5;
    let sum = 0;
    for (const { tree } of grown) sum += treeProba(tree, x);
    return sum / grown.length;
  }

  // Out-of-bag accuracy: score each training point using only the trees
  // that never saw it. A free estimate of accuracy on new data.
  function outOfBag() {
    if (!bootstrap || !data) return { accuracy: null, counted: 0 };
    let counted = 0;
    let correct = 0;
    for (let i = 0; i < data.y.length; i++) {
      if (oobCount[i] === 0) continue;
      counted += 1;
      if ((oobSum[i] / oobCount[i] >= 0.5) === (data.y[i] === 1)) correct += 1;
    }
    return { accuracy: counted ? correct / counted : null, counted };
  }

  // Accuracy on any set of points after 1 tree, 2 trees, 3 trees and so on
  // (with the decision threshold at 0.5).
  function accuracyCurve(X, y) {
    if (X.length === 0) return [];
    const sums = new Float64Array(X.length);
    return grown.map(({ tree }, t) => {
      let correct = 0;
      for (let j = 0; j < X.length; j++) {
        sums[j] += treeProba(tree, X[j]);
        if ((sums[j] / (t + 1) >= 0.5) === (y[j] === 1)) correct += 1;
      }
      return correct / X.length;
    });
  }

  function explain(point = null) {
    let example = null;
    if (point && grown.length) {
      const votes = grown.map(({ tree }, t) => {
        const p = treeProba(tree, point);
        return { tree: t + 1, p, blue: p >= 0.5 };
      });
      const blueVotes = votes.filter((v) => v.blue).length;
      example = {
        x: point,
        votes,
        blueVotes,
        redVotes: votes.length - blueVotes,
        p: votes.reduce((sum, v) => sum + v.p, 0) / votes.length,
      };
    }
    const overlay = [];
    // While training, ring the bootstrap sample of the newest tree.
    const newest = grown[grown.length - 1];
    if (!finished && newest && newest.sample) overlay.push({ kind: 'rings', points: newest.sample });

    const oob = outOfBag();
    return {
      model: 'forest',
      target: trees,
      grown: grown.length,
      maxDepth,
      bootstrap,
      featuresPerSplit,
      trained: grown.length > 0,
      finished,
      oobAccuracy: oob.accuracy,
      oobCounted: oob.counted,
      firstTrees: grown.slice(0, THUMBNAILS).map((g) => g.view),
      accuracyCurve,
      example,
      overlay,
    };
  }

  return { train, predictProba, explain };
}
