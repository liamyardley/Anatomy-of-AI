// The list of models the lab offers. main.js and plot.js only ever talk to a
// model through this list and the common model interface, so adding a model
// (including, later, a neural network) means adding one entry here plus its
// model and panel modules. Nothing else needs to change.
//
// Each entry has:
//   label          name shown in the model picker
//   create(params) builds a new, untrained model
//   createPanel    builds the model's part of the maths panel
//   params         the settings shown as controls; `values` gives a slider
//                  fixed stops, `type: 'toggle'` gives an on/off switch,
//                  `type: 'choice'` a drop-down list of `options` (each
//                  { value, label, short }), otherwise min, max and step
//                  are used. `advanced: true` puts a setting behind "More
//                  settings"; `showIf(params)` shows it only when relevant
//   speeds         training steps per tick, and milliseconds between ticks
//   animationGrid  resolution of the shaded regions while training animates
//   progress(s)    one line describing a training snapshot
//   finished(s)    one line describing the finished model
//
// Every model is created with its settings plus `seed`, the lab's current
// seed, so that a model which needs randomness can stay reproducible.

import { createLogistic } from './models/logistic.js';
import { createLogisticPanel } from './panels/logistic.js';
import { createKnn } from './models/knn.js';
import { createKnnPanel } from './panels/knn.js';
import { createTree } from './models/tree.js';
import { createTreePanel } from './panels/tree.js';
import { createForest } from './models/forest.js';
import { createForestPanel } from './panels/forest.js';

const onOff = (value) => (value ? 'on' : 'off');
const plural = (count, one, many = `${one}s`) => `${count} ${count === 1 ? one : many}`;
const ODD_K = Array.from({ length: 16 }, (_, i) => 2 * i + 1); // 1, 3, 5 ... 31

export const MODELS = {
  logistic: {
    label: 'Logistic regression',
    create: createLogistic,
    createPanel: createLogisticPanel,
    params: [
      { key: 'degree', label: 'Feature degree', min: 1, max: 5, step: 1, initial: 1 },
      { key: 'learningRate', label: 'Learning rate', values: [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30], initial: 3 },
      {
        key: 'l2', label: 'Regularisation strength',
        values: [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1], initial: 0,
        format: (v) => (v === 0 ? 'off' : String(v)),
      },
    ],
    speeds: {
      slow: { steps: 1, interval: 120 },
      normal: { steps: 4, interval: 0 },
      fast: { steps: 30, interval: 0 },
    },
    animationGrid: 60,
    progress: (s) => `Training: iteration ${s.iteration}, loss ${s.loss.toFixed(3)}`,
    finished: (s) => (s.iteration >= s.maxIterations
      ? `Stopped at the limit of ${s.maxIterations} iterations (loss ${s.loss.toFixed(3)}).`
      : `Finished: the loss stopped improving after ${s.iteration} iterations (loss ${s.loss.toFixed(3)}).`),
  },

  knn: {
    label: 'k-nearest neighbours',
    create: createKnn,
    createPanel: createKnnPanel,
    params: [
      { key: 'k', label: 'Neighbours (k)', values: ODD_K, initial: 5 },
      { key: 'weighted', label: 'Distance weighting', type: 'toggle', initial: false, format: onOff },
      {
        key: 'metric', label: 'Distance measure', type: 'choice', initial: 'euclidean', advanced: true,
        options: [
          { value: 'euclidean', label: 'Euclidean (straight line)', short: 'Euclidean' },
          { value: 'manhattan', label: 'Manhattan (city blocks)', short: 'Manhattan' },
          { value: 'chebyshev', label: 'Chebyshev (biggest gap)', short: 'Chebyshev' },
          { value: 'minkowski', label: 'Minkowski (adjustable power)', short: 'Minkowski' },
          { value: 'cosine', label: 'Cosine (direction only)', short: 'cosine' },
        ],
      },
      {
        key: 'p', label: 'Minkowski power (p)', values: [1, 1.25, 1.5, 2, 3, 4, 6, 10, 20], initial: 2,
        advanced: true, showIf: (params) => params.metric === 'minkowski',
      },
    ],
    speeds: {
      slow: { steps: 1, interval: 160 },
      normal: { steps: 1, interval: 50 },
      fast: { steps: 3, interval: 0 },
    },
    animationGrid: 60,
    progress: (s) => {
      if (s.phase === 'memorise') return `Storing the training points: ${s.stored} of ${s.total} copied into memory`;
      if (s.phase === 'project') return 'Only direction counts: sliding a copy of each point onto a circle around the centre';
      return `Predicting: asking the neighbours, ${Math.round(s.reveal * 100)}% of the plot filled in`;
    },
    finished: (s) => `Finished: ${plural(s.total, 'point')} stored. Nothing was learned; every prediction is worked out by asking the neighbours.`,
  },

  tree: {
    label: 'Decision tree',
    create: createTree,
    createPanel: createTreePanel,
    params: [
      { key: 'maxDepth', label: 'Maximum depth', min: 1, max: 10, step: 1, initial: 3 },
      { key: 'minLeaf', label: 'Minimum samples per leaf', min: 1, max: 20, step: 1, initial: 1 },
      {
        key: 'criterion', label: 'Impurity measure', type: 'choice', initial: 'gini', advanced: true,
        options: [
          { value: 'gini', label: 'Gini impurity', short: 'Gini' },
          { value: 'entropy', label: 'Entropy', short: 'entropy' },
        ],
      },
      {
        key: 'minSplit', label: 'Minimum samples to split', advanced: true, initial: 2,
        values: [2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40],
      },
      {
        key: 'maxLeaves', label: 'Maximum number of leaves', advanced: true, initial: 0,
        values: [0, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50], format: (v) => (v === 0 ? 'no limit' : String(v)),
      },
      {
        key: 'ccpAlpha', label: 'Pruning strength', advanced: true, initial: 0,
        values: [0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1], format: (v) => (v === 0 ? 'off' : String(v)),
      },
    ],
    speeds: {
      slow: { steps: 1, interval: 1000 },
      normal: { steps: 1, interval: 400 },
      fast: { steps: 1, interval: 60 },
    },
    animationGrid: 120,
    progress: (s) => (s.pruned > 0
      ? `Pruning: ${plural(s.pruned, 'branch', 'branches')} cut back so far, ${plural(s.leaves, 'leaf', 'leaves')} left`
      : `Training: ${plural(s.splits, 'split')} made, ${plural(s.leaves, 'leaf', 'leaves')}`),
    finished: (s) => {
      const pruned = s.pruned > 0 ? ` Pruning cut back ${plural(s.pruned, 'branch', 'branches')}.` : '';
      if (s.splits === 0) {
        return s.pruned > 0
          ? `Finished: pruning removed every split, so the tree is a single leaf.`
          : 'Finished: no split made the groups any purer, so the tree is a single leaf.';
      }
      return `Finished: ${plural(s.splits, 'split')}, ${s.leaves} leaves, ${plural(s.depth, 'question')} deep.${pruned}`;
    },
  },

  forest: {
    label: 'Random forest',
    create: createForest,
    createPanel: createForestPanel,
    params: [
      { key: 'trees', label: 'Number of trees', min: 1, max: 200, step: 1, initial: 50 },
      { key: 'maxDepth', label: 'Maximum depth', min: 1, max: 10, step: 1, initial: 6 },
      { key: 'bootstrap', label: 'Bootstrap samples', type: 'toggle', initial: true, format: onOff },
      { key: 'featuresPerSplit', label: 'Features considered per split', min: 1, max: 2, step: 1, initial: 1 },
      { key: 'minLeaf', label: 'Minimum samples per leaf', min: 1, max: 20, step: 1, initial: 1, advanced: true },
      {
        key: 'sampleShare', label: 'Sample size per tree', advanced: true, initial: 1,
        values: [0.1, 0.25, 0.5, 0.75, 1], format: (v) => `${Math.round(v * 100)}% of the training points`,
        showIf: (params) => params.bootstrap,
      },
    ],
    speeds: {
      slow: { steps: 1, interval: 500 },
      normal: { steps: 1, interval: 80 },
      fast: { steps: 5, interval: 0 },
    },
    animationGrid: 60,
    progress: (s) => `Training: ${s.trees} of ${plural(s.total, 'tree')} grown`,
    finished: (s) => `Finished: ${plural(s.trees, 'tree')} grown and averaged.`,
  },
};
