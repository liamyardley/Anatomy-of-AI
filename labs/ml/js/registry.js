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
//                  fixed stops, otherwise min, max and step are used
//   speeds         training steps per tick, and milliseconds between ticks
//   animationGrid  resolution of the shaded regions while training animates
//   progress(s)    one line describing a training snapshot
//   finished(s)    one line describing the finished model

import { createLogistic } from './models/logistic.js';
import { createLogisticPanel } from './panels/logistic.js';

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
};

// Shown in the picker, greyed out, until they are built.
export const COMING_SOON = ['k-nearest neighbours', 'Decision tree', 'Random forest'];
