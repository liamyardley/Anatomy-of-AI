// Editable copy for the training lab ("How a model finds its numbers").
//
// Everything a visitor reads that is likely to be reworded lives here, so
// the wording can change without touching the code.
//
// House style: plain English, UK spelling, no em dashes. Text may contain
// simple HTML such as <strong> and <sub>.

export const ONE_WEIGHT = {
  model: 'The model is a straight line through the bottom-left corner: y = w &times; x. It has one setting, the number w, called its <strong>weight</strong>: how steep the line is. Each dotted bar is a <strong>miss</strong>: the gap between a point and the line.',
  loss: 'The <strong>loss</strong> is a single number that says how wrong the model is. Here it is the average of the squared misses, so one big miss counts for much more than several small ones. Every weight you try leaves a dot. Can you find the weight with the lowest loss?',
  showCurve: 'Show the whole loss curve',
  descent: '<strong>Gradient descent</strong> finds the lowest point without ever seeing the whole curve. From wherever the weight is now, it measures how steeply the loss curve slopes at that spot (a number called the <strong>gradient</strong>, drawn as the short straight line through the ring) and takes a step downhill. The <strong>learning rate</strong> sets how big that step is. Each step is one <strong>iteration</strong>. Training starts from wherever you leave the weight.',
  learningRate: 'The learning rate multiplies the gradient to give the size of each step. A small rate takes tiny steps and can run out of steps before it arrives. At 1, on this curve, a single step lands on the bottom. Above 1 each step jumps past the bottom to the other side; it still settles as long as each landing is closer than the last. Above 2 each landing is further away than the one before, and the loss grows without limit.',
  lossKey: 'Here n is the number of points, and for each point (x<sub>i</sub>, y<sub>i</sub>) the miss is the line\'s prediction w x<sub>i</sub> minus the true value y<sub>i</sub>.',
  gradientKey: 'The gradient is the slope of the loss curve at the current weight. A negative gradient means the curve runs downhill to the right, so the weight should go up. A positive gradient means the weight should come down.',
  updateKey: 'The Greek letter &eta; (eta) is the learning rate. Subtracting moves the weight downhill: against the slope.',
  stepsEmpty: 'Press Train or Step to fill in the steps.',
  note: 'The Machine Learning Lab\'s logistic regression is trained in exactly this way. It has more weights, and a different loss (log-loss), but each press of Train there was this same loop: measure the gradient, step downhill, repeat.',
};

export const TRAIN_COPY = {
  untrained: 'Drag the weight yourself first. Then press Train to watch gradient descent find it.',
  stale: 'Something changed: press Train to start again from the current weight',
  progress: (s) => `Step ${s.step}: weight ${s.w.toFixed(2)}, loss ${s.loss.toFixed(3)}`,
  settled: (s) => `Settled after ${s.step} ${s.step === 1 ? 'step' : 'steps'}: weight ${s.w.toFixed(2)}, loss ${s.loss.toFixed(3)}.`,
  diverged: (s) => `Stopped after ${s.step} steps: each step jumped past the bottom and landed further away, so the loss kept growing. The learning rate is too large. Try a smaller one.`,
  bouncing: (s) => `Stopped after ${s.step} steps without settling (weight ${s.w.toFixed(2)}, loss ${s.loss.toFixed(3)}). Each step jumps past the bottom and lands a little closer, but too slowly. The learning rate is too large. Try a smaller one.`,
  limit: (s) => `Stopped after ${s.step} steps without settling (weight ${s.w.toFixed(2)}, loss ${s.loss.toFixed(3)}). It was still creeping downhill. The learning rate is too small to get there in time. Try a larger one, or start closer.`,
};

// The table of finished runs, which shows that the starting weight matters.
export const RUNS_COPY = {
  intro: 'Every finished run is listed here. <strong>Where you start matters.</strong> Try it: train from one weight, then press <strong>Start somewhere else</strong> (or drag the weight) and train again with the same learning rate. On this curve, which has a single dip, every start that settles ends at the same weight, but a start further from the bottom needs more steps, and with a small learning rate it may not arrive at all. Later parts of this lab have bumpier curves, where the start can change the answer itself.',
  empty: 'No finished runs yet.',
  startButton: 'Start somewhere else',
  results: {
    settled: 'Settled',
    diverged: 'Loss kept growing',
    bouncing: 'Ran out of steps, still bouncing',
    limit: 'Ran out of steps, still creeping',
  },
};

export const SEED_NOTE = 'The points are random, but a computer\'s randomness starts from a number called a seed. The same seed always gives exactly the same points, so a whole class can type in the same seed and see the same thing. <strong>New data</strong> picks a new seed for you.';
