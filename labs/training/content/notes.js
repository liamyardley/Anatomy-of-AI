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
  intro: 'Every finished run is listed here. <strong>Where you start matters.</strong> Try it: train from one weight, then press <strong>Start somewhere else</strong> (or drag the weight) and train again with the same learning rate. On this curve, which has a single dip, every start that settles ends at the same weight, but a start further from the bottom needs more steps, and with a small learning rate it may not arrive at all. Part 2 has bumpier ground, where the start can change the answer itself.',
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

// ===== Part 2: two weights =====

// A number with a proper minus sign.
const num = (value, places) => value.toFixed(places).replace('-', '−');

export const TWO_WEIGHTS = {
  map: 'Real models have many weights. With two, the loss is no longer a curve but a <strong>landscape</strong>: every point on this map is one pair of weights, and its height is the loss there. The map is drawn like a walker\'s map, with <strong>contour lines</strong> joining places of equal height. The stronger the colour, the lower the ground. <strong>Click the map</strong> to choose where training starts (the cross).',
  keys: 'With a keyboard: select the map, then use the arrow keys to move the start. Hold Shift for bigger moves.',
  surfaces: {
    bowl: '<strong>Bowl.</strong> One round dip, the same steepness in every direction. This is the curve from Part 1 with a second weight added. Every start leads straight to the bottom.',
    valley: '<strong>Narrow valley.</strong> Steep from side to side, nearly flat along the floor. Watch the path: it drops quickly to the valley floor, then crawls. Try raising the learning rate to speed up the crawl and see what the steep sides do.',
    saddle: '<strong>Saddle.</strong> Two dips with a pass between them, shaped like a horse\'s saddle. In the middle the ground is flat, but it is not a dip. Try a start exactly on the middle line (w<sub>2</sub> = 0), then one a single click above it.',
    dips: '<strong>Several dips.</strong> Three dips of different depths. Start in several places and see where each run ends. Which starts find the deepest dip?',
  },
  descent: 'Gradient descent works exactly as before, in two directions at once. The <strong>gradient</strong> is now a pair of numbers: the slope along w<sub>1</sub> and the slope along w<sub>2</sub>. Together they point straight uphill, so the step goes the opposite way. The arrow on the map is the next step. Paths from earlier runs stay on the map, faintly, so you can compare starts.',
  learningRate: 'One learning rate is shared by both weights. On the bowl that is fine: 1 still lands on the bottom in a single step. On the narrow valley it is a problem. The steep sides can only stand a rate up to 1 before each step lands further up the opposite side, while the nearly flat floor would need a much larger rate to make good progress. No single rate suits both directions. Part 3 looks at methods built to deal with this.',
  gradientKey: 'The gradient is the pair (slope along w<sub>1</sub>, slope along w<sub>2</sub>). Each weight is moved against its own slope, with the same learning rate &eta; (eta).',
  runs: 'Every finished run on this surface is listed here. On the saddle and on the several dips, compare where the runs end: <strong>gradient descent only ever feels the slope where it stands</strong>, so it settles in whichever dip is downhill from its start, and cannot know whether a deeper one exists.',
  runsEmpty: 'No finished runs on this surface yet.',
};

export const TWO_TRAIN_COPY = {
  untrained: 'Click the map to choose a start, then press Train.',
  stale: 'Something changed: press Train to start again from the cross',
  progress: (s) => `Step ${s.step}: weights (${num(s.w[0], 2)}, ${num(s.w[1], 2)}), loss ${num(s.loss, 3)}`,
  lowest: (s) => `Settled after ${s.step} ${s.step === 1 ? 'step' : 'steps'} at the lowest point on the map: weights (${num(s.w[0], 2)}, ${num(s.w[1], 2)}), loss ${num(s.loss, 3)}.`,
  dip: (s) => `Settled after ${s.step} steps in a dip, but not the deepest one (loss ${num(s.loss, 3)}). From inside this dip every direction is uphill, so gradient descent stops here. Try another start.`,
  saddle: (s) => `Stopped after ${s.step} steps on the saddle (loss ${num(s.loss, 3)}). The ground is flat here, so there is no slope to follow, but it is not a dip: lower ground lies to either side. Move the start just off the middle line.`,
  flat: (s) => `Settled after ${s.step} steps at weights (${num(s.w[0], 2)}, ${num(s.w[1], 2)}), loss ${num(s.loss, 3)}.`,
  diverged: (s) => `Stopped after ${s.step} steps: each step landed higher than the last, so the loss kept growing. The learning rate is too large for the steepest direction. Try a smaller one.`,
  bouncing: (s) => `Stopped after ${s.step} steps without settling (loss ${num(s.loss, 3)}). It keeps jumping back and forth across the slope. The learning rate is too large. Try a smaller one.`,
  limit: (s) => `Stopped after ${s.step} steps without settling (loss ${num(s.loss, 3)}). It was still creeping downhill. The ground here is nearly flat, so each step is tiny. Try a larger learning rate, or start closer.`,
  results: {
    lowest: 'Lowest point',
    dip: 'A dip, not the deepest',
    saddle: 'Stuck on the saddle',
    flat: 'Settled',
    diverged: 'Loss kept growing',
    bouncing: 'Ran out of steps, still bouncing',
    limit: 'Ran out of steps, still creeping',
  },
};
