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
  learningRate: 'One learning rate is shared by both weights. On the bowl that is fine: 1 still lands on the bottom in a single step. On the narrow valley it is a problem. The steep sides can only stand a rate up to 1 before each step lands further up the opposite side, while the nearly flat floor would need a much larger rate to make good progress. No single rate suits both directions. Part 3 races methods built to deal with this.',
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

// ===== Part 3: better ways to step =====

export const RACE_COPY = {
  intro: 'Plain gradient descent has one rule: step against the slope, in proportion to how steep it is. Part 2 showed where that struggles. An <strong>optimiser</strong> is a rule for turning the gradient into a step, and there are cleverer rules than the plain one. Here four of them start from the same place on the same map. Press <strong>Train</strong> to race them.',
  map: 'Each optimiser has its own line style and its own shape at the head of its path. Click the map to move the start (the cross), or select it and use the arrow keys.',
  fair: '<strong>Is it a fair race?</strong> Not quite, and it cannot be. Each optimiser has its own learning rate, because the same number means different things to each of them, and each one\'s result depends on that choice. Change the rates and the order can change. That is the honest lesson: there is no best optimiser, only one that suits a surface and has been given a sensible rate.',
  surfaces: {
    bowl: '<strong>Bowl.</strong> The easy case. Which one wins here? It may not be the one you expect.',
    valley: '<strong>Narrow valley.</strong> The surface that plain gradient descent found hardest in Part 2: steep sides and a nearly flat floor.',
    saddle: '<strong>Saddle.</strong> Two dips with flat ground between them. Try a start close to the middle line.',
    dips: '<strong>Several dips.</strong> Three dips of different depths. Do they all end in the same dip? Try a start in the bottom-left corner.',
  },
  optimisers: {
    gd: 'Steps against the slope, in proportion to how steep it is. Steep ground gives big steps, nearly flat ground gives tiny ones. It has no memory: each step depends only on where it stands now.',
    momentum: 'Keeps most of its speed from the step before, like a heavy ball rolling downhill. Along a steady slope the speed builds up, so it crosses flat ground much faster. Across a narrow valley the back-and-forth pushes cancel out. The cost: it can roll straight past the bottom and has to come back.',
    rmsprop: 'Gives each weight its own step size. It keeps a running average of how steep each direction has been lately, and divides by it: a direction that has been steep gets smaller steps, a direction that has been gentle gets larger ones. In a narrow valley this evens the two directions out. Because of the dividing, its steps stay about one learning rate long however gentle the slope, so near the bottom it hovers instead of stopping.',
    adam: 'Momentum and RMSProp together: it keeps a running average of the gradient (for direction and speed) and of its square (to give each weight its own step size). It is the usual first choice for training neural networks, because it works reasonably well on most surfaces without much tuning.',
  },
  untrained: 'Press Train to race the optimisers from the cross.',
  stale: 'Something changed: press Train to race again',
  noneChosen: 'Tick at least one optimiser to race.',
  empty: 'No race yet.',
  progress: (step, going) => `Step ${step}: ${going} still going`,
  first: (label, steps) => `${label} arrived first, in ${steps} ${steps === 1 ? 'step' : 'steps'}.`,
  last: (label, steps) => `${label} was last to arrive, in ${steps}.`,
  differentPlaces: 'They did not all end in the same place: see the table.',
  notLowest: {
    dip: 'But they all ended in a dip that is not the deepest.',
    saddle: 'But they all stopped on the saddle, which is flat but not a dip.',
  },
  nobody: 'Nobody arrived.',
  diverged: (names) => `${names} went off the map: the learning rate is too large.`,
  limit: (names) => `${names} ran out of steps before arriving.`,
  results: {
    going: 'Still going',
    lowest: 'Arrived at the lowest point',
    dip: 'Arrived in a dip, not the deepest',
    saddle: 'Stopped on the saddle',
    diverged: 'Went off the map',
    limit: 'Ran out of steps',
  },
};

export const BATCH_COPY = {
  intro: 'So far, every step has looked at <strong>all</strong> the data to measure the gradient. With a few dozen points that is easy. With millions it is slow: one step costs a whole pass through the data. The alternative is to measure the gradient on a small handful of points, called a <strong>batch</strong>, and step straight away. The gradient from a handful is only an estimate, so each step is a little off, but it is far cheaper, so many more steps fit into the same amount of work.',
  data: 'The model is a straight line again, now with two weights: its <strong>steepness</strong> a and its <strong>height</strong> b where it meets the left edge (y = a x + b). Once training starts, each run\'s current line is drawn here.',
  map: 'The loss for every pair of weights. The plus sign marks the best possible line. Three runs start from the cross with the same learning rate. They differ only in how many points they look at before each step. Click the map to move the start.',
  loss: 'The loss of each run, always measured on all the data, against how much data it has looked at. One <strong>pass</strong> (often called an <strong>epoch</strong>) means every point has been used once. All three runs get the same number of passes, so they do the same amount of work.',
  legend: {
    full: 'Full batch: all the points, one step per pass',
    mini: 'Mini-batch: a handful of points per step',
    single: 'Single point: one point per step',
  },
  labels: {
    full: () => 'Full batch',
    mini: () => 'Mini-batch',
    single: () => 'Single point',
  },
  learningRate: 'All three runs share this learning rate. With a small rate the single-point run gets furthest, because it takes so many more steps. With a large rate its noisy steps throw it around: it reaches the right area quickly but never settles, and with a rate that is too large it flies off. Full batch can stand a larger rate than the others, because its gradient is exact.',
  miniSize: 'How many points the mini-batch run looks at before each step. Small batches behave like the single-point run: many noisy steps. Large batches behave like full batch: few, accurate steps. Using single points is often called <strong>stochastic gradient descent</strong> (stochastic means random), and mini-batches are what almost all real training uses.',
  seed: 'The seed decides the data points and the order in which the batches are drawn, so the same seed always gives the same race.',
  untrained: 'Press Train to run all three from the cross.',
  stale: 'Something changed: press Train to run again',
  empty: 'No runs yet.',
  progress: (pass, passes) => `Pass ${pass.toFixed(1)} of ${passes} through the data`,
  finished: (passes) => `All three have made ${passes} passes through the data.`,
  closest: (label, updates) => `${label} ended closest to the best line, after ${updates} steps.`,
  diverged: (names) => `${names} flew off: the learning rate is too large for steps this noisy.`,
};
