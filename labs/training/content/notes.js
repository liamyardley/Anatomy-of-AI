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

// ===== Part 4: the real thing =====

const share = (value) => `${Math.round(value * 100)}%`;

export const REAL_COPY = {
  intro: 'Everything so far used made-up surfaces. This one is real. The points are the <strong>Two clouds</strong> dataset from the Machine Learning Lab, and the model is its <strong>logistic regression</strong>: a straight line that splits the plot into a red side and a blue side. The map on the right is that model\'s true loss for every possible pair of weights. Press <strong>Train</strong> and watch both at once: each step on the map is a small turn of the line on the data. This is what Train was doing in the Machine Learning Lab.',
  data: 'Red circles and blue triangles, as in the Machine Learning Lab. The thick line is the model\'s boundary, and the short arrow points to the side it calls blue. The dot is the centre of the data. To keep the model to two weights that fit on a map, its line always passes through that dot (the full model has a third number, the bias, which lets the line sit anywhere).',
  map: 'The plus sign is the lowest loss: the best line this model can draw. Click the map to choose different starting weights (the cross). The Machine Learning Lab always starts from zero, the middle of the map.',
  noLine: 'Both weights are zero: no line yet',
  offMap: 'the lowest point is off the map',
  modelKey: 'The model turns each point into a probability of blue: a weighted sum of the point\'s two coordinates, squashed into the range 0 to 1 by the function &sigma; (sigma). The loss is the <strong>log-loss</strong>: small when the true colour gets a high probability, large when the model is confidently wrong.',
  learningRate: 'The same learning rate as in Part 1, but the useful values are much larger here. The coordinates of the points are small numbers (between about &minus;1 and 1), so the loss surface slopes gently and small rates crawl. That is why the Machine Learning Lab offers rates up to 30.',
  overlap: 'How much the two clouds overlap (the "noise" setting in the Machine Learning Lab). When they overlap a lot, no line is very good and the best weights are small: the model stays unsure. When they barely overlap, the model can afford to be confident, and the best weights are large. If a line separates the colours perfectly there is no lowest point at all: bigger weights always score a little better, for ever. That runaway is one reason for the penalty in Part 5.',
  seed: 'The same seed gives the same clouds here as the Two clouds dataset gives in the Machine Learning Lab (with 80 points).',
  link: 'In the Machine Learning Lab the same loop runs with more weights: a bias, and at higher feature degrees a weight for every squared and multiplied term. The map would need more dimensions than can be drawn, but nothing else changes.',
  untrained: 'Press Train to watch gradient descent turn the line.',
  stale: 'Something changed: press Train to start again from the cross',
  progress: (s, acc) => `Step ${s.step}: loss ${s.loss.toFixed(3)}, ${share(acc)} of points on the right side`,
  settled: (s, acc) => `Settled after ${s.step} steps: loss ${s.loss.toFixed(3)}, with ${share(acc)} of the points on the right side of the line. It cannot reach 100%, because the clouds overlap.`,
  limit: (s, acc) => `Stopped after ${s.step} steps, still creeping downhill (loss ${s.loss.toFixed(3)}, ${share(acc)} right). The line is close, but the weights are still growing. Try a larger learning rate.`,
  noBottom: (s, acc) => `Stopped after ${s.step} steps (loss ${s.loss.toFixed(3)}, ${share(acc)} right). These clouds can be separated perfectly, so there is no lowest point: the weights would keep growing for ever.`,
  bouncing: (s, acc) => `Stopped after ${s.step} steps without settling (loss ${s.loss.toFixed(3)}, ${share(acc)} right). The steps keep jumping across the valley. Try a smaller learning rate.`,
  diverged: (s) => `Stopped after ${s.step} steps: the loss kept growing. The learning rate is too large.`,
};

// ===== Part 5: regularisation =====

export const REG_COPY = {
  intro: 'Left alone, training chases the lowest loss on the data it has, and large weights are often how it gets there: Part 4 showed weights growing without limit. Large weights tend to mean a model that has fitted the quirks of its training data. <strong>Regularisation</strong> adds a second cost, a <strong>penalty</strong> for large weights, so training has to balance two things: fit the data, and keep the weights small. The <strong>strength</strong> sets how much the penalty counts.',
  map: 'The contours show the data loss alone, and the plus sign is its lowest point: the best fit with no penalty. The penalty pulls towards zero, where the two dotted lines cross. Training settles somewhere between. Once a run starts, a shape is drawn through the current weights: every pair of weights on that shape has the same penalty. Click the map to move the start.',
  kinds: { none: 'No penalty', l2: 'L2 (ridge)', l1: 'L1 (lasso)', elastic: 'Elastic net' },
  shapes: {
    none: 'With no penalty, training ends at the plus sign: the best fit to this data.',
    l2: '<strong>L2</strong> (also called <strong>ridge</strong>) charges for the <em>square</em> of each weight. Weights with equal penalty lie on a <strong>circle</strong>. Large weights are charged heavily and small ones almost nothing, so every weight is shrunk, but none is pushed all the way to zero.',
    l1: '<strong>L1</strong> (also called <strong>lasso</strong>) charges for the <em>size</em> of each weight. Weights with equal penalty lie on a <strong>diamond</strong>, with its corners on the dotted lines, where one weight is exactly zero. The charge for a small weight is as steep as for a large one, so a weight that is not earning its keep is pushed all the way to zero.',
    elastic: 'The <strong>elastic net</strong> charges half of each. Its shape is a diamond with rounded sides: it still has corners, so it can still set weights to zero, but it shrinks the others more smoothly.',
  },
  strength: 'The strength (written &lambda;, lambda) multiplies the penalty. Near zero the penalty hardly counts and training ends close to the plus sign. As the strength grows, the answer is pulled towards zero and fits the data less well. The right strength is a matter of trying several and checking each on data the model has not seen.',
  why: '<strong>Why does the diamond give exact zeros?</strong> The answer is where the penalty shape first touches a loss contour. A circle is equally round everywhere, so it can touch anywhere, and almost never exactly on a dotted line. A diamond sticks out at its corners, so a contour usually meets a corner first, and a corner sits on a dotted line, where one weight is zero. In the training loop this shows up as an extra move after every step: each weight is shrunk towards zero by a fixed amount, and a weight that would cross zero stops there.',
  seed: 'The seed decides the data for the map and for the sweep below.',
  untrained: 'Choose a penalty and a strength, then press Train.',
  stale: 'Something changed: press Train to start again from the cross',
  progress: (s) => `Step ${s.step}: weights (${num(s.w[0], 2)}, ${num(s.w[1], 2)}), data loss ${s.loss.toFixed(3)}`,
  finished: (s, settled) => `${settled ? 'Settled' : 'Stopped'} after ${s.step} steps at (${num(s.w[0], 2)}, ${num(s.w[1], 2)}), data loss ${s.loss.toFixed(3)}.`,
  endings: {
    none: 'This is the best fit to the data, with nothing holding the weights back.',
    l2: 'Both weights have been shrunk towards zero, but neither is exactly zero.',
    l1: 'Both weights have been shrunk, and neither has reached zero yet. Try a larger strength.',
    elastic: 'Both weights have been shrunk, and neither has reached zero yet. Try a larger strength.',
    oneZero: (which) => `Weight w${which} is exactly zero: the model has dropped that feature altogether.`,
    bothZero: 'Both weights are exactly zero: the penalty is so strong that the model ignores the data.',
  },
  sweepIntro: 'Two weights make a good picture, but the effect matters most with many. This model has <strong>six</strong> features. Only the first three affect the answer; the last three are irrelevant, though the model is not told that. The sweep trains the model again and again, with a stronger penalty each time, and plots where each weight ends up.',
  sweepLegend: 'Solid lines are the three features that matter (w1, w2, w3). Dashed lines are the three irrelevant ones (w4, w5, w6). The penalty grows from left to right.',
  sweepEmpty: 'Press Run the sweep.',
  sweepDone: 'The sweep has finished: three charts drawn.',
  sweepNotes: {
    l2: () => 'Ridge shrinks every weight smoothly towards zero, and none ever reaches it. The irrelevant features are kept, with small weights.',
    l1: (first, idle, kept, all) => `Lasso selects. ${idle !== null ? `By a strength of ${idle.toPrecision(2)} all three irrelevant weights are exactly zero${kept === 3 ? ', while the three that matter are still in use' : ''}.` : 'The irrelevant weights fall to exactly zero one by one.'}${all !== null ? ` By ${all.toPrecision(2)} every weight is zero.` : ''}`,
    elastic: (first, idle) => `The elastic net does some of each: it shrinks like ridge and also sets weights to exactly zero, but needs a stronger penalty than lasso to do it${idle !== null ? ` (all three irrelevant weights are gone by ${idle.toPrecision(2)})` : ''}.`,
  },
  link: 'The Machine Learning Lab\'s logistic regression has an L2 setting, called regularisation strength. Raise its feature degree to 5 on a noisy dataset, watch the boundary twist to fit single points, then turn the strength up and watch it relax.',
};

// ===== Predict, observe, explain: one short activity for each part =====

export const ACTIVITIES = {
  one: {
    predict: 'With a learning rate of 0.3, training from a weight of 0 takes a certain number of steps. If you double the rate to 0.6, will it take about half as many steps, fewer than half, or more?',
    observe: 'Set the weight to 0 and train at 0.3, then at 0.6, then at 1.5. Read the three rows in the table of runs.',
    explain: 'Each step closes a fixed share of the remaining gap: 30% at a rate of 0.3, and 60% at 0.6. Closing 60% each time gets there in well under half the steps (12 against 26 with the starting data). But more is not always better: at 1.5 each step jumps past the bottom and lands half as far away on the other side, which takes 16 steps. The fastest rate is the one that lands on the bottom, not the largest one that still works.',
  },
  two: {
    predict: 'Choose the Saddle. If training starts exactly on the middle line (w<sub>2</sub> = 0), where will it end? And if it starts one click above the line?',
    observe: 'Select the map and use the arrow keys to put the start exactly on w<sub>2</sub> = 0, then train. Move the start up one step and train again.',
    explain: 'On the middle line the slope along w<sub>2</sub> is exactly zero, so gradient descent never moves off the line. It slides along to the saddle point and stops there, on flat ground that is not a dip. One click above, there is a tiny slope away from the line, and it grows with every step, so the run curves off into a real dip. In practice almost every start is slightly off the line, which is why saddles slow training down more often than they stop it.',
  },
  race: {
    predict: 'Choose the Bowl. Which of the four optimisers will arrive first?',
    observe: 'Race them on the Bowl, then on the Narrow valley, from the same start.',
    explain: 'On the Bowl, plain gradient descent wins easily. The ground is equally steep in every direction, so a step in proportion to the slope is exactly right, and the cleverer rules only add caution or overshoot. On the Narrow valley the order reverses: the plain rule crawls along the flat floor while the others, built for ground that is steeper one way than the other, get there in about half the steps or fewer. An optimiser is a tool for a kind of surface, not a ranking.',
  },
  batch: {
    predict: 'All three runs look at the same amount of data. After 20 passes, which will be closest to the best line: full batch, mini-batch or single point?',
    observe: 'Train at a learning rate of 0.1. Then try 0.03 and 0.4.',
    explain: 'At 0.1 the mini-batch run usually ends closest. Full batch moves smoothly but has taken only 20 steps and is still on its way. The single-point run got near the best line within the first pass or two, but its steps are so noisy that it keeps being knocked about and never settles. At 0.03 the noise matters less and the single-point run wins on sheer number of steps. At 0.4 its noise is too much. Mini-batches are the usual compromise: many steps, each accurate enough.',
  },
  real: {
    predict: 'Set the overlap to 1, so the clouds barely overlap. Will the lowest point on the map be closer to the middle than before, or further away?',
    observe: 'Change the overlap from 2 to 1, then to 3, and watch where the plus sign goes. Train at each setting.',
    explain: 'Further away, usually right off the map. When the clouds hardly overlap, a line can put nearly every point on the right side, and the loss rewards being confident about it: larger weights push the probabilities closer to 0% and 100%. When the clouds overlap heavily, confidence is punished by all the points on the wrong side, so the best weights are small. The size of the weights is the model\'s confidence.',
  },
  reg: {
    predict: 'Choose L1 and raise the strength a step at a time. Which weight will reach exactly zero first, w<sub>1</sub> or w<sub>2</sub>? Will L2 ever do the same?',
    observe: 'Train with L1 at 0.4, 0.7, 1 and 1.5. Then switch to L2 and train at the same strengths.',
    explain: 'The weaker weight, w<sub>2</sub>, goes first: it was contributing least to fitting the data, so it is the cheapest to give up. L1 takes the same fixed amount off every weight at every step, and a small weight cannot stand that for long. L2 never gets there: it takes off an amount in proportion to the weight\'s size, so as a weight gets smaller the pull on it fades away. Lasso chooses features; ridge only quietens them.',
  },
};

// Short practical notes, shown at the end of some parts.
export const PRACTICE_NOTES = {
  one: {
    title: 'The learning rate is the first thing to tune',
    text: 'If the loss jumps about or blows up, the rate is too large: divide it by three or ten. If the loss falls steadily but slowly, it is too small. Rates are tried in multiples, such as 0.001, 0.003, 0.01, 0.03 and so on, because what matters is the rough size, not the exact figure.',
  },
  batch: {
    title: 'Mini-batches and Adam are the usual starting point',
    text: 'Most real training uses mini-batches of a few dozen to a few hundred points, with Adam or with gradient descent plus momentum. The batch size and the learning rate affect each other, so if one is changed the other often needs changing too.',
  },
  reg: {
    title: 'Put the features on the same scale first',
    text: 'A penalty treats every weight alike. A feature measured in large numbers needs only a small weight, so it slips past the penalty, while one measured in small numbers is punished for needing a large weight. Rescaling every feature to a similar range before training makes the penalty fair. The strength itself is chosen by trying several and scoring each on data kept back from training.',
  },
};
