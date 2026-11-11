// Editable copy for the Machine Learning Lab.
//
// Everything a visitor reads that is likely to be reworded lives here, so
// the wording can change without touching the code. Parameter explanations,
// the predict-observe-explain activities and the field notes join this file
// in later phases.
//
// House style: plain English, UK spelling, no em dashes.

// One short, neutral description per dataset. These deliberately do not say
// which model suits which dataset: working that out is the activity.
export const DATASET_NOTES = {
  clouds: {
    title: 'Two clouds',
    text: 'Two overlapping clouds of points, one mostly red and one mostly blue. Where they meet, the colours mix, so no rule can get every point right.',
  },
  ring: {
    title: 'Ring',
    text: 'One colour sits in a disc in the middle and the other forms a ring around it. Ask yourself: could a single straight line ever separate them?',
  },
  checker: {
    title: 'Checkerboard',
    text: 'Four quarters in alternating colours, a little blurred where they meet. Each colour appears in two opposite corners.',
  },
  moons: {
    title: 'Moons',
    text: 'Two crescents that curl around each other. The colours are close together but follow two clear curves.',
  },
  islands: {
    title: 'Noisy islands',
    text: 'Several small clusters of each colour. About one point in ten has been given the wrong colour on purpose, as happens in real data when labels are recorded wrongly.',
  },
  spiral: {
    title: 'Spiral',
    text: 'Two spiral arms wound around each other. This one is hard: the colours alternate again and again as you move out from the centre.',
  },
  noise: {
    title: 'Pure noise',
    text: 'Points scattered evenly, with each colour decided by the toss of a coin. There is no pattern here to find, which makes it a useful test of whether a model is fooling itself.',
  },
  mystery: {
    title: 'Mystery',
    text: 'An unknown pattern, chosen at random. Study the points and decide what you think it is, then press Reveal.',
  },
  custom: {
    title: 'Custom',
    text: 'Your own dataset. Switch the plot to Edit points, choose a colour, then click or tap to place points.',
  },
  edited: {
    title: 'Custom (edited)',
    text: 'A premade dataset that you have changed by hand. "Add more points" now scatters new points close to the ones already here.',
  },
};

export const COPY = {
  mysteryRevealed: (name) => `This mystery dataset was: ${name}.`,
  queryEmpty: 'No query point yet. Hover over the plot, or tap it, to drop one.',
  customEmpty: 'The plot is empty. Switch to Edit points and click or tap to add some.',
  addMoreEmpty: 'Add a few points by hand first, then "Add more points" can scatter more like them.',
};

// --- training and threshold messages ---

export const TRAIN_COPY = {
  untrained: 'Choose your settings, then press Train to watch the model learn.',
  stale: 'Settings changed: press Train to see the new model',
  noData: 'There are no training points yet. Add some points, then press Train.',
  threshold: 'Blue counts as the positive class: a point is predicted blue when its probability of blue is at least the threshold. Moving the threshold does not retrain the model. The shading, which shows the model\'s probabilities, stays exactly the same; only the boundary line and the predictions move.',
  accuracyWarning: 'Accuracy alone can mislead. On data that is 60% blue, a model that calls every point blue scores 60% while never getting a single red point right. The confusion matrix shows which kind of mistake is being made.',
  experiment: 'Try changing a setting and training again. Can you work out what it does before you reveal the explanation?',
};

// --- the models ---
// Text may contain simple HTML such as <strong> and <sub>.

export const MODEL_NOTES = {
  logistic: {
    title: 'Logistic regression',
    idea: 'Logistic regression gives every feature a <strong>weight</strong>: a number saying how strongly that feature pushes the answer towards blue (a positive weight) or towards red (a negative weight). It multiplies each feature by its weight and adds everything up, together with a starting value w<sub>0</sub> called the <strong>bias</strong>. The total is a score, z. The <strong>sigmoid</strong> function, an S-shaped curve written &sigma;, then squashes z into a probability between 0 and 1: the probability that the point is blue.',
    degreeNote: 'With a feature degree above 1, extra terms such as x<sub>1</sub><sup>2</sup> and x<sub>1</sub>x<sub>2</sub> join the sum, each with a weight of its own.',
    training: '<strong>Training</strong> means finding the weights that make the <strong>log-loss</strong> as small as possible. Log-loss is a penalty score: small when the model gives the right colour a high probability, large when it is confidently wrong. The method is <strong>gradient descent</strong>: nudge every weight a little in the direction that lowers the loss, then repeat. Each round of nudges is one <strong>iteration</strong>.',
    lossKey: 'Here n is the number of training points, y<sub>i</sub> is 1 for a blue point and 0 for a red one, and p<sub>i</sub> is the probability of blue the model gives that point.',
    fittedUntrained: 'Every weight starts at zero, so the model begins undecided everywhere. Press Train and watch the numbers move.',
    fitted: 'These are the model\'s actual weights, updated at every training step.',
    sigmoid: 'Each training point is placed on the sigmoid curve by its score z. Points far to the right are confidently blue, points far to the left are confidently red, and the ones near the middle are the uncertain ones close to the boundary. For a straight-line boundary, z grows in step with how far the point is from the line, so distance becomes probability. (Blue points are drawn just above the curve and red just below, so they do not hide each other.)',
    lossCurve: 'The loss after each iteration. A curve that falls and flattens means training has settled.',
  },
};

// --- "What does this do?" explanations, one per setting ---
// Two or three sentences each: what it controls, what small and large values
// do, and what to look for on the plot.

export const PARAM_NOTES = {
  logistic: {
    degree: 'The degree decides which features the model is given. At 1 it sees only x<sub>1</sub> and x<sub>2</sub>, so its boundary has to be a straight line. From 2 upwards it also gets squares and products such as x<sub>1</sub><sup>2</sup> and x<sub>1</sub>x<sub>2</sub>, which let the boundary curve; a high degree on few points can bend around individual points, so watch training accuracy rise while test accuracy falls.',
    learningRate: 'The learning rate is the size of each nudge to the weights. A small value learns slowly and may run out of iterations before it settles. A large value learns quickly, and a very large one overshoots, so the loss curve jumps about instead of falling smoothly.',
    l2: 'Regularisation adds a penalty for large weights, which discourages the model from bending itself around individual points. Off means no penalty; larger values pull every weight towards zero, giving a smoother and more cautious boundary with paler shading. Too much and the model can barely learn at all.',
  },
};
