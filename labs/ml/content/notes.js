// Editable copy for the Machine Learning Lab.
//
// Everything a visitor reads that is likely to be reworded lives here, so
// the wording can change without touching the code: dataset descriptions,
// model and setting explanations, the predict-observe-explain activities
// and the field notes.
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
    text: 'Four rectangular blocks in alternating colours, a little blurred where they meet. Each colour appears in two opposite corners.',
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
  roc: 'Moving the threshold trades one kind of mistake for another. The <strong>ROC curve</strong> (short for receiver operating characteristic, a name left over from radar engineering) shows that trade for every possible threshold at once. Each point on the curve is one threshold: how many blue points it catches, plotted against how many red points it wrongly calls blue. The <strong>dot is the current threshold</strong>, so it slides along the curve as you move the slider. A curve that hugs the top-left corner belongs to a model that tells the colours apart well; the dashed diagonal is what guessing would give. The <strong>area under the curve</strong> (AUC) sums it up in one number: 1 is perfect and 0.5 is guessing.',
  rocEmpty: 'Train a model to draw its ROC curve.',
  rocOneClass: 'An ROC curve needs both red and blue points, and these points are all one colour.',
};

// --- comparing the models ---

export const COMPARE_COPY = {
  empty: 'Press Train all to fill in the table. Nothing is animated, so it takes a moment at most.',
  done: 'All models trained on the current dataset.',
  stale: 'The data or a setting has changed: press Train all to update the table.',
  noData: 'There are no training points yet. Add some points first.',
  note: 'Accuracy is measured at the current decision threshold and follows the slider. Training time is how long your browser took, without animation; k-nearest neighbours looks quickest because it only stores the points and does its real work later, each time it predicts. Select a model\'s name to switch to it.',
};

// --- predict, observe, explain ---
// One activity per dataset. `explain` is hidden until the visitor asks for it.

export const ACTIVITY_COPY = {
  question: 'Which model do you think will do best on the test points here, and what will its boundary look like?',
  questionMystery: 'Here is an unknown dataset. Study the points: which model will win?',
  notRun: 'Nothing trained yet.',
  stale: 'The data or a setting has changed since the models were trained. Train them again.',
  mysteryHidden: 'Press Reveal (beside Regenerate) to find out what this dataset was. The explanation will then appear here.',
};

export const ACTIVITY_NOTES = {
  clouds: 'Logistic regression usually comes first or equal first. The two clouds sit either side of one straight, slanting line, and a straight line is exactly what logistic regression draws. k-nearest neighbours and the random forest come close, but they bend around stray points in the overlap. The decision tree has the hardest time: it can only cut straight across or straight down, so it has to copy a slanting line with a staircase. Nobody reaches 100%, because the clouds overlap.',
  ring: 'With a feature degree of 1, logistic regression fails completely: no straight line can separate a disc from the ring around it. k-nearest neighbours and the random forest cope well, because they work locally instead of drawing one line. A shallow decision tree manages only a boxy approximation of the circle. Now set the feature degree to 2 and train logistic regression again. Given the squares of the features it can draw a circle, and it becomes one of the best, with the smoothest boundary of all. A "linear" model can draw curves once it is given the right features.',
  checker: 'Expect a near tie between the decision tree, the random forest and k-nearest neighbours. What sets the tree apart is how little it needs: the true boundaries run straight across and straight down, which are the only cuts a tree can make, so two questions are enough. Set its maximum depth to 2 and see. Logistic regression with a feature degree of 1 fails, since no single straight line separates opposite corners. Try a degree of 2 and watch what the x<sub>1</sub>x<sub>2</sub> term does.',
  moons: 'k-nearest neighbours usually wins. The crescents are curved, and each point\'s colour is best judged from the points right beside it, which is all k-nearest neighbours ever does. The random forest is close behind. The decision tree builds a staircase around the curves. Logistic regression with a feature degree of 1 draws one straight line through both moons; raising the degree to 3 lets it bend, and it improves.',
  islands: 'Expect a near tie, with nobody much above 90%. About one label in ten is wrong on purpose, and the wrong labels are wrong in the test points too, so that is a ceiling no model can break. The interesting part is who falls below it. Set the decision tree\'s maximum depth to 10 and train it: it chases the wrong labels, drawing small boxes around single points, so its training accuracy soars while its test accuracy drops. The random forest also grows fairly deep trees, yet it holds up, because averaging many trees smooths those mistakes away. A simple model such as logistic regression often does as well as anything here, precisely because it cannot chase single points.',
  spiral: 'k-nearest neighbours copes far better than the rest, because it only needs nearby points to agree with each other. A deep tree or a large forest gets partway, with a blocky imitation of the arms. Logistic regression cannot follow them at any feature degree on offer. Now make it harder: lower the number of points, or raise the noise, and watch k-nearest neighbours start to struggle too. A pattern like this really needs a model that can build its own features out of many small pieces, which is what a neural network does: the subject of the next lab.',
  noise: 'A trick question: no model can beat a coin toss here. The colours were decided at random, so every model scores about 50% on the test points. Now look at the training accuracy. k-nearest neighbours with k = 1, or a decision tree with a large maximum depth, scores 100% on the training points while learning nothing at all. That gap between training and test accuracy is called <strong>overfitting</strong>, and this is its plainest form.',
  blobs: 'This one was a handful of random blobs of each colour. k-nearest neighbours and the random forest usually do best, because blobs are local clumps and both models judge a point by what is near it. Logistic regression only does well if the blobs happen to fall either side of a single line.',
};

// --- field notes ---
// Short lessons from real project work, shown beside each model.
// Each is { title, text }.

export const FIELD_NOTES = {
  logistic: [
    {
      title: 'Look past the accuracy figure',
      text: 'In a project predicting stock market moves from headline sentiment, a logistic model reported respectable accuracy while predicting "up" for every single day. Only the confusion matrix revealed it.',
    },
    {
      title: 'More features, more overfitting',
      text: 'Fitting every squared and interaction term at once overfitted badly. Trimming terms back one at a time gave a model that held up on new data.',
    },
  ],
  knn: [
    {
      title: 'The choice of "closeness" matters as much as k',
      text: 'On text, k-nearest neighbours using cosine similarity (which compares the direction of two sets of numbers, not their size) beat the same method using straight-line distance by a wide margin.',
    },
  ],
  tree: [
    {
      title: 'Easy to explain, quick to change its mind',
      text: 'A single tree was the easiest model to explain to anyone, and the least stable: small changes to the data produced a noticeably different tree.',
    },
  ],
  forest: [
    {
      title: 'More trees is not the lever',
      text: 'Bagging (training each tree on its own bootstrap sample) and random forests cut the test error of a single tree substantially. But growing a forest from 100 to 5,000 trees did not reliably improve accuracy; tree depth, and how similar the trees are to each other, mattered more.',
    },
  ],
  general: {
    title: 'Test performance is the score that counts',
    text: 'When several models were compared on the same training data, the most flexible one won on the training data and was the least stable on the test data.',
  },
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
    threshold: '',
  },

  knn: {
    title: 'k-nearest neighbours',
    idea: 'k-nearest neighbours (k-NN for short) has no formula to fit. To classify a point, it finds the <strong>k</strong> training points closest to it, its <strong>neighbours</strong>, and lets them vote. How "closest" is measured is a choice. The usual one is the ordinary straight-line distance between two points, called the <strong>Euclidean distance</strong>.',
    distanceKey: 'Here (x<sub>1</sub>, x<sub>2</sub>) is the point being classified, (a<sub>1</sub>, a<sub>2</sub>) is a stored training point and d is the distance between them.',
    vote: 'The probability of blue is simply the share of the k neighbours that are blue. With <strong>distance weighting</strong> switched on, each neighbour\'s vote counts for more the closer it is.',
    training: 'There is <strong>no training</strong> in the usual sense. Pressing Train only copies the training points into memory. All the real work happens later, at <strong>prediction time</strong>: every single prediction means measuring the distance to every stored point.',
    memory: 'This table is the whole model: a copy of the training points and nothing else. No weights, no rules.',
    memoryUntrained: 'The memory is empty. Press Train to watch the training points being copied in.',
    filling: 'Once the points are stored, the plot fills in from the top. Each patch of the plot asks its own k nearest neighbours which colour it should be.',
    picture: 'The stored points nearest to your query point, lined up by distance. Everything inside the shaded band is one of the k nearest and gets a vote; everything beyond it is ignored. On the plot, the same k points are joined to the query point, and the dashed outline reaches exactly as far as the furthest of them. Its shape is the shape of "equally far away" under the chosen distance measure.',
    // One line per distance measure, shown under its formula.
    metrics: {
      euclidean: 'Euclidean distance is the straight-line distance. Everything equally far from a point lies on a <strong>circle</strong>.',
      manhattan: 'Manhattan distance adds the gap across to the gap up, like walking along a grid of city streets. Everything equally far from a point lies on a <strong>diamond</strong>.',
      chebyshev: 'Chebyshev distance takes only the bigger of the two gaps, across or up. Everything equally far from a point lies on a <strong>square</strong>.',
      minkowski: 'Minkowski distance is a whole family, set by a power p. With p = 1 it is Manhattan distance, with p = 2 it is Euclidean, and as p grows it approaches Chebyshev: the outline swells from a diamond, through a circle, towards a square.',
      cosine: 'Cosine distance ignores how far a point is from the centre of the plot and compares only its <strong>direction</strong>. Here &theta; (theta) is the angle between the two points as seen from the centre: the distance is 0 when they lie in the same direction. Everything equally far from a point lies along the edges of a <strong>wedge</strong> spreading out from the centre.',
    },
    threshold: 'With k-nearest neighbours the probability can only take a few values (with k = 5: 0, 0.2, 0.4, 0.6, 0.8 or 1), so as the threshold moves the boundary stays put and then jumps, instead of gliding.',
  },

  tree: {
    title: 'Decision tree',
    idea: 'A decision tree asks a series of yes/no questions, such as "is x<sub>1</sub> less than 0.3?". Each question <strong>splits</strong> a group of training points into two smaller groups. The tree picks the question that leaves the two groups as <strong>pure</strong> as possible, meaning as close as possible to all one colour, then repeats the process on each group.',
    giniKey: 'Purity is measured with <strong>Gini impurity</strong>, G. Here p<sub>red</sub> and p<sub>blue</sub> are the shares of red and blue points in a group. G is 0 when the group is all one colour and 0.5 when it is half and half.',
    gainKey: 'For every possible question the tree works out the <strong>impurity drop</strong>: the impurity of the group before the split, minus the impurity of the two new groups (each counted in proportion to its size, n<sub>yes</sub> and n<sub>no</sub> out of n). The question with the biggest drop wins.',
    training: 'A group that is not split any further is called a <strong>leaf</strong>. To classify a new point, follow the questions down to a leaf: the probability of blue is the share of blue training points in that leaf. Each training step here is <strong>one split</strong>.',
    diagram: 'The tree, growing one split at a time. Yes goes left and no goes right. Hover over a box, or focus the diagram and use the arrow keys, to light up its rectangle on the plot. Move the query point on the plot and its route through the tree is outlined here.',
    diagramUntrained: 'Press Train to watch the tree grow, one question at a time.',
    candidates: 'For the selected group, the best question the tree could find for each feature. It keeps the one with the bigger impurity drop.',
    picture: 'The curve shows Gini impurity for every possible mix of colours: highest for an even mix, zero for a pure group. The ring marks the selected group, and the two dots are the groups its question creates. A good question moves the dots down the curve, towards the pure ends.',
    threshold: 'A decision tree gives every point in a leaf the same probability, so only a handful of values ever occur. As the threshold moves, the boundary stays put and then jumps as whole leaves change colour, instead of gliding.',
  },

  forest: {
    title: 'Random forest',
    idea: 'A single decision tree is unreliable: change a few training points and it can grow quite differently. A random forest grows <strong>many trees</strong>, makes each one a little different on purpose, and averages their answers. The individual mistakes tend to cancel out.',
    different: 'Two things make the trees differ. Each tree learns from its own <strong>bootstrap sample</strong>: a random sample of the training points, the same size as the original, drawn <strong>with replacement</strong>, so some points appear several times and about a third are left out. And at each split a tree may be allowed to look at only <strong>some of the features</strong>, chosen at random.',
    formulaKey: 'Here T is the number of trees and p<sub>t</sub> is the probability of blue given by tree number t. Each tree is a decision tree, grown by choosing the questions that leave the purest groups. Each training step here grows <strong>one whole tree</strong>.',
    oob: '<strong>Out-of-bag accuracy</strong> is a free test. Every training point was left out of some trees\' bootstrap samples, so it can be scored using only the trees that never saw it.',
    oobOff: 'Out-of-bag accuracy needs bootstrap samples, so it is not available with bootstrap switched off.',
    thumbnails: 'The first nine trees, each drawn on its own. Every one is different, and every one is rough. The main plot shows their average, which is smoother than any of them. The dot marks your query point.',
    thumbnailsUntrained: 'Press Train to watch the trees appear, one at a time.',
    curve: 'Accuracy on the test points as trees are added (with the threshold at 0.5). It climbs quickly, then levels off: beyond a certain point, extra trees stop helping.',
    curveNoTest: 'There are no test points to measure accuracy on.',
    picture: 'One dot for every tree, placed at the probability of blue it gives your query point. Where the dots pile up at one end, the trees agree. Where they are spread out, the forest is unsure. The solid line is their average: the forest\'s answer.',
    threshold: 'A random forest averages many trees, so its probability moves in small steps (with 50 trees, steps of about 0.02). The boundary glides more smoothly than a single tree\'s, and more smoothly still with more trees.',
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
  knn: {
    k: 'k is how many neighbours get a vote. With k = 1 each point is given the colour of its single nearest neighbour, so the boundary is jagged and wraps around every stray point. Larger values smooth the boundary; very large values smooth away real detail as well.',
    metric: 'This decides what "closest" means. Each measure gives the neighbourhood a different shape, drawn as a dashed outline around your query point: a circle, a diamond, a square, or (for cosine) a wedge spreading out from the centre. Try each on the same dataset, then try cosine on the Ring: can you see why it cannot work there?',
    p: 'The power p bends the neighbourhood from one shape to another. At 1 it is a diamond (Manhattan), at 2 a circle (Euclidean), and as p grows it squares off towards Chebyshev. Slide it and watch the outline around your query point change.',
    weighted: 'When this is on, a neighbour\'s vote counts for more the closer it is (each vote is worth 1 divided by its distance). It matters most when k is large: far-off neighbours still vote, but quietly. Look for the boundary hugging nearby points more tightly.',
  },
  tree: {
    maxDepth: 'The maximum depth is how many questions in a row the tree may ask. A shallow tree can only draw a few large rectangles. A deep tree keeps cutting until it has boxed in individual points, so watch training accuracy climb towards 100% while test accuracy stalls or falls.',
    minLeaf: 'This is the smallest number of training points a leaf may hold. At 1 the tree is free to make a leaf for a single odd point. Larger values forbid such tiny boxes, giving a simpler tree that ignores stray points.',
  },
  forest: {
    trees: 'How many trees to grow. One tree alone is rough and unreliable; adding trees steadies the average. Watch the accuracy chart: it rises quickly and then flattens, after which more trees only cost time.',
    maxDepth: 'How many questions in a row each tree may ask. Deep trees each fit their own sample closely, mistakes included, and the forest relies on averaging to cancel those mistakes out. Shallow trees are simpler but may all miss the same detail.',
    bootstrap: 'When this is on, each tree learns from its own random sample of the training points, drawn with replacement. When it is off, every tree sees exactly the same points, so the trees can only differ through the features they are allowed to look at.',
    featuresPerSplit: 'How many of the two features a tree may look at when choosing each question. With 1, each split is made on a feature picked at random, which makes the trees more different from one another; that difference is what makes averaging work. With 2, every tree always takes the best split, so the trees look more alike.',
  },
};
