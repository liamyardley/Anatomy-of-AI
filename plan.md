# Anatomy of AI: project plan

The living plan for turning the Anatomy of AI diagram into a hands-on course. Claude Code should read this file before starting any new piece of work, and update the status boxes when a piece is finished.

Status key: `[x]` built · `[~]` in progress · `[ ]` planned · `(idea)` not yet committed to

## 1. Vision

Anatomy of AI shows that AI is far more than chatbots. The home page is an Euler diagram of the fields of AI. Each region of the diagram becomes a door into one or more **labs**: interactive pages where people learn how that part of AI actually works by playing with it.

Audience: anyone curious about AI, with no assumed background. Every technical term is explained the first time it appears.

## 2. Principles (apply to every lab)

- **Play first, explanation second.** Users experiment, predict and work things out before the page tells them. Explanations of parameters are hidden behind toggles until revealed.
- **Train to see.** Models never update silently. Set parameters, press Train, watch it learn (with pause, step, speed and skip).
- **The maths is the model.** Formulas show the real fitted numbers, and a worked example shows the full calculation for any point the user picks.
- **Honest about limits.** When a technique fails, or a famous claim is shakier than the folklore, the page says so and turns it into an activity.
- **Predict, observe, explain** activities in every lab.
- **Field notes:** short, plain-English lessons from real projects in my MSc. These are lessons only, never published assignment solutions (see section 4).
- **Plain HTML, CSS and JavaScript, no build step,** hosted on GitHub Pages. Shared design tokens in `assets/css/theme.css`. Any new dependency needs agreeing first.
- **Accessible:** light and dark mode, keyboard operable, reduced motion respected, colour never the only cue.
- **Reproducible:** all randomness through one seeded generator, with a visible, editable seed.
- **UK spelling. No em dashes anywhere.**

## 3. Site structure

```
index.html                 home page: the Euler diagram, the map of the course
labs/<lab-name>/           one folder per lab
assets/                    shared CSS, shared JS engine (plot, training controls, RNG, optimisers)
notebooks/                 Colab projects (section 4)
datasets/                  small public datasets with data cards (section 4)
tests/                     Node unit tests
plan.md                    this file
```

Each region's panel on the home page lists its labs with a "Try it" link. A region can have several labs, and a lab can be linked from more than one region.

Shared engine, built once in the ML lab and reused everywhere: the 2D plot, the training controls, the seeded RNG, the model interface (`train` returning a stepper, `predictProba`, `explain`), and pluggable **optimisers** (gradient descent, SGD, momentum, RMSProp, Adam).

## 4. Coding labs

Two tiers, doing different jobs.

### 4a. Code-along (separate pages, alongside the labs)

The gap this fills: coding courses show code with no picture of what it is doing, and visual demos show pictures with no code. The code-along joins them, so a novice coder can see what each line does and what each argument tunes.

- **Where it lives (decided):** code-alongs are their own pages, not a panel inside the labs. The labs already hold a lot and stay as places to explore. Each code-along page takes one aspect of a lab, with a visualisation built for it (reused from the lab's engine, or new), so the existing labs do not need adapting.
- Each page builds up the Python for one idea in small stages. Each stage is a few lines with a plain-English explanation of what they do.
- **Each stage lights up the part of the visualisation it creates.** For example, for classification:
  1. load the data → the points appear on the plot;
  2. split into train and test → test points turn hollow;
  3. create the model → the parameter panel highlights;
  4. fit → the training animation runs, with the line currently executing highlighted;
  5. predict and evaluate → the regions shade and the confusion matrix fills in.
- **Two-way link between code and controls.** Moving a slider rewrites the argument in the code (`KNeighborsClassifier(n_neighbors=7)`), and editing the argument in the code moves the slider and marks the model as needing retraining.
- **Two versions of each model's code:** "from scratch" (numpy, matching the maths panel line for line) and "with scikit-learn" (what you would write in practice), so learners see that the library is doing the maths they have just watched.
- **Fill-in-the-gap exercises** with instant checks, in the same staged structure: worked example, then gaps, then an open extension.
- **Running real Python:** use Pyodide (Python with numpy and scikit-learn, running in the browser, nothing to install), loaded only on the code-along pages, as it is a large first download. To be confirmed with a prototype: load time, and whether scikit-learn is fast enough to drive the animations, or whether the JS engine animates while Python checks the learner's answers.
- **No hooks needed in the labs.** Because the code-alongs are separate pages, the labs do not have to emit stage events. The code-along pages reuse the shared engine (plot, models, training controls) directly.

### 4b. Colab projects (full projects on real data)

One full project per section, in a Colab notebook stored in `notebooks/` with an "Open in Colab" link from the relevant labs. These are where learners leave the toy 2D world and work on a real public dataset end to end.

Shape of every project: the question, exploring the data, a baseline, the models from the matching lab, tuning (with cross-validation), evaluation, and a short reflection. Built-in checks after each part. Solutions in a separate notebook.

**Datasets ready to play with.** Small ones stored in `datasets/` (CSV plus a data card: source, licence, columns, known quirks); larger ones loaded from their official source inside the notebook. A small helper (`load_dataset("penguins")`) so every notebook starts the same way. **Check each licence before committing a copy to the repo.**

| Dataset | Good for | Section |
| --- | --- | --- |
| Palmer Penguins | Trees, k-NN, clustering; a friendly first dataset | ML |
| Titanic | Trees and forests on categorical data, dummy variables, data leakage | ML |
| Breast Cancer Wisconsin (in scikit-learn) | Thresholds, sensitivity and specificity, ROC curves | ML |
| California Housing (in scikit-learn) | Regression, ridge and lasso, regularisation paths | ML, Optimisation |
| Wine (in scikit-learn) | PCA, clustering, feature scaling | ML (unsupervised) |
| Adult Income (UCI) | Categorical data, class imbalance, fairness | ML |
| SMS Spam Collection (UCI) | Naive Bayes, text features | ML, NLP |
| MNIST / Fashion-MNIST (in Keras) | Neural networks, CNNs | DL, CV |
| IMDB reviews (in Keras) | Sentiment classification | NLP |
| 20 Newsgroups (in scikit-learn) | TF-IDF, topic modelling | NLP |
| GloVe word vectors (Stanford) | Word vectors, similarity, analogies | NLP |

### 4c. Academic integrity

Coursework informs the lessons and field notes only. No assignment solutions are published, adapted or not. This matters especially for CS50 AI projects (such as Minesweeper), whose policy asks students not to post solutions publicly. Every lab and project uses its own problems and datasets, and its code is written fresh.

**Sources of ideas.** The CS50 AI syllabus (https://cs50.harvard.edu/extension/ai/2025/fall/) is a useful map of topics: search, knowledge, uncertainty, optimisation, learning, neural networks, language. The topics and the standard ways of picturing them are common across textbooks and courses, so they can inform what a lab covers. Its projects, starter code and solutions are not reused.

## 5. Sections by region of the diagram

### 5.1 Artificial intelligence (outside machine learning)

AI that does not learn from data: search, logic, planning and optimisation.

- [ ] **Search lab:** a maze with editable walls; breadth-first, depth-first, greedy and A* search race, showing the frontier expanding and the path found. Compare nodes explored.
- [ ] **Game-playing lab:** minimax on noughts and crosses with a visible game tree; alpha-beta pruning greys out branches it never needs to look at.
- [ ] **Logic lab:** a knowledge-based agent for a Minesweeper-style puzzle (or another logic puzzle), showing the facts it knows and the conclusions it draws. Written from scratch (see 4c).
- [ ] **Optimisation lab:** hill climbing and simulated annealing on a landscape; constraint satisfaction on a relatable problem such as an exam timetable or Sudoku.
- (idea) **Uncertainty lab:** a small Bayesian network or Markov chain to play with.
- (idea) **Expert system:** a rule-based system, to show the "AI before learning" era.

### 5.2 Machine learning

**Lab 1: Classification (2D)**: see `ML-Lab-brief.md` for the full specification.

Done so far (specified in the brief), live at https://liamyardley.github.io/Anatomy-of-AI/labs/ml/:
- [x] Phase 1: scaffold, shared theme, seeded RNG, premade datasets (Two clouds, Ring, Checkerboard, Moons, Noisy islands, Spiral), Pure noise, Mystery, Regenerate, Add more points, Explore and Edit modes, train/test split
- [x] Phase 2: logistic regression, training engine, region shading, decision threshold, sensitivity, specificity, confusion matrix, live maths, worked example, hidden parameter explanations
- [x] Phase 3: k-nearest neighbours
- [x] Phase 4: decision tree with linked tree diagram
- [x] Phase 5: random forest with thumbnails and accuracy-against-trees chart
- [x] Phase 6: comparison table, ROC curve, predict-observe-explain activities, field notes, accessibility and mobile passes, README

Notes on what was built (where it differs from, or adds to, the brief):
- The page is four headed blocks: (1) the model, the data and training; (2) from weights to probabilities; (3) results; (4) compare the models.
- Results also show precision, recall and F1.
- k-NN already has distance weighting (on or off). Its only distance so far is Euclidean.
- The Checkerboard's crossing point is always off-centre. With four equal blocks a decision tree has no first split worth making.
- Noisy islands does not single out the random forest: at default settings logistic regression and k-NN do as well. The activity text says so. Open: change the generator, or leave it.
- The RNG, formatting helpers and training engine now live in `assets/js/` (moved for Lab 2). The plot and the models are still in `labs/ml/js/`.
- Still to do from the brief's checks: try it with reduced motion, on a real touch device, and with a screen reader.
- Not built yet from section 3: the optimiser module (planned with Lab 2, below).

Step 1: **more parameters on the existing models** (done, apart from one tidy-up)

The lab is about what each model does to classify the data, so its settings are the ones that change what the model is and the shape of its boundary. Settings about how a model is fitted, and how settings are chosen, move out to Lab 2 (see "The split" below).
- [x] k-NN **distance metrics**, each visualised the way Euclidean is (the user sees the "neighbourhood shape" around a query point):
  - Euclidean: a circle;
  - Manhattan: a diamond;
  - Chebyshev: a square;
  - **Minkowski with a p slider**, morphing the shape smoothly from diamond (p = 1) to circle (p = 2) towards square (large p);
  - **Cosine:** ignores how far a point is from the origin and compares only direction. Visualise it by sliding every point onto the unit circle (animated), then finding neighbours by angle, so the neighbourhood is a wedge from the origin and the regions become slices radiating outwards. The Ring dataset becomes impossible (inner and outer points share directions), which is the lesson. This also sets up the word vectors lab, where similar words cluster on a sphere by angle.
- [x] Decision tree: Gini or entropy; minimum samples to split; maximum number of leaves; **cost-complexity pruning** (tree regularisation).
- [x] Random forest: minimum samples per leaf; sample size per tree.
- [ ] Tidy-up: move logistic regression's **learning rate** behind "More settings". (The link from it to Lab 2 is done, along with one in the lab's introduction.) Regularisation strength (L2) stays on show, because the "high degree overfits, regularisation pulls it back" lesson needs it.

Moved to Lab 2 (were in step 1): L1 and elastic net penalties, the choice of optimiser, k-fold cross-validation, the train/test split ratio, and the sweep.

Decided for step 1:
- The settings each model has today stay on show; the new ones sit behind a **"More settings"** toggle, so the lab stays approachable.
- With a maximum number of leaves, the tree splits the group with the biggest impurity drop first (as scikit-learn does) instead of growing level by level.
- Groundwork done: settings can be drop-downs, can appear only when relevant, and can sit behind "More settings"; the plot can draw outlines of any shape.
- The explanation for k-NN and the tree says which distance or impurity measure is in use, with a link to the setting.

Tabled for later (not in step 1):
- A **"stretch x₁ by 10"** switch on the data, plus a **scale features** option: shows k-NN (and later SVM and neural networks) breaking on unscaled data while trees are unaffected.
- **Class weights**, with an unbalanced dataset option. Touches all four models.

Next, step 2: **more models**, in this order
- [ ] Gradient boosting: each step adds one tree fitted to the previous trees' mistakes; the contrast with the forest (average of independent trees vs a chain of corrections).
- [ ] Support vector machine: margin band and support vectors; C and gamma; the kernel trick shown by lifting the Ring dataset into 3D (x₁² + x₂²) where a flat plane separates it.
- [ ] Perceptron: the simplest learner and the step into neural networks; on data a line cannot separate, it never settles.
- [ ] Naive Bayes: curved boundaries, and its independence assumption failing on the diagonal Two clouds.

Next, step 3: **regression mode** (one input, a continuous output, click to place points): polynomial regression, k-NN regression, a tree as a step function, a forest as smoothed steps. Bridge into the neural network lab.

**The split between Lab 1 and Lab 2**

- **Lab 1 (classification): what a model does.** Settings that change the model itself and the shape of its boundary: k and the distance measure, depth and pruning, feature degree, number of trees.
- **Lab 2 (training): how a model gets there.** Two topics, which may or may not end up on one page:
  1. **Fitting: how a model finds its numbers.** Gradient descent, learning rate, optimisers, batch size, L1 and L2. This applies to models with weights (logistic regression now, neural networks later). Trees and k-NN do not train this way, and the lab says so.
  2. **Choosing: how settings are picked honestly.** Train/test split, k-fold cross-validation, sweeps, grid search, the bias-variance curve. This applies to every model.
- **Decision deferred:** build Fitting first as its own page, look at it, and only then decide whether Choosing joins it or becomes a separate lab.

**Lab 2: How a model finds its numbers (fitting)** at `labs/training/` (linked from Machine learning and Deep learning on the home page, and from Lab 1). Built.

Built in phases, each committed and reviewed, as Lab 1 was:
- [x] Phase 0, shared engine: the seeded RNG, the number and formula helpers and the training engine moved from `labs/ml/js/` into `assets/js/`, and the labs' shared styles into `assets/css/lab.css`, with Lab 1 still passing its tests and working as before. The **optimiser module** is there too (`assets/js/optimisers.js`): gradient descent, momentum, RMSProp and Adam behind one small interface, pure and unit tested. Not moved yet: the logistic regression model, which stays in Lab 1 until phase 4 needs it.
- [x] Phase 1, one weight: a tiny dataset and a model with a single number to find. Drag the weight by hand and watch the loss change; try to find the lowest point before pressing Train. Then gradient descent takes steps down the curve. The learning rate goes from crawling, to overshooting, to blowing up.
- [x] Phase 2, two weights: the loss as a contour map. Click a start point and watch the path. Surfaces: a bowl, a long narrow valley, a saddle, and one with several dips (where the start point decides the answer).
- [x] Phase 3, optimisers and batch size: race gradient descent, momentum, RMSProp and Adam from the same start. Full batch, mini-batch and single-point updates (smooth against noisy paths).
- [x] Phase 4, the real thing: the true loss surface of a logistic regression on a Lab 1 dataset, with the path on the contour map and the boundary moving on the data plot side by side: "this is what Train was doing".
- [x] Phase 5, regularisation: the L2 circle and L1 diamond meeting the loss contours, a strength slider, and why L1 (lasso) sets weights to exactly zero while L2 (ridge) only shrinks them. Coefficient paths as the strength grows (ridge shrinks, lasso selects, elastic net compromises).
- [x] Phase 6: predict-observe-explain activities, field notes, accessibility and mobile passes, home page link, README.

Notes on what was built (where it differs from, or adds to, the plan):
- Phase 3: each optimiser in the race has its own learning rate, and the page says the race is not a fair one. A runner counts as arrived after three steps in a row within 0.05 of a flat spot, because RMSProp hovers instead of stopping.
- Phase 4: the model has no bias, so that its two weights fit on a map; its line always passes through the centre of the data. The page says so. The logistic regression model itself was not moved out of Lab 1: Part 4 has its own small two-weight version (`logistic2.js`).
- Phase 5: the map uses a two-weight linear model with a mean squared loss (clean elliptical contours), and the sweep uses six features of which three are irrelevant. The elastic net is fixed at half L1, half L2.
- Phase 6: the notes at the end of some parts are general practical notes labelled "In practice", written by Claude. They are not field notes from coursework; replace or add real ones in `labs/training/content/notes.js` (`PRACTICE_NOTES`).
- Not done: a check with a screen reader or on a real touch device.

Decided for Lab 2:
- L1 uses a soft-threshold step after each update, so lasso sets weights to exactly zero.
- Mini-batch and single-point training stop on a smoothed loss or after a fixed number of passes through the data, and have a batch size setting.

**Choosing settings honestly** (where it lives is undecided: part of Lab 2 or its own lab)
- [ ] Train/test split ratio.
- [ ] **k-fold cross-validation** with the folds shown on the plot.
- [ ] **Sweep:** train across a range of one setting and plot training and test accuracy against it (the bias-variance U-curve).
- [ ] Grid search over two settings at once, shown as a heat map.
- [ ] Early stopping as regularisation; the bias-variance trade-off with a polynomial and a strength slider.

**Lab 3: Finding groups without labels** (unsupervised learning)
- [ ] Reuses the Lab 1 plot and datasets with the colours removed: "can the algorithm recover the groups?", then reveal the true colours and compare.
- [ ] k-means: animate the assign-points, move-centres loop; k slider; the elbow and silhouette scores for choosing k; fails on Moons and Ring.
- [ ] DBSCAN: neighbourhood radius and minimum points; shows core, border and noise points; succeeds on Moons and Ring, struggles when clusters have different densities.
- [ ] Hierarchical clustering with a dendrogram you can cut at different heights.
- (idea) PCA: finding the directions of greatest spread; projecting 3D data down to 2D.

**Lab 4: Reinforcement learning** (linked from Agentic AI)
- [ ] A gridworld where an agent learns by trial and error (Q-learning), with the value of each square shading in as it learns; exploration rate and reward controls.

**Colab projects:** Titanic (trees, forests, categorical data, leakage); Breast Cancer (thresholds and ROC); California Housing (regularisation); Penguins or Wine (clustering).

### 5.3 Deep learning

- [ ] **Neural network lab:**
  - 1D first: each hidden ReLU neuron is a hinge, shown on its own and summed, building a curve from straight pieces;
  - XOR: why one straight line fails;
  - 2D builder: inputs, number of layers and neurons per layer, activation function; hand-drag weights before training; reuses the Lab 1 engine, datasets and optimisers (the Spiral finally solved).
- [ ] Backpropagation as the chain rule on a visible computational graph.
- [ ] Dropout and early stopping (linking back to ML Lab 2).
- **Colab project:** a dense network on Fashion-MNIST, tuning architecture, optimiser and regularisation.

### 5.4 Computer vision

- [ ] **Convolution playground:** apply hand-set filters (blur, sharpen, edge detection) to an image and see the feature map, before any learning.
- [ ] Pooling and stacking layers.
- [ ] Draw a digit and watch a small trained CNN classify it, with its feature maps shown. Model trained in Colab and loaded in the browser (needs a runtime dependency: to be agreed).
- [ ] Link to CNN Explainer (Polo Club) for the full architecture.
- **Colab project:** a CNN on MNIST, then Fashion-MNIST or CIFAR-10.

### 5.5 Natural language processing

- [ ] **From words to numbers:** tokenisation, bag of words and TF-IDF on short texts.
- [ ] **Word vectors lab** (pretrained GloVe subset):
  - similar words cluster; cosine similarity (linking to the ML cosine distance);
  - user-defined axes from word pairs (man to woman, man to king), where analogies show up as parallelograms;
  - a 3D unit sphere view, with an honest note about what squashing hundreds of dimensions into three distorts;
  - honest caveats as activities: opposites usually sit close together, not opposite; famous analogies only work when the input words are excluded.
- (idea) Topic modelling on a small corpus.
- **Colab project:** sentiment on IMDB reviews, from TF-IDF with classical models to a small neural network.

### 5.6 Generative AI

- [ ] **Next-character generator:** a small model trained on a text, generating one character at a time; temperature slider from repetitive to chaotic.
- [ ] **Denoising in 2D:** points scattered as noise gradually denoised into a spiral or moons, as a picture of how diffusion models work (reuses the 2D plot).
- (idea) A tiny GAN in 2D: a generator and a judge competing.

### 5.7 Large language models

- [ ] **Attention lab:** a toy attention calculation on one sentence with hand-made vectors (what does "it" attend to?), then real attention patterns.
- [ ] **Masked words (BERT):** hide a word, see the model's top guesses, running in the browser (transformers.js: dependency to be agreed).
- [ ] **Next-token prediction (GPT-style):** the probability of each next token, with temperature and top-k sampling.
- **Colab project:** fine-tune a small BERT model for text classification and compare it with the classical models from the NLP project.

### 5.8 Agentic AI

- [ ] **Agent loop:** a scripted walk-through of goal, plan, tool call, observation, repeat, with each part labelled (no live API needed).
- [ ] Link to the reinforcement learning gridworld (ML Lab 4) and the search lab (5.1): agents without language models.

## 6. Roadmap

1. ML Lab 1, phases 1 to 6 (`ML-Lab-brief.md`): done
2. ML Lab 1, step 1: more parameters, including distance metrics: done, apart from one tidy-up
3. ML Lab 2, fitting (how a model finds its numbers), phases 0 to 6: done
4. **Decide where "choosing settings honestly" lives, then build it**: next
5. ML Lab 1, step 2: more models
6. **Code-along prototype** as its own page (k-NN first): tests the most novel idea before many more labs are built around it
7. ML Lab 3: unsupervised
8. Deep learning: neural network lab
9. First Colab project (Titanic) and the `datasets/` folder
10. Remaining regions, ordered by interest

## 7. Open questions

- Does "choosing settings honestly" (cross-validation, sweeps, grid search) belong in Lab 2 with fitting, or in a lab of its own? Decide after the fitting lab is built.
- Pyodide or another approach for running Python in the code-along: decide after the prototype.
- Runtime dependencies for the CV and LLM labs (TensorFlow.js, ONNX Runtime Web, transformers.js).
- Whether the home page needs a course-style route through the labs as well as the map.
