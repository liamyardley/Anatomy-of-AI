// Wiring for the Machine Learning Lab: page state, controls, the training
// loop and the plot.
//
// This file never mentions a particular model. It works through the list in
// registry.js and the common model interface, so a new model needs no
// changes here.

import { createRng } from './rng.js';
import {
  DATASET_TYPES, DEFAULT_POINT_COUNT, createDataset, buildPoints, morePoints, kindLabel,
} from './datasets.js';
import {
  DEFAULT_TRAIN_SHARE, drawIsTest, countPoints, confusion, summarise,
} from './metrics.js';
import { createPlot } from './plot.js';
import { createEngine } from './engine.js';
import { MODELS } from './registry.js';
import { fmt, pct, setValue, prefersReducedMotion } from './format.js';
import {
  DATASET_NOTES, COPY, TRAIN_COPY, PARAM_NOTES, MODEL_NOTES,
} from '../content/notes.js';

const $ = (id) => document.getElementById(id);
const els = {
  dataset: $('datasetSel'), model: $('modelSel'), seed: $('seedInput'),
  regen: $('regenBtn'), more: $('moreBtn'), reveal: $('revealBtn'),
  noise: $('noiseRange'), noiseOut: $('noiseOut'),
  count: $('countRange'), countOut: $('countOut'),
  classToggle: $('classToggle'), clear: $('clearBtn'), showTest: $('showTest'),
  modeHint: $('modeHint'), status: $('status'), announce: $('announce'),
  datasetTitle: $('datasetTitle'), datasetText: $('datasetText'), revealNote: $('revealNote'),
  train: $('trainBtn'), pause: $('pauseBtn'), step: $('stepBtn'), skip: $('skipBtn'),
  trainMessage: $('trainMessage'),
  threshold: $('thresholdRange'), thresholdOut: $('thresholdOut'),
  main: $('labMain'),
  modelIntro: $('modelIntro'), modelDetail: $('modelDetail'), modelWide: $('modelWide'),
  params: $('paramControls'), showAll: $('showAll'),
};

const MAX_SEED = 999999;
const REMOVE_REACH = 0.2; // how close a point must be to be removed
const FULL_GRID = 120;    // resolution of the shaded regions once training has finished

const state = {
  // --- data ---
  type: 'clouds',        // a dataset key, or 'edited' for a premade set changed by hand
  seed: 42,
  dataset: null,         // { type, kind, seed, settings } from datasets.js
  points: [],
  noise: 0.5,            // slider position, 0 to 1 (0.5 is the default amount)
  count: DEFAULT_POINT_COUNT,
  trainShare: DEFAULT_TRAIN_SHARE,
  showTest: false,       // test points start hidden
  mode: 'explore',       // 'explore' or 'edit'
  activeClass: 1,        // colour of new points in Edit mode: 0 red, 1 blue
  query: null,           // { x1, x2 }
  batch: 0,              // how many times "Add more points" has been pressed
  revealed: false,       // Mystery only
  manualRng: null,       // decides train or test for points added by hand

  // --- model ---
  modelKey: Object.keys(MODELS)[0],
  params: {},            // per model: { settingKey: value }
  model: null,           // the model on show: training, trained, or out of date
  phase: 'untrained',    // 'untrained', 'training' or 'done'
  stale: false,          // the model on show no longer matches the settings or data
  snapshot: null,        // the latest training snapshot
  scores: null,          // probabilities for every point, captured when training finished
  threshold: 0.5,
  speed: 'normal',
  matrixSet: 'test',     // which points the confusion matrix counts
  panelOverlay: [],      // extra shapes the maths panel wants on the plot
};

for (const [key, spec] of Object.entries(MODELS)) {
  state.params[key] = Object.fromEntries(spec.params.map((p) => [p.key, p.initial]));
}

const spec = () => MODELS[state.modelKey];
// A new, untrained model with the current settings. Every model is also given
// the lab's seed, so one that needs randomness stays reproducible.
const createModel = () => spec().create({ ...state.params[state.modelKey], seed: state.seed });
const isCustom = () => state.type === 'custom' || state.type === 'edited';
const noiseMultiplier = () => state.noise * 2;
const className = (y) => (y === 1 ? 'blue' : 'red');

function announce(message) {
  els.announce.textContent = message;
}

// ===== Data =====

// Start a fresh dataset of the current type from the current seed.
function rebuild() {
  state.dataset = createDataset(state.type, state.seed);
  state.points = buildPoints(state.dataset, state.count, noiseMultiplier(), state.trainShare);
  state.batch = 0;
  state.revealed = false;
  state.manualRng = createRng(state.seed, 'manual');
  invalidate();
  render();
}

// The first hand edit turns a premade dataset into "Custom (edited)".
function markEdited() {
  if (isCustom()) return;
  state.type = 'edited';
  state.dataset = { type: 'custom', kind: 'custom', seed: state.seed, settings: null };
  syncDatasetPicker();
}

function syncDatasetPicker() {
  let option = els.dataset.querySelector('option[value="edited"]');
  if (state.type === 'edited' && !option) {
    option = new Option('Custom (edited)', 'edited');
    els.dataset.add(option);
  } else if (state.type !== 'edited' && option) {
    option.remove();
  }
  els.dataset.value = state.type;
}

function addPoint(where, otherClass) {
  markEdited();
  const y = otherClass ? 1 - state.activeClass : state.activeClass;
  const test = drawIsTest(state.trainShare, state.manualRng);
  state.points.push({ x1: where.x1, x2: where.x2, y, test });
  const hiddenNote = test && !state.showTest ? ' It is hidden: tick "Show test points" to see it.' : '';
  announce(`Added a ${className(y)} ${test ? 'test' : 'training'} point at ${fmt(where.x1)}, ${fmt(where.x2)}.${hiddenNote}`);
  invalidate();
  render();
}

function removeNearest(where) {
  // Hidden test points cannot be removed: you should see what you delete.
  let best = -1;
  let bestDistance = REMOVE_REACH;
  state.points.forEach((p, i) => {
    if (p.test && !state.showTest) return;
    const d = Math.hypot(p.x1 - where.x1, p.x2 - where.x2);
    if (d < bestDistance) { best = i; bestDistance = d; }
  });
  if (best === -1) {
    announce('No point close enough to remove.');
    return;
  }
  markEdited();
  const [removed] = state.points.splice(best, 1);
  announce(`Removed a ${className(removed.y)} point.`);
  invalidate();
  render();
}

// Everything about the data and its controls.
function render() {
  const custom = isCustom();
  const counts = countPoints(state.points);

  plot.update({
    points: state.points, showTest: state.showTest, query: state.query, mode: state.mode,
  });

  syncDatasetPicker();
  els.seed.value = state.seed;
  els.noiseOut.textContent = state.noise === 0 ? 'none' : `${Math.round(state.noise * 200)}%`;
  els.countOut.textContent = custom ? counts.total : state.count;
  els.regen.disabled = custom;
  els.noise.disabled = custom || state.dataset.kind === 'noise';
  els.count.disabled = custom;
  els.regen.title = custom ? 'Regenerate works on the premade datasets' : '';
  els.reveal.hidden = state.type !== 'mystery';
  els.reveal.disabled = state.revealed;
  els.classToggle.hidden = state.mode !== 'edit';
  els.clear.hidden = state.mode !== 'edit';
  els.clear.disabled = counts.total === 0;

  els.modeHint.textContent = state.mode === 'edit'
    ? 'Click or tap to add a point. Shift-click adds the other colour. Right-click or long-press removes the nearest point.'
    : 'Hover or tap to move the query point. Switch to Edit points to add or remove points.';

  els.status.textContent = counts.total === 0
    ? COPY.customEmpty
    : `${counts.total} points: ${counts.train} training, ${counts.test} test${state.showTest ? '' : ' (hidden)'}. ${counts.red} red, ${counts.blue} blue.`;

  const note = DATASET_NOTES[state.type];
  els.datasetTitle.textContent = note.title;
  els.datasetText.textContent = note.text;
  els.revealNote.hidden = !(state.type === 'mystery' && state.revealed);
  if (!els.revealNote.hidden) {
    els.revealNote.textContent = COPY.mysteryRevealed(kindLabel(state.dataset));
  }
}

// ===== Training =====
//
// Nothing is ever fitted automatically. The flow is always: set the
// parameters, press Train, watch the model learn.

let panel = null; // the current model's part of the maths panel

const engine = createEngine({
  // Called after each tick of training steps.
  onProgress(snapshot) {
    state.snapshot = snapshot;
    // Some models fill in their predictions gradually and say how far they
    // have got with snapshot.reveal (0 to 1).
    plot.setModel(state.model, spec().animationGrid, snapshot.reveal ?? 1);
    updatePanel();
    updateTrainUI();
  },
  // Called once, when training is complete.
  onFinish(snapshot) {
    state.snapshot = snapshot;
    state.phase = 'done';
    plot.setModel(state.model, FULL_GRID);
    captureScores();
    updatePanel();
    updateTrainUI();
    updateMetrics();
    announce(spec().finished(snapshot));
  },
});

const trainingPoints = () => state.points.filter((p) => !p.test);

// The training and test points as plain arrays, for the maths panels.
let dataCache = null;
function splitData() {
  if (!dataCache) {
    const collect = (wantTest) => {
      const chosen = state.points.filter((p) => p.test === wantTest);
      return { X: chosen.map((p) => [p.x1, p.x2]), y: chosen.map((p) => p.y) };
    };
    dataCache = { train: collect(false), test: collect(true) };
  }
  return dataCache;
}

// Start training from scratch. Returns false if there is nothing to train on.
function startTraining({ paused = false } = {}) {
  const training = trainingPoints();
  if (training.length === 0) {
    els.trainMessage.textContent = TRAIN_COPY.noData;
    announce(TRAIN_COPY.noData);
    return false;
  }
  state.model = createModel();
  const trainer = state.model.train(
    training.map((p) => [p.x1, p.x2]),
    training.map((p) => p.y),
  );
  state.phase = 'training';
  state.stale = false;
  state.snapshot = null;
  state.scores = null;
  plot.update({ stale: false });
  plot.setModel(state.model, spec().animationGrid, 0);
  engine.start(trainer, spec().speeds[state.speed], { paused });
  updatePanel();
  updateTrainUI();
  updateMetrics();
  return true;
}

// Called whenever the data or a setting changes. The model on show is now
// out of date: stop any training, fade it, and wait for Train.
function invalidate() {
  engine.cancel();
  dataCache = null;
  if (state.model) {
    state.stale = true;
    state.phase = 'done';
  }
  plot.update({ stale: state.stale });
  updatePanel();
  updateTrainUI();
  updateMetrics();
}

// Remember every point's probability, so that moving the threshold can
// update the scores instantly without asking the model again.
function captureScores() {
  const collect = (wantTest) => {
    const chosen = state.points.filter((p) => p.test === wantTest);
    return {
      probabilities: chosen.map((p) => state.model.predictProba([p.x1, p.x2])),
      labels: chosen.map((p) => p.y),
    };
  };
  state.scores = { train: collect(false), test: collect(true) };
}

const panelStatus = () => {
  if (!state.model) return 'untrained';
  if (state.stale) return 'stale';
  return state.phase === 'training' ? 'training' : 'done';
};

// `flash` is false when only the query point moved, so that hovering does
// not make every number blink.
function updatePanel(flash = true) {
  if (!panel) return;
  const point = state.query ? [state.query.x1, state.query.x2] : null;
  // Before any training, show a fresh model so the panel has its starting numbers.
  const model = state.model || createModel();
  const info = model.explain(point);
  panel.update(info, {
    threshold: state.threshold, status: panelStatus(), flash, data: splitData(),
  });
  // A model may ask for extra drawings on the plot (neighbour lines, split
  // lines and so on), and so may its panel.
  plot.update({ overlay: [...(info.overlay || []), ...state.panelOverlay] });
}

function updateTrainUI() {
  const status = panelStatus();
  els.pause.disabled = !engine.active;
  els.pause.textContent = engine.active && !engine.running ? 'Resume' : 'Pause';
  els.step.disabled = engine.running;
  els.skip.disabled = !engine.active;

  let message = TRAIN_COPY.untrained;
  if (status === 'stale') message = TRAIN_COPY.stale;
  else if (status === 'training') {
    message = state.snapshot ? spec().progress(state.snapshot) : 'Training: ready for the first step';
    if (!engine.running) message += ' (paused)';
  } else if (status === 'done') message = spec().finished(state.snapshot);
  els.trainMessage.textContent = message;
  els.trainMessage.dataset.status = status;
  els.main.dataset.status = status; // fades every number that is out of date
}

// The results at the foot of the page. They are shown only once training has finished,
// and they follow the threshold instantly.
function updateMetrics() {
  const cells = {
    train: ['accTrain', 'precTrain', 'sensTrain', 'specTrain', 'f1Train'].map($),
    test: ['accTest', 'precTest', 'sensTest', 'specTest', 'f1Test'].map($),
  };
  const matrix = [$('cmTP'), $('cmFN'), $('cmFP'), $('cmTN')];
  if (!state.scores) {
    const blank = state.phase === 'training' ? '...' : '';
    for (const cell of [...cells.train, ...cells.test, ...matrix]) cell.textContent = blank;
    return;
  }
  for (const set of ['train', 'test']) {
    const counts = confusion(state.scores[set].probabilities, state.scores[set].labels, state.threshold);
    const summary = summarise(counts);
    const values = [summary.accuracy, summary.precision, summary.recall, summary.specificity, summary.f1];
    values.forEach((value, i) => setValue(cells[set][i], pct(value)));
    if (set === state.matrixSet) {
      [counts.tp, counts.fn, counts.fp, counts.tn].forEach((n, i) => setValue(matrix[i], String(n)));
    }
  }
}

// ===== Model picker and settings =====

// Build the controls for the current model's settings. Each has a hidden
// "What does this do?" explanation, so people can experiment first.
function buildParamControls() {
  els.params.innerHTML = '';
  const values = state.params[state.modelKey];
  for (const def of spec().params) {
    const id = `param-${def.key}`;
    const stops = def.values;
    const format = def.format || String;
    const toggle = def.type === 'toggle';
    const wrap = document.createElement('div');
    wrap.className = 'param';
    const control = toggle
      ? `<label class="switch" for="${id}">
           <span class="switch-label">${def.label}</span>
           <input id="${id}" type="checkbox" role="switch">
           <output for="${id}"></output>
         </label>`
      : `<div class="field">
           <label for="${id}">${def.label} <output for="${id}"></output></label>
           <input id="${id}" type="range"
             min="${stops ? 0 : def.min}" max="${stops ? stops.length - 1 : def.max}" step="${stops ? 1 : def.step}">
         </div>`;
    wrap.innerHTML = `${control}
      <details class="explain">
        <summary>What does this do?</summary>
        <p>${PARAM_NOTES[state.modelKey][def.key]}</p>
      </details>`;
    const input = wrap.querySelector('input');
    const output = wrap.querySelector('output');
    const show = (value) => {
      output.textContent = format(value);
      if (!toggle) input.setAttribute('aria-valuetext', format(value));
    };
    if (toggle) input.checked = values[def.key];
    else input.value = stops ? stops.indexOf(values[def.key]) : values[def.key];
    show(values[def.key]);
    input.addEventListener(toggle ? 'change' : 'input', () => {
      if (toggle) values[def.key] = input.checked;
      else values[def.key] = stops ? stops[Number(input.value)] : Number(input.value);
      show(values[def.key]);
      invalidate(); // changing a setting never refits: it waits for Train
    });
    els.params.append(wrap);
  }
  applyShowAll();
}

function applyShowAll() {
  for (const details of document.querySelectorAll('details.explain')) {
    details.open = els.showAll.checked;
  }
}
els.showAll.addEventListener('change', applyShowAll);

function selectModel(key) {
  engine.cancel();
  state.modelKey = key;
  state.model = null;
  state.phase = 'untrained';
  state.stale = false;
  state.snapshot = null;
  state.scores = null;
  plot.update({ stale: false });
  plot.setModel(null);
  state.panelOverlay = [];
  // The old panel is about to be removed from the page; make sure nothing it
  // does on the way out (losing focus, for example) reaches the new model.
  panel = null;
  const created = spec().createPanel({
    intro: els.modelIntro, detail: els.modelDetail, wide: els.modelWide,
    // Lets the panel highlight part of the plot, for example the rectangle
    // belonging to the tree node under the mouse.
    setOverlay(shapes) {
      if (panel !== created) return;
      state.panelOverlay = shapes;
      updatePanel(false);
    },
  });
  panel = created;
  $('thresholdNote').textContent = [TRAIN_COPY.threshold, MODEL_NOTES[key].threshold]
    .filter(Boolean).join(' ');
  buildParamControls();
  updatePanel();
  updateTrainUI();
  updateMetrics();
}

for (const [key, model] of Object.entries(MODELS)) els.model.add(new Option(model.label, key));
els.model.addEventListener('change', () => selectModel(els.model.value));

// ===== The plot =====

const plot = createPlot($('plot'), {
  onQuery(where) {
    state.query = where;
    plot.update({ query: where });
    updatePanel(false);
  },
  onAdd: addPoint,
  onRemove: removeNearest,
});

// ===== Controls =====

// --- data ---
for (const { key, label } of DATASET_TYPES) els.dataset.add(new Option(label, key));

els.dataset.addEventListener('change', () => {
  state.type = els.dataset.value;
  if (state.type === 'custom') setMode('edit'); // an empty plot is for drawing on
  rebuild();
});

function setSeed(value) {
  const seed = Math.max(0, Math.min(MAX_SEED, Math.floor(Number(value))));
  if (!Number.isFinite(seed)) { els.seed.value = state.seed; return; }
  state.seed = seed;
  if (isCustom()) {
    // Hand-made points stay; the seed still drives "Add more points".
    state.dataset.seed = seed;
    state.batch = 0;
    state.manualRng = createRng(seed, 'manual');
    render();
  } else {
    rebuild();
  }
}
els.seed.addEventListener('change', () => setSeed(els.seed.value));

els.regen.addEventListener('click', () => {
  // Choosing the next seed is the one place that is not seeded: it is the
  // source of a fresh variation. Everything after it follows from the seed.
  setSeed(Math.floor(Math.random() * 100000));
  announce(`New ${DATASET_NOTES[state.type].title} dataset, seed ${state.seed}.`);
});

els.more.addEventListener('click', () => {
  const added = morePoints(state.dataset, state.points, state.batch + 1, undefined,
    noiseMultiplier(), state.trainShare);
  if (added.length === 0) {
    announce(COPY.addMoreEmpty);
    els.status.textContent = COPY.addMoreEmpty;
    return;
  }
  state.batch += 1;
  state.points.push(...added);
  announce(`Added ${added.length} more points.`);
  invalidate();
  render();
});

els.reveal.addEventListener('click', () => {
  state.revealed = true;
  render();
  announce(COPY.mysteryRevealed(kindLabel(state.dataset)));
});

els.noise.addEventListener('input', () => {
  state.noise = Number(els.noise.value);
  if (!isCustom()) rebuildKeepingReveal();
});
els.count.addEventListener('input', () => {
  state.count = Number(els.count.value);
  if (!isCustom()) rebuildKeepingReveal();
});

// Moving a slider redraws the same variation, so a revealed Mystery stays revealed.
function rebuildKeepingReveal() {
  const revealed = state.revealed;
  rebuild();
  state.revealed = revealed;
  render();
}

function setMode(mode) {
  state.mode = mode;
  document.querySelector(`input[name="mode"][value="${mode}"]`).checked = true;
}
for (const radio of document.querySelectorAll('input[name="mode"]')) {
  radio.addEventListener('change', () => { setMode(radio.value); render(); });
}
for (const radio of document.querySelectorAll('input[name="cls"]')) {
  radio.addEventListener('change', () => { state.activeClass = Number(radio.value); });
}

els.clear.addEventListener('click', () => {
  markEdited();
  state.points = [];
  announce('Cleared all points.');
  invalidate();
  render();
});

els.showTest.addEventListener('change', () => {
  state.showTest = els.showTest.checked;
  render();
});

// --- training ---
els.train.addEventListener('click', () => {
  if (!startTraining()) return;
  announce('Training started.');
  // With reduced motion, skip the animation and show the finished model.
  if (prefersReducedMotion()) engine.skip();
});

els.pause.addEventListener('click', () => {
  if (engine.running) engine.pause(); else engine.resume();
  updateTrainUI();
});

els.step.addEventListener('click', () => {
  // Step also works from a standing start: it begins training, paused.
  if (!engine.active && !startTraining({ paused: true })) return;
  engine.stepOnce();
  updateTrainUI();
});

els.skip.addEventListener('click', () => engine.skip());

for (const radio of document.querySelectorAll('input[name="speed"]')) {
  radio.addEventListener('change', () => {
    state.speed = radio.value;
    engine.setSpeed(spec().speeds[state.speed]);
  });
}

// --- decision threshold ---
// Moving it never retrains: the shading stays put, and only the boundary
// line, the predictions and the scores move.
function showThreshold() {
  els.thresholdOut.textContent = state.threshold.toFixed(2);
}
els.threshold.addEventListener('input', () => {
  state.threshold = Number(els.threshold.value);
  showThreshold();
  plot.update({ threshold: state.threshold });
  updatePanel();
  updateMetrics();
});

for (const radio of document.querySelectorAll('input[name="matrixSet"]')) {
  radio.addEventListener('change', () => {
    state.matrixSet = radio.value;
    updateMetrics();
  });
}

// ===== Start =====

$('accuracyWarning').textContent = TRAIN_COPY.accuracyWarning;
$('experimentNote').textContent = TRAIN_COPY.experiment;
showThreshold();

// Optional address-bar settings, for example ?data=moons&seed=42.
const query = new URLSearchParams(location.search);
if (DATASET_TYPES.some((t) => t.key === query.get('data'))) state.type = query.get('data');
if (/^\d{1,6}$/.test(query.get('seed') || '')) state.seed = Number(query.get('seed'));
if (state.type === 'custom') setMode('edit');

$('loadNotice').remove(); // the scripts loaded, so the server note is not needed
selectModel(state.modelKey);
rebuild();
