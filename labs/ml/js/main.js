// Wiring for the Machine Learning Lab: page state, controls and the plot.
// Models and the training engine plug in here in the next phase.

import { createRng } from './rng.js';
import {
  DATASET_TYPES, DEFAULT_POINT_COUNT, createDataset, buildPoints, morePoints, kindLabel,
} from './datasets.js';
import { DEFAULT_TRAIN_SHARE, drawIsTest, countPoints } from './metrics.js';
import { createPlot } from './plot.js';
import { DATASET_NOTES, COPY } from '../content/notes.js';

const $ = (id) => document.getElementById(id);
const els = {
  dataset: $('datasetSel'), seed: $('seedInput'),
  regen: $('regenBtn'), more: $('moreBtn'), reveal: $('revealBtn'),
  noise: $('noiseRange'), noiseOut: $('noiseOut'),
  count: $('countRange'), countOut: $('countOut'),
  classToggle: $('classToggle'), clear: $('clearBtn'), showTest: $('showTest'),
  modeHint: $('modeHint'), status: $('status'), announce: $('announce'),
  datasetTitle: $('datasetTitle'), datasetText: $('datasetText'), revealNote: $('revealNote'),
  queryReadout: $('queryReadout'),
};

const MAX_SEED = 999999;
const REMOVE_REACH = 0.2; // how close a point must be to be removed

const state = {
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
};

const isCustom = () => state.type === 'custom' || state.type === 'edited';
const noiseMultiplier = () => state.noise * 2;

// Two decimal places, with a proper minus sign.
const fmt = (v) => (Math.abs(v) < 0.005 ? 0 : v).toFixed(2).replace('-', '−');
const className = (y) => (y === 1 ? 'blue' : 'red');

function announce(message) {
  els.announce.textContent = message;
}

// --- building datasets ---

// Start a fresh dataset of the current type from the current seed.
function rebuild() {
  state.dataset = createDataset(state.type, state.seed);
  state.points = buildPoints(state.dataset, state.count, noiseMultiplier(), state.trainShare);
  state.batch = 0;
  state.revealed = false;
  state.manualRng = createRng(state.seed, 'manual');
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

// --- editing by hand ---

function addPoint(where, otherClass) {
  markEdited();
  const y = otherClass ? 1 - state.activeClass : state.activeClass;
  const test = drawIsTest(state.trainShare, state.manualRng);
  state.points.push({ x1: where.x1, x2: where.x2, y, test });
  const hiddenNote = test && !state.showTest ? ' It is hidden: tick "Show test points" to see it.' : '';
  announce(`Added a ${className(y)} ${test ? 'test' : 'training'} point at ${fmt(where.x1)}, ${fmt(where.x2)}.${hiddenNote}`);
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
  render();
}

// --- rendering ---

function render() {
  const custom = isCustom();
  const counts = countPoints(state.points);

  plot.update({
    points: state.points, showTest: state.showTest, query: state.query, mode: state.mode,
  });

  // Controls
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

  // Status line
  els.status.textContent = counts.total === 0
    ? COPY.customEmpty
    : `${counts.total} points: ${counts.train} training, ${counts.test} test${state.showTest ? '' : ' (hidden)'}. ${counts.red} red, ${counts.blue} blue.`;

  // Dataset card
  const note = DATASET_NOTES[state.type];
  els.datasetTitle.textContent = note.title;
  els.datasetText.textContent = note.text;
  els.revealNote.hidden = !(state.type === 'mystery' && state.revealed);
  if (!els.revealNote.hidden) {
    els.revealNote.textContent = COPY.mysteryRevealed(kindLabel(state.dataset));
  }

  renderQuery();
}

function renderQuery() {
  els.queryReadout.innerHTML = state.query
    ? `Query point: x<sub>1</sub> = ${fmt(state.query.x1)}, x<sub>2</sub> = ${fmt(state.query.x2)}`
    : COPY.queryEmpty;
}

// --- the plot ---

const plot = createPlot($('plot'), {
  onQuery(where) {
    state.query = where;
    plot.update({ query: where });
    renderQuery();
  },
  onAdd: addPoint,
  onRemove: removeNearest,
});

// --- controls ---

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
  render();
});

els.showTest.addEventListener('change', () => {
  state.showTest = els.showTest.checked;
  render();
});

// --- start ---

// Optional address-bar settings, for example ?data=moons&seed=42.
const params = new URLSearchParams(location.search);
if (DATASET_TYPES.some((t) => t.key === params.get('data'))) state.type = params.get('data');
if (/^\d{1,6}$/.test(params.get('seed') || '')) state.seed = Number(params.get('seed'));
if (state.type === 'custom') setMode('edit');

$('loadNotice').remove(); // the scripts loaded, so the server note is not needed
rebuild();
