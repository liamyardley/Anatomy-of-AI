// Part 4 of the training lab: the real thing. The loss surface of a
// logistic regression on the Machine Learning Lab's Two clouds dataset, with
// the path on the map and the boundary moving on the data, side by side.
// Page wiring only. The model is in logistic2.js.

import { createEngine } from '../../../assets/js/engine.js';
import { createRng } from '../../../assets/js/rng.js';
import { fmt, pct, texNum, setValue, renderTex } from '../../../assets/js/format.js';
import { createDataset, samplePoints } from '../../ml/js/datasets.js';
import { createDescent2 } from './twoweights.js';
import { centreData, logisticSurface, accuracy } from './logistic2.js';
import { BOX, createMapView } from './mapview.js';
import {
  $, pair, onMap, startMark, startPicker, trainBar, stopSlider, seedControls, clampSeed,
} from './ui.js';
import { REAL_COPY } from '../content/notes.js';

const POINTS = 80;
const WEIGHT_RANGE = [-8, 8];
const RATES = [0.3, 1, 3, 10, 30];
const OVERLAPS = [1, 1.5, 2, 3];
const MAX_STEPS = 300;
const SPEEDS = {
  slow: { steps: 1, interval: 350 },
  normal: { steps: 1, interval: 90 },
  fast: { steps: 3, interval: 25 },
};

const state = {
  seed: 42,
  overlap: 2,
  data: null,      // { X, y, centre }
  surface: null,
  start: [0, 0],
  at: [0, 0],
  rate: 10,
  speed: 'normal',
  run: null,
  phase: 'untrained',
  stale: false,
  view: null,
};

const announce = (message) => { $('realAnnounce').textContent = message; };

// The points and the model's boundary: the line where the score is zero,
// which always passes through the centre of the data.
function drawData() {
  const SIZE = 360; const L = 12; const PLOT = 336;
  const REACH = 1.6; // the centred points fit comfortably inside this
  const x = (v) => L + ((v + REACH) / (2 * REACH)) * PLOT;
  const y = (v) => L + ((REACH - v) / (2 * REACH)) * PLOT;
  const { X, y: labels } = state.data;
  const [a, b] = state.at;
  const marks = X.map(([p, q], i) => (labels[i] === 1
    ? `<path class="pt-blue" d="M${x(p).toFixed(1)} ${(y(q) - 5).toFixed(1)}l4.6 8h-9.2Z"/>`
    : `<circle class="pt-red" cx="${x(p).toFixed(1)}" cy="${y(q).toFixed(1)}" r="3.8"/>`)).join('');
  let boundary = `<text class="tick strong" x="${SIZE / 2}" y="${L + 14}" text-anchor="middle">${REAL_COPY.noLine}</text>`;
  const size = Math.hypot(a, b);
  if (size > 1e-6) {
    // The line runs at right angles to the weights; the arrow points to the
    // side the model calls blue.
    const along = [-b / size, a / size];
    const towards = [a / size, b / size];
    const far = 3;
    boundary = `<line class="model-line" x1="${x(-far * along[0]).toFixed(1)}" y1="${y(-far * along[1]).toFixed(1)}" x2="${x(far * along[0]).toFixed(1)}" y2="${y(far * along[1]).toFixed(1)}"/>
      <line class="blue-side" x1="${x(0)}" y1="${y(0)}" x2="${x(0.42 * towards[0]).toFixed(1)}" y2="${y(0.42 * towards[1]).toFixed(1)}" marker-end="url(#realArrow)"/>`;
  }
  $('realData').innerHTML = `<svg viewBox="0 0 ${SIZE} ${SIZE}" role="img" aria-label="${X.length} points in two colours and the model's boundary line. ${pct(accuracy(state.at, X, labels))} of the points are on the right side.">
    <defs>
      <clipPath id="realDataClip"><rect x="${L}" y="${L}" width="${PLOT}" height="${PLOT}"/></clipPath>
      <marker id="realArrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="blue-head" d="M0 0 10 5 0 10Z"/></marker>
    </defs>
    <rect class="map-frame" x="${L}" y="${L}" width="${PLOT}" height="${PLOT}"/>
    <g clip-path="url(#realDataClip)" opacity="${state.stale ? 0.4 : 1}">${marks}${boundary}</g>
    <circle class="centre-dot" cx="${x(0)}" cy="${y(0)}" r="3"/>
  </svg>`;
}

function drawMap() {
  const { view } = state;
  const ticks = [-8, -4, 0, 4, 8];
  let path = '';
  if (state.run) {
    const history = state.run.history;
    path = `<g opacity="${state.stale ? 0.35 : 1}">
      <polyline class="path" points="${view.line(history.map((h) => h.w))}"/>
      ${history.length <= 60 ? history.map((h) => `<circle class="path-dot" cx="${view.px(h.w[0]).toFixed(1)}" cy="${view.py(h.w[1]).toFixed(1)}" r="2.4"/>`).join('') : ''}
    </g>`;
  }
  const { best } = state.surface;
  const bestOnMap = best && onMap(best, WEIGHT_RANGE, WEIGHT_RANGE);
  const bestMark = bestOnMap
    ? `<path class="best-mark" d="M${view.px(best[0]) - 6} ${view.py(best[1])}h12M${view.px(best[0])} ${view.py(best[1]) - 6}v12"/>`
    : `<text class="tick strong" x="${BOX.L + BOX.PLOT - 6}" y="${BOX.T + 14}" text-anchor="end">${REAL_COPY.offMap}</text>`;
  const [a, b] = state.at;
  $('realOverlay').innerHTML = `<svg viewBox="0 0 ${BOX.SIZE} ${BOX.SIZE}" role="img" aria-label="Contour map of the log-loss against the two weights. The weights are at ${pair(state.at)}, where the loss is ${fmt(state.surface.loss(a, b), 3)}.">
    <defs>${view.clip('realClip')}</defs>
    ${view.axes(ticks, ticks, 'weight w₁', 'weight w₂')}
    <g clip-path="url(#realClip)">
      ${path}
      ${bestMark}
      ${startMark(view.px(state.start[0]), view.py(state.start[1]))}
      <circle class="here" cx="${view.px(a).toFixed(1)}" cy="${view.py(b).toFixed(1)}" r="5.5"/>
    </g>
  </svg>`;
}

// How a finished run ended, as a key of REAL_COPY.
function resultOf(snapshot) {
  if (snapshot.outcome === 'settled') return 'settled';
  if (snapshot.outcome === 'limit' && !state.surface.best) return 'noBottom';
  return snapshot.outcome;
}

function update(flash = true) {
  drawData();
  drawMap();
  const [a, b] = state.at;
  const { X, y } = state.data;
  setValue($('realWeightsOut'), pair(state.at), flash);
  setValue($('realLossOut'), fmt(state.surface.loss(a, b), 3), flash);
  setValue($('realAccuracyOut'), pct(accuracy(state.at, X, y)), flash);
  const [ga, gb] = state.surface.gradient(a, b);
  renderTex($('realNextTex'),
    `\\begin{aligned} \\text{gradient} &= (${texNum(ga)},\\ ${texNum(gb)}) \\\\ w_1 &: ${texNum(a)} - ${state.rate} \\times (${texNum(ga)}) = ${texNum(a - state.rate * ga)} \\\\ w_2 &: ${texNum(b)} - ${state.rate} \\times (${texNum(gb)}) = ${texNum(b - state.rate * gb)} \\end{aligned}`,
    { display: true });

  let message = REAL_COPY.untrained;
  let status = 'untrained';
  if (state.run && state.stale) {
    message = REAL_COPY.stale;
    status = 'stale';
  } else if (state.run) {
    const last = state.run.history[state.run.history.length - 1];
    if (state.phase === 'training') {
      message = REAL_COPY.progress(last, accuracy(last.w, X, y)) + (engine.running ? '' : ' (paused)');
      status = 'training';
    } else {
      message = REAL_COPY[resultOf(last)](last, accuracy(last.w, X, y));
      status = 'done';
    }
  }
  $('realMessage').textContent = message;
  $('realMessage').dataset.status = status;
  refreshBar();
}

const engine = createEngine({
  onProgress(snapshot) {
    state.at = snapshot.w;
    update();
  },
  onFinish(snapshot) {
    state.phase = 'done';
    state.at = snapshot.w;
    update();
    announce(REAL_COPY[resultOf(snapshot)](snapshot, accuracy(snapshot.w, state.data.X, state.data.y)));
  },
});
let refreshBar = () => {};

function begin(paused) {
  state.at = [...state.start];
  state.run = createDescent2({
    surface: state.surface, start: state.start, learningRate: state.rate, maxSteps: MAX_STEPS,
  });
  state.phase = 'training';
  state.stale = false;
  engine.start(state.run, SPEEDS[state.speed], { paused });
  update();
}

// A setting changed: stop and wait for Train. `clear` drops the old path.
function changed(clear) {
  engine.cancel();
  if (clear) {
    state.run = null;
    state.phase = 'untrained';
    state.stale = false;
    state.at = [...state.start];
  } else if (state.run) {
    state.stale = true;
    state.phase = 'done';
  }
  update(false);
}

function load(seed) {
  state.seed = clampSeed(seed);
  const dataset = createDataset('clouds', state.seed);
  const points = samplePoints(dataset, POINTS, state.overlap, createRng(state.seed, 'points'));
  state.data = centreData(points);
  state.surface = logisticSurface(state.data.X, state.data.y);
  $('realSeed').value = state.seed;
  state.view.draw(state.surface);
  changed(true);
}

export function startPartFour() {
  state.view = createMapView({ canvas: $('realCanvas'), xRange: WEIGHT_RANGE, yRange: WEIGHT_RANGE });
  for (const [id, key] of [
    ['realNote', 'intro'], ['realDataNote', 'data'], ['realMapNote', 'map'], ['realRateNote', 'learningRate'],
    ['realOverlapNote', 'overlap'], ['realModelKey', 'modelKey'], ['realLinkNote', 'link'], ['realSeedNote', 'seed'],
  ]) $(id).innerHTML = REAL_COPY[key];
  renderTex($('realModelTex'), 'P(\\text{blue}) = \\sigma(w_1 x_1 + w_2 x_2) \\qquad \\sigma(z) = \\frac{1}{1 + e^{-z}}', { display: true });
  renderTex($('realLossTex'), 'L = -\\frac{1}{n}\\sum_{i=1}^{n}\\ln\\big(\\text{probability given to the true colour of point } i\\big)', { display: true });

  stopSlider('realRate', 'realRateOut', RATES, () => state.rate, (v) => { state.rate = v; }, () => changed(false));
  stopSlider('realOverlap', 'realOverlapOut', OVERLAPS, () => state.overlap, (v) => { state.overlap = v; }, () => load(state.seed));
  seedControls('realSeed', 'realNewBtn', () => state.seed, load, announce);
  startPicker($('realMap'), state.view, WEIGHT_RANGE, WEIGHT_RANGE, () => state.start, (start) => {
    state.start = start;
    changed(true);
    announce(`Start at ${pair(state.start)}. Press Train.`);
  }, 0.25);

  refreshBar = trainBar('real', engine, SPEEDS, begin, () => state.speed, (v) => { state.speed = v; });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => state.view.draw(state.surface));
  load(state.seed);
}
