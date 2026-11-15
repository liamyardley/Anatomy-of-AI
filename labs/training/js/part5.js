// Part 5 of the training lab: regularisation.
//   The picture   the penalty's shape (circle or diamond) meeting the loss
//                 contours on a two-weight map
//   The sweep     six weights against the strength of the penalty
// Page wiring only. The model is in regularise.js.

import { createEngine } from '../../../assets/js/engine.js';
import { fmt, setValue, renderTex } from '../../../assets/js/format.js';
import {
  makeRegression, dataSurface, createProximal, sweep, penaltyOutline,
} from './regularise.js';
import { BOX, createMapView } from './mapview.js';
import {
  $, pair, startMark, startPicker, trainBar, stopSlider, seedControls, clampSeed,
} from './ui.js';
import { REG_COPY } from '../content/notes.js';

const A_RANGE = [-0.75, 2.25];
const B_RANGE = [-1.5, 1.5];
const STRENGTHS = [0.03, 0.1, 0.2, 0.4, 0.7, 1, 1.5, 2.5, 4];
const SPEEDS = {
  slow: { steps: 1, interval: 450 },
  normal: { steps: 1, interval: 160 },
  fast: { steps: 1, interval: 40 },
};
const FORMULAS = {
  none: '\\text{minimise}\\quad \\text{data loss}',
  l2: '\\text{minimise}\\quad \\text{data loss} + \\lambda \\times \\tfrac{1}{2}\\big(w_1^2 + w_2^2\\big)',
  l1: '\\text{minimise}\\quad \\text{data loss} + \\lambda \\times \\big(|w_1| + |w_2|\\big)',
  elastic: '\\text{minimise}\\quad \\text{data loss} + \\lambda \\times \\Big(\\tfrac{1}{2}\\big(|w_1| + |w_2|\\big) + \\tfrac{1}{4}\\big(w_1^2 + w_2^2\\big)\\Big)',
};

// The sweep's model: six features, of which only the first three matter.
const TRUTH = [2, -1.5, 1, 0, 0, 0];
const SWEEP_KINDS = ['l2', 'l1', 'elastic'];
// Strengths from 0.02 to 8, evenly spaced on a logarithmic scale.
const SWEEP_STRENGTHS = Array.from({ length: 27 }, (_, i) => 0.02 * (8 / 0.02) ** (i / 26));

const state = {
  seed: 42,
  data: null,
  surface: null,
  kind: 'l1',
  strength: 1,
  start: [2, -1],
  at: [2, -1],
  speed: 'normal',
  run: null,
  phase: 'untrained',
  stale: false,
  view: null,
  wide: null,       // the six-feature data for the sweep
  sweeps: null,     // { l2: [...], l1: [...], elastic: [...] } once run
  sweepStale: false,
};

const announce = (message) => { $('regAnnounce').textContent = message; };
// A weight as text, saying so when it is exactly zero.
// A tiny weight that is not zero is shown with enough decimal places to see it.
const weightText = (value) => {
  if (value === 0) return 'exactly 0';
  return Math.abs(value) < 0.005 ? fmt(value, 4) : fmt(value);
};

function drawMap() {
  const { view } = state;
  const [a, b] = state.at;
  let path = '';
  let outline = '';
  if (state.run) {
    const history = state.run.history;
    path = `<polyline class="path" points="${view.line(history.map((h) => h.w))}"/>
      ${history.map((h) => `<circle class="path-dot" cx="${view.px(h.w[0]).toFixed(1)}" cy="${view.py(h.w[1]).toFixed(1)}" r="2.4"/>`).join('')}`;
    // Every pair of weights with the same penalty as the current pair.
    const shape = penaltyOutline(state.kind, state.at);
    if (shape.length) outline = `<polygon class="penalty-shape" points="${view.line(shape)}"/>`;
  }
  const [ba, bb] = state.surface.best;
  $('regOverlay').innerHTML = `<svg viewBox="0 0 ${BOX.SIZE} ${BOX.SIZE}" role="img" aria-label="Contour map of the data loss against two weights. The weights are at ${pair(state.at)}. The best fit with no penalty is at ${pair(state.surface.best)}.">
    <defs>${view.clip('regClip')}</defs>
    ${view.axes([-0.5, 0, 0.5, 1, 1.5, 2], [-1, 0, 1], 'weight w₁', 'weight w₂')}
    <g clip-path="url(#regClip)">
      <line class="zero-line" x1="${view.px(0)}" y1="${BOX.T}" x2="${view.px(0)}" y2="${BOX.T + BOX.PLOT}"/>
      <line class="zero-line" x1="${BOX.L}" y1="${view.py(0)}" x2="${BOX.L + BOX.PLOT}" y2="${view.py(0)}"/>
      <g opacity="${state.stale ? 0.35 : 1}">${outline}${path}</g>
      <path class="best-mark" d="M${view.px(ba) - 6} ${view.py(bb)}h12M${view.px(ba)} ${view.py(bb) - 6}v12"/>
      ${startMark(view.px(state.start[0]), view.py(state.start[1]))}
      <circle class="here" cx="${view.px(a).toFixed(1)}" cy="${view.py(b).toFixed(1)}" r="5.5"/>
    </g>
  </svg>`;
}

// The sentence for a finished run: which penalty, and what it did.
function summary(last) {
  const zero = last.w.map((value, k) => (value === 0 ? k + 1 : 0)).filter(Boolean);
  const head = REG_COPY.finished(last, last.outcome === 'settled');
  if (state.kind === 'none') return `${head} ${REG_COPY.endings.none}`;
  if (zero.length === 2) return `${head} ${REG_COPY.endings.bothZero}`;
  if (zero.length === 1) return `${head} ${REG_COPY.endings.oneZero(zero[0])}`;
  return `${head} ${REG_COPY.endings[state.kind]}`;
}

function update(flash = true) {
  drawMap();
  const now = state.run ? state.run.history[state.run.history.length - 1] : null;
  setValue($('regW1Out'), weightText(state.at[0]), flash);
  setValue($('regW2Out'), weightText(state.at[1]), flash);
  setValue($('regLossOut'), fmt(state.surface.loss(...state.at), 3), flash);
  setValue($('regBestOut'), `${pair(state.surface.best)}, loss ${fmt(state.surface.loss(...state.surface.best), 3)}`, false);
  $('regStrengthField').hidden = state.kind === 'none';
  $('regShapeNote').innerHTML = REG_COPY.shapes[state.kind];
  renderTex($('regTex'), FORMULAS[state.kind], { display: true });

  let message = REG_COPY.untrained;
  let status = 'untrained';
  if (state.run && state.stale) {
    message = REG_COPY.stale;
    status = 'stale';
  } else if (state.run && state.phase === 'training') {
    message = REG_COPY.progress(now) + (engine.running ? '' : ' (paused)');
    status = 'training';
  } else if (state.run) {
    message = summary(now);
    status = 'done';
  }
  $('regMessage').textContent = message;
  $('regMessage').dataset.status = status;
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
    announce(summary(snapshot));
  },
});
let refreshBar = () => {};

function begin(paused) {
  state.at = [...state.start];
  state.run = createProximal({
    X: state.data.X, y: state.data.y, kind: state.kind, strength: state.kind === 'none' ? 0 : state.strength, start: state.start,
  });
  state.phase = 'training';
  state.stale = false;
  engine.start(state.run, SPEEDS[state.speed], { paused });
  update();
}

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

// ===== The sweep: six weights against the strength of the penalty =====

function drawSweep() {
  const W = 360; const H = 250; const L = 34; const R = 352; const T = 10; const B = 214;
  const low = Math.log(SWEEP_STRENGTHS[0]);
  const high = Math.log(SWEEP_STRENGTHS[SWEEP_STRENGTHS.length - 1]);
  const x = (strength) => L + ((Math.log(strength) - low) / (high - low)) * (R - L);
  const y = (v) => B - ((v + 2) / 4.5) * (B - T);
  const xTicks = [0.02, 0.1, 0.5, 2, 8];
  const frame = `${xTicks.map((v) => `<text class="tick" x="${x(v).toFixed(1)}" y="${B + 13}" text-anchor="middle">${v}</text>`).join('')}
    ${[-2, -1, 0, 1, 2].map((v) => `<line class="grid" x1="${L}" y1="${y(v)}" x2="${R}" y2="${y(v)}"/><text class="tick" x="${L - 5}" y="${y(v) + 3}" text-anchor="end">${v}</text>`).join('')}
    <line class="axis" x1="${L}" y1="${T}" x2="${L}" y2="${B}"/>
    <line class="zero-line strong" x1="${L}" y1="${y(0)}" x2="${R}" y2="${y(0)}"/>
    <text class="axis-title" x="${(L + R) / 2}" y="${H - 6}" text-anchor="middle">strength of the penalty (λ)</text>`;

  for (const kind of SWEEP_KINDS) {
    const box = $(`regSweep-${kind}`);
    const rows = state.sweeps ? state.sweeps[kind] : null;
    let lines = '';
    let note = REG_COPY.sweepEmpty;
    if (rows) {
      lines = TRUTH.map((truth, k) => {
        const cls = truth === 0 ? 'coef idle' : 'coef';
        // A label at the left end, where the penalty is weakest and the
        // lines are furthest apart. The irrelevant three share one label.
        const label = truth === 0 ? '' : `<text class="coef-label" x="${L + 5}" y="${(y(rows[0].w[k]) - 5).toFixed(1)}">w${k + 1}</text>`;
        return `<polyline class="${cls}" points="${rows.map((r) => `${x(r.strength).toFixed(1)},${y(r.w[k]).toFixed(1)}`).join(' ')}"/>${label}`;
      }).join('') + `<text class="coef-label idle" x="${L + 5}" y="${y(0) + 13}">w4, w5, w6</text>`;
      const firstZero = rows.find((r) => r.zeros > 0);
      const idleGone = rows.find((r) => r.w.slice(3).every((v) => v === 0));
      const allGone = rows.find((r) => r.zeros === TRUTH.length);
      note = REG_COPY.sweepNotes[kind](
        firstZero ? firstZero.strength : null,
        idleGone ? idleGone.strength : null,
        idleGone ? idleGone.w.slice(0, 3).filter((v) => v !== 0).length : 0,
        allGone ? allGone.strength : null,
      );
    }
    box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${REG_COPY.kinds[kind]}: each weight against the strength of the penalty.">
      ${frame}
      <g opacity="${state.sweepStale ? 0.35 : 1}">${lines}</g>
    </svg>`;
    $(`regSweepNote-${kind}`).textContent = note;
  }
}

function runSweep() {
  state.sweeps = Object.fromEntries(SWEEP_KINDS.map((kind) => [
    kind, sweep({ X: state.wide.X, y: state.wide.y, kind, strengths: SWEEP_STRENGTHS }),
  ]));
  state.sweepStale = false;
  drawSweep();
  announce(REG_COPY.sweepDone);
}

function load(seed) {
  state.seed = clampSeed(seed);
  state.data = makeRegression(state.seed);
  state.surface = dataSurface(state.data.X, state.data.y);
  state.wide = makeRegression(state.seed, { truth: TRUTH, noise: 0.8 });
  // An earlier sweep no longer matches the data: fade it until it is run again.
  state.sweepStale = state.sweeps !== null;
  $('regSeed').value = state.seed;
  state.view.draw(state.surface);
  changed(true);
  drawSweep();
}

export function startPartFive() {
  state.view = createMapView({ canvas: $('regCanvas'), xRange: A_RANGE, yRange: B_RANGE });
  for (const [id, key] of [
    ['regNote', 'intro'], ['regMapNote', 'map'], ['regStrengthNote', 'strength'], ['regWhyNote', 'why'],
    ['regSweepIntro', 'sweepIntro'], ['regSweepLegend', 'sweepLegend'], ['regSeedNote', 'seed'], ['regLinkNote', 'link'],
  ]) $(id).innerHTML = REG_COPY[key];
  for (const kind of SWEEP_KINDS) $(`regSweepTitle-${kind}`).textContent = REG_COPY.kinds[kind];

  for (const radio of document.querySelectorAll('input[name="regKind"]')) {
    if (radio.checked) state.kind = radio.value;
    radio.addEventListener('change', () => {
      state.kind = radio.value;
      changed(false);
      announce(`${REG_COPY.kinds[state.kind]}.`);
    });
  }
  stopSlider('regStrength', 'regStrengthOut', STRENGTHS, () => state.strength, (v) => { state.strength = v; }, () => changed(false));
  seedControls('regSeed', 'regNewBtn', () => state.seed, load, announce);
  startPicker($('regMap'), state.view, A_RANGE, B_RANGE, () => state.start, (start) => {
    state.start = start;
    changed(true);
    announce(`Start at ${pair(state.start)}. Press Train.`);
  });
  $('regSweepBtn').addEventListener('click', runSweep);

  refreshBar = trainBar('reg', engine, SPEEDS, begin, () => state.speed, (v) => { state.speed = v; });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => state.view.draw(state.surface));
  load(state.seed);
}
