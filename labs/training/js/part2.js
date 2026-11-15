// Part 2 of the training lab: two weights and the loss as a contour map.
// Page wiring only. The surfaces and gradient descent are in twoweights.js,
// which knows nothing about the page.

import { createEngine } from '../../../assets/js/engine.js';
import { fmt, texNum, setValue, renderTex, prefersReducedMotion } from '../../../assets/js/format.js';
import { SURFACES, RANGE, createDescent2 } from './twoweights.js';
import { TWO_WEIGHTS, TWO_TRAIN_COPY } from '../content/notes.js';

const $ = (id) => document.getElementById(id);

const RATES = [0.01, 0.03, 0.1, 0.2, 0.3, 0.6, 0.9, 1, 1.5, 2.05]; // learning rate stops
const SPEEDS = {
  slow: { steps: 1, interval: 500 },
  normal: { steps: 1, interval: 160 },
  fast: { steps: 2, interval: 30 },
};
const GRID = 0.05;     // clicks and arrow keys place the start on this grid
const BANDS = 16;      // how many height bands the map is drawn in
const CELLS = 160;     // the map is coloured in CELLS x CELLS squares
const MAX_GHOSTS = 8;  // earlier paths kept on the map

// Each surface's loss, written out.
const FORMULAS = {
  bowl: 'L(w_1, w_2) = \\tfrac{1}{2}w_1^2 + \\tfrac{1}{2}w_2^2',
  valley: 'L(w_1, w_2) = \\tfrac{1}{20}w_1^2 + w_2^2',
  saddle: 'L(w_1, w_2) = \\tfrac{1}{2}w_1^2 + \\tfrac{1}{4}w_2^4 - \\tfrac{1}{2}w_2^2 + \\tfrac{1}{4}',
  dips: 'L(w_1, w_2) = \\text{a shallow bowl with three dips pressed into it}',
};

const state = {
  key: 'bowl',
  start: [1.5, 1.2],  // where training starts: set by clicking the map
  at: [1.5, 1.2],     // the point on show
  rate: 0.3,
  speed: 'normal',
  run: null,
  phase: 'untrained', // 'untrained', 'training' or 'done'
  stale: false,
  ghosts: [],         // paths of earlier finished runs on this surface
  runs: [],           // the table of finished runs on this surface
};

const surface = () => SURFACES[state.key];
const announce = (message) => { $('twoAnnounce').textContent = message; };
const pair = ([a, b]) => `(${fmt(a)}, ${fmt(b)})`;
const big = (value, places = 3) => (Math.abs(value) >= 1000 ? value.toExponential(1).replace('-', '−') : fmt(value, places));

// ===== The map =====
//
// A canvas holds the coloured height bands; an SVG on top holds everything
// that changes (the paths, the start, the current point and its next step).

const BOX = { SIZE: 360, L: 36, T: 8, PLOT: 316 };
const px = (a) => BOX.L + ((a + RANGE) / (2 * RANGE)) * BOX.PLOT;
const py = (b) => BOX.T + ((RANGE - b) / (2 * RANGE)) * BOX.PLOT;

// Lowest and highest loss on the map, for scaling the bands.
function heights() {
  let low = Infinity;
  let high = -Infinity;
  for (let i = 0; i <= 80; i++) {
    for (let j = 0; j <= 80; j++) {
      const value = surface().loss(-RANGE + (i / 80) * 2 * RANGE, -RANGE + (j / 80) * 2 * RANGE);
      if (value < low) low = value;
      if (value > high) high = value;
    }
  }
  for (const spot of surface().spots) low = Math.min(low, surface().loss(spot.at[0], spot.at[1]));
  return { low, high };
}

// Which band a height falls in. The square root packs more bands near the
// bottom, where the interesting shapes are.
const band = (value, low, high) => Math.min(BANDS - 1, Math.floor(BANDS * Math.sqrt((value - low) / (high - low))));

// Colour the map: the lower the ground, the stronger the tint, with a thin
// line wherever one band meets the next (the contour lines).
function drawMap() {
  const canvas = $('twoCanvas');
  const size = canvas.width;
  const cell = size / CELLS;
  const ctx = canvas.getContext('2d');
  const style = getComputedStyle(document.documentElement);
  const tint = style.getPropertyValue('--ml-line').trim();
  const line = style.getPropertyValue('--muted').trim();
  const { low, high } = heights();
  const levels = [];
  for (let j = 0; j < CELLS; j++) {
    const row = [];
    for (let i = 0; i < CELLS; i++) {
      const a = -RANGE + ((i + 0.5) / CELLS) * 2 * RANGE;
      const b = RANGE - ((j + 0.5) / CELLS) * 2 * RANGE;
      row.push(band(surface().loss(a, b), low, high));
    }
    levels.push(row);
  }
  ctx.clearRect(0, 0, size, size);
  ctx.fillStyle = tint;
  for (let j = 0; j < CELLS; j++) {
    for (let i = 0; i < CELLS; i++) {
      ctx.globalAlpha = 0.62 * (1 - levels[j][i] / (BANDS - 1)) ** 1.3 + 0.03;
      ctx.fillRect(i * cell, j * cell, cell + 0.5, cell + 0.5);
    }
  }
  ctx.globalAlpha = 0.5;
  ctx.fillStyle = line;
  for (let j = 0; j < CELLS; j++) {
    for (let i = 0; i < CELLS; i++) {
      if (i + 1 < CELLS && levels[j][i + 1] !== levels[j][i]) ctx.fillRect((i + 1) * cell - 0.5, j * cell, 1, cell);
      if (j + 1 < CELLS && levels[j + 1][i] !== levels[j][i]) ctx.fillRect(i * cell, (j + 1) * cell - 0.5, cell, 1);
    }
  }
  ctx.globalAlpha = 1;
}

const points = (history) => history.map((h) => `${px(h.w[0]).toFixed(1)},${py(h.w[1]).toFixed(1)}`).join(' ');

function drawOverlay() {
  const { SIZE, L, T, PLOT } = BOX;
  const ticks = [-2, -1, 0, 1, 2];
  const axes = ticks.map((v) => `
    <text class="tick" x="${px(v)}" y="${T + PLOT + 13}" text-anchor="middle">${v}</text>
    <text class="tick" x="${L - 5}" y="${py(v) + 3}" text-anchor="end">${v}</text>`).join('');

  const ghosts = state.ghosts.map((history) => `<polyline class="ghost" points="${points(history)}"/>
    <circle class="ghost-end" cx="${px(history[history.length - 1].w[0]).toFixed(1)}" cy="${py(history[history.length - 1].w[1]).toFixed(1)}" r="3"/>`).join('');

  let path = '';
  if (state.run) {
    const history = state.run.history;
    path = `<g opacity="${state.stale ? 0.35 : 1}">
      <polyline class="path" points="${points(history)}"/>
      ${history.length <= 80 ? history.map((h) => `<circle class="path-dot" cx="${px(h.w[0]).toFixed(1)}" cy="${py(h.w[1]).toFixed(1)}" r="2.6"/>`).join('') : ''}
    </g>`;
  }

  // The next step from the point on show: against the gradient, scaled by
  // the learning rate.
  const [a, b] = state.at;
  const [ga, gb] = surface().gradient(a, b);
  const next = [a - state.rate * ga, b - state.rate * gb];
  const arrow = Math.hypot(next[0] - a, next[1] - b) > 0.02
    ? `<line class="tangent" x1="${px(a).toFixed(1)}" y1="${py(b).toFixed(1)}" x2="${px(next[0]).toFixed(1)}" y2="${py(next[1]).toFixed(1)}" marker-end="url(#twoArrow)"/>`
    : '';
  const [sa, sb] = state.start;
  const offMap = Math.abs(a) > RANGE || Math.abs(b) > RANGE;

  $('twoOverlay').innerHTML = `<svg viewBox="0 0 ${SIZE} ${SIZE}" role="img" aria-label="Contour map of the loss for two weights. The point is at ${pair(state.at)}, where the loss is ${fmt(surface().loss(a, b), 3)}.">
    <defs>
      <clipPath id="twoClip"><rect x="${L}" y="${T}" width="${PLOT}" height="${PLOT}"/></clipPath>
      <marker id="twoArrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-head" d="M0 0 10 5 0 10Z"/></marker>
    </defs>
    <rect class="map-frame" x="${L}" y="${T}" width="${PLOT}" height="${PLOT}"/>
    ${axes}
    <text class="axis-title" x="${L + PLOT / 2}" y="${SIZE - 4}" text-anchor="middle">weight w₁</text>
    <text class="axis-title" transform="translate(10 ${T + PLOT / 2}) rotate(-90)" text-anchor="middle">weight w₂</text>
    <g clip-path="url(#twoClip)">
      ${ghosts}${path}
      <path class="start-mark" d="M${px(sa) - 5} ${py(sb) - 5}l10 10m0 -10l-10 10"/>
      ${arrow}
      <circle class="here" cx="${px(a).toFixed(1)}" cy="${py(b).toFixed(1)}" r="5.5"/>
    </g>
    ${offMap ? `<text class="tick strong" x="${L + PLOT - 6}" y="${T + 14}" text-anchor="end">the weights have left the map</text>` : ''}
  </svg>`;
}

// ===== Numbers, mathematics and tables =====

function updateNumbers(flash = true) {
  const [a, b] = state.at;
  const [ga, gb] = surface().gradient(a, b);
  setValue($('twoWeightsOut'), Math.abs(a) < 1000 && Math.abs(b) < 1000 ? pair(state.at) : 'off the map', flash);
  setValue($('twoLossOut'), big(surface().loss(a, b)), flash);
  $('twoStartOut').textContent = pair(state.start);
  if (Math.abs(a) < 1000 && Math.abs(b) < 1000) {
    renderTex($('twoNextTex'),
      `\\begin{aligned} \\text{gradient} &= (${texNum(ga)},\\ ${texNum(gb)}) \\\\ w_1 &: ${texNum(a)} - ${state.rate} \\times (${texNum(ga)}) = ${texNum(a - state.rate * ga)} \\\\ w_2 &: ${texNum(b)} - ${state.rate} \\times (${texNum(gb)}) = ${texNum(b - state.rate * gb)} \\end{aligned}`,
      { display: true });
  }
}

function updateRuns() {
  $('twoRunRows').innerHTML = state.runs.map((r, i) => `<tr>
    <td>${i + 1}</td>
    <td class="num">${pair(r.start)}</td>
    <td class="num">${r.rate}</td>
    <td class="num">${r.steps}</td>
    <td class="num">${Math.abs(r.end[0]) < 1000 && Math.abs(r.end[1]) < 1000 ? pair(r.end) : 'off the map'}</td>
    <td class="num">${big(r.loss)}</td>
    <td>${TWO_TRAIN_COPY.results[r.result]}</td>
  </tr>`).join('');
  $('twoRunsScroll').hidden = state.runs.length === 0;
  $('twoRunsEmpty').hidden = state.runs.length > 0;
  $('twoClearBtn').disabled = state.runs.length === 0 && state.ghosts.length === 0;
  $('twoRunsScroll').scrollTop = $('twoRunsScroll').scrollHeight;
}

// How a finished run ended, in one word: the outcome, or for a run that
// settled, the kind of place it settled in.
const resultOf = (snapshot) => (snapshot.outcome === 'settled' ? (snapshot.spot ? snapshot.spot.kind : 'flat') : snapshot.outcome);

function updateTrainUI() {
  $('twoPauseBtn').disabled = !engine.active;
  $('twoPauseBtn').textContent = engine.active && !engine.running ? 'Resume' : 'Pause';
  $('twoStepBtn').disabled = engine.running;
  $('twoSkipBtn').disabled = !engine.active;

  let message = TWO_TRAIN_COPY.untrained;
  let status = 'untrained';
  if (state.run && state.stale) {
    message = TWO_TRAIN_COPY.stale;
    status = 'stale';
  } else if (state.run) {
    const last = state.run.history[state.run.history.length - 1];
    if (state.phase === 'training') {
      message = TWO_TRAIN_COPY.progress(last) + (engine.running ? '' : ' (paused)');
      status = 'training';
    } else {
      message = TWO_TRAIN_COPY[resultOf(last)](last);
      status = 'done';
    }
  }
  $('twoTrainMessage').textContent = message;
  $('twoTrainMessage').dataset.status = status;
}

function render(flash = true) {
  drawOverlay();
  updateNumbers(flash);
  updateRuns();
  updateTrainUI();
}

// ===== Training =====

const engine = createEngine({
  onProgress(snapshot) {
    state.at = snapshot.w;
    render();
  },
  onFinish(snapshot) {
    state.phase = 'done';
    state.at = snapshot.w;
    state.runs.push({
      start: [...state.start], rate: state.rate, steps: snapshot.step, end: snapshot.w, loss: snapshot.loss, result: resultOf(snapshot),
    });
    if (state.runs.length > 30) state.runs.shift();
    render();
    announce(TWO_TRAIN_COPY[resultOf(snapshot)](snapshot));
  },
});

// Keep the path of a finished run on the map, faintly, when a new one begins.
function keepGhost() {
  if (state.run && state.phase === 'done' && !state.stale && state.run.history.length > 1) {
    state.ghosts.push(state.run.history);
    if (state.ghosts.length > MAX_GHOSTS) state.ghosts.shift();
  }
}

function startTraining({ paused = false } = {}) {
  keepGhost();
  state.at = [...state.start];
  state.run = createDescent2({ surface: surface(), start: state.start, learningRate: state.rate });
  state.phase = 'training';
  state.stale = false;
  engine.start(state.run, SPEEDS[state.speed], { paused });
  render();
}

// A new start point: stop, and wait for Train.
function setStart(a, b) {
  const snap = (v) => Math.max(-RANGE, Math.min(RANGE, Math.round(v / GRID) * GRID));
  engine.cancel();
  keepGhost();
  state.run = null;
  state.phase = 'untrained';
  state.stale = false;
  state.start = [snap(a), snap(b)];
  state.at = [...state.start];
  render(false);
}

function setSurface(key) {
  engine.cancel();
  state.key = key;
  state.run = null;
  state.phase = 'untrained';
  state.stale = false;
  state.ghosts = [];
  state.runs = [];
  state.at = [...state.start];
  $('twoSurfaceNote').innerHTML = TWO_WEIGHTS.surfaces[key];
  renderTex($('twoLossTex'), FORMULAS[key], { display: true });
  drawMap();
  render(false);
}

// ===== Controls =====

export function startPartTwo() {
  $('twoMapNote').innerHTML = TWO_WEIGHTS.map;
  $('twoDescentNote').innerHTML = TWO_WEIGHTS.descent;
  $('twoRateNote').innerHTML = TWO_WEIGHTS.learningRate;
  $('twoGradientKey').innerHTML = TWO_WEIGHTS.gradientKey;
  $('twoRunsNote').innerHTML = TWO_WEIGHTS.runs;
  $('twoRunsEmpty').textContent = TWO_WEIGHTS.runsEmpty;
  $('twoKeysNote').textContent = TWO_WEIGHTS.keys;
  renderTex($('twoUpdateTex'), '\\begin{aligned} w_1 &\\leftarrow w_1 - \\eta \\times \\text{slope along } w_1 \\\\ w_2 &\\leftarrow w_2 - \\eta \\times \\text{slope along } w_2 \\end{aligned}', { display: true });

  for (const radio of document.querySelectorAll('input[name="twoSurface"]')) {
    radio.addEventListener('change', () => {
      setSurface(radio.value);
      announce(`${SURFACES[radio.value].label} surface.`);
    });
  }

  // Click (or tap) the map to choose where training starts.
  const map = $('twoMap');
  map.addEventListener('click', (event) => {
    const box = map.getBoundingClientRect();
    const scale = BOX.SIZE / box.width;
    const x = (event.clientX - box.left) * scale;
    const y = (event.clientY - box.top) * scale;
    if (x < BOX.L || x > BOX.L + BOX.PLOT || y < BOX.T || y > BOX.T + BOX.PLOT) return;
    setStart(-RANGE + ((x - BOX.L) / BOX.PLOT) * 2 * RANGE, RANGE - ((y - BOX.T) / BOX.PLOT) * 2 * RANGE);
    announce(`Start at ${pair(state.start)}. Press Train.`);
  });
  // The arrow keys move the start one grid square at a time.
  map.addEventListener('keydown', (event) => {
    const moves = { ArrowLeft: [-GRID, 0], ArrowRight: [GRID, 0], ArrowUp: [0, GRID], ArrowDown: [0, -GRID] };
    const move = moves[event.key];
    if (!move) return;
    event.preventDefault();
    const size = event.shiftKey ? 5 : 1;
    setStart(state.start[0] + move[0] * size, state.start[1] + move[1] * size);
    announce(`Start at ${pair(state.start)}, loss ${fmt(surface().loss(...state.start), 3)}.`);
  });

  const showRate = () => {
    $('twoRateOut').textContent = String(state.rate);
    $('twoRateRange').setAttribute('aria-valuetext', String(state.rate));
  };
  $('twoRateRange').max = RATES.length - 1;
  $('twoRateRange').value = RATES.indexOf(state.rate);
  showRate();
  $('twoRateRange').addEventListener('input', () => {
    state.rate = RATES[Number($('twoRateRange').value)];
    showRate();
    engine.cancel();
    if (state.run) { state.stale = true; state.phase = 'done'; }
    render(false);
  });

  $('twoTrainBtn').addEventListener('click', () => {
    startTraining();
    announce('Training started.');
    if (prefersReducedMotion()) engine.skip();
  });
  $('twoPauseBtn').addEventListener('click', () => {
    if (engine.running) engine.pause(); else engine.resume();
    updateTrainUI();
  });
  $('twoStepBtn').addEventListener('click', () => {
    if (!engine.active) startTraining({ paused: true });
    engine.stepOnce();
    updateTrainUI();
  });
  $('twoSkipBtn').addEventListener('click', () => engine.skip());
  for (const radio of document.querySelectorAll('input[name="twoSpeed"]')) {
    radio.addEventListener('change', () => {
      state.speed = radio.value;
      engine.setSpeed(SPEEDS[state.speed]);
    });
  }

  $('twoClearBtn').addEventListener('click', () => {
    state.ghosts = [];
    state.runs = [];
    render(false);
    announce('Earlier runs cleared.');
  });

  // The map's colours come from the theme, so redraw it when that changes.
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { drawMap(); });

  // A browser may bring back earlier choices when the page is reloaded.
  const chosen = document.querySelector('input[name="twoSurface"]:checked');
  const pace = document.querySelector('input[name="twoSpeed"]:checked');
  if (pace) state.speed = pace.value;
  setSurface(chosen ? chosen.value : state.key);
}
