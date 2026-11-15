// Part 3 of the training lab: better ways to step.
//   The race        four optimisers from the same start on the same map
//   Batch size      full batch, mini-batch and single points on real data
// Page wiring only. The models are in twoweights.js and batches.js.

import { createEngine } from '../../../assets/js/engine.js';
import { fmt, prefersReducedMotion } from '../../../assets/js/format.js';
import { OPTIMISERS } from '../../../assets/js/optimisers.js';
import { SURFACES, RANGE, createRace } from './twoweights.js';
import {
  SLOPE_RANGE, INTERCEPT_RANGE, makeScatter, lineSurface, createBatchRace,
} from './batches.js';
import { BOX, createMapView } from './mapview.js';
import { RACE_COPY, BATCH_COPY } from '../content/notes.js';

const $ = (id) => document.getElementById(id);
const pair = ([a, b]) => `(${fmt(a)}, ${fmt(b)})`;
const GRID = 0.05;
const snapTo = (v, [low, high]) => Math.max(low, Math.min(high, Math.round(v / GRID) * GRID));
const onMap = ([a, b], [x0, x1], [y0, y1]) => a >= x0 && a <= x1 && b >= y0 && b <= y1;

// The shape drawn at the head of each path, so colour is never the only cue.
const HEADS = {
  circle: (x, y) => `<circle cx="${x}" cy="${y}" r="5"/>`,
  square: (x, y) => `<rect x="${x - 4.5}" y="${y - 4.5}" width="9" height="9"/>`,
  triangle: (x, y) => `<path d="M${x} ${y - 6}L${x + 5.6} ${y + 4.2}H${x - 5.6}Z"/>`,
  diamond: (x, y) => `<path d="M${x} ${y - 6.2}L${x + 6.2} ${y}L${x} ${y + 6.2}L${x - 6.2} ${y}Z"/>`,
};
const head = (shape, x, y, cls) => `<g class="head ${cls}">${HEADS[shape](Number(x.toFixed(1)), Number(y.toFixed(1)))}</g>`;
const swatch = (shape, cls) => `<svg class="swatch" viewBox="0 0 34 14" aria-hidden="true"><line class="lane ${cls}" x1="1" y1="7" x2="33" y2="7"/>${head(shape, 17, 7, cls)}</svg>`;
const startMark = (x, y) => `<path class="start-mark" d="M${x - 5} ${y - 5}l10 10m0 -10l-10 10"/>`;

// Wire up a Train / Pause / Step / Skip bar and its speed buttons.
function trainBar(prefix, engine, speeds, begin, getSpeed, setSpeed) {
  const refresh = () => {
    $(`${prefix}PauseBtn`).disabled = !engine.active;
    $(`${prefix}PauseBtn`).textContent = engine.active && !engine.running ? 'Resume' : 'Pause';
    $(`${prefix}StepBtn`).disabled = engine.running;
    $(`${prefix}SkipBtn`).disabled = !engine.active;
  };
  $(`${prefix}TrainBtn`).addEventListener('click', () => {
    begin(false);
    if (prefersReducedMotion()) engine.skip();
  });
  $(`${prefix}PauseBtn`).addEventListener('click', () => {
    if (engine.running) engine.pause(); else engine.resume();
    refresh();
  });
  $(`${prefix}StepBtn`).addEventListener('click', () => {
    if (!engine.active) begin(true);
    engine.stepOnce();
    refresh();
  });
  $(`${prefix}SkipBtn`).addEventListener('click', () => engine.skip());
  for (const radio of document.querySelectorAll(`input[name="${prefix}Speed"]`)) {
    if (radio.checked) setSpeed(radio.value);
    radio.addEventListener('change', () => {
      setSpeed(radio.value);
      engine.setSpeed(speeds[getSpeed()]);
    });
  }
  return refresh;
}

// ===================================================================
// The race
// ===================================================================

const RUNNERS = {
  gd: { shape: 'circle', rates: [0.03, 0.1, 0.3, 0.6, 0.9, 1.5], rate: 0.3 },
  momentum: { shape: 'square', rates: [0.01, 0.03, 0.1, 0.2, 0.3], rate: 0.1 },
  rmsprop: { shape: 'triangle', rates: [0.01, 0.03, 0.05, 0.1, 0.3], rate: 0.05 },
  adam: { shape: 'diamond', rates: [0.03, 0.1, 0.2, 0.3, 0.6], rate: 0.1 },
};
const RACE_SPEEDS = {
  slow: { steps: 1, interval: 400 },
  normal: { steps: 1, interval: 110 },
  fast: { steps: 2, interval: 25 },
};

const race = {
  key: 'valley',
  start: [1.5, 1.2],
  rates: Object.fromEntries(Object.entries(RUNNERS).map(([name, r]) => [name, r.rate])),
  on: { gd: true, momentum: true, rmsprop: true, adam: true },
  speed: 'normal',
  snapshot: null,
  phase: 'untrained',
  stale: false,
  view: null,
};

const raceResult = (r) => (r.outcome === 'arrived' ? r.spot.kind : r.outcome);

function drawRace() {
  const { view } = race;
  const ticks = [-2, -1, 0, 1, 2];
  const range = [-RANGE, RANGE];
  let paths = '';
  let heads = '';
  if (race.snapshot) {
    for (const r of race.snapshot.runners) {
      paths += `<polyline class="lane run-${r.name}" points="${view.line(r.history)}"/>`;
      if (onMap(r.w, range, range)) heads += head(RUNNERS[r.name].shape, view.px(r.w[0]), view.py(r.w[1]), `run-${r.name}`);
    }
  }
  const [sa, sb] = race.start;
  const summary = race.snapshot
    ? race.snapshot.runners.map((r) => `${OPTIMISERS[r.name].label} at ${pair(r.w)}`).join('; ')
    : `Start at ${pair(race.start)}`;
  $('raceOverlay').innerHTML = `<svg viewBox="0 0 ${BOX.SIZE} ${BOX.SIZE}" role="img" aria-label="Contour map with the path of each optimiser. ${summary}.">
    <defs>${view.clip('raceClip')}</defs>
    ${view.axes(ticks, ticks, 'weight w₁', 'weight w₂')}
    <g clip-path="url(#raceClip)" opacity="${race.stale ? 0.35 : 1}">${paths}${heads}</g>
    ${startMark(view.px(sa), view.py(sb))}
  </svg>`;
}

function updateRace() {
  drawRace();
  const rows = race.snapshot ? race.snapshot.runners : [];
  $('raceRows').innerHTML = rows.map((r) => `<tr>
    <th scope="row">${swatch(RUNNERS[r.name].shape, `run-${r.name}`)} ${OPTIMISERS[r.name].label}</th>
    <td class="num">${r.learningRate}</td>
    <td class="num">${r.steps}</td>
    <td class="num">${Math.abs(r.w[0]) < 1000 && Math.abs(r.w[1]) < 1000 ? pair(r.w) : 'off the map'}</td>
    <td>${r.outcome ? RACE_COPY.results[raceResult(r)] : RACE_COPY.results.going}</td>
  </tr>`).join('');
  $('raceScroll').hidden = rows.length === 0;
  $('raceScroll').classList.toggle('fitted-stale', race.stale);
  $('raceEmpty').hidden = rows.length > 0;

  let message = RACE_COPY.untrained;
  let status = 'untrained';
  if (race.snapshot && race.stale) {
    message = RACE_COPY.stale;
    status = 'stale';
  } else if (race.snapshot && race.phase === 'training') {
    const going = rows.filter((r) => !r.outcome).length;
    message = RACE_COPY.progress(race.snapshot.step, going) + (raceEngine.running ? '' : ' (paused)');
    status = 'training';
  } else if (race.snapshot) {
    message = raceSummary(rows);
    status = 'done';
  }
  $('raceMessage').textContent = message;
  $('raceMessage').dataset.status = status;
  refreshRaceBar();
}

// One or two sentences on how the race went.
function raceSummary(rows) {
  const arrived = rows.filter((r) => r.outcome === 'arrived').sort((a, b) => a.steps - b.steps);
  const parts = [];
  if (arrived.length) {
    const first = arrived[0];
    parts.push(RACE_COPY.first(OPTIMISERS[first.name].label, first.steps));
    if (arrived.length > 1) {
      const last = arrived[arrived.length - 1];
      parts.push(RACE_COPY.last(OPTIMISERS[last.name].label, last.steps));
    }
    const places = new Set(arrived.map((r) => r.spot.at.join()));
    if (places.size > 1) parts.push(RACE_COPY.differentPlaces);
    else if (arrived[0].spot.kind !== 'lowest') parts.push(RACE_COPY.notLowest[arrived[0].spot.kind]);
  } else {
    parts.push(RACE_COPY.nobody);
  }
  const lost = rows.filter((r) => r.outcome === 'diverged').map((r) => OPTIMISERS[r.name].label);
  if (lost.length) parts.push(RACE_COPY.diverged(lost.join(' and ')));
  const slow = rows.filter((r) => r.outcome === 'limit').map((r) => OPTIMISERS[r.name].label);
  if (slow.length) parts.push(RACE_COPY.limit(slow.join(' and ')));
  return parts.join(' ');
}

const raceEngine = createEngine({
  onProgress(snapshot) {
    race.snapshot = snapshot;
    updateRace();
  },
  onFinish(snapshot) {
    race.snapshot = snapshot;
    race.phase = 'done';
    updateRace();
    $('raceAnnounce').textContent = raceSummary(snapshot.runners);
  },
});
let refreshRaceBar = () => {};

function beginRace(paused) {
  const runners = Object.keys(RUNNERS).filter((name) => race.on[name]).map((name) => ({ name, learningRate: race.rates[name] }));
  if (runners.length === 0) {
    $('raceAnnounce').textContent = RACE_COPY.noneChosen;
    $('raceMessage').textContent = RACE_COPY.noneChosen;
    return;
  }
  const trainer = createRace({ surface: SURFACES[race.key], start: race.start, runners });
  race.snapshot = trainer.snapshot();
  race.phase = 'training';
  race.stale = false;
  raceEngine.start(trainer, RACE_SPEEDS[race.speed], { paused });
  updateRace();
}

// A setting changed: stop and wait for Train. `clear` drops the old paths.
function raceChanged(clear) {
  raceEngine.cancel();
  if (clear) { race.snapshot = null; race.phase = 'untrained'; race.stale = false; }
  else if (race.snapshot) { race.stale = true; race.phase = 'done'; }
  updateRace();
}

function startRaceSection() {
  race.view = createMapView({ canvas: $('raceCanvas'), xRange: [-RANGE, RANGE], yRange: [-RANGE, RANGE] });
  $('raceNote').innerHTML = RACE_COPY.intro;
  $('raceMapNote').innerHTML = RACE_COPY.map;
  $('raceEmpty').textContent = RACE_COPY.empty;
  $('raceFairNote').innerHTML = RACE_COPY.fair;

  // One row of controls for each optimiser: a tick box, its learning rate
  // and an explanation.
  $('raceRunners').innerHTML = Object.entries(RUNNERS).map(([name, r]) => `<div class="param runner">
    <label class="check"><input type="checkbox" id="raceOn-${name}" checked><span>${swatch(r.shape, `run-${name}`)} <strong>${OPTIMISERS[name].label}</strong></span></label>
    <div class="field">
      <label for="raceRate-${name}">Learning rate <output id="raceRateOut-${name}" for="raceRate-${name}"></output></label>
      <input id="raceRate-${name}" type="range" min="0" max="${r.rates.length - 1}" step="1" value="${r.rates.indexOf(r.rate)}">
    </div>
    <details class="explain"><summary>How does it step?</summary><p>${RACE_COPY.optimisers[name]}</p></details>
  </div>`).join('');
  for (const [name, r] of Object.entries(RUNNERS)) {
    const range = $(`raceRate-${name}`);
    const show = () => {
      $(`raceRateOut-${name}`).textContent = String(race.rates[name]);
      range.setAttribute('aria-valuetext', String(race.rates[name]));
    };
    show();
    range.addEventListener('input', () => {
      race.rates[name] = r.rates[Number(range.value)];
      show();
      raceChanged(false);
    });
    $(`raceOn-${name}`).addEventListener('change', () => {
      race.on[name] = $(`raceOn-${name}`).checked;
      raceChanged(false);
    });
  }

  const setSurface = (key) => {
    race.key = key;
    $('raceSurfaceNote').innerHTML = RACE_COPY.surfaces[key];
    race.view.draw(SURFACES[key]);
    raceChanged(true);
  };
  for (const radio of document.querySelectorAll('input[name="raceSurface"]')) {
    if (radio.checked) race.key = radio.value;
    radio.addEventListener('change', () => {
      setSurface(radio.value);
      $('raceAnnounce').textContent = `${SURFACES[radio.value].label} surface.`;
    });
  }

  const map = $('raceMap');
  const setStart = (a, b) => {
    race.start = [snapTo(a, [-RANGE, RANGE]), snapTo(b, [-RANGE, RANGE])];
    raceChanged(true);
    $('raceAnnounce').textContent = `Start at ${pair(race.start)}. Press Train.`;
  };
  map.addEventListener('click', (event) => {
    const at = race.view.weightsAt(event, map);
    if (at) setStart(at[0], at[1]);
  });
  map.addEventListener('keydown', (event) => {
    const move = { ArrowLeft: [-GRID, 0], ArrowRight: [GRID, 0], ArrowUp: [0, GRID], ArrowDown: [0, -GRID] }[event.key];
    if (!move) return;
    event.preventDefault();
    const size = event.shiftKey ? 5 : 1;
    setStart(race.start[0] + move[0] * size, race.start[1] + move[1] * size);
  });

  refreshRaceBar = trainBar('race', raceEngine, RACE_SPEEDS, beginRace, () => race.speed, (v) => { race.speed = v; });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => race.view.draw(SURFACES[race.key]));
  setSurface(race.key);
}

// ===================================================================
// Batch size
// ===================================================================

const BATCH_RATES = [0.01, 0.03, 0.1, 0.2, 0.4];
const MINI_SIZES = [2, 4, 8, 20];
const PASSES = 20;
const BATCH_SPEEDS = {
  slow: { steps: 1, interval: 160 },
  normal: { steps: 2, interval: 50 },
  fast: { steps: 6, interval: 25 },
};
const BATCH_LOOK = {
  full: { shape: 'circle', cls: 'run-gd' },
  mini: { shape: 'square', cls: 'run-momentum' },
  single: { shape: 'diamond', cls: 'run-adam' },
};
const MAX_SEED = 999999;

const batch = {
  seed: 42,
  data: null,
  surface: null,
  start: [-0.5, 2],
  rate: 0.1,
  mini: 8,
  speed: 'normal',
  snapshot: null,
  phase: 'untrained',
  stale: false,
  view: null,
};

const batchLabel = (r) => BATCH_COPY.labels[r.key](r.batchSize);

// The data and each runner's current line.
function drawBatchData() {
  const W = 360; const H = 230; const L = 30; const R = 352; const T = 8; const B = 204;
  const x = (v) => L + (v / 2) * (R - L);
  const y = (v) => B - ((v + 0.5) / 5) * (B - T);
  const lines = (batch.snapshot ? batch.snapshot.runners : []).map((r) => {
    const cls = BATCH_LOOK[r.key].cls;
    return `<line class="lane ${cls}" x1="${x(0)}" y1="${y(r.w[1]).toFixed(1)}" x2="${x(2)}" y2="${y(2 * r.w[0] + r.w[1]).toFixed(1)}"/>`;
  }).join('');
  const first = batch.snapshot ? '' : `<line class="lane run-start" x1="${x(0)}" y1="${y(batch.start[1]).toFixed(1)}" x2="${x(2)}" y2="${y(2 * batch.start[0] + batch.start[1]).toFixed(1)}"/>`;
  $('batchData').innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${batch.data.points.length} data points and the line each run has reached.">
    <defs><clipPath id="batchDataClip"><rect x="${L}" y="${T}" width="${R - L}" height="${B - T}"/></clipPath></defs>
    ${[0, 1, 2].map((v) => `<text class="tick" x="${x(v)}" y="${B + 13}" text-anchor="middle">${v}</text>`).join('')}
    ${[0, 1, 2, 3, 4].map((v) => `<line class="grid" x1="${L}" y1="${y(v)}" x2="${R}" y2="${y(v)}"/><text class="tick" x="${L - 5}" y="${y(v) + 3}" text-anchor="end">${v}</text>`).join('')}
    <line class="axis" x1="${L}" y1="${B}" x2="${R}" y2="${B}"/><line class="axis" x1="${L}" y1="${T}" x2="${L}" y2="${B}"/>
    <text class="axis-title" x="${(L + R) / 2}" y="${H - 4}" text-anchor="middle">x</text>
    ${batch.data.points.map((p) => `<circle class="point" cx="${x(p.x).toFixed(1)}" cy="${y(p.y).toFixed(1)}" r="3"/>`).join('')}
    <g clip-path="url(#batchDataClip)" opacity="${batch.stale ? 0.35 : 1}">${first}${lines}</g>
  </svg>`;
}

function drawBatchMap() {
  const { view } = batch;
  // The noisiest run is drawn first, so the steadier ones stay visible on top.
  const runners = batch.snapshot ? [...batch.snapshot.runners].reverse() : [];
  const paths = runners.map((r) => `<polyline class="lane thin ${BATCH_LOOK[r.key].cls}" points="${view.line(r.history.map((h) => h.w))}"/>`).join('');
  const heads = runners.filter((r) => onMap(r.w, SLOPE_RANGE, INTERCEPT_RANGE))
    .map((r) => head(BATCH_LOOK[r.key].shape, view.px(r.w[0]), view.py(r.w[1]), BATCH_LOOK[r.key].cls)).join('');
  const [ba, bb] = batch.surface.best;
  $('batchOverlay').innerHTML = `<svg viewBox="0 0 ${BOX.SIZE} ${BOX.SIZE}" role="img" aria-label="Contour map of the loss against the line's steepness and height, with the path of each batch size. The best line is at ${pair(batch.surface.best)}.">
    <defs>${view.clip('batchClip')}</defs>
    ${view.axes([-1, 0, 1, 2, 3], [-1, 0, 1, 2], 'steepness a', 'height b')}
    <g clip-path="url(#batchClip)" opacity="${batch.stale ? 0.35 : 1}">${paths}${heads}</g>
    <path class="best-mark" d="M${view.px(ba) - 6} ${view.py(bb)}h12M${view.px(ba)} ${view.py(bb) - 6}v12"/>
    ${startMark(view.px(batch.start[0]), view.py(batch.start[1]))}
  </svg>`;
}

// The loss (on all the data) against how much data each run has looked at.
function drawBatchLoss() {
  const W = 360; const H = 210; const L = 34; const R = 352; const T = 8; const B = 184;
  const n = batch.data.points.length;
  const floor = batch.surface.loss(...batch.surface.best);
  const top = Math.max(floor * 1.5, batch.surface.loss(...batch.start)) * 1.05;
  const x = (seen) => L + (seen / (PASSES * n)) * (R - L);
  const y = (v) => B - (Math.min(v, top * 1.2) / top) * (B - T);
  const step = top > 4 ? 2 : top > 2 ? 1 : 0.5;
  const yTicks = [];
  for (let v = 0; v <= top; v += step) yTicks.push(v);
  const curves = (batch.snapshot ? [...batch.snapshot.runners].reverse() : []).map((r) => `<polyline class="lane thin ${BATCH_LOOK[r.key].cls}" points="${r.history.map((h) => `${x(h.seen).toFixed(1)},${y(h.loss).toFixed(1)}`).join(' ')}"/>`).join('');
  $('batchLoss').innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Loss against passes through the data, one curve for each batch size.">
    <defs><clipPath id="batchLossClip"><rect x="${L}" y="${T}" width="${R - L}" height="${B - T}"/></clipPath></defs>
    ${[0, 5, 10, 15, 20].map((v) => `<text class="tick" x="${x(v * n)}" y="${B + 13}" text-anchor="middle">${v}</text>`).join('')}
    ${yTicks.map((v) => `<line class="grid" x1="${L}" y1="${y(v)}" x2="${R}" y2="${y(v)}"/><text class="tick" x="${L - 5}" y="${y(v) + 3}" text-anchor="end">${v}</text>`).join('')}
    <line class="axis" x1="${L}" y1="${B}" x2="${R}" y2="${B}"/><line class="axis" x1="${L}" y1="${T}" x2="${L}" y2="${B}"/>
    <line class="threshold-line" x1="${L}" y1="${y(floor)}" x2="${R}" y2="${y(floor)}"/>
    <text class="tick" x="${R - 2}" y="${y(floor) + 10}" text-anchor="end">lowest possible loss</text>
    <text class="axis-title" x="${(L + R) / 2}" y="${H - 2}" text-anchor="middle">passes through the data</text>
    <g clip-path="url(#batchLossClip)" opacity="${batch.stale ? 0.35 : 1}">${curves}</g>
  </svg>`;
}

function updateBatch() {
  drawBatchData();
  drawBatchMap();
  drawBatchLoss();
  const rows = batch.snapshot ? batch.snapshot.runners : [];
  const [ba, bb] = batch.surface.best;
  $('batchRows').innerHTML = rows.map((r) => `<tr>
    <th scope="row">${swatch(BATCH_LOOK[r.key].shape, BATCH_LOOK[r.key].cls)} ${batchLabel(r)}</th>
    <td class="num">${r.batchSize}</td>
    <td class="num">${r.updates}</td>
    <td class="num">${r.outcome === 'diverged' ? 'grew without limit' : fmt(r.loss, 3)}</td>
    <td class="num">${r.outcome === 'diverged' ? '' : fmt(Math.hypot(r.w[0] - ba, r.w[1] - bb), 3)}</td>
  </tr>`).join('');
  $('batchScroll').hidden = rows.length === 0;
  $('batchScroll').classList.toggle('fitted-stale', batch.stale);
  $('batchEmpty').hidden = rows.length > 0;

  let message = BATCH_COPY.untrained;
  let status = 'untrained';
  if (batch.snapshot && batch.stale) {
    message = BATCH_COPY.stale;
    status = 'stale';
  } else if (batch.snapshot && batch.phase === 'training') {
    message = BATCH_COPY.progress(batch.snapshot.pass, PASSES) + (batchEngine.running ? '' : ' (paused)');
    status = 'training';
  } else if (batch.snapshot) {
    message = batchSummary(rows);
    status = 'done';
  }
  $('batchMessage').textContent = message;
  $('batchMessage').dataset.status = status;
  refreshBatchBar();
}

function batchSummary(rows) {
  const [ba, bb] = batch.surface.best;
  const ok = rows.filter((r) => r.outcome !== 'diverged')
    .map((r) => ({ r, off: Math.hypot(r.w[0] - ba, r.w[1] - bb) }))
    .sort((p, q) => p.off - q.off);
  const lost = rows.filter((r) => r.outcome === 'diverged').map(batchLabel);
  const parts = [BATCH_COPY.finished(PASSES)];
  if (ok.length) parts.push(BATCH_COPY.closest(batchLabel(ok[0].r), ok[0].r.updates));
  if (lost.length) parts.push(BATCH_COPY.diverged(lost.join(' and ')));
  return parts.join(' ');
}

const batchEngine = createEngine({
  onProgress(snapshot) {
    batch.snapshot = snapshot;
    updateBatch();
  },
  onFinish(snapshot) {
    batch.snapshot = snapshot;
    batch.phase = 'done';
    updateBatch();
    $('batchAnnounce').textContent = batchSummary(snapshot.runners);
  },
});
let refreshBatchBar = () => {};

function beginBatch(paused) {
  const n = batch.data.points.length;
  const trainer = createBatchRace({
    points: batch.data.points,
    start: batch.start,
    learningRate: batch.rate,
    seed: batch.seed,
    passes: PASSES,
    runners: [{ key: 'full', batchSize: n }, { key: 'mini', batchSize: batch.mini }, { key: 'single', batchSize: 1 }],
  });
  batch.snapshot = trainer.snapshot();
  batch.phase = 'training';
  batch.stale = false;
  batchEngine.start(trainer, BATCH_SPEEDS[batch.speed], { paused });
  updateBatch();
}

function batchChanged(clear) {
  batchEngine.cancel();
  if (clear) { batch.snapshot = null; batch.phase = 'untrained'; batch.stale = false; }
  else if (batch.snapshot) { batch.stale = true; batch.phase = 'done'; }
  updateBatch();
}

function newBatchData(seed) {
  batch.seed = Math.max(0, Math.min(MAX_SEED, Math.floor(seed)));
  batch.data = makeScatter(batch.seed);
  batch.surface = lineSurface(batch.data.points);
  $('batchSeed').value = batch.seed;
  batch.view.draw(batch.surface);
  batchChanged(true);
}

function startBatchSection() {
  batch.view = createMapView({ canvas: $('batchCanvas'), xRange: SLOPE_RANGE, yRange: INTERCEPT_RANGE });
  $('batchNote').innerHTML = BATCH_COPY.intro;
  $('batchDataNote').innerHTML = BATCH_COPY.data;
  $('batchMapNote').innerHTML = BATCH_COPY.map;
  $('batchLossNote').innerHTML = BATCH_COPY.loss;
  $('batchEmpty').textContent = BATCH_COPY.empty;
  $('batchRateNote').innerHTML = BATCH_COPY.learningRate;
  $('batchMiniNote').innerHTML = BATCH_COPY.miniSize;
  $('batchSeedNote').innerHTML = BATCH_COPY.seed;
  $('batchLegend').innerHTML = Object.entries(BATCH_LOOK)
    .map(([key, look]) => `<span>${swatch(look.shape, look.cls)} ${BATCH_COPY.legend[key]}</span>`).join('');

  const slider = (id, out, stops, get, set) => {
    const range = $(id);
    range.max = stops.length - 1;
    range.value = stops.indexOf(get());
    const show = () => {
      $(out).textContent = String(get());
      range.setAttribute('aria-valuetext', String(get()));
    };
    show();
    range.addEventListener('input', () => {
      set(stops[Number(range.value)]);
      show();
      batchChanged(false);
    });
  };
  slider('batchRate', 'batchRateOut', BATCH_RATES, () => batch.rate, (v) => { batch.rate = v; });
  slider('batchMini', 'batchMiniOut', MINI_SIZES, () => batch.mini, (v) => { batch.mini = v; });

  $('batchSeed').addEventListener('change', () => {
    const seed = Number($('batchSeed').value);
    if (Number.isFinite(seed)) newBatchData(seed); else $('batchSeed').value = batch.seed;
  });
  $('batchNewBtn').addEventListener('click', () => {
    // Choosing the next seed is the one place that is not seeded.
    newBatchData(Math.floor(Math.random() * 100000));
    $('batchAnnounce').textContent = `New data, seed ${batch.seed}.`;
  });

  const map = $('batchMap');
  const setStart = (a, b) => {
    batch.start = [snapTo(a, SLOPE_RANGE), snapTo(b, INTERCEPT_RANGE)];
    batchChanged(true);
    $('batchAnnounce').textContent = `Start at steepness ${fmt(batch.start[0])}, height ${fmt(batch.start[1])}. Press Train.`;
  };
  map.addEventListener('click', (event) => {
    const at = batch.view.weightsAt(event, map);
    if (at) setStart(at[0], at[1]);
  });
  map.addEventListener('keydown', (event) => {
    const move = { ArrowLeft: [-GRID, 0], ArrowRight: [GRID, 0], ArrowUp: [0, GRID], ArrowDown: [0, -GRID] }[event.key];
    if (!move) return;
    event.preventDefault();
    const size = event.shiftKey ? 5 : 1;
    setStart(batch.start[0] + move[0] * size, batch.start[1] + move[1] * size);
  });

  refreshBatchBar = trainBar('batch', batchEngine, BATCH_SPEEDS, beginBatch, () => batch.speed, (v) => { batch.speed = v; });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => batch.view.draw(batch.surface));
  newBatchData(batch.seed);
}

export function startPartThree() {
  startRaceSection();
  startBatchSection();
}
