// Wiring for the training lab: page state, controls, the two charts and the
// training loop. The model itself (one weight, its loss and gradient descent)
// is in oneweight.js, which knows nothing about the page.

import { createEngine } from '../../../assets/js/engine.js';
import {
  fmt, texNum, setValue, renderTex, prefersReducedMotion,
} from '../../../assets/js/format.js';
import {
  WEIGHT_MIN, WEIGHT_MAX, makeLineData, loss, gradient, createDescent,
} from './oneweight.js';
import { createRng } from '../../../assets/js/rng.js';
import { startPartTwo } from './part2.js';
import { startPartThree } from './part3.js';
import { startPartFour } from './part4.js';
import { startPartFive } from './part5.js';
import {
  ONE_WEIGHT, TRAIN_COPY, RUNS_COPY, SEED_NOTE, ACTIVITIES, PRACTICE_NOTES,
} from '../content/notes.js';

const $ = (id) => document.getElementById(id);

const MAX_SEED = 999999;
const RATES = [0.01, 0.03, 0.1, 0.3, 0.6, 1, 1.5, 1.9, 2.05, 2.3]; // learning rate stops
const SPEEDS = {
  slow: { steps: 1, interval: 900 },
  normal: { steps: 1, interval: 350 },
  fast: { steps: 1, interval: 70 },
};

const state = {
  seed: 42,
  data: null,        // { seed, slope, points } from oneweight.js
  w: 0,              // the weight on show
  start: 0,          // where the weight was last put by hand: training starts here
  tried: [],         // every weight tried by hand, for the dots on the loss chart
  rate: 0.3,
  speed: 'normal',
  run: null,         // the current or latest gradient descent run
  phase: 'untrained', // 'untrained', 'training' or 'done'
  stale: false,      // the run on show no longer matches the settings
  showCurve: false,
  runs: [],          // every finished run: { start, rate, steps, end, outcome }
  starts: null,      // seeded source of new starting weights
};

function announce(message) {
  $('announce').textContent = message;
}

// ===== The two charts (SVG, coloured by the theme's CSS variables) =====

const BOX = { W: 360, H: 300, L: 40, R: 350, T: 12, B: 258 };
const X_MAX = 1.4;  // the data plot's x axis
const Y_MAX = 4.2;  // and its y axis

function frame(xTicks, yTicks, xTitle, yTitle, x, y) {
  const { W, H, L, R, T, B } = BOX;
  const lines = [
    ...xTicks.map((v) => `<line class="grid" x1="${x(v)}" y1="${T}" x2="${x(v)}" y2="${B}"/>
      <text class="tick" x="${x(v)}" y="${B + 13}" text-anchor="middle">${v}</text>`),
    ...yTicks.map((v) => `<line class="grid" x1="${L}" y1="${y(v)}" x2="${R}" y2="${y(v)}"/>
      <text class="tick" x="${L - 5}" y="${y(v) + 3}" text-anchor="end">${v}</text>`),
  ];
  return `${lines.join('')}
    <line class="axis" x1="${L}" y1="${B}" x2="${R}" y2="${B}"/><line class="axis" x1="${L}" y1="${T}" x2="${L}" y2="${B}"/>
    <text class="axis-title" x="${(L + R) / 2}" y="${H - 6}" text-anchor="middle">${xTitle}</text>
    <text class="axis-title" transform="translate(11 ${(T + B) / 2}) rotate(-90)" text-anchor="middle">${yTitle}</text>`;
}

const clip = (id) => `<clipPath id="${id}"><rect x="${BOX.L}" y="${BOX.T}" width="${BOX.R - BOX.L}" height="${BOX.B - BOX.T}"/></clipPath>`;

// The points, the line y = w x, and a dotted bar for every miss.
function drawData() {
  const { L, R, T, B, W, H } = BOX;
  const x = (v) => L + (v / X_MAX) * (R - L);
  const y = (v) => B - (v / Y_MAX) * (B - T);
  const { points } = state.data;
  const w = state.w;
  const misses = points.map((p) => `<line class="miss" x1="${x(p.x).toFixed(1)}" y1="${y(p.y).toFixed(1)}" x2="${x(p.x).toFixed(1)}" y2="${y(w * p.x).toFixed(1)}"/>`).join('');
  const dots = points.map((p) => `<circle class="point" cx="${x(p.x).toFixed(1)}" cy="${y(p.y).toFixed(1)}" r="4.2"/>`).join('');
  $('dataChart').innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Ten points and the line y = ${w.toFixed(2)} times x. Its loss is ${loss(w, points).toFixed(3)}.">
    <defs>${clip('dataClip')}</defs>
    ${frame([0, 0.5, 1], [0, 1, 2, 3, 4], 'x', 'y', x, y)}
    <g clip-path="url(#dataClip)">
      ${misses}
      <line class="model-line" x1="${x(0)}" y1="${y(0)}" x2="${x(X_MAX)}" y2="${y(w * X_MAX).toFixed(1)}"/>
    </g>
    ${dots}
  </svg>`;
}

// The loss against the weight: dots for the weights tried by hand, the path
// gradient descent has taken, and a ring with the slope at the current weight.
function drawLoss() {
  const { L, R, T, B, W, H } = BOX;
  const { points } = state.data;
  const top = Math.max(loss(WEIGHT_MIN, points), loss(WEIGHT_MAX, points)) * 1.05;
  const x = (v) => L + ((v - WEIGHT_MIN) / (WEIGHT_MAX - WEIGHT_MIN)) * (R - L);
  const y = (v) => B - (v / top) * (B - T);
  const at = (w) => `${x(w).toFixed(1)},${y(loss(w, points)).toFixed(1)}`;
  const yTicks = [0, 1, 2, 3, 4, 5, 6, 7, 8].filter((v) => v <= top);

  let curve = '';
  if (state.showCurve) {
    const samples = [];
    for (let w = WEIGHT_MIN; w <= WEIGHT_MAX + 1e-9; w += 0.05) samples.push(at(w));
    curve = `<polyline class="loss-curve" points="${samples.join(' ')}"/>`;
  }
  const tried = state.tried.map((w) => `<circle class="tried" cx="${x(w).toFixed(1)}" cy="${y(loss(w, points)).toFixed(1)}" r="2.4"/>`).join('');

  let path = '';
  if (state.run) {
    const history = state.run.history;
    path = `<g opacity="${state.stale ? 0.35 : 1}">
      <polyline class="path" points="${history.map((h) => at(h.w)).join(' ')}"/>
      ${history.map((h) => `<circle class="path-dot" cx="${x(h.w).toFixed(1)}" cy="${y(h.loss).toFixed(1)}" r="3"/>`).join('')}
    </g>`;
  }

  // The slope at the current weight, drawn as a short straight line.
  const w = state.w;
  const here = loss(w, points);
  const slope = gradient(w, points);
  const reach = 0.4;
  const tangent = `<line class="tangent" x1="${x(w - reach).toFixed(1)}" y1="${y(here - reach * slope).toFixed(1)}" x2="${x(w + reach).toFixed(1)}" y2="${y(here + reach * slope).toFixed(1)}"/>`;
  const offChart = w < WEIGHT_MIN || w > WEIGHT_MAX || here > top;

  $('lossChart').innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Loss against weight. At weight ${w.toFixed(2)} the loss is ${here.toFixed(3)} and the gradient is ${slope.toFixed(2)}.">
    <defs>${clip('lossClip')}</defs>
    ${frame([0, 1, 2, 3, 4], yTicks, 'weight w', 'loss', x, y)}
    <g clip-path="url(#lossClip)">
      ${curve}${tried}${path}${tangent}
      <circle class="here" cx="${x(w).toFixed(1)}" cy="${y(here).toFixed(1)}" r="6"/>
    </g>
    ${offChart ? `<text class="tick strong" x="${R - 4}" y="${T + 12}" text-anchor="end">the weight has left the chart</text>` : ''}
  </svg>`;
}

// ===== The numbers and the mathematics =====

function updateNumbers(flash = true) {
  const { points } = state.data;
  const w = state.w;
  const here = loss(w, points);
  const slope = gradient(w, points);
  const big = (value) => (Math.abs(value) >= 1000 ? value.toExponential(1).replace('-', '−') : fmt(value, 3));

  $('weightOut').textContent = fmt(w);
  setValue($('lossOut'), big(here), flash);
  const best = state.tried.length ? Math.min(...state.tried.map((t) => loss(t, points))) : here;
  setValue($('bestOut'), fmt(best, 3), flash);

  // The update rule with the current numbers in.
  if (Number.isFinite(w) && Math.abs(w) < 1000) {
    const next = w - state.rate * slope;
    renderTex($('nextTex'),
      `\\begin{aligned} \\text{gradient} &= ${texNum(slope)} \\\\ w_{\\text{new}} &= ${texNum(w)} - ${state.rate} \\times (${texNum(slope)}) = ${texNum(next)} \\end{aligned}`,
      { display: true });
  }
}

function updateSteps() {
  const body = $('stepRows');
  const history = state.run ? state.run.history : [];
  const cell = (value, places = 3) => (Math.abs(value) >= 1000 ? value.toExponential(1).replace('-', '−') : fmt(value, places));
  body.innerHTML = history.map((h) => `<tr>
    <td>${h.step === 0 ? 'start' : h.step}</td>
    <td class="num">${cell(h.w, 2)}</td>
    <td class="num">${cell(h.loss)}</td>
    <td class="num">${cell(h.gradient, 2)}</td>
    <td class="num">${h.move === null ? '' : cell(h.move, 2)}</td>
  </tr>`).join('');
  body.closest('.table-scroll').classList.toggle('fitted-stale', state.stale);
  body.closest('.table-scroll').hidden = history.length === 0;
  $('stepsNote').hidden = history.length > 0;
  const scroller = body.closest('.table-scroll');
  scroller.scrollTop = scroller.scrollHeight;
}

// The finished runs, so runs from different starts can be compared.
function updateRuns() {
  const cell = (value) => (Math.abs(value) >= 1000 ? value.toExponential(1).replace('-', '−') : fmt(value, 2));
  $('runRows').innerHTML = state.runs.map((r, i) => `<tr>
    <td>${i + 1}</td>
    <td class="num">${cell(r.start)}</td>
    <td class="num">${r.rate}</td>
    <td class="num">${r.steps}</td>
    <td class="num">${cell(r.end)}</td>
    <td>${RUNS_COPY.results[r.outcome]}</td>
  </tr>`).join('');
  $('runsScroll').hidden = state.runs.length === 0;
  $('runsEmpty').hidden = state.runs.length > 0;
  $('runsScroll').scrollTop = $('runsScroll').scrollHeight;
}

function updateTrainUI() {
  $('pauseBtn').disabled = !engine.active;
  $('pauseBtn').textContent = engine.active && !engine.running ? 'Resume' : 'Pause';
  $('stepBtn').disabled = engine.running;
  $('skipBtn').disabled = !engine.active;

  let message = TRAIN_COPY.untrained;
  let status = 'untrained';
  if (state.run && state.stale) {
    message = TRAIN_COPY.stale;
    status = 'stale';
  } else if (state.run) {
    const last = state.run.history[state.run.history.length - 1];
    if (state.phase === 'training') {
      message = TRAIN_COPY.progress(last) + (engine.running ? '' : ' (paused)');
      status = 'training';
    } else {
      message = TRAIN_COPY[last.outcome](last);
      status = 'done';
    }
  }
  $('trainMessage').textContent = message;
  $('trainMessage').dataset.status = status;
}

function render(flash = true) {
  drawData();
  drawLoss();
  updateNumbers(flash);
  updateSteps();
  updateRuns();
  updateTrainUI();
}

// ===== Training =====
//
// Nothing happens until Train (or Step) is pressed. Training always starts
// from the weight last set by hand.

const engine = createEngine({
  onProgress(snapshot) {
    showWeight(snapshot.w);
    render();
  },
  onFinish(snapshot) {
    state.phase = 'done';
    showWeight(snapshot.w);
    state.runs.push({ start: state.start, rate: state.rate, steps: snapshot.step, end: snapshot.w, outcome: snapshot.outcome });
    if (state.runs.length > 30) state.runs.shift();
    render();
    announce(TRAIN_COPY[snapshot.outcome](snapshot));
  },
});

// Put a weight on show, keeping the slider in step (it stops at its ends).
function showWeight(w) {
  state.w = w;
  $('weightRange').value = Math.max(WEIGHT_MIN, Math.min(WEIGHT_MAX, w));
}

function startTraining({ paused = false } = {}) {
  showWeight(state.start);
  state.run = createDescent({ points: state.data.points, start: state.start, learningRate: state.rate });
  state.phase = 'training';
  state.stale = false;
  engine.start(state.run, SPEEDS[state.speed], { paused });
  render();
}

// A setting changed: stop, fade the old run, and wait for Train.
function invalidate() {
  engine.cancel();
  if (state.run) {
    state.stale = true;
    state.phase = 'done';
  }
  render(false);
}

// The weight was moved by hand: any earlier run no longer starts from here.
function setWeightByHand(w) {
  engine.cancel();
  state.run = null;
  state.phase = 'untrained';
  state.stale = false;
  state.w = w;
  state.start = w;
  if (state.tried.length < 600) state.tried.push(w);
  render(false);
}

function newData(seed) {
  state.seed = Math.max(0, Math.min(MAX_SEED, Math.floor(seed)));
  state.data = makeLineData(state.seed);
  state.tried = [];
  state.runs = [];
  state.starts = createRng(state.seed, 'starts');
  $('seedInput').value = state.seed;
  $('weightRange').value = 0;
  setWeightByHand(0);
}

// ===== Controls =====

$('weightRange').addEventListener('input', () => setWeightByHand(Number($('weightRange').value)));

$('seedInput').addEventListener('change', () => {
  const seed = Number($('seedInput').value);
  if (Number.isFinite(seed)) newData(seed); else $('seedInput').value = state.seed;
});
$('newDataBtn').addEventListener('click', () => {
  // Choosing the next seed is the one place that is not seeded: it is the
  // source of a fresh variation. Everything after it follows from the seed.
  newData(Math.floor(Math.random() * 100000));
  announce(`New data, seed ${state.seed}.`);
});

// A new starting weight, at least a little way from the current one. It
// comes from the seed, so the same seed offers the same starts in order.
$('startBtn').addEventListener('click', () => {
  let w = state.start;
  while (Math.abs(w - state.start) < 0.6) {
    w = Math.round(state.starts.uniform(WEIGHT_MIN, WEIGHT_MAX) * 100) / 100;
  }
  $('weightRange').value = w;
  setWeightByHand(w);
  announce(`Starting weight ${fmt(w)}. Press Train.`);
});

$('showCurve').addEventListener('change', () => {
  state.showCurve = $('showCurve').checked;
  drawLoss();
});

function showRate() {
  $('rateOut').textContent = String(state.rate);
  $('rateRange').setAttribute('aria-valuetext', String(state.rate));
}
$('rateRange').addEventListener('input', () => {
  state.rate = RATES[Number($('rateRange').value)];
  showRate();
  invalidate(); // changing a setting never retrains: it waits for Train
});

$('trainBtn').addEventListener('click', () => {
  startTraining();
  announce('Training started.');
  // With reduced motion, skip the animation and show the finished run.
  if (prefersReducedMotion()) engine.skip();
});
$('pauseBtn').addEventListener('click', () => {
  if (engine.running) engine.pause(); else engine.resume();
  updateTrainUI();
});
$('stepBtn').addEventListener('click', () => {
  // Step also works from a standing start: it begins training, paused.
  if (!engine.active) startTraining({ paused: true });
  engine.stepOnce();
  updateTrainUI();
});
$('skipBtn').addEventListener('click', () => engine.skip());
for (const radio of document.querySelectorAll('input[name="speed"]')) {
  radio.addEventListener('change', () => {
    state.speed = radio.value;
    engine.setSpeed(SPEEDS[state.speed]);
  });
}

// ===== Start =====

$('modelNote').innerHTML = ONE_WEIGHT.model;
$('lossNote').innerHTML = ONE_WEIGHT.loss;
$('showCurveLabel').textContent = ONE_WEIGHT.showCurve;
$('descentNote').innerHTML = ONE_WEIGHT.descent;
$('rateNote').innerHTML = ONE_WEIGHT.learningRate;
$('lossKey').innerHTML = ONE_WEIGHT.lossKey;
$('gradientKey').innerHTML = ONE_WEIGHT.gradientKey;
$('updateKey').innerHTML = ONE_WEIGHT.updateKey;
$('stepsNote').textContent = ONE_WEIGHT.stepsEmpty;
$('linkNote').innerHTML = ONE_WEIGHT.note;
$('seedNote').innerHTML = SEED_NOTE;
$('runsNote').innerHTML = RUNS_COPY.intro;
$('runsEmpty').textContent = RUNS_COPY.empty;
$('startBtn').textContent = RUNS_COPY.startButton;

renderTex($('lossTex'), 'L(w) = \\frac{1}{n}\\sum_{i=1}^{n}\\big(w\\,x_i - y_i\\big)^2', { display: true });
renderTex($('gradientTex'), '\\text{gradient} = \\frac{2}{n}\\sum_{i=1}^{n}\\big(w\\,x_i - y_i\\big)\\,x_i', { display: true });
renderTex($('updateTex'), 'w_{\\text{new}} = w - \\eta \\times \\text{gradient}', { display: true });

$('rateRange').value = RATES.indexOf(state.rate);
showRate();

// Optional address-bar setting, for example ?seed=42.
const query = new URLSearchParams(location.search);
if (/^\d{1,6}$/.test(query.get('seed') || '')) state.seed = Number(query.get('seed'));

$('loadNotice').remove(); // the scripts loaded, so the server note is not needed
newData(state.seed);
startPartTwo();
startPartThree();
startPartFour();
startPartFive();

// Predict, observe, explain: one activity for each part, and a practical
// note for some. The wording is in content/notes.js.
for (const [key, activity] of Object.entries(ACTIVITIES)) {
  $(`activity-${key}`).innerHTML = `<details class="activity">
    <summary>Activity: predict, observe, explain</summary>
    <ol class="steps">
      <li><strong>Predict.</strong> ${activity.predict}</li>
      <li><strong>Observe.</strong> ${activity.observe}</li>
      <li><strong>Explain.</strong> Decide for yourself what happened and why. Then check.
        <details class="reveal"><summary>Reveal the explanation</summary><p>${activity.explain}</p></details>
      </li>
    </ol>
  </details>`;
}
for (const [key, note] of Object.entries(PRACTICE_NOTES)) {
  $(`practice-${key}`).innerHTML = `<aside class="field-note" aria-label="Practical note">
    <p class="crumb">In practice</p>
    <p class="field-note-title">${note.title}</p>
    <p>${note.text}</p>
  </aside>`;
}
