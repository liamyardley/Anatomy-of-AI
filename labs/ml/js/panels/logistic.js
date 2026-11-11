// The maths panel for logistic regression: the idea, the formulas, the
// model's live numbers, a worked example for the query point, and two small
// charts (the loss curve and the sigmoid).
//
// A panel is created with two places to draw into:
//   intro   above the settings: what the model is and its formula
//   detail  below the settings: its internals, worked example and visuals
// and returns { update(info, context) }, where `info` comes from the model's
// explain() and `context` is { threshold, status, flash }.

import { MODEL_NOTES, COPY } from '../../content/notes.js';
import { fmt, texNum, setValue, renderTex, MARK } from '../format.js';

const note = MODEL_NOTES.logistic;

// "0.42 + 1.31 x_1 - 0.80 x_2" from a list of { tex, weight }.
function sumTex(terms) {
  return terms.map((t, i) => {
    const size = `${texNum(Math.abs(t.weight))}${t.tex ? `\\,${t.tex}` : ''}`;
    const negative = Number(texNum(t.weight)) < 0;
    if (i === 0) return (negative ? '-' : '') + size;
    return `${negative ? '-' : '+'} ${size}`;
  });
}

// The formula with the model's current weights in. Short sums go on one
// line; long ones (higher degrees) are broken over several.
function fittedTex(terms) {
  const parts = sumTex(terms);
  if (parts.length <= 3) return `p = \\sigma\\left(${parts.join(' ')}\\right)`;
  const lines = [];
  for (let i = 0; i < parts.length; i += 3) lines.push(parts.slice(i, i + 3).join(' '));
  return `\\begin{aligned} z ={} & ${lines.join(' \\\\ & ')} \\\\ p ={} & \\sigma(z) \\end{aligned}`;
}

// --- the two small charts (SVG, coloured by the theme's CSS variables) ---

function lossChart(history) {
  const W = 320, H = 84, L = 30, R = 312, T = 8, B = 66;
  if (history.length < 2) {
    return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Loss curve: no training yet"><line class="axis" x1="${L}" y1="${B}" x2="${R}" y2="${B}"/><text class="tick" x="${L}" y="${H - 4}">Press Train to draw the loss curve</text></svg>`;
  }
  const top = Math.max(0.75, Math.min(3, Math.max(...history)));
  const last = history.length - 1;
  const stride = Math.max(1, Math.floor(history.length / 300));
  const points = [];
  for (let i = 0; i <= last; i += stride) points.push(i);
  if (points[points.length - 1] !== last) points.push(last);
  const x = (i) => L + (i / Math.max(last, 20)) * (R - L);
  const y = (v) => B - (Math.min(v, top) / top) * (B - T);
  const path = points.map((i) => `${x(i).toFixed(1)},${y(history[i]).toFixed(1)}`).join(' ');
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Loss curve: the loss is ${history[last].toFixed(2)} after ${last} iterations">
    <line class="axis" x1="${L}" y1="${B}" x2="${R}" y2="${B}"/><line class="axis" x1="${L}" y1="${T}" x2="${L}" y2="${B}"/>
    <text class="tick" x="${L - 4}" y="${T + 4}" text-anchor="end">${top.toFixed(1)}</text>
    <text class="tick" x="${L - 4}" y="${B + 3}" text-anchor="end">0</text>
    <text class="tick" x="${L}" y="${H - 4}">iteration 0</text>
    <text class="tick" x="${R}" y="${H - 4}" text-anchor="end">${Math.max(last, 20)}</text>
    <polyline class="curve" points="${path}"/>
    <circle class="curve-end" cx="${x(last).toFixed(1)}" cy="${y(history[last]).toFixed(1)}" r="3"/>
  </svg>`;
}

function sigmoidChart(scores, threshold, example) {
  const W = 320, H = 176, L = 30, R = 312, T = 14, B = 146, Z = 8;
  const x = (z) => L + ((Math.max(-Z, Math.min(Z, z)) + Z) / (2 * Z)) * (R - L);
  const y = (p) => B - p * (B - T);
  const sig = (z) => 1 / (1 + Math.exp(-z));
  const curve = [];
  for (let z = -Z; z <= Z + 1e-9; z += 0.25) curve.push(`${x(z).toFixed(1)},${y(sig(z)).toFixed(1)}`);
  const marks = scores.map(({ z, y: label }) => {
    const px = x(z);
    const py = y(sig(z));
    if (label === 1) {
      const cy = py - 6;
      return `<path class="pt-blue" d="M${px.toFixed(1)} ${(cy - 3.4).toFixed(1)}l3 5.2h-6z"/>`;
    }
    return `<circle class="pt-red" cx="${px.toFixed(1)}" cy="${(py + 6).toFixed(1)}" r="2.5"/>`;
  }).join('');
  const query = example
    ? `<line class="query-line" x1="${x(example.z).toFixed(1)}" y1="${B}" x2="${x(example.z).toFixed(1)}" y2="${y(example.p).toFixed(1)}"/>
       <circle class="query-ring" cx="${x(example.z).toFixed(1)}" cy="${y(example.p).toFixed(1)}" r="5.5"/>`
    : '';
  const ticks = [-8, -4, 0, 4, 8].map((z) =>
    `<text class="tick" x="${x(z)}" y="${B + 12}" text-anchor="middle">${z < 0 ? `−${-z}` : z}</text>`).join('');
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="The sigmoid curve, with each training point placed on it by its score z">
    <line class="axis" x1="${L}" y1="${B}" x2="${R}" y2="${B}"/><line class="axis" x1="${L}" y1="${T}" x2="${L}" y2="${B}"/>
    <text class="tick" x="${L - 4}" y="${T + 4}" text-anchor="end">1</text>
    <text class="tick" x="${L - 4}" y="${y(0.5) + 3}" text-anchor="end">0.5</text>
    <text class="tick" x="${L - 4}" y="${B + 3}" text-anchor="end">0</text>
    ${ticks}
    <text class="tick" x="${R}" y="${H - 4}" text-anchor="end">score z</text>
    <text class="tick" x="${L + 4}" y="${T - 3}">probability of blue</text>
    <line class="threshold-line" x1="${L}" y1="${y(threshold).toFixed(1)}" x2="${R}" y2="${y(threshold).toFixed(1)}"/>
    <text class="tick" x="${R}" y="${(y(threshold) - 4).toFixed(1)}" text-anchor="end">threshold ${threshold.toFixed(2)}</text>
    <polyline class="curve" points="${curve.join(' ')}"/>
    ${marks}${query}
  </svg>`;
}

export function createLogisticPanel({ intro, detail }) {
  intro.innerHTML = `
    <div class="card">
      <p class="crumb">The model</p>
      <h3 class="card-title">${note.title}</h3>
      <p class="blurb">${note.idea}</p>
      <div class="formula" data-role="formula"></div>
      <div class="formula" data-role="sigmoid"></div>
      <p class="blurb">${note.degreeNote}</p>
      <p class="blurb">${note.training}</p>
      <div class="formula" data-role="loss"></div>
      <p class="small-print">${note.lossKey}</p>
      <p class="crumb sub">With the current numbers in</p>
      <div class="formula fitted" data-role="fitted"></div>
      <p class="small-print" data-role="fittedNote"></p>
    </div>`;

  detail.innerHTML = `
    <div class="card">
      <p class="crumb">Inside the model</p>
      <h3 class="card-title">Weights and loss</h3>
      <div class="facts fitted">
        <div><span class="fact-label">Iteration</span><span class="num" data-role="iteration"></span></div>
        <div><span class="fact-label">Loss (log-loss)</span><span class="num" data-role="lossValue"></span></div>
      </div>
      <div class="chart fitted" data-role="lossChart"></div>
      <p class="small-print">${note.lossCurve}</p>
      <div class="table-scroll fitted" tabindex="0" role="group" aria-label="Table of weights">
        <table class="data-table">
          <thead><tr><th scope="col">Feature term</th><th scope="col" class="num">Weight</th></tr></thead>
          <tbody data-role="weights"></tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <p class="crumb">Worked example</p>
      <h3 class="card-title">The calculation for your query point</h3>
      <p class="blurb" data-role="exampleEmpty">${COPY.queryEmpty}</p>
      <ol class="steps fitted" data-role="example" hidden>
        <li><strong>Start with the inputs.</strong> <span class="num-line">x<sub>1</sub> = <span class="num" data-role="x1"></span>, x<sub>2</sub> = <span class="num" data-role="x2"></span></span></li>
        <li><strong>Multiply each term by its weight.</strong>
          <div class="table-scroll" tabindex="0" role="group" aria-label="Each term multiplied by its weight">
            <table class="data-table">
              <thead><tr><th scope="col">Term</th><th scope="col" class="num">Value</th><th scope="col" class="num">&times; weight</th><th scope="col" class="num">= amount</th></tr></thead>
              <tbody data-role="exampleRows"></tbody>
            </table>
          </div>
        </li>
        <li><strong>Add the amounts up</strong> to get the score. <span class="num-line">z = <span class="num" data-role="z"></span></span></li>
        <li><strong>Squash the score</strong> into a probability with the sigmoid. <span class="num-line">p = 1 / (1 + e<sup>&minus;z</sup>) = <span class="num" data-role="p"></span></span></li>
        <li><strong>Compare with the threshold.</strong> <span class="num-line" data-role="verdict"></span></li>
      </ol>
    </div>

    <div class="card">
      <p class="crumb">See it</p>
      <h3 class="card-title">From score to probability</h3>
      <div class="chart fitted" data-role="sigmoidChart"></div>
      <p class="small-print">${note.sigmoid}</p>
    </div>`;

  const el = {};
  for (const node of [...intro.querySelectorAll('[data-role]'), ...detail.querySelectorAll('[data-role]')]) {
    el[node.dataset.role] = node;
  }

  // The symbolic formulas never change, so render them once.
  renderTex(el.formula, 'p = \\sigma\\left(w_0 + w_1 x_1 + w_2 x_2\\right)', { display: true });
  renderTex(el.sigmoid, '\\sigma(z) = \\frac{1}{1 + e^{-z}}', { display: true });
  renderTex(el.loss, 'L = -\\frac{1}{n}\\sum_{i=1}^{n}\\Big[\\,y_i \\ln p_i + (1 - y_i)\\ln(1 - p_i)\\Big]', { display: true });

  // Make a table body have exactly `count` rows of `cells` cells each.
  function sizeRows(body, count, cells) {
    while (body.rows.length > count) body.deleteRow(-1);
    while (body.rows.length < count) {
      const row = body.insertRow();
      for (let c = 0; c < cells; c++) {
        const cell = row.insertCell();
        if (c > 0) cell.className = 'num';
      }
    }
  }

  const termLabel = (cell, tex) => {
    if (tex) renderTex(cell, tex);
    else if (cell.dataset.tex !== 'bias') { cell.dataset.tex = 'bias'; cell.textContent = 'bias (always 1)'; }
  };

  function update(info, { threshold, status, flash = true }) {
    // The formula with numbers in
    renderTex(el.fitted, fittedTex(info.terms), { display: true });
    el.fittedNote.textContent = info.trained ? note.fitted : note.fittedUntrained;

    // Internals
    setValue(el.iteration, info.trained ? String(info.iteration) : '0', flash);
    setValue(el.lossValue, info.loss === null ? 'not trained' : fmt(info.loss, 3), flash);
    el.lossChart.innerHTML = lossChart(info.lossHistory);
    sizeRows(el.weights, info.terms.length, 2);
    info.terms.forEach((t, i) => {
      const row = el.weights.rows[i];
      termLabel(row.cells[0], t.tex);
      setValue(row.cells[1], fmt(t.weight), flash);
    });

    // Worked example
    const ex = info.example;
    el.example.hidden = !ex;
    el.exampleEmpty.hidden = !!ex;
    if (ex) {
      setValue(el.x1, fmt(ex.x[0]), false);
      setValue(el.x2, fmt(ex.x[1]), false);
      sizeRows(el.exampleRows, ex.rows.length, 4);
      ex.rows.forEach((r, i) => {
        const row = el.exampleRows.rows[i];
        termLabel(row.cells[0], r.tex);
        setValue(row.cells[1], fmt(r.value), false);
        setValue(row.cells[2], fmt(r.weight), flash);
        setValue(row.cells[3], fmt(r.product), flash);
      });
      setValue(el.z, fmt(ex.z), flash);
      setValue(el.p, fmt(ex.p), flash);
      const blue = ex.p >= threshold;
      const verdict = `${fmt(ex.p)} is ${blue ? 'at least' : 'below'} ${fmt(threshold)}, so the model predicts `
        + `<strong>${blue ? `${MARK.blue} blue` : `${MARK.red} red`}</strong>.`;
      if (el.verdict.dataset.html !== verdict) {
        el.verdict.dataset.html = verdict;
        el.verdict.innerHTML = verdict;
      }
    }

    // Sigmoid chart
    el.sigmoidChart.innerHTML = sigmoidChart(info.trainingScores, threshold, ex);
  }

  return { update };
}
