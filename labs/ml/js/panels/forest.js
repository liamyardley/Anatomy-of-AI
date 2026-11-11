// The maths panel for the random forest: the idea, thumbnails of the first
// few trees, a chart of accuracy against the number of trees, a worked
// example listing every tree's answer, and a picture of how much the trees
// disagree.
//
// See panels/logistic.js for how a panel is put together.

import { MODEL_NOTES, COPY } from '../../content/notes.js';
import { fmt, pct, setValue, renderTex } from '../format.js';
import { roles, sizeRows, setHtml, classLabel, verdict } from './shared.js';

const note = MODEL_NOTES.forest;

const THUMBNAILS = 9;
const THUMB_GRID = 48; // each thumbnail is drawn on a 48 x 48 grid

// --- thumbnails: one small picture of each tree's own predictions ---

function classColours() {
  const css = getComputedStyle(document.documentElement);
  const rgb = (name) => {
    const hex = css.getPropertyValue(name).trim();
    return [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  };
  return { red: rgb('--class-red'), blue: rgb('--class-blue') };
}

function paintThumbnail(canvas, tree) {
  const ctx = canvas.getContext('2d');
  const image = ctx.createImageData(THUMB_GRID, THUMB_GRID);
  const colours = classColours();
  for (let j = 0; j < THUMB_GRID; j++) {
    const x2 = 1 - (2 * (j + 0.5)) / THUMB_GRID;
    for (let i = 0; i < THUMB_GRID; i++) {
      const p = tree.predictProba([-1 + (2 * (i + 0.5)) / THUMB_GRID, x2]);
      const colour = p >= 0.5 ? colours.blue : colours.red;
      const at = (j * THUMB_GRID + i) * 4;
      image.data[at] = colour[0];
      image.data[at + 1] = colour[1];
      image.data[at + 2] = colour[2];
      image.data[at + 3] = Math.round(255 * 0.75 * Math.abs(2 * p - 1));
    }
  }
  ctx.putImageData(image, 0, 0);
}

// --- accuracy against the number of trees ---

function accuracyChart(curve, target) {
  const W = 320, H = 120, L = 34, R = 312, T = 10, B = 92;
  const frame = `<line class="axis" x1="${L}" y1="${B}" x2="${R}" y2="${B}"/><line class="axis" x1="${L}" y1="${T}" x2="${L}" y2="${B}"/>`;
  if (curve.length === 0) {
    return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Accuracy chart: nothing to show yet">${frame}<text class="tick" x="${L + 6}" y="${H - 6}">Press Train to draw test accuracy against the number of trees</text></svg>`;
  }
  const low = Math.min(0.5, Math.floor(Math.min(...curve) * 10) / 10);
  const span = Math.max(target, 2) - 1;
  const x = (t) => L + (span === 0 ? 0 : (t / span)) * (R - L);
  const y = (a) => B - ((a - low) / (1 - low)) * (B - T);
  const points = curve.map((a, t) => `${x(t).toFixed(1)},${y(a).toFixed(1)}`).join(' ');
  const last = curve.length - 1;
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Test accuracy is ${Math.round(curve[last] * 100)}% with ${curve.length} trees">
    ${frame}
    <text class="tick" x="${L - 4}" y="${T + 4}" text-anchor="end">100%</text>
    <text class="tick" x="${L - 4}" y="${B + 3}" text-anchor="end">${Math.round(low * 100)}%</text>
    <text class="tick" x="${L}" y="${B + 12}">1 tree</text>
    <text class="tick" x="${R}" y="${B + 12}" text-anchor="end">${Math.max(target, 2)} trees</text>
    <text class="tick" x="${(L + R) / 2}" y="${H - 4}" text-anchor="middle">test accuracy as trees are added</text>
    <polyline class="curve" points="${points}"/>
    <circle class="curve-end" cx="${x(last).toFixed(1)}" cy="${y(curve[last]).toFixed(1)}" r="3"/>
    <text class="tick strong" x="${Math.min(x(last) + 6, R - 22).toFixed(1)}" y="${(y(curve[last]) - 6).toFixed(1)}">${Math.round(curve[last] * 100)}%</text>
  </svg>`;
}

// --- how much the trees disagree about the query point ---

function spreadChart(example, threshold) {
  const W = 320, H = 150, L = 14, R = 306, T = 20, B = 116;
  const x = (p) => L + p * (R - L);
  const axis = `<line class="axis" x1="${L}" y1="${B}" x2="${R}" y2="${B}"/>
    <text class="tick" x="${L}" y="${B + 12}">0 (red)</text>
    <text class="tick" x="${x(0.5)}" y="${B + 12}" text-anchor="middle">0.5</text>
    <text class="tick" x="${R}" y="${B + 12}" text-anchor="end">1 (blue)</text>
    <text class="tick" x="${(L + R) / 2}" y="${H - 4}" text-anchor="middle">each tree's probability of blue</text>`;
  if (!example) {
    return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="No query point yet">${axis}<text class="tick" x="${L}" y="${T}">Drop a query point on the plot to see every tree's answer</text></svg>`;
  }
  // Stack the trees into 21 columns by their answer.
  const columns = Array.from({ length: 21 }, () => 0);
  for (const vote of example.votes) columns[Math.round(vote.p * 20)] += 1;
  const tallest = Math.max(...columns);
  const gap = Math.min(7, (B - T - 4) / tallest);
  const radius = Math.max(1, Math.min(2.8, gap / 2));
  let dots = '';
  columns.forEach((count, c) => {
    const px = x(c / 20).toFixed(1);
    for (let i = 0; i < count; i++) {
      dots += `<circle class="${c >= 10 ? 'pt-blue' : 'pt-red'}" cx="${px}" cy="${(B - 4 - i * gap).toFixed(1)}" r="${radius}"/>`;
    }
  });
  const mean = x(example.p).toFixed(1);
  const cut = x(threshold).toFixed(1);
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${example.blueVotes} trees lean blue and ${example.redVotes} lean red; their average is ${example.p.toFixed(2)}">
    ${axis}${dots}
    <line class="threshold-line" x1="${cut}" y1="${T}" x2="${cut}" y2="${B}"/>
    <line class="mean-line" x1="${mean}" y1="${T}" x2="${mean}" y2="${B}"/>
    <text class="tick strong" x="${mean}" y="${T - 6}" text-anchor="middle">average ${example.p.toFixed(2)}</text>
  </svg>`;
}

export function createForestPanel({ intro, detail, wide }) {
  intro.innerHTML = `
      <h3 class="card-title">${note.title}</h3>
      <p class="blurb">${note.idea}</p>
      <p class="blurb">${note.different}</p>
      <div class="formula" data-role="average"></div>
      <p class="small-print">${note.formulaKey}</p>`;

  const slots = Array.from({ length: THUMBNAILS }, (_, i) => `
    <figure class="thumb" data-empty="true">
      <div class="thumb-frame"><canvas width="${THUMB_GRID}" height="${THUMB_GRID}" aria-hidden="true"></canvas><span class="thumb-dot" hidden></span></div>
      <figcaption>Tree ${i + 1}</figcaption>
    </figure>`).join('');

  detail.innerHTML = `
    <div class="card">
      <p class="crumb">Inside the model</p>
      <h3 class="card-title">The trees</h3>
      <div class="facts fitted">
        <div><span class="fact-label">Trees grown</span><span class="num" data-role="grown"></span></div>
        <div><span class="fact-label">Out-of-bag accuracy</span><span class="num" data-role="oob"></span></div>
      </div>
      <p class="small-print" data-role="oobNote"></p>
      <p class="crumb sub">The first nine trees, one by one</p>
      <div class="thumbs fitted" data-role="thumbs" role="img" aria-label="Small plots of the first nine trees' own predictions">${slots}</div>
      <p class="small-print" data-role="thumbNote"></p>
      <p class="crumb sub">Do more trees help?</p>
      <div class="chart fitted" data-role="accuracyChart"></div>
      <p class="small-print" data-role="curveNote"></p>
    </div>`;

  wide.innerHTML = `
    <div class="split">
      <div>
      <p class="crumb">Worked example</p>
      <h3 class="card-title">Every tree's answer for your query point</h3>
      <p class="blurb" data-role="exampleEmpty"></p>
      <ol class="steps fitted" data-role="example" hidden>
        <li><strong>Start with the query point.</strong> <span class="num-line">x<sub>1</sub> = <span class="num" data-role="x1"></span>, x<sub>2</sub> = <span class="num" data-role="x2"></span></span></li>
        <li><strong>Ask every tree.</strong> Each one follows its own questions down to a leaf and reports that leaf's share of blue points.
          <div class="table-scroll" tabindex="0" role="group" aria-label="Each tree's answer">
            <table class="data-table">
              <thead><tr><th scope="col">Tree</th><th scope="col" class="num">Probability of blue</th><th scope="col" class="num">Leans</th></tr></thead>
              <tbody data-role="votes"></tbody>
            </table>
          </div>
        </li>
        <li><strong>Average the answers.</strong> <span class="num-line" data-role="tally"></span></li>
        <li><strong>Compare with the threshold.</strong> <span class="num-line" data-role="verdict"></span></li>
      </ol>
      </div>
      <div>
      <p class="crumb">See it</p>
      <h3 class="card-title">How much the trees disagree</h3>
      <div class="chart fitted" data-role="spreadChart"></div>
      <p class="small-print">${note.picture}</p>
      </div>
    </div>`;

  const el = roles(intro, detail, wide);
  renderTex(el.average, 'p = \\frac{1}{T}\\sum_{t=1}^{T} p_t', { display: true });

  const figures = [...el.thumbs.querySelectorAll('.thumb')];
  let painted = [];      // which tree each thumbnail currently shows
  let curveKey = null;   // what the accuracy curve was last worked out from
  let curve = [];
  let lastCall = null;

  function update(info, context) {
    const { threshold, flash = true, data } = context;
    lastCall = [info, context];

    setValue(el.grown, `${info.grown} of ${info.target}`, flash);
    setValue(el.oob, info.trained ? pct(info.oobAccuracy) : 'not trained', flash);
    setHtml(el.oobNote, info.bootstrap ? note.oob : note.oobOff);

    // Thumbnails: paint each tree once, when it first appears.
    figures.forEach((figure, i) => {
      const tree = info.firstTrees[i] || null;
      if (painted[i] !== tree) {
        painted[i] = tree;
        figure.dataset.empty = String(!tree);
        const canvas = figure.querySelector('canvas');
        if (tree) paintThumbnail(canvas, tree);
        else canvas.getContext('2d').clearRect(0, 0, THUMB_GRID, THUMB_GRID);
      }
      const dot = figure.querySelector('.thumb-dot');
      const ex = info.example;
      dot.hidden = !tree || !ex;
      if (tree && ex) {
        dot.style.left = `${((ex.x[0] + 1) / 2) * 100}%`;
        dot.style.top = `${((1 - ex.x[1]) / 2) * 100}%`;
      }
    });
    el.thumbNote.textContent = info.trained ? note.thumbnails : note.thumbnailsUntrained;

    // Accuracy against the number of trees: reworked only when the forest
    // or the test points change, not on every mouse move.
    const first = info.firstTrees[0] || null;
    if (!curveKey || curveKey.grown !== info.grown || curveKey.first !== first || curveKey.test !== data.test) {
      curveKey = { grown: info.grown, first, test: data.test };
      curve = info.accuracyCurve(data.test.X, data.test.y);
      el.accuracyChart.innerHTML = accuracyChart(curve, info.target);
    }
    el.curveNote.textContent = data.test.X.length ? note.curve : note.curveNoTest;

    // Worked example
    const ex = info.example;
    el.example.hidden = !ex;
    el.exampleEmpty.hidden = !!ex;
    el.exampleEmpty.textContent = info.trained ? COPY.queryEmpty : note.thumbnailsUntrained;
    if (ex) {
      setValue(el.x1, fmt(ex.x[0]), false);
      setValue(el.x2, fmt(ex.x[1]), false);
      sizeRows(el.votes, ex.votes.length, 3);
      ex.votes.forEach((vote, i) => {
        const row = el.votes.rows[i];
        setValue(row.cells[0], String(vote.tree), false);
        setValue(row.cells[1], fmt(vote.p), false);
        setHtml(row.cells[2], classLabel(vote.blue ? 1 : 0));
      });
      const trees = ex.votes.length;
      setHtml(el.tally, `${ex.blueVotes} of the ${trees} ${trees === 1 ? 'tree leans' : 'trees lean'} blue and ${ex.redVotes} red. `
        + `Adding up all ${trees === 1 ? 'the' : trees} probabilities and dividing by ${trees} gives p = <span class="num">${fmt(ex.p)}</span>`);
      setHtml(el.verdict, verdict(ex.p, threshold));
    }

    el.spreadChart.innerHTML = spreadChart(ex, threshold);
  }

  // The thumbnails are painted in the class colours, so repaint them when
  // the colour scheme changes.
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    requestAnimationFrame(() => {
      painted = [];
      if (lastCall && el.thumbs.isConnected) update(...lastCall);
    });
  });

  return { update };
}
