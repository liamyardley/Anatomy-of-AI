// The maths panel for k-nearest neighbours: the idea, the "memory" that is
// the whole model, a worked example for the query point, and a picture of
// which stored points get a vote.
//
// See panels/logistic.js for how a panel is put together.

import { MODEL_NOTES, COPY } from '../../content/notes.js';
import { fmt, texNum, setValue, renderTex } from '../format.js';
import { roles, sizeRows, setHtml, classLabel, verdict } from './shared.js';

const note = MODEL_NOTES.knn;

// The distance formula for each measure, in symbols.
function distanceTex(metric, power) {
  const dx = '|x_1 - a_1|';
  const dy = '|x_2 - a_2|';
  if (metric === 'manhattan') return `d = ${dx} + ${dy}`;
  if (metric === 'chebyshev') return `d = \\max\\big(${dx},\\; ${dy}\\big)`;
  if (metric === 'minkowski') return `d = \\Big(${dx}^{p} + ${dy}^{p}\\Big)^{1/p}, \\quad p = ${power}`;
  if (metric === 'cosine') return 'd = 1 - \\cos\\theta = 1 - \\frac{x_1 a_1 + x_2 a_2}{\\sqrt{x_1^2 + x_2^2}\\,\\sqrt{a_1^2 + a_2^2}}';
  return 'd = \\sqrt{(x_1 - a_1)^2 + (x_2 - a_2)^2}';
}

// The same formula with one query point and one neighbour's numbers in.
function workedTex(metric, power, x, a, distance) {
  const n = texNum;
  const gap = (i) => `|${n(x[i])} - ${n(a[i])}|`;
  // Cosine distances between near neighbours are tiny, so they get more decimal places.
  const end = ` = ${metric === 'cosine' ? distance.toFixed(4) : n(distance)}`;
  if (metric === 'manhattan') return `d = ${gap(0)} + ${gap(1)}${end}`;
  if (metric === 'chebyshev') return `d = \\max\\big(${gap(0)},\\; ${gap(1)}\\big)${end}`;
  if (metric === 'minkowski') return `d = \\Big(${gap(0)}^{${power}} + ${gap(1)}^{${power}}\\Big)^{1/${power}}${end}`;
  if (metric === 'cosine') {
    const dot = x[0] * a[0] + x[1] * a[1];
    const sizes = Math.hypot(x[0], x[1]) * Math.hypot(a[0], a[1]);
    return `d = 1 - \\frac{${n(dot)}}{${n(sizes)}}${end}`;
  }
  return `d = \\sqrt{(${n(x[0])} - ${n(a[0])})^2 + (${n(x[1])} - ${n(a[1])})^2}${end}`;
}

// The stored points nearest the query, placed along a distance axis. Blue
// sits on the upper row and red on the lower one. The shaded band covers the
// k nearest: the ones that vote.
function voteChart(example) {
  const W = 320, H = 130, L = 12, R = 308, B = 100;
  if (!example) {
    return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="No query point yet"><line class="axis" x1="${L}" y1="${B}" x2="${R}" y2="${B}"/><text class="tick" x="${L}" y="${H - 8}">Drop a query point on the plot to see who votes</text></svg>`;
  }
  const far = example.nearby[example.nearby.length - 1].distance;
  const reach = Math.max(far, example.radius * 1.15, 1e-6);
  const x = (d) => L + (d / reach) * (R - L);
  const k = example.neighbours.length;
  const marks = example.nearby.map((n, i) => {
    const px = x(n.distance).toFixed(1);
    const faded = i >= k ? ' faded' : '';
    return n.y === 1
      ? `<path class="pt-blue${faded}" d="M${px} 38l5 8.6h-10z"/>`
      : `<circle class="pt-red${faded}" cx="${px}" cy="72" r="4"/>`;
  }).join('');
  const edge = x(example.radius);
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="The ${k} nearest stored points vote: ${example.blueVotes} blue and ${example.redVotes} red">
    <rect class="band" x="${L}" y="18" width="${(edge - L + 5).toFixed(1)}" height="${B - 18}"/>
    <line class="axis" x1="${L}" y1="${B}" x2="${R}" y2="${B}"/>
    <line class="threshold-line" x1="${(edge + 5).toFixed(1)}" y1="14" x2="${(edge + 5).toFixed(1)}" y2="${B}"/>
    <text class="tick" x="${L + 3}" y="13">the ${k} nearest vote</text>
    ${marks}
    <text class="tick" x="${L}" y="${B + 12}">0</text>
    <text class="tick" x="${R}" y="${B + 12}" text-anchor="end">${reach < 0.1 ? reach.toPrecision(2) : reach.toFixed(2)}</text>
    <text class="tick" x="${(L + R) / 2}" y="${H - 6}" text-anchor="middle">distance from the query point</text>
  </svg>`;
}

export function createKnnPanel({ intro, detail, wide }) {
  intro.innerHTML = `
      <h3 class="card-title">${note.title}</h3>
      <p class="blurb">${note.idea}</p>
      <div class="formula" data-role="distance"></div>
      <p class="small-print" data-role="metricNote"></p>
      <p class="small-print">${note.distanceKey}</p>
      <div class="formula" data-role="share"></div>
      <p class="blurb">${note.vote}</p>
      <p class="blurb">${note.training}</p>`;

  detail.innerHTML = `
    <div class="card">
      <p class="crumb">Inside the model</p>
      <h3 class="card-title">The memory</h3>
      <div class="facts fitted">
        <div><span class="fact-label">Points stored</span><span class="num" data-role="stored"></span></div>
        <div><span class="fact-label">Neighbours asked (k)</span><span class="num" data-role="k"></span></div>
      </div>
      <p class="small-print" data-role="memoryNote"></p>
      <div class="table-scroll fitted" tabindex="0" role="group" aria-label="The stored training points" data-role="memoryScroll">
        <table class="data-table">
          <thead><tr><th scope="col">Stored point</th><th scope="col" class="num">x<sub>1</sub></th><th scope="col" class="num">x<sub>2</sub></th><th scope="col" class="num">Class</th></tr></thead>
          <tbody data-role="memory"></tbody>
        </table>
      </div>
      <p class="small-print">${note.filling}</p>
    </div>`;

  wide.innerHTML = `
    <div class="split">
      <div>
      <p class="crumb">Worked example</p>
      <h3 class="card-title">The calculation for your query point</h3>
      <p class="blurb" data-role="exampleEmpty"></p>
      <ol class="steps fitted" data-role="example" hidden>
        <li><strong>Start with the query point.</strong> <span class="num-line">x<sub>1</sub> = <span class="num" data-role="x1"></span>, x<sub>2</sub> = <span class="num" data-role="x2"></span></span></li>
        <li><strong>Measure the distance to every stored point.</strong> For the nearest one:
          <div class="formula" data-role="nearestTex"></div>
        </li>
        <li><strong>Keep the <span data-role="kWord"></span> closest.</strong>
          <div class="table-scroll" tabindex="0" role="group" aria-label="The nearest neighbours">
            <table class="data-table">
              <thead><tr><th scope="col">Neighbour</th><th scope="col" class="num">Class</th><th scope="col" class="num">Distance</th><th scope="col" class="num" data-role="weightHead">Vote weight</th></tr></thead>
              <tbody data-role="neighbours"></tbody>
            </table>
          </div>
        </li>
        <li><strong>Count the votes.</strong> <span class="num-line" data-role="tally"></span></li>
        <li><strong>Compare with the threshold.</strong> <span class="num-line" data-role="verdict"></span></li>
      </ol>
      </div>
      <div>
      <p class="crumb">See it</p>
      <h3 class="card-title">Who gets a vote?</h3>
      <div class="chart fitted" data-role="voteChart"></div>
      <p class="small-print">${note.picture}</p>
      </div>
    </div>`;

  const el = roles(intro, detail, wide);
  renderTex(el.share, 'p = \\frac{\\text{number of blue neighbours}}{k}', { display: true });

  let shownMemory = null; // the memory array and length last drawn in the table

  function update(info, { threshold, flash = true }) {
    // The distance measure in use
    renderTex(el.distance, distanceTex(info.metric, info.power), { display: true });
    setHtml(el.metricNote, note.metrics[info.metric]);

    // The memory
    setValue(el.stored, info.trained ? `${info.stored} of ${info.total}` : '0', flash);
    setValue(el.k, String(info.k), false);
    el.memoryNote.textContent = info.trained ? note.memory : note.memoryUntrained;
    if (!shownMemory || shownMemory.list !== info.memory || shownMemory.length !== info.memory.length) {
      const grew = shownMemory && shownMemory.list === info.memory;
      shownMemory = { list: info.memory, length: info.memory.length };
      el.memory.innerHTML = info.memory.map((item, i) => `<tr><td>${i + 1}</td>`
        + `<td class="num">${fmt(item.x[0])}</td><td class="num">${fmt(item.x[1])}</td>`
        + `<td class="num">${classLabel(item.y)}</td></tr>`).join('');
      // While points are being copied in, keep the newest in view.
      if (grew && !info.finished) el.memoryScroll.scrollTop = el.memoryScroll.scrollHeight;
      else el.memoryScroll.scrollTop = 0;
    }

    // Worked example
    const ex = info.example;
    el.example.hidden = !ex;
    el.exampleEmpty.hidden = !!ex;
    el.exampleEmpty.textContent = info.trained ? COPY.queryEmpty : note.memoryUntrained;
    if (ex) {
      setValue(el.x1, fmt(ex.x[0]), false);
      setValue(el.x2, fmt(ex.x[1]), false);
      const first = ex.neighbours[0];
      renderTex(el.nearestTex, workedTex(info.metric, info.power, ex.x, first.x, first.distance), { display: true });
      const count = ex.neighbours.length;
      el.kWord.textContent = count === 1 ? 'single' : String(count);
      el.weightHead.hidden = !info.weighted;
      sizeRows(el.neighbours, count, 4);
      ex.neighbours.forEach((n, i) => {
        const row = el.neighbours.rows[i];
        setHtml(row.cells[0], String(i + 1));
        setHtml(row.cells[1], classLabel(n.y));
        setValue(row.cells[2], fmt(n.distance, info.metric === 'cosine' ? 4 : 2), false);
        row.cells[3].hidden = !info.weighted;
        setValue(row.cells[3], fmt(n.weight), false);
      });
      const votes = `${ex.blueVotes} blue and ${ex.redVotes} red`;
      setHtml(el.tally, info.weighted
        ? `${votes}. Counting each vote by its weight, blue has ${fmt(ex.blueWeight)} out of ${fmt(ex.totalWeight)}, so p = <span class="num">${fmt(ex.p)}</span>`
        : `${votes}, so p = ${ex.blueVotes} / ${count} = <span class="num">${fmt(ex.p)}</span>`);
      setHtml(el.verdict, verdict(ex.p, threshold));
    }

    // Who gets a vote
    el.voteChart.innerHTML = voteChart(ex);
  }

  return { update };
}
