// The maths panel for the decision tree: the idea, a tree diagram that grows
// as training runs and is linked to the plot, the candidate splits for the
// selected group, a worked example for the query point, and a picture of
// Gini impurity.
//
// See panels/logistic.js for how a panel is put together. This one also uses
// setOverlay() to light up a group's rectangle on the plot.

import { MODEL_NOTES, COPY } from '../../content/notes.js';
import { fmt, setValue, renderTex } from '../format.js';
import {
  roles, setHtml, classLabel, verdict, settingLink, followSettingLinks,
} from './shared.js';

const note = MODEL_NOTES.tree;

const FEATURE_HTML = ['x<sub>1</sub>', 'x<sub>2</sub>'];
const FEATURE_TEXT = ['x₁', 'x₂'];
const LEAF_REASONS = {
  depth: 'It is at the maximum depth, so the tree did not look for another question.',
  pure: 'Every point in it is the same colour, so there is nothing left to split.',
  small: 'It holds too few points to be split under the current minimum settings.',
  nogain: 'No question made its two halves any purer, so it stays a leaf.',
  leaves: 'The tree reached its maximum number of leaves before this group\'s turn came.',
  pruned: 'It was split while the tree grew, but pruning removed the split: it did not remove enough impurity to be worth its extra leaves.',
};
const WAITING = 'It can be split, and is waiting its turn.';

// How impurity is measured, in words and symbols.
const CRITERIA = {
  gini: { name: 'Gini impurity', top: 0.5, value: (p) => 2 * p * (1 - p), tex: 'G = 1 - p_{\\text{red}}^2 - p_{\\text{blue}}^2' },
  entropy: {
    name: 'entropy',
    top: 1,
    value: (p) => (p <= 0 || p >= 1 ? 0 : -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p))),
    tex: 'G = -\\,p_{\\text{red}}\\log_2 p_{\\text{red}} - p_{\\text{blue}}\\log_2 p_{\\text{blue}}',
  },
};

// --- the tree diagram ---

const SLOT = 70;   // width given to each leaf
const LEVEL = 66;  // height of each level
const BOX_W = 64;
const BOX_H = 38;

// Give every node a position: leaves are spread left to right, and each
// question sits halfway between its two answers.
function layout(root) {
  const at = new Map();
  let nextLeaf = 0;
  let deepest = 0;
  (function place(node) {
    deepest = Math.max(deepest, node.depth);
    if (!node.split) {
      at.set(node.id, { x: (nextLeaf++ + 0.5) * SLOT, y: node.depth * LEVEL + 6 });
      return;
    }
    place(node.left);
    place(node.right);
    at.set(node.id, {
      x: (at.get(node.left.id).x + at.get(node.right.id).x) / 2,
      y: node.depth * LEVEL + 6,
    });
  }(root));
  return { at, width: nextLeaf * SLOT, height: deepest * LEVEL + BOX_H + 12 };
}

function diagramSvg(nodes) {
  const { at, width, height } = layout(nodes[0]);
  const edges = [];
  const boxes = [];
  for (const node of nodes) {
    const { x, y } = at.get(node.id);
    if (node.split) {
      for (const [child, word] of [[node.left, 'yes'], [node.right, 'no']]) {
        const to = at.get(child.id);
        edges.push(`<line class="edge" x1="${x}" y1="${y + BOX_H}" x2="${to.x}" y2="${to.y}"/>`);
        if (node.depth === 0) {
          edges.push(`<text class="edge-label" x="${(x + to.x) / 2}" y="${(y + BOX_H + to.y) / 2 - 3}" text-anchor="middle">${word}</text>`);
        }
      }
    }
    const total = node.red + node.blue;
    const lean = total === 0 || node.red === node.blue ? 'even' : (node.blue > node.red ? 'blue' : 'red');
    const counts = `<tspan class="count-red">● ${node.red}</tspan>  <tspan class="count-blue">▲ ${node.blue}</tspan>`;
    const lines = node.split
      ? `<text class="question" x="${x}" y="${y + 15}" text-anchor="middle">${FEATURE_TEXT[node.split.feature]} &lt; ${fmt(node.split.threshold)}</text>
         <text class="counts" x="${x}" y="${y + 30}" text-anchor="middle" xml:space="preserve">${counts}</text>`
      : `<text class="counts" x="${x}" y="${y + 23}" text-anchor="middle" xml:space="preserve">${counts}</text>`;
    boxes.push(`<g class="node ${node.split ? 'question-node' : 'leaf'} lean-${lean}" data-id="${node.id}">
      <rect x="${x - BOX_W / 2}" y="${y}" width="${BOX_W}" height="${BOX_H}" rx="8"/>${lines}</g>`);
  }
  return {
    at,
    svg: `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" aria-hidden="true">${edges.join('')}${boxes.join('')}</svg>`,
  };
}

// --- impurity picture ---

function giniChart(node, criterion) {
  const measure = CRITERIA[criterion];
  const title = measure.name.charAt(0).toUpperCase() + measure.name.slice(1);
  const W = 320, H = 176, L = 30, R = 312, T = 14, B = 146;
  const x = (p) => L + p * (R - L);
  const y = (g) => B - (g / measure.top) * (B - T);
  const curve = [];
  for (let p = 0; p <= 1.0001; p += 0.025) curve.push(`${x(p).toFixed(1)},${y(measure.value(Math.min(p, 1))).toFixed(1)}`);
  const share = (n) => (n.red + n.blue === 0 ? 0.5 : n.blue / (n.red + n.blue));
  let marks = '';
  if (node) {
    const p = share(node);
    if (node.split) {
      for (const [child, word] of [[node.left, 'yes'], [node.right, 'no']]) {
        const cp = share(child);
        marks += `<line class="query-line" x1="${x(p).toFixed(1)}" y1="${y(node.impurity).toFixed(1)}" x2="${x(cp).toFixed(1)}" y2="${y(child.impurity).toFixed(1)}"/>
          <circle class="curve-end" cx="${x(cp).toFixed(1)}" cy="${y(child.impurity).toFixed(1)}" r="4"/>
          <text class="tick" x="${x(cp).toFixed(1)}" y="${(y(child.impurity) - 8).toFixed(1)}" text-anchor="middle">${word}</text>`;
      }
    }
    marks += `<circle class="query-ring" cx="${x(p).toFixed(1)}" cy="${y(node.impurity).toFixed(1)}" r="6"/>`;
  }
  const label = node
    ? `${title} of the selected group is ${node.impurity.toFixed(2)}`
    : `${title} for every mix of red and blue`;
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${label}">
    <line class="axis" x1="${L}" y1="${B}" x2="${R}" y2="${B}"/><line class="axis" x1="${L}" y1="${T}" x2="${L}" y2="${B}"/>
    <text class="tick" x="${L - 4}" y="${T + 4}" text-anchor="end">${measure.top}</text>
    <text class="tick" x="${L - 4}" y="${B + 3}" text-anchor="end">0</text>
    <text class="tick" x="${L + 4}" y="${T - 3}">${title}</text>
    <text class="tick" x="${L}" y="${B + 12}">all red</text>
    <text class="tick" x="${(L + R) / 2}" y="${B + 12}" text-anchor="middle">half and half</text>
    <text class="tick" x="${R}" y="${B + 12}" text-anchor="end">all blue</text>
    <text class="tick" x="${R}" y="${H - 4}" text-anchor="end">share of blue points in the group</text>
    <polyline class="curve" points="${curve.join(' ')}"/>
    ${marks}
  </svg>`;
}

export function createTreePanel({ intro, detail, wide, setOverlay }) {
  intro.innerHTML = `
      <h3 class="card-title">${note.title}</h3>
      <p class="blurb">${note.idea}</p>
      <p class="blurb" data-role="using"></p>
      <div class="formula" data-role="gini"></div>
      <p class="small-print" data-role="giniKey"></p>
      <div class="formula" data-role="gain"></div>
      <p class="small-print">${note.gainKey}</p>
      <p class="blurb">${note.training}</p>`;

  detail.innerHTML = `
    <div class="card">
      <p class="crumb">Inside the model</p>
      <h3 class="card-title">The tree</h3>
      <div class="facts fitted">
        <div><span class="fact-label">Splits</span><span class="num" data-role="splits"></span></div>
        <div><span class="fact-label">Leaves</span><span class="num" data-role="leaves"></span></div>
        <div><span class="fact-label">Depth</span><span class="num" data-role="depth"></span></div>
        <div><span class="fact-label">Branches pruned</span><span class="num" data-role="pruned"></span></div>
      </div>
      <div class="tree-diagram fitted" data-role="diagram" tabindex="0" role="group"
        aria-label="Tree diagram. Use the arrow keys to move between groups: up for the parent, left for the yes answer, right for the no answer."></div>
      <p class="small-print" data-role="diagramNote"></p>
      <p class="crumb sub">The selected group</p>
      <p class="blurb" data-role="selectedText" aria-live="polite"></p>
      <div class="table-scroll fitted" data-role="candidateWrap">
        <table class="data-table">
          <thead><tr><th scope="col">Feature</th><th scope="col">Best question</th><th scope="col" class="num">Impurity drop</th><th scope="col" class="num">Result</th></tr></thead>
          <tbody data-role="candidates"></tbody>
        </table>
      </div>
      <p class="small-print" data-role="candidateNote">${note.candidates}</p>
    </div>`;

  wide.innerHTML = `
    <div class="split">
      <div>
      <p class="crumb">Worked example</p>
      <h3 class="card-title">The questions asked of your query point</h3>
      <p class="blurb" data-role="exampleEmpty"></p>
      <ol class="steps fitted" data-role="example" hidden></ol>
      </div>
      <div>
      <p class="crumb">See it</p>
      <h3 class="card-title">What a good question does</h3>
      <div class="chart fitted" data-role="giniChart"></div>
      <p class="small-print">${note.picture}</p>
      </div>
    </div>`;

  const el = roles(intro, detail, wide);
  followSettingLinks(intro);
  renderTex(el.gain, '\\text{drop} = G - \\frac{n_{\\text{yes}}}{n}\\,G_{\\text{yes}} - \\frac{n_{\\text{no}}}{n}\\,G_{\\text{no}}', { display: true });

  let nodes = [];          // the tree last shown
  let drawn = null;        // { list, version } of the tree last drawn
  let selected = null;     // id of the selected group
  let lit = false;         // is the selected group's rectangle lit on the plot?
  let positions = new Map(); // where each box sits in the diagram

  const nodeById = (id) => nodes[id] || null;

  // Light up (or stop lighting) the selected group's rectangle on the plot.
  // This asks the page to redraw, which calls update() below again.
  function light(on) {
    lit = on;
    const node = nodeById(selected);
    setOverlay(on && node ? [{ kind: 'box', x1: node.region.x1, x2: node.region.x2, style: 'focus' }] : []);
  }

  function select(id) {
    if (nodeById(id) === null) return;
    selected = id;
    light(true);
  }

  // Keep the selected box in view inside the scrolling diagram (without
  // scrolling the page itself).
  function reveal() {
    const at = positions.get(selected);
    if (!at) return;
    const view = el.diagram;
    if (at.x - BOX_W < view.scrollLeft || at.x + BOX_W > view.scrollLeft + view.clientWidth) {
      view.scrollLeft = at.x - view.clientWidth / 2;
    }
    if (at.y < view.scrollTop || at.y + BOX_H + 8 > view.scrollTop + view.clientHeight) {
      view.scrollTop = at.y - view.clientHeight / 2;
    }
  }

  el.diagram.addEventListener('pointerover', (event) => {
    const box = event.target.closest('.node');
    if (box) select(Number(box.dataset.id));
  });
  el.diagram.addEventListener('click', (event) => {
    const box = event.target.closest('.node');
    if (box) select(Number(box.dataset.id));
  });
  el.diagram.addEventListener('pointerleave', () => { if (lit) light(false); });
  el.diagram.addEventListener('focus', () => { if (nodes.length) light(true); });
  el.diagram.addEventListener('blur', () => { if (lit) light(false); });
  el.diagram.addEventListener('keydown', (event) => {
    const node = nodeById(selected);
    if (!node) return;
    const target = {
      ArrowUp: node.parent,
      ArrowLeft: node.left,
      ArrowDown: node.left,
      ArrowRight: node.right,
      Home: nodes[0],
    }[event.key];
    if (target === undefined) return;
    event.preventDefault();
    if (target) { select(target.id); reveal(); }
  });

  function update(info, { threshold, flash = true }) {
    nodes = info.nodes;

    setValue(el.splits, String(info.splits), flash);
    setValue(el.leaves, info.trained ? String(info.leaves) : '0', flash);
    setValue(el.depth, String(info.depth), flash);
    setValue(el.pruned, String(info.pruned), flash);
    const measure = CRITERIA[info.criterion];
    const link = settingLink('criterion', `<strong>${note.criterionNames[info.criterion]}</strong>`);
    setHtml(el.using, info.criterion === 'gini' ? note.usingDefault(link) : note.usingChosen(link));
    renderTex(el.gini, measure.tex, { display: true });
    setHtml(el.giniKey, info.criterion === 'entropy' ? note.entropyKey : note.giniKey);
    el.diagramNote.textContent = info.trained ? note.diagram : note.diagramUntrained;

    // Redraw the diagram only when the tree itself has changed.
    if (!drawn || drawn.list !== nodes || drawn.version !== info.version) {
      const sameTree = drawn && drawn.list === nodes;
      const changed = sameTree && info.lastSplit !== null;
      drawn = { list: nodes, version: info.version };
      const picture = nodes.length ? diagramSvg(nodes) : { at: new Map(), svg: '' };
      positions = picture.at;
      el.diagram.innerHTML = picture.svg;
      el.diagram.hidden = nodes.length === 0;
      if (!sameTree) {
        selected = nodes.length ? 0 : null;
        // A highlight left over from the previous tree no longer means anything.
        if (lit) queueMicrotask(() => light(false));
      }
      // While the tree grows or is pruned, follow the newest change.
      if (changed || (!sameTree && info.lastSplit !== null)) { selected = info.lastSplit; reveal(); }
    }

    // Mark the selected group and the query point's route.
    const ex = info.example;
    const route = new Set(ex ? [...ex.path.map((q) => q.id), ex.leaf.id] : []);
    for (const box of el.diagram.querySelectorAll('.node')) {
      const id = Number(box.dataset.id);
      box.classList.toggle('selected', id === selected);
      box.classList.toggle('on-route', route.has(id));
    }

    // The selected group and the questions that were compared for it
    const node = nodeById(selected);
    el.candidateWrap.hidden = !node || !node.candidates;
    el.candidateNote.hidden = el.candidateWrap.hidden;
    if (!node) {
      el.selectedText.textContent = note.diagramUntrained;
    } else {
      const kind = node.split ? 'This group' : 'This leaf';
      let text = `${kind} holds ${node.red} red and ${node.blue} blue training points. Its ${measure.name} is ${fmt(node.impurity)}.`;
      if (!node.split) text += ` ${LEAF_REASONS[node.leafReason] || WAITING}`;
      el.selectedText.textContent = text;
      if (node.candidates) {
        setHtml(el.candidates, node.candidates.map((c) => {
          const chosen = node.split && node.split.feature === c.feature;
          let question = 'not looked at';
          let drop = '';
          if (c.considered && c.gain === null) question = 'no possible cut';
          else if (c.considered) {
            question = `${FEATURE_HTML[c.feature]} &lt; ${fmt(c.threshold)}`;
            drop = fmt(c.gain, 3);
          }
          return `<tr class="${chosen ? 'chosen' : ''}"><th scope="row">${FEATURE_HTML[c.feature]}</th><td>${question}</td>`
            + `<td class="num">${drop}</td><td class="num">${chosen ? 'chosen' : ''}</td></tr>`;
        }).join(''));
      }
    }

    // Worked example
    el.example.hidden = !ex;
    el.exampleEmpty.hidden = !!ex;
    el.exampleEmpty.textContent = info.trained ? COPY.queryEmpty : note.diagramUntrained;
    if (ex) {
      const questions = ex.path.map((q) => `<li><strong>Is ${FEATURE_HTML[q.feature]} less than ${fmt(q.threshold)}?</strong> `
        + `<span class="num-line">Here ${FEATURE_HTML[q.feature]} = <span class="num">${fmt(q.value)}</span>, `
        + `so <strong>${q.yes ? 'yes' : 'no'}</strong>: go ${q.yes ? 'left' : 'right'}.</span></li>`);
      const total = ex.leaf.red + ex.leaf.blue;
      const arrive = ex.path.length
        ? '<strong>Arrive at a leaf.</strong>'
        : '<strong>No questions yet.</strong> The tree has not split, so every point lands in the same group.';
      setHtml(el.example, `
        <li><strong>Start with the query point.</strong> <span class="num-line">x<sub>1</sub> = <span class="num">${fmt(ex.x[0])}</span>, x<sub>2</sub> = <span class="num">${fmt(ex.x[1])}</span></span></li>
        ${questions.join('')}
        <li>${arrive} <span class="num-line">It holds ${ex.leaf.red} ${classLabel(0)} and ${ex.leaf.blue} ${classLabel(1)} training points, so p = ${ex.leaf.blue} / ${total} = <span class="num">${fmt(ex.p)}</span></span></li>
        <li><strong>Compare with the threshold.</strong> <span class="num-line">${verdict(ex.p, threshold)}</span></li>`);
    }

    el.giniChart.innerHTML = giniChart(node, info.criterion);
  }

  return { update };
}
