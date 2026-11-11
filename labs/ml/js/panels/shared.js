// Helpers shared by the models' maths panels.

import { fmt, MARK } from '../format.js';

// Collect every element marked data-role="name" inside the given parts into
// one object, so a panel can write el.name.
export function roles(...parts) {
  const el = {};
  for (const part of parts) {
    for (const node of part.querySelectorAll('[data-role]')) el[node.dataset.role] = node;
  }
  return el;
}

// Make a table body have exactly `count` rows of `cells` cells each. Every
// cell after the first is right-aligned, for numbers.
export function sizeRows(body, count, cells) {
  while (body.rows.length > count) body.deleteRow(-1);
  while (body.rows.length < count) {
    const row = body.insertRow();
    for (let c = 0; c < cells; c++) {
      const cell = row.insertCell();
      if (c > 0) cell.className = 'num';
    }
  }
}

// Replace an element's HTML only if it has changed, so unchanged content is
// not rebuilt on every mouse move.
export function setHtml(element, html) {
  if (element.dataset.html === html) return;
  element.dataset.html = html;
  element.innerHTML = html;
}

// "red" or "blue" with its shape beside it.
export const classLabel = (y) => (y === 1 ? `${MARK.blue} blue` : `${MARK.red} red`);

// The last line of every worked example: compare the probability with the
// decision threshold.
export function verdict(p, threshold) {
  const blue = p >= threshold;
  return `${fmt(p)} is ${blue ? 'at least' : 'below'} ${fmt(threshold)}, so the model predicts `
    + `<strong>${classLabel(blue ? 1 : 0)}</strong>.`;
}
