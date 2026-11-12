// Small SVG charts that belong to the page as a whole, not to one model.
// They are coloured by the theme's CSS variables (see .chart in lab.css).

// The ROC curve, with a dot for the current threshold.
//   points  from rocCurve() in metrics.js: [{ fpr, tpr }]
//   dot     { fpr, tpr } at the current threshold, or null
//   auc     area under the curve, or null
export function rocChart(points, dot, auc) {
  const W = 300, H = 290, L = 40, R = 286, T = 12, B = 250;
  const x = (v) => L + v * (R - L);
  const y = (v) => B - v * (B - T);
  const frame = `
    <line class="axis" x1="${L}" y1="${B}" x2="${R}" y2="${B}"/><line class="axis" x1="${L}" y1="${T}" x2="${L}" y2="${B}"/>
    <line class="threshold-line" x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}"/>
    <text class="tick" x="${L - 5}" y="${T + 4}" text-anchor="end">100%</text>
    <text class="tick" x="${L - 5}" y="${B + 3}" text-anchor="end">0%</text>
    <text class="tick" x="${R}" y="${B + 13}" text-anchor="end">100%</text>
    <text class="tick" x="${(L + R) / 2}" y="${H - 20}" text-anchor="middle">red points wrongly called blue</text>
    <text class="tick" x="${(L + R) / 2}" y="${H - 8}" text-anchor="middle">(1 − specificity)</text>
    <text class="tick" transform="translate(12 ${(T + B) / 2}) rotate(-90)" text-anchor="middle">blue points caught (sensitivity)</text>`;
  if (points.length === 0) {
    return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="ROC curve: nothing to show yet">${frame}</svg>`;
  }
  const line = points.map((p) => `${x(p.fpr).toFixed(1)},${y(p.tpr).toFixed(1)}`).join(' ');
  const marker = dot
    ? `<circle class="roc-dot" cx="${x(dot.fpr).toFixed(1)}" cy="${y(dot.tpr).toFixed(1)}" r="6"/>`
    : '';
  const label = `ROC curve. Area under the curve ${auc.toFixed(2)}.`
    + (dot ? ` At the current threshold the model catches ${Math.round(dot.tpr * 100)}% of blue points and wrongly calls ${Math.round(dot.fpr * 100)}% of red points blue.` : '');
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${label}">
    ${frame}
    <polyline class="curve" points="${line}"/>
    ${marker}
    <text class="tick strong" x="${R - 4}" y="${B - 8}" text-anchor="end">area under the curve ${auc.toFixed(2)}</text>
  </svg>`;
}
