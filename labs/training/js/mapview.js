// A contour map of a loss with two weights, shared by the parts of the
// training lab that need one.
//
// A canvas holds the coloured height bands. An SVG laid over it (drawn by
// each part) holds whatever changes: paths, markers, arrows. This file does
// the canvas, and the arithmetic for placing things on the SVG.

const BANDS = 16;  // how many height bands the map is drawn in
const CELLS = 160; // the map is coloured in CELLS x CELLS squares

// The SVG's coordinate system: a square, with the plotting area inside it.
// (The canvas is placed over the plotting area by the stylesheet.)
export const BOX = { SIZE: 360, L: 36, T: 8, PLOT: 316 };

// xRange and yRange are [low, high] for the weight across and the weight up.
export function createMapView({ canvas, xRange, yRange }) {
  const [x0, x1] = xRange;
  const [y0, y1] = yRange;
  const px = (a) => BOX.L + ((a - x0) / (x1 - x0)) * BOX.PLOT;
  const py = (b) => BOX.T + ((y1 - b) / (y1 - y0)) * BOX.PLOT;

  // Colour the map: the lower the ground, the stronger the tint, with a thin
  // line wherever one band meets the next (the contour lines).
  function draw(surface) {
    const size = canvas.width;
    const cell = size / CELLS;
    const ctx = canvas.getContext('2d');
    const style = getComputedStyle(document.documentElement);
    const tint = style.getPropertyValue('--ml-line').trim();
    const line = style.getPropertyValue('--muted').trim();

    const heights = [];
    let low = Infinity;
    let high = -Infinity;
    for (let j = 0; j < CELLS; j++) {
      const row = [];
      for (let i = 0; i < CELLS; i++) {
        const a = x0 + ((i + 0.5) / CELLS) * (x1 - x0);
        const b = y1 - ((j + 0.5) / CELLS) * (y1 - y0);
        const value = surface.loss(a, b);
        if (value < low) low = value;
        if (value > high) high = value;
        row.push(value);
      }
      heights.push(row);
    }
    for (const spot of surface.spots || []) low = Math.min(low, surface.loss(spot.at[0], spot.at[1]));
    // Which band a height falls in. The square root packs more bands near
    // the bottom, where the interesting shapes are.
    const band = (value) => Math.min(BANDS - 1, Math.floor(BANDS * Math.sqrt((value - low) / (high - low))));
    const levels = heights.map((row) => row.map(band));

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

  // The frame, tick labels and axis titles, as SVG.
  function axes(xTicks, yTicks, xTitle, yTitle) {
    const { SIZE, L, T, PLOT } = BOX;
    return `<rect class="map-frame" x="${L}" y="${T}" width="${PLOT}" height="${PLOT}"/>
      ${xTicks.map((v) => `<text class="tick" x="${px(v)}" y="${T + PLOT + 13}" text-anchor="middle">${v}</text>`).join('')}
      ${yTicks.map((v) => `<text class="tick" x="${L - 5}" y="${py(v) + 3}" text-anchor="end">${v}</text>`).join('')}
      <text class="axis-title" x="${L + PLOT / 2}" y="${SIZE - 4}" text-anchor="middle">${xTitle}</text>
      <text class="axis-title" transform="translate(10 ${T + PLOT / 2}) rotate(-90)" text-anchor="middle">${yTitle}</text>`;
  }

  // A clip path that keeps drawing inside the plotting area.
  const clip = (id) => `<clipPath id="${id}"><rect x="${BOX.L}" y="${BOX.T}" width="${BOX.PLOT}" height="${BOX.PLOT}"/></clipPath>`;

  // The pair of weights under a click or tap on `element` (the map's outer
  // box), or null if it landed outside the plotting area.
  function weightsAt(event, element) {
    const box = element.getBoundingClientRect();
    const scale = BOX.SIZE / box.width;
    const x = (event.clientX - box.left) * scale;
    const y = (event.clientY - box.top) * scale;
    if (x < BOX.L || x > BOX.L + BOX.PLOT || y < BOX.T || y > BOX.T + BOX.PLOT) return null;
    return [x0 + ((x - BOX.L) / BOX.PLOT) * (x1 - x0), y1 - ((y - BOX.T) / BOX.PLOT) * (y1 - y0)];
  }

  // A list of [a, b] points as the `points` of an SVG polyline.
  const line = (list) => list.map(([a, b]) => `${px(a).toFixed(1)},${py(b).toFixed(1)}`).join(' ');

  return { px, py, draw, axes, clip, weightsAt, line };
}
