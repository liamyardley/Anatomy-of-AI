// The canvas plot: shaded decision regions, the boundary line, the points,
// the query point, and mouse, touch and keyboard input.
//
// It knows nothing about datasets or particular models. main.js tells it
// what to draw, hands it any object with a predictProba([x1, x2]) method to
// shade the regions, and receives callbacks when the user does something.

const PAD = 14;            // space around the [-1, 1] square, in CSS pixels
const POINT_RADIUS = 4.4;
const LONG_PRESS_MS = 550;
const TAP_SLOP = 10;       // how far a finger may drift and still count as a tap
const REGION_STRENGTH = 0.55; // opacity of the shading where the model is certain

export function createPlot(canvas, handlers) {
  const ctx = canvas.getContext('2d');
  let size = 0;            // canvas width and height in CSS pixels
  let view = {
    points: [], showTest: false, query: null, mode: 'explore',
    threshold: 0.5,        // where the boundary line is drawn
    stale: false,          // fade the regions: they no longer match the settings
  };
  let keyboardActive = false;
  let press = null;        // the touch or click currently in progress
  let lastPointerType = 'mouse';

  // The model's probability of blue, sampled on a grid of (n + 1) x (n + 1)
  // nodes covering the plot. Row 0 is the top edge (x2 = 1).
  let field = null;
  let fieldSize = 0;
  const regionImage = document.createElement('canvas');
  let regionKey = '';      // what the cached image was painted from

  // --- coordinates: data space [-1, 1] to pixels and back ---
  const span = () => size - 2 * PAD;
  const toPx = (x1, x2) => [PAD + ((x1 + 1) / 2) * span(), PAD + ((1 - x2) / 2) * span()];
  const clamp = (v) => Math.max(-1, Math.min(1, v));
  function fromEvent(event) {
    const rect = canvas.getBoundingClientRect();
    return {
      x1: clamp(((event.clientX - rect.left - PAD) / span()) * 2 - 1),
      x2: clamp(1 - ((event.clientY - rect.top - PAD) / span()) * 2),
    };
  }

  function colours() {
    const css = getComputedStyle(document.documentElement);
    const get = (name) => css.getPropertyValue(name).trim();
    return {
      surface: get('--surface'), border: get('--border'), muted: get('--muted'),
      ink: get('--ink'), red: get('--class-red'), blue: get('--class-blue'),
    };
  }

  const hexToRgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));

  // --- decision regions ---

  // Paint the probability grid as a small image: red where the model says
  // red, blue where it says blue, and fading to nothing near 0.5, so
  // uncertainty shows as paleness. Stretching it over the plot smooths it.
  function paintRegions(c) {
    const key = `${fieldSize}|${c.red}|${c.blue}|${field.version}`;
    if (key === regionKey) return;
    regionKey = key;
    const n = fieldSize + 1;
    regionImage.width = n;
    regionImage.height = n;
    const imageCtx = regionImage.getContext('2d');
    const image = imageCtx.createImageData(n, n);
    const red = hexToRgb(c.red);
    const blue = hexToRgb(c.blue);
    for (let k = 0; k < field.values.length; k++) {
      const p = field.values[k];
      const rgb = p >= 0.5 ? blue : red;
      image.data[k * 4] = rgb[0];
      image.data[k * 4 + 1] = rgb[1];
      image.data[k * 4 + 2] = rgb[2];
      image.data[k * 4 + 3] = Math.round(255 * REGION_STRENGTH * Math.abs(2 * p - 1));
    }
    imageCtx.putImageData(image, 0, 0);
  }

  // Marching squares: find where the probability crosses the threshold
  // inside each grid cell, giving short line pieces that join into the
  // boundary. Positions are in grid units.
  function boundarySegments(threshold) {
    const n = fieldSize;
    const row = n + 1;
    const v = field.values;
    const cross = (a, b) => (threshold - a) / (b - a);
    const segments = [];
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const a = v[j * row + i];           // top left
        const b = v[j * row + i + 1];       // top right
        const c = v[(j + 1) * row + i + 1]; // bottom right
        const d = v[(j + 1) * row + i];     // bottom left
        const code = (a >= threshold ? 8 : 0) | (b >= threshold ? 4 : 0)
          | (c >= threshold ? 2 : 0) | (d >= threshold ? 1 : 0);
        if (code === 0 || code === 15) continue;
        const top = () => [i + cross(a, b), j];
        const right = () => [i + 1, j + cross(b, c)];
        const bottom = () => [i + cross(d, c), j + 1];
        const left = () => [i, j + cross(a, d)];
        switch (code) {
          case 1: case 14: segments.push([left(), bottom()]); break;
          case 2: case 13: segments.push([bottom(), right()]); break;
          case 3: case 12: segments.push([left(), right()]); break;
          case 4: case 11: segments.push([top(), right()]); break;
          case 6: case 9: segments.push([top(), bottom()]); break;
          case 7: case 8: segments.push([left(), top()]); break;
          case 5: segments.push([left(), top()], [bottom(), right()]); break;
          default: segments.push([top(), right()], [left(), bottom()]); break; // 10
        }
      }
    }
    return segments;
  }

  function drawRegions(c) {
    paintRegions(c);
    const cell = span() / fieldSize;
    ctx.save();
    ctx.beginPath();
    ctx.rect(PAD, PAD, span(), span());
    ctx.clip();
    ctx.globalAlpha = view.stale ? 0.3 : 1;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    // Each image pixel is one grid node, so the image overhangs the plot by
    // half a cell on every side.
    ctx.drawImage(regionImage, PAD - cell / 2, PAD - cell / 2, span() + cell, span() + cell);

    // The boundary: a dark line with a pale edge, so it reads on any shading.
    const segments = boundarySegments(view.threshold);
    ctx.lineCap = 'round';
    for (const [width, colour] of [[4.5, c.surface], [2, c.ink]]) {
      ctx.lineWidth = width;
      ctx.strokeStyle = colour;
      ctx.beginPath();
      for (const [from, to] of segments) {
        ctx.moveTo(PAD + from[0] * cell, PAD + from[1] * cell);
        ctx.lineTo(PAD + to[0] * cell, PAD + to[1] * cell);
      }
      ctx.stroke();
    }
    ctx.restore();
  }

  // --- drawing ---
  function shapePath(px, py, y, radius) {
    ctx.beginPath();
    if (y === 1) {
      // Blue: a triangle with about the same area as the red circle.
      const r = radius * 1.35;
      ctx.moveTo(px, py - r);
      ctx.lineTo(px + r * 0.866, py + r * 0.5);
      ctx.lineTo(px - r * 0.866, py + r * 0.5);
      ctx.closePath();
    } else {
      ctx.arc(px, py, radius, 0, Math.PI * 2);
    }
  }

  function drawPoint(p, c) {
    const [px, py] = toPx(p.x1, p.x2);
    const colour = p.y === 1 ? c.blue : c.red;
    if (p.test) {
      // Test points are hollow and grey: the model is not told their class.
      // The shape still records the true answer, so predictions can be checked.
      shapePath(px, py, p.y, POINT_RADIUS - 0.6);
      ctx.fillStyle = c.surface;
      ctx.fill();
      ctx.lineWidth = 1.7;
      ctx.strokeStyle = c.muted;
      ctx.stroke();
    } else {
      shapePath(px, py, p.y, POINT_RADIUS);
      ctx.fillStyle = colour;
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = c.surface;
      ctx.stroke();
    }
  }

  function drawGrid(c) {
    ctx.lineWidth = 1;
    for (const v of [-1, -0.5, 0, 0.5, 1]) {
      const [x] = toPx(v, 0);
      const [, y] = toPx(0, v);
      ctx.strokeStyle = c.border;
      ctx.globalAlpha = v === 0 ? 1 : 0.6;
      ctx.beginPath();
      ctx.moveTo(Math.round(x) + 0.5, PAD); ctx.lineTo(Math.round(x) + 0.5, size - PAD);
      ctx.moveTo(PAD, Math.round(y) + 0.5); ctx.lineTo(size - PAD, Math.round(y) + 0.5);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    // A few axis labels, tucked just inside the frame.
    ctx.fillStyle = c.muted;
    ctx.font = "11px 'Karla', system-ui, sans-serif";
    ctx.textBaseline = 'bottom';
    ctx.textAlign = 'left';
    ctx.fillText('−1', PAD + 3, size - PAD - 2);
    ctx.textAlign = 'right';
    ctx.fillText('x₁ = 1', size - PAD - 3, size - PAD - 2);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('x₂ = 1', PAD + 3, PAD + 3);
  }

  function drawQuery(q, c) {
    const [px, py] = toPx(q.x1, q.x2);
    if (keyboardActive) {
      // Crosshair lines make the keyboard cursor easy to find.
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = c.muted;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(px, PAD); ctx.lineTo(px, size - PAD);
      ctx.moveTo(PAD, py); ctx.lineTo(size - PAD, py);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    // A ring with a centre dot, outlined so it shows on any background.
    for (const [width, colour] of [[4.5, c.surface], [2, c.ink]]) {
      ctx.lineWidth = width;
      ctx.strokeStyle = colour;
      ctx.beginPath();
      ctx.arc(px, py, 9, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.fillStyle = c.ink;
    ctx.beginPath();
    ctx.arc(px, py, 2, 0, Math.PI * 2);
    ctx.fill();
  }

  function draw() {
    if (size === 0) return;
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const c = colours();
    ctx.fillStyle = c.surface;
    ctx.fillRect(0, 0, size, size);
    if (field) drawRegions(c);
    drawGrid(c);
    // Training points on top of test points, so the solid ones stay readable.
    if (view.showTest) for (const p of view.points) if (p.test) drawPoint(p, c);
    for (const p of view.points) if (!p.test) drawPoint(p, c);
    if (view.query) drawQuery(view.query, c);
  }

  function resize() {
    const width = canvas.getBoundingClientRect().width;
    if (width === 0) return;
    const dpr = window.devicePixelRatio || 1;
    size = width;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(width * dpr);
    draw();
  }

  // --- mouse and touch ---
  function cancelPress() {
    if (press) clearTimeout(press.timer);
    press = null;
  }

  canvas.addEventListener('pointerdown', (event) => {
    lastPointerType = event.pointerType;
    keyboardActive = false;
    cancelPress();
    if (event.pointerType === 'mouse' && event.button !== 0) return;
    press = { x: event.clientX, y: event.clientY, long: false, timer: null };
    if (event.pointerType !== 'mouse') {
      // Long-press removes the nearest point (Edit mode only).
      const where = fromEvent(event);
      press.timer = setTimeout(() => {
        if (!press) return;
        press.long = true;
        if (view.mode === 'edit') handlers.onRemove(where);
      }, LONG_PRESS_MS);
    }
  });

  canvas.addEventListener('pointermove', (event) => {
    if (event.pointerType === 'mouse') {
      // Hovering sets the query point in both modes.
      handlers.onQuery(fromEvent(event));
    } else if (press && Math.hypot(event.clientX - press.x, event.clientY - press.y) > TAP_SLOP) {
      cancelPress(); // the finger is scrolling, not tapping
    }
  });

  canvas.addEventListener('pointerup', (event) => {
    if (!press) return;
    const wasLong = press.long;
    const moved = Math.hypot(event.clientX - press.x, event.clientY - press.y);
    cancelPress();
    if (wasLong || moved > TAP_SLOP) return;
    const where = fromEvent(event);
    if (view.mode === 'edit') handlers.onAdd(where, event.shiftKey);
    else handlers.onQuery(where);
  });

  canvas.addEventListener('pointercancel', cancelPress);

  canvas.addEventListener('contextmenu', (event) => {
    if (view.mode !== 'edit') return;
    event.preventDefault(); // right-click removes instead of opening a menu
    if (lastPointerType === 'mouse') handlers.onRemove(fromEvent(event));
  });

  // --- keyboard ---
  canvas.addEventListener('keydown', (event) => {
    const moves = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] };
    const at = view.query || { x1: 0, x2: 0 };
    if (moves[event.key]) {
      event.preventDefault();
      keyboardActive = true;
      const step = event.shiftKey ? 0.01 : 0.05;
      handlers.onQuery({
        x1: clamp(at.x1 + moves[event.key][0] * step),
        x2: clamp(at.x2 + moves[event.key][1] * step),
      });
    } else if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      keyboardActive = true;
      if (view.mode === 'edit') handlers.onAdd(at, event.shiftKey);
      else handlers.onQuery(at);
    } else if ((event.key === 'Delete' || event.key === 'Backspace') && view.mode === 'edit') {
      event.preventDefault();
      handlers.onRemove(at);
    }
  });
  canvas.addEventListener('blur', () => { keyboardActive = false; draw(); });

  // Keep the drawing sharp and correctly coloured as things change.
  new ResizeObserver(resize).observe(canvas);
  // Redraw after the browser has applied the new colours.
  const colourScheme = matchMedia('(prefers-color-scheme: dark)');
  colourScheme.addEventListener('change', () => requestAnimationFrame(draw));
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(draw);

  let fieldVersion = 0;

  return {
    // Change what is shown: any of { points, showTest, query, mode, threshold, stale }.
    // Moving the threshold only moves the boundary line; the shading is untouched.
    update(next) {
      view = { ...view, ...next };
      draw();
    },

    // Shade the plot with a model's probabilities, sampled on a grid of
    // gridSize x gridSize cells. Pass null to remove the shading.
    setModel(model, gridSize = 120) {
      if (!model) {
        field = null;
      } else {
        const row = gridSize + 1;
        const values = new Float32Array(row * row);
        for (let j = 0; j < row; j++) {
          const x2 = 1 - (2 * j) / gridSize;
          for (let i = 0; i < row; i++) {
            values[j * row + i] = model.predictProba([-1 + (2 * i) / gridSize, x2]);
          }
        }
        fieldSize = gridSize;
        field = { values, version: ++fieldVersion };
      }
      draw();
    },
    draw,
  };
}
