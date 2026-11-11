// The canvas plot: points, the query point, and mouse, touch and keyboard
// input. It knows nothing about datasets or models; main.js tells it what to
// draw and receives callbacks when the user does something.
//
// Decision regions and model overlays are added in the next phase.

const PAD = 14;            // space around the [-1, 1] square, in CSS pixels
const POINT_RADIUS = 4.4;
const LONG_PRESS_MS = 550;
const TAP_SLOP = 10;       // how far a finger may drift and still count as a tap

export function createPlot(canvas, handlers) {
  const ctx = canvas.getContext('2d');
  let size = 0;            // canvas width and height in CSS pixels
  let view = { points: [], showTest: true, query: null, mode: 'explore' };
  let keyboardActive = false;
  let press = null;        // the touch or click currently in progress
  let lastPointerType = 'mouse';

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
      // Test points are hollow.
      shapePath(px, py, p.y, POINT_RADIUS - 0.6);
      ctx.fillStyle = c.surface;
      ctx.fill();
      ctx.lineWidth = 1.7;
      ctx.strokeStyle = colour;
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
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(draw);

  return {
    // Replace what is shown: { points, showTest, query, mode }.
    update(next) {
      view = { ...view, ...next };
      draw();
    },
    draw,
  };
}
