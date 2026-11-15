// Small pieces of page wiring shared by the parts of the training lab.

import { fmt, prefersReducedMotion } from '../../../assets/js/format.js';

export const $ = (id) => document.getElementById(id);

// A pair of weights as text: (1.50, −0.20).
export const pair = ([a, b]) => `(${fmt(a)}, ${fmt(b)})`;

// Clicks and arrow keys place a start point on this grid.
export const GRID = 0.05;
export const snapTo = (v, [low, high], grid = GRID) => Math.max(low, Math.min(high, Math.round(v / grid) * grid));
export const onMap = ([a, b], [x0, x1], [y0, y1]) => a >= x0 && a <= x1 && b >= y0 && b <= y1;

// The cross that marks where training starts.
export const startMark = (x, y) => `<path class="start-mark" d="M${x - 5} ${y - 5}l10 10m0 -10l-10 10"/>`;

// Let a map choose a start point: click or tap it, or focus it and use the
// arrow keys (Shift for bigger moves). `get` returns the current start and
// `set` is given the new one, already snapped to the grid and kept on the map.
export function startPicker(map, view, xRange, yRange, get, set, grid = GRID) {
  const place = (a, b) => set([snapTo(a, xRange, grid), snapTo(b, yRange, grid)]);
  map.addEventListener('click', (event) => {
    const at = view.weightsAt(event, map);
    if (at) place(at[0], at[1]);
  });
  map.addEventListener('keydown', (event) => {
    const move = { ArrowLeft: [-grid, 0], ArrowRight: [grid, 0], ArrowUp: [0, grid], ArrowDown: [0, -grid] }[event.key];
    if (!move) return;
    event.preventDefault();
    const size = event.shiftKey ? 5 : 1;
    const [a, b] = get();
    place(a + move[0] * size, b + move[1] * size);
  });
}

// Wire up a Train / Pause / Step / Skip bar and its speed buttons, whose
// ids all start with `prefix`. `begin(paused)` starts a new run. Returns a
// function that brings the buttons up to date.
export function trainBar(prefix, engine, speeds, begin, getSpeed, setSpeed) {
  const refresh = () => {
    $(`${prefix}PauseBtn`).disabled = !engine.active;
    $(`${prefix}PauseBtn`).textContent = engine.active && !engine.running ? 'Resume' : 'Pause';
    $(`${prefix}StepBtn`).disabled = engine.running;
    $(`${prefix}SkipBtn`).disabled = !engine.active;
  };
  $(`${prefix}TrainBtn`).addEventListener('click', () => {
    begin(false);
    // With reduced motion, skip the animation and show the finished run.
    if (prefersReducedMotion()) engine.skip();
  });
  $(`${prefix}PauseBtn`).addEventListener('click', () => {
    if (engine.running) engine.pause(); else engine.resume();
    refresh();
  });
  $(`${prefix}StepBtn`).addEventListener('click', () => {
    if (!engine.active) begin(true);
    engine.stepOnce();
    refresh();
  });
  $(`${prefix}SkipBtn`).addEventListener('click', () => engine.skip());
  for (const radio of document.querySelectorAll(`input[name="${prefix}Speed"]`)) {
    if (radio.checked) setSpeed(radio.value);
    radio.addEventListener('change', () => {
      setSpeed(radio.value);
      engine.setSpeed(speeds[getSpeed()]);
    });
  }
  return refresh;
}

// A slider that moves between fixed stops. `get` and `set` read and write
// the value; `changed` is called after a move.
export function stopSlider(id, outId, stops, get, set, changed, show = String) {
  const range = $(id);
  range.max = stops.length - 1;
  range.value = stops.indexOf(get());
  const update = () => {
    $(outId).textContent = show(get());
    range.setAttribute('aria-valuetext', show(get()));
  };
  update();
  range.addEventListener('input', () => {
    set(stops[Number(range.value)]);
    update();
    changed();
  });
}

// A seed box and a New data button. `load(seed)` builds the data.
export function seedControls(inputId, buttonId, getSeed, load, announce) {
  $(inputId).addEventListener('change', () => {
    const seed = Number($(inputId).value);
    if (Number.isFinite(seed)) load(seed); else $(inputId).value = getSeed();
  });
  $(buttonId).addEventListener('click', () => {
    // Choosing the next seed is the one place that is not seeded: it is the
    // source of a fresh variation. Everything after it follows from the seed.
    load(Math.floor(Math.random() * 100000));
    announce(`New data, seed ${getSeed()}.`);
  });
}
export const clampSeed = (seed) => Math.max(0, Math.min(999999, Math.floor(seed)));
