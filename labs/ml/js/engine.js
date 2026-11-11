// The training engine: runs any model's trainer step by step, so the page can
// animate it. It knows nothing about what a "step" means; each model decides
// that for itself. The same engine will drive every model in the lab.
//
// A speed is { steps, interval }: how many training steps to run per tick,
// and the shortest time between ticks in milliseconds (0 means every frame).

export function createEngine({ onProgress, onFinish }) {
  let trainer = null;
  let running = false;
  let speed = { steps: 1, interval: 0 };
  let lastTick = -Infinity;
  let frame = 0;

  function complete(result) {
    trainer = null;
    running = false;
    cancelAnimationFrame(frame);
    onFinish(result.snapshot);
  }

  // Run a batch of steps, then report progress (or the finished model).
  function advance(count) {
    let result = null;
    for (let i = 0; i < count; i++) {
      result = trainer.step();
      if (result.done) break;
    }
    if (result.done) complete(result);
    else onProgress(result.snapshot);
  }

  function tick(now) {
    if (!running) return;
    if (now - lastTick >= speed.interval) {
      lastTick = now;
      advance(speed.steps);
    }
    if (running) frame = requestAnimationFrame(tick);
  }

  function resume() {
    if (!trainer || running) return;
    running = true;
    lastTick = -Infinity;
    frame = requestAnimationFrame(tick);
  }

  return {
    // Begin animating a new trainer. Any training already under way is dropped.
    start(newTrainer, newSpeed, { paused = false } = {}) {
      cancelAnimationFrame(frame);
      trainer = newTrainer;
      speed = newSpeed;
      running = false;
      if (!paused) resume();
    },
    pause() {
      running = false;
      cancelAnimationFrame(frame);
    },
    resume,
    // One step while paused.
    stepOnce() {
      if (trainer && !running) advance(1);
    },
    // Jump straight to the finished model.
    skip() {
      if (trainer) complete(trainer.finish());
    },
    setSpeed(newSpeed) {
      speed = newSpeed;
    },
    // Abandon training without finishing it.
    cancel() {
      trainer = null;
      running = false;
      cancelAnimationFrame(frame);
    },
    get active() { return trainer !== null; },   // training has started and not finished
    get running() { return running; },           // active and not paused
  };
}
