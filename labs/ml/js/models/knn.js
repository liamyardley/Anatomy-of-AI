// k-nearest neighbours. Pure code: no DOM access, so it can be tested in Node.
//
// To classify a point, find the k stored training points closest to it and
// let them vote. There is nothing to fit, so "training" is only storing the
// points; all of the work happens when a prediction is asked for.
//
// The model has the same shape as every other model in the lab (see
// logistic.js). Its training steps exist to make that lesson visible:
//   phase 1, "memorise": the training points are copied into memory;
//   phase 2, "predict":  the plot fills in, a band of rows at a time, as each
//                        grid cell asks its neighbours (snapshot.reveal says
//                        how much of the plot has been filled so far).

const MEMORISE_STEPS = 20; // steps taken to copy the points into memory
const PREDICT_STEPS = 30;  // steps taken to fill in the plot
const NEARBY = 40;         // how many of the closest points explain() lists

const distanceBetween = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

export function createKnn({ k = 5, weighted = false } = {}) {
  let memory = [];        // the stored training points: { x: [x1, x2], y }
  let total = 0;          // how many points there are to store
  let phase = 'idle';     // 'idle', 'memorise', 'predict' or 'done'
  let reveal = 0;         // share of the plot filled in so far
  let recent = [];        // the points stored by the latest step
  let finished = false;

  // With distance weighting, a closer neighbour gets a bigger say.
  const weightOf = (distance) => (weighted ? 1 / Math.max(distance, 1e-6) : 1);

  // The k stored points closest to x, nearest first.
  function nearest(x) {
    const best = [];
    for (const item of memory) {
      const distance = distanceBetween(x, item.x);
      if (best.length === k && distance >= best[k - 1].distance) continue;
      let at = best.length;
      while (at > 0 && best[at - 1].distance > distance) at--;
      best.splice(at, 0, { x: item.x, y: item.y, distance });
      if (best.length > k) best.pop();
    }
    return best;
  }

  // Count the votes: the probability of blue is blue's share of them.
  function tally(neighbours) {
    let blueWeight = 0;
    let totalWeight = 0;
    let blueVotes = 0;
    for (const n of neighbours) {
      const weight = weightOf(n.distance);
      totalWeight += weight;
      if (n.y === 1) { blueWeight += weight; blueVotes++; }
    }
    return {
      blueVotes,
      redVotes: neighbours.length - blueVotes,
      blueWeight,
      totalWeight,
      p: totalWeight > 0 ? blueWeight / totalWeight : 0.5,
    };
  }

  function train(X, y) {
    memory = [];
    total = X.length;
    reveal = 0;
    recent = [];
    finished = total === 0;
    phase = finished ? 'done' : 'memorise';
    const batch = Math.max(1, Math.ceil(total / MEMORISE_STEPS));
    let rowsDone = 0;

    const snapshot = () => ({ phase, stored: memory.length, total, reveal, done: finished });

    function step() {
      if (finished) return { done: true, snapshot: snapshot() };
      if (phase === 'memorise') {
        recent = [];
        const end = Math.min(total, memory.length + batch);
        for (let i = memory.length; i < end; i++) {
          memory.push({ x: X[i], y: y[i] });
          recent.push(X[i]);
        }
        if (memory.length === total) phase = 'predict';
      } else {
        recent = [];
        rowsDone += 1;
        reveal = rowsDone / PREDICT_STEPS;
        if (rowsDone >= PREDICT_STEPS) { finished = true; phase = 'done'; }
      }
      return { done: finished, snapshot: snapshot() };
    }

    function finish() {
      memory = X.map((x, i) => ({ x, y: y[i] }));
      recent = [];
      reveal = 1;
      finished = true;
      phase = 'done';
      return { done: true, snapshot: snapshot() };
    }

    return { step, finish };
  }

  function predictProba(x) {
    if (memory.length === 0) return 0.5;
    return tally(nearest(x)).p;
  }

  // The model's internals, plus the full calculation for one point.
  function explain(point = null) {
    let example = null;
    const overlay = [];
    if (phase === 'memorise' && recent.length) overlay.push({ kind: 'rings', points: recent });

    if (point && memory.length) {
      const neighbours = nearest(point).map((n) => ({ ...n, weight: weightOf(n.distance) }));
      const radius = neighbours[neighbours.length - 1].distance;
      // The closest stored points in order, voters or not, for the picture.
      const nearby = memory
        .map((item) => ({ y: item.y, distance: distanceBetween(point, item.x) }))
        .sort((a, b) => a.distance - b.distance)
        .slice(0, NEARBY);
      example = { x: point, neighbours, radius, nearby, ...tally(neighbours) };
      for (const n of neighbours) overlay.push({ kind: 'segment', from: point, to: n.x, style: 'link' });
      overlay.push({ kind: 'circle', centre: point, radius });
    }

    return {
      model: 'knn',
      k,
      weighted,
      trained: memory.length > 0,
      finished,
      phase,
      stored: memory.length,
      total,
      memory,
      example,
      overlay,
    };
  }

  return { train, predictProba, explain };
}
