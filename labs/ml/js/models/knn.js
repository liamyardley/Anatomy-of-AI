// k-nearest neighbours. Pure code: no DOM access, so it can be tested in Node.
//
// To classify a point, find the k stored training points closest to it and
// let them vote. There is nothing to fit, so "training" is only storing the
// points; all of the work happens when a prediction is asked for.
//
// "Closest" depends on how distance is measured, and that is a setting:
//   euclidean  straight-line distance
//   manhattan  distance along a grid of streets: across plus up
//   chebyshev  the bigger of the two gaps, across or up
//   minkowski  a family with a power p: p = 1 is Manhattan, p = 2 is
//              Euclidean, and a large p approaches Chebyshev
//   cosine     compares direction from the centre only, ignoring how far
//              out a point is
//
// The model has the same shape as every other model in the lab (see
// logistic.js). Its training steps exist to make the lesson visible:
//   "memorise"  the training points are copied into memory;
//   "project"   cosine only: a copy of each point slides out (or in) to a
//               circle around the centre, because only direction counts;
//   "predict"   the plot fills in, a band of rows at a time, as each grid
//               cell asks its neighbours (snapshot.reveal says how much of
//               the plot has been filled so far).

const MEMORISE_STEPS = 20; // steps taken to copy the points into memory
const PROJECT_STEPS = 16;  // steps taken to slide the copies onto the circle
const PREDICT_STEPS = 30;  // steps taken to fill in the plot
const NEARBY = 40;         // how many of the closest points explain() lists
const OUTLINE_POINTS = 96; // points used to draw a neighbourhood's outline
const CIRCLE_RADIUS = 0.92; // the circle that cosine distance slides points onto
const FAR = 4;             // further than any corner of the plot

// The power p of each member of the Minkowski family.
const POWERS = { manhattan: 1, euclidean: 2, chebyshev: Infinity };

// The "length" of the step (dx, dy) under a Minkowski power.
function length(dx, dy, p) {
  const a = Math.abs(dx);
  const b = Math.abs(dy);
  if (p === Infinity) return Math.max(a, b);
  if (p === 1) return a + b;
  if (p === 2) return Math.hypot(a, b);
  return (a ** p + b ** p) ** (1 / p);
}

// Cosine distance: 1 minus the cosine of the angle between the two points,
// seen from the centre. 0 means the same direction, 1 a right angle, 2
// opposite. A point exactly at the centre has no direction, so it is
// treated as being at a right angle to everything.
function cosineDistance(a, b) {
  const sizes = Math.hypot(a[0], a[1]) * Math.hypot(b[0], b[1]);
  if (sizes === 0) return 1;
  const cosine = (a[0] * b[0] + a[1] * b[1]) / sizes;
  return 1 - Math.max(-1, Math.min(1, cosine));
}

// Returns the distance function for a setting.
export function distanceFor(metric = 'euclidean', p = 2) {
  if (metric === 'cosine') return cosineDistance;
  const power = metric === 'minkowski' ? p : POWERS[metric];
  return (a, b) => length(a[0] - b[0], a[1] - b[1], power);
}

// The edge of the neighbourhood around `centre`: every place that is exactly
// `radius` away. A circle for Euclidean, a diamond for Manhattan, a square
// for Chebyshev, and a wedge spreading out from the centre of the plot for
// cosine. Returns a closed list of points, or null if there is no edge to
// draw.
export function neighbourhoodOutline(centre, radius, metric = 'euclidean', p = 2) {
  if (metric === 'cosine') {
    if (radius >= 2 || Math.hypot(centre[0], centre[1]) === 0) return null; // everything is inside
    const direction = Math.atan2(centre[1], centre[0]);
    const spread = Math.acos(1 - radius); // the widest angle that still counts
    const points = [[0, 0]];
    for (let i = 0; i <= 24; i++) {
      const angle = direction - spread + (2 * spread * i) / 24;
      points.push([FAR * Math.cos(angle), FAR * Math.sin(angle)]);
    }
    return points;
  }
  const power = metric === 'minkowski' ? p : POWERS[metric];
  const points = [];
  for (let i = 0; i < OUTLINE_POINTS; i++) {
    const angle = (2 * Math.PI * i) / OUTLINE_POINTS;
    const dx = Math.cos(angle);
    const dy = Math.sin(angle);
    const scale = radius / length(dx, dy, power);
    points.push([centre[0] + dx * scale, centre[1] + dy * scale]);
  }
  return points;
}

// Where a point lands when it is slid onto the circle around the centre.
function onCircle(x) {
  const size = Math.hypot(x[0], x[1]);
  if (size === 0) return [0, 0];
  return [(x[0] / size) * CIRCLE_RADIUS, (x[1] / size) * CIRCLE_RADIUS];
}

export function createKnn({
  k = 5, weighted = false, metric = 'euclidean', p = 2,
} = {}) {
  const distanceBetween = distanceFor(metric, p);
  const cosine = metric === 'cosine';

  let memory = [];        // the stored training points: { x: [x1, x2], y }
  let total = 0;          // how many points there are to store
  let phase = 'idle';     // 'idle', 'memorise', 'project', 'predict' or 'done'
  let reveal = 0;         // share of the plot filled in so far
  let slide = 0;          // cosine only: how far the copies have slid, 0 to 1
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
    slide = 0;
    recent = [];
    finished = total === 0;
    phase = finished ? 'done' : 'memorise';
    const batch = Math.max(1, Math.ceil(total / MEMORISE_STEPS));
    let slideSteps = 0;
    let rowsDone = 0;

    const snapshot = () => ({ phase, stored: memory.length, total, reveal, done: finished });

    function step() {
      if (finished) return { done: true, snapshot: snapshot() };
      recent = [];
      if (phase === 'memorise') {
        const end = Math.min(total, memory.length + batch);
        for (let i = memory.length; i < end; i++) {
          memory.push({ x: X[i], y: y[i] });
          recent.push(X[i]);
        }
        if (memory.length === total) phase = cosine ? 'project' : 'predict';
      } else if (phase === 'project') {
        slideSteps += 1;
        slide = slideSteps / PROJECT_STEPS;
        if (slideSteps >= PROJECT_STEPS) phase = 'predict';
      } else {
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
      slide = 1;
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

    // Cosine: a copy of every stored point, part or all of the way to the circle.
    if (cosine && slide > 0) {
      const circle = [];
      for (let i = 0; i < OUTLINE_POINTS; i++) {
        const angle = (2 * Math.PI * i) / OUTLINE_POINTS;
        circle.push([CIRCLE_RADIUS * Math.cos(angle), CIRCLE_RADIUS * Math.sin(angle)]);
      }
      overlay.push({ kind: 'path', points: circle, closed: true, style: 'guide' });
      overlay.push({
        kind: 'marks',
        points: memory.map((item) => {
          const to = onCircle(item.x);
          return [item.x[0] + (to[0] - item.x[0]) * slide, item.x[1] + (to[1] - item.x[1]) * slide, item.y];
        }),
      });
    }

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
      const outline = neighbourhoodOutline(point, radius, metric, p);
      if (outline) overlay.push({ kind: 'path', points: outline, closed: true, style: 'reach' });
    }

    return {
      model: 'knn',
      k,
      weighted,
      metric,
      power: p,
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
