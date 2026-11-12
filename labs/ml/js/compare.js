// Train every model on the same data, without animation, so they can be
// compared side by side. Pure code: no DOM access.
//
// It works through the list in registry.js and the common model interface,
// so a new model joins the comparison automatically.

// A model's settings as a short line of text, for example
// "Feature degree 2, Learning rate 3, Regularisation strength off".
export function describeSettings(spec, params) {
  return spec.params
    .map((def) => `${def.label} ${(def.format || String)(params[def.key])}`)
    .join(', ');
}

// `models` is the registry; `params` holds each model's current settings;
// `data` is { train: { X, y }, test: { X, y } }; `now` returns the time in
// milliseconds. Returns one entry per model:
//   { key, label, settings, ms, scores: { train, test } }
// where each of scores.train and scores.test is { probabilities, labels },
// ready for metrics.js to turn into accuracy at any threshold.
export function trainAll(models, params, seed, data, now = () => performance.now()) {
  return Object.entries(models).map(([key, spec]) => {
    const model = spec.create({ ...params[key], seed });
    const started = now();
    model.train(data.train.X, data.train.y).finish();
    const ms = now() - started;
    const score = (set) => ({
      probabilities: set.X.map((x) => model.predictProba(x)),
      labels: set.y,
    });
    return {
      key,
      label: spec.label,
      settings: describeSettings(spec, params[key]),
      ms,
      scores: { train: score(data.train), test: score(data.test) },
    };
  });
}
