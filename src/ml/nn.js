/**
 * ml/nn — forward pass of a small trained network, written out by hand (no ML library in the browser).
 *
 * A policy file (JSON, written by train/quadlab.py: export_policy) looks like
 *
 *   { "format": "quadlab-policy/1", "name": "...",
 *     "level": "wrench", "obs": "token" | "window",
 *     "ml": { ...the config.ml settings it was trained with (policyDt, history, obsScale, action) },
 *     "layers": [ { "type": "lstm",  "weight_ih": [[...]], "weight_hh": [[...]], "bias_ih": [...], "bias_hh": [...] },
 *                 { "type": "gru",   ...same names },
 *                 { "type": "dense", "weight": [[...]], "bias": [...], "activation": "tanh" | "relu" | "none" } ] }
 *
 * Weight layouts are PyTorch's (nn.LSTM gate order i, f, g, o; nn.GRU gate order r, z, n; nn.Linear
 * weight [out][in]), so a state_dict exports as is.
 *
 *   obs 'token'   one token per call; recurrent layers keep their state between calls (reset() clears it)
 *   obs 'window'  the whole history window per call; recurrent layers run over it from a zero state, and the
 *                 last output goes on to the dense layers (no state is kept between calls)
 *
 *   const policy = loadPolicy(json);  policy.reset();  const a = policy.act(obs);   // a ∈ [−1, 1]⁴
 */
const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const ACT = { tanh: Math.tanh, relu: (x) => (x > 0 ? x : 0), none: (x) => x };

/** y = W x + b for W given as rows. */
function affine(W, b, x, out = new Float64Array(W.length)) {
  for (let i = 0; i < W.length; i++) {
    const row = W[i];
    let s = b ? b[i] : 0;
    for (let j = 0; j < row.length; j++) s += row[j] * x[j];
    out[i] = s;
  }
  return out;
}

function lstmCell(L) {
  const H = L.weight_hh[0].length;
  return {
    hidden: H,
    zero: () => ({ h: new Float64Array(H), c: new Float64Array(H) }),
    step(x, s) {
      const gi = affine(L.weight_ih, L.bias_ih, x), gh = affine(L.weight_hh, L.bias_hh, s.h);
      const h = new Float64Array(H), c = new Float64Array(H);
      for (let k = 0; k < H; k++) {
        const i = sigmoid(gi[k] + gh[k]), f = sigmoid(gi[H + k] + gh[H + k]);
        const g = Math.tanh(gi[2 * H + k] + gh[2 * H + k]), o = sigmoid(gi[3 * H + k] + gh[3 * H + k]);
        c[k] = f * s.c[k] + i * g;
        h[k] = o * Math.tanh(c[k]);
      }
      return { h, c };
    },
  };
}

function gruCell(L) {
  const H = L.weight_hh[0].length;
  return {
    hidden: H,
    zero: () => ({ h: new Float64Array(H) }),
    step(x, s) {
      const gi = affine(L.weight_ih, L.bias_ih, x), gh = affine(L.weight_hh, L.bias_hh, s.h);
      const h = new Float64Array(H);
      for (let k = 0; k < H; k++) {
        const r = sigmoid(gi[k] + gh[k]), z = sigmoid(gi[H + k] + gh[H + k]);
        const n = Math.tanh(gi[2 * H + k] + r * gh[2 * H + k]);
        h[k] = (1 - z) * n + z * s.h[k];
      }
      return { h };
    },
  };
}

export function loadPolicy(json) {
  const spec = typeof json === 'string' ? JSON.parse(json) : json;
  if (spec.format !== 'quadlab-policy/1') throw new Error(`not a quadlab policy file (format: ${spec.format})`);
  const obsMode = spec.obs || 'token';
  const layers = spec.layers.map((L) => {
    if (L.type === 'lstm') return { type: 'rnn', cell: lstmCell(L) };
    if (L.type === 'gru') return { type: 'rnn', cell: gruCell(L) };
    if (L.type === 'dense') {
      if (!ACT[L.activation || 'none']) throw new Error(`unknown activation ${L.activation}`);
      return { type: 'dense', L, f: ACT[L.activation || 'none'] };
    }
    throw new Error(`unknown layer type ${L.type}`);
  });
  const inputSize = spec.layers[0].weight_ih ? spec.layers[0].weight_ih[0].length : spec.layers[0].weight[0].length;
  let state = [];

  const dense = (l, x) => { const y = affine(l.L.weight, l.L.bias, x); for (let i = 0; i < y.length; i++) y[i] = l.f(y[i]); return y; };

  /** Run every layer on a sequence of inputs (rows); `keep` = continue from and keep the recurrent state. */
  function forward(seq, keep) {
    let xs = seq;
    layers.forEach((l, i) => {
      if (l.type === 'dense') { xs = xs.map((x) => dense(l, x)); return; }
      let s = keep && state[i] ? state[i] : l.cell.zero();
      xs = xs.map((x) => { s = l.cell.step(x, s); return s.h; });
      if (keep) state[i] = s;
    });
    return xs[xs.length - 1];
  }

  return {
    name: spec.name || 'policy', level: spec.level, obs: obsMode, ml: spec.ml || {}, inputSize, spec,
    reset() { state = []; },
    /** obs: one token ('token') or the flat window ('window') → action, clipped to [−1, 1]. */
    act(obs) {
      let y;
      if (obsMode === 'token') y = forward([obs], true);
      else if (layers[0].type === 'rnn') {
        const rows = [];
        for (let k = 0; k < obs.length; k += inputSize) rows.push(obs.subarray ? obs.subarray(k, k + inputSize) : obs.slice(k, k + inputSize));
        y = forward(rows, false);
      } else y = forward([obs], false);
      return Array.from(y, (v) => Math.max(-1, Math.min(1, v)));
    },
  };
}
