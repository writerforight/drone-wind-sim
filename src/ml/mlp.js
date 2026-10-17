/**
 * ml/mlp — a small multi-layer perceptron with backpropagation, SGD and Adam. Ported from the MLP of
 * "Neural Nets Space Bending" (mlp-space-deformation/src/nn.js); only what a controller needs is kept.
 *
 * Layer l maps a_{l−1} (size dims[l]) to
 *   z_l = W_l a_{l−1} + b_l        (linear)
 *   a_l = act_l(z_l)               (activation)
 * All parameters live in one Float64Array `theta`; for layer l, W_l (out × in, row-major) starts at
 * layout[l].w and b_l follows at layout[l].b. No dependencies: runs in the browser and in Node.
 */
import { createRandom } from '../util/random.js';

const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);

/** Activations: value f(z) and derivative f′(z). */
export const ACTIVATIONS = {
  identity: { f: (z) => z, df: () => 1 },
  relu: { f: (z) => (z > 0 ? z : 0), df: (z) => (z > 0 ? 1 : 0) },
  tanh: { f: Math.tanh, df: (z) => { const t = Math.tanh(z); return 1 - t * t; } },
  sigmoid: { f: (z) => 1 / (1 + Math.exp(-z)), df: (z) => { const s = 1 / (1 + Math.exp(-z)); return s * (1 - s); } },
  // GELU, tanh approximation: 0.5 z (1 + tanh(√(2/π)(z + 0.044715 z³)))
  gelu: {
    f: (z) => 0.5 * z * (1 + Math.tanh(SQRT_2_OVER_PI * (z + 0.044715 * z * z * z))),
    df: (z) => {
      const u = SQRT_2_OVER_PI * (z + 0.044715 * z * z * z), t = Math.tanh(u);
      return 0.5 * (1 + t) + 0.5 * z * (1 - t * t) * SQRT_2_OVER_PI * (1 + 3 * 0.044715 * z * z);
    },
  },
  sin: { f: Math.sin, df: Math.cos },
};
export const ACTIVATION_NAMES = Object.keys(ACTIVATIONS);

export class MLP {
  /** dims = [in, hidden…, out]; acts = one activation name per layer. */
  constructor(dims, acts) {
    if (acts.length !== dims.length - 1) throw new Error('need one activation per layer');
    for (const a of acts) if (!ACTIVATIONS[a]) throw new Error(`unknown activation ${a}`);
    this.dims = dims.slice();
    this.acts = acts.slice();
    this.layout = [];
    let off = 0;
    for (let l = 0; l < dims.length - 1; l++) {
      const nin = dims[l], nout = dims[l + 1];
      this.layout.push({ nin, nout, w: off, b: off + nin * nout });
      off += nin * nout + nout;
    }
    this.nParams = off;
    this.theta = new Float64Array(off);
  }

  get inDim() { return this.dims[0]; }
  get outDim() { return this.dims[this.dims.length - 1]; }

  clone() {
    const m = new MLP(this.dims, this.acts);
    m.theta.set(this.theta);
    return m;
  }

  /**
   * Random initialisation. dist: 'xavier' N(0, s²·2/(in+out)) | 'he' N(0, s²·2/in) | 'normal' N(0, s²).
   * Biases are drawn the same way, scaled by 0.25. `rng` is a createRandom() (or a seed).
   */
  init(dist = 'xavier', scale = 1, rng = 1) {
    const r = typeof rng === 'number' ? createRandom(rng) : rng;
    for (const L of this.layout) {
      const std = dist === 'he' ? scale * Math.sqrt(2 / L.nin) : dist === 'normal' ? scale : scale * Math.sqrt(2 / (L.nin + L.nout));
      for (let i = 0; i < L.nin * L.nout; i++) this.theta[L.w + i] = std * r.gauss();
      for (let i = 0; i < L.nout; i++) this.theta[L.b + i] = 0.25 * std * r.gauss();
    }
    return this;
  }

  /** Forward pass keeping every intermediate value (backprop needs them): { zs, as }, as[0] = x. */
  forward(x) {
    const zs = [], as = [Float64Array.from(x)];
    let a = as[0];
    for (let l = 0; l < this.layout.length; l++) {
      const L = this.layout[l], th = this.theta, act = ACTIVATIONS[this.acts[l]];
      const z = new Float64Array(L.nout), out = new Float64Array(L.nout);
      for (let i = 0; i < L.nout; i++) {
        let s = th[L.b + i];
        const row = L.w + i * L.nin;
        for (let j = 0; j < L.nin; j++) s += th[row + j] * a[j];
        z[i] = s;
        out[i] = act.f(s);
      }
      zs.push(z);
      as.push(out);
      a = out;
    }
    return { zs, as };
  }

  predict(x) {
    const c = this.forward(x);
    return c.as[c.as.length - 1];
  }

  /**
   * Backpropagation. Given the forward cache and gOut = ∂L/∂(output), ACCUMULATES ∂L/∂θ into `grad`
   * (Float64Array of length nParams, may be null) and returns ∂L/∂x.
   */
  backward(cache, gOut, grad) {
    const th = this.theta;
    let gA = Float64Array.from(gOut);
    for (let l = this.layout.length - 1; l >= 0; l--) {
      const L = this.layout[l], act = ACTIVATIONS[this.acts[l]];
      const z = cache.zs[l], aPrev = cache.as[l];
      const gZ = new Float64Array(L.nout);
      for (let i = 0; i < L.nout; i++) gZ[i] = gA[i] * act.df(z[i]);   // through the activation
      const gPrev = new Float64Array(L.nin);
      for (let i = 0; i < L.nout; i++) {
        const row = L.w + i * L.nin, g = gZ[i];
        if (grad) {
          grad[L.b + i] += g;                                            // ∂L/∂b_i = gZ_i
          for (let j = 0; j < L.nin; j++) grad[row + j] += g * aPrev[j]; // ∂L/∂W_ij = gZ_i a_j
        }
        for (let j = 0; j < L.nin; j++) gPrev[j] += th[row + j] * g;     // ∂L/∂a_prev = Wᵀ gZ
      }
      gA = gPrev;
    }
    return gA;
  }
}

/** Plain gradient descent: θ ← θ − lr g. */
export class SGD {
  constructor(n, lr) { this.lr = lr; }
  step(theta, grad) { for (let i = 0; i < theta.length; i++) theta[i] -= this.lr * grad[i]; }
}

/** Adam (Kingma & Ba): per-parameter step sizes from running gradient moments. */
export class Adam {
  constructor(n, lr, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
    Object.assign(this, { lr, b1, b2, eps, t: 0 });
    this.m = new Float64Array(n);
    this.v = new Float64Array(n);
  }
  step(theta, grad) {
    this.t++;
    const { b1, b2 } = this, c1 = 1 - b1 ** this.t, c2 = 1 - b2 ** this.t;
    for (let i = 0; i < theta.length; i++) {
      this.m[i] = b1 * this.m[i] + (1 - b1) * grad[i];
      this.v[i] = b2 * this.v[i] + (1 - b2) * grad[i] * grad[i];
      theta[i] -= (this.lr * (this.m[i] / c1)) / (Math.sqrt(this.v[i] / c2) + this.eps);
    }
  }
}

export const createOptimizer = (name, n, lr) => (name === 'sgd' ? new SGD(n, lr) : new Adam(n, lr));

/** Scales `grad` in place so its norm is at most `maxNorm`; returns the norm before clipping. */
export function clipGradient(grad, maxNorm) {
  let s = 0;
  for (let i = 0; i < grad.length; i++) s += grad[i] * grad[i];
  const n = Math.sqrt(s);
  if (n > maxNorm) for (let i = 0; i < grad.length; i++) grad[i] *= maxNorm / n;
  return n;
}
