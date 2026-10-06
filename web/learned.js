/* Learned control loop in the browser: the network trained with learn/train.py, exported by
   learn/export_web.py.  Same observation and actuation as learn/loop.py and learn/actuation.py:

     [observation_t, memory_{t-1}]  --MLP-->  [u_t, memory_t]  ->  motor speeds

   The network runs at the rate it was trained at (50 Hz); in between, u is held and only the
   actuation (in 'thrust_rates' mode the inner rate loop) is recomputed with the latest gyro reading. */
(function (root) {
  'use strict';
  const S = root.DroneSim || (typeof require !== 'undefined' ? require('./sim.js') : null);
  const ACT = {
    tanh: Math.tanh, relu: (x) => Math.max(0, x), elu: (x) => (x > 0 ? x : Math.exp(x) - 1), identity: (x) => x,
    sigmoid: (x) => 1 / (1 + Math.exp(-x)), silu: (x) => x / (1 + Math.exp(-x)),
    gelu: (x) => 0.5 * x * (1 + erf(x / Math.SQRT2)),
  };
  function erf(x) {      // Abramowitz-Stegun 7.1.26 (|error| < 1.5e-7)
    const s = Math.sign(x); x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x));
  }
  function linear(L, x) {
    const y = new Array(L.b.length);
    for (let i = 0; i < y.length; i++) { const w = L.W[i]; let s = L.b[i]; for (let j = 0; j < x.length; j++) s += w[j] * x[j]; y[i] = s; }
    return y;
  }

  class LearnedNet {
    constructor(m) {
      this.m = m;
      this.f = ACT[m.activation] || Math.tanh;
      this.fm = ACT[m.memory_activation] || Math.tanh;
    }
    forward(obs, h) {
      let x = obs.concat(h);
      for (const L of this.m.layers) x = linear(L, x).map(this.f);
      const y = linear(this.m.out, x), k = this.m.act_dim;
      return { u: y.slice(0, k).map(Math.tanh), h: y.slice(k).map(this.fm) };
    }
    observation(est, ref, uPrev) {
      const R = S.quatToRot(est.q), rv = ref.v || [0, 0, 0];
      return [0, 1, 2].map((i) => (ref.p[i] - est.p[i]) / 5).concat(est.v.map((x) => x / 5), rv.map((x) => x / 5),
        R[0], R[1], R[2], est.w.map((x) => x / 5), uPrev);
    }
    speeds(u, w) {
      const a = this.m.actuation;
      let T;
      if (this.m.control === 'motors') {
        T = a.t_hover.map((t, i) => t + u[i] * a.margin[i]);
      } else {
        const F = a.mass * 9.80665 * (1 + u[0]);
        const wsp = [0, 1, 2].map((i) => u[i + 1] * a.rate_max[i]);
        const Jv = (v) => [0, 1, 2].map((i) => a.J[i][0] * v[0] + a.J[i][1] * v[1] + a.J[i][2] * v[2]);
        const Jw = Jv(w), e = Jv([0, 1, 2].map((i) => a.rate_kp[i] * (wsp[i] - w[i])));
        const c = [w[1] * Jw[2] - w[2] * Jw[1], w[2] * Jw[0] - w[0] * Jw[2], w[0] * Jw[1] - w[1] * Jw[0]];
        const wrench = [F, e[0] + c[0], e[1] + c[1], e[2] + c[2]];
        T = a.Bp.map((row, i) => Math.min(a.t_max[i], Math.max(a.t_min[i], row[0] * wrench[0] + row[1] * wrench[1] + row[2] * wrench[2] + row[3] * wrench[3])));
      }
      return T.map((t, i) => Math.sqrt(Math.max(t, 1e-6) / a.k[i]));
    }
    /* recompute the exported PyTorch outputs; returns the largest difference */
    selfCheck() {
      let worst = 0;
      for (const c of this.m.checks) {
        const o = this.forward(this.observation({ p: c.p, v: c.v, q: c.q, w: c.w }, { p: c.ref_p, v: c.ref_v }, c.u_prev), c.h);
        const n = this.speeds(o.u, c.w);
        const d = (a, b) => Math.max(...a.map((x, i) => Math.abs(x - b[i])));
        worst = Math.max(worst, d(o.u, c.u), d(o.h, c.h_next), d(n, c.n) / 100);
      }
      return worst;
    }
  }

  /* same interface as CascadedPID: reset(), update(estimate, reference, dt) -> motor speeds */
  class LearnedController {
    constructor(net) { this.net = net; this.reset(); }
    reset() { this.h = new Array(this.net.m.memory).fill(0); this.u = new Array(this.net.m.act_dim).fill(0); this.tNext = 0; this.t = 0; }
    update(est, ref, dt) {
      if (this.t >= this.tNext - 1e-9) {
        const o = this.net.forward(this.net.observation(est, ref, this.u), this.h);
        this.u = o.u; this.h = o.h; this.tNext += this.net.m.ctrl_dt;
      }
      this.t += dt;
      return this.net.speeds(this.u, est.w);
    }
    memoryNorm() { return Math.sqrt(this.h.reduce((s, x) => s + x * x, 0) / this.h.length); }
  }

  const api = { LearnedNet, LearnedController };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.DroneLearned = api;
})(typeof window !== 'undefined' ? window : globalThis);
