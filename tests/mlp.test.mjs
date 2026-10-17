import { ACTIVATION_NAMES, ACTIVATIONS, clipGradient, createOptimizer, MLP } from '../src/ml/mlp.js';
import { createRandom } from '../src/util/random.js';
import { close, ok } from './assert.mjs';

/** L = ½‖y − t‖², so ∂L/∂y = y − t. */
const loss = (net, x, t) => 0.5 * net.predict(x).reduce((s, y, i) => s + (y - t[i]) ** 2, 0);

export const tests = {
  'activation derivatives match finite differences'() {
    for (const name of ACTIVATION_NAMES) {
      const { f, df } = ACTIVATIONS[name];
      for (const z of [-2.3, -0.7, 0.31, 1.9]) close(df(z), (f(z + 1e-6) - f(z - 1e-6)) / 2e-6, 1e-6, `${name}'(${z})`);
    }
  },

  'backprop gradients (weights, biases, input) match finite differences, every activation'() {
    const rng = createRandom(7);
    for (const act of ACTIVATION_NAMES) {
      const net = new MLP([13, 9, 7, 4], [act, act, 'identity']).init('xavier', 1, rng);
      const x = Array.from({ length: 13 }, () => rng.range(-1, 1)), t = [0.3, -0.2, 0.5, 0.1];
      const cache = net.forward(x), y = cache.as.at(-1);
      const grad = new Float64Array(net.nParams);
      const gx = net.backward(cache, y.map((v, i) => v - t[i]), grad);
      const h = 1e-6;
      let worst = 0;
      for (let i = 0; i < net.nParams; i++) {
        const keep = net.theta[i];
        net.theta[i] = keep + h; const lp = loss(net, x, t);
        net.theta[i] = keep - h; const lm = loss(net, x, t);
        net.theta[i] = keep;
        worst = Math.max(worst, Math.abs((lp - lm) / (2 * h) - grad[i]) / Math.max(1, Math.abs(grad[i])));
      }
      ok(worst < 1e-6, `${act}: worst parameter gradient error ${worst.toExponential(2)}`);
      for (let j = 0; j < 13; j++) {
        const xp = x.slice(), xm = x.slice(); xp[j] += h; xm[j] -= h;
        close(gx[j], (loss(net, xp, t) - loss(net, xm, t)) / (2 * h), 1e-6, `${act}: ∂L/∂x${j}`);
      }
    }
  },

  'backward accumulates into grad (two calls give twice the gradient)'() {
    const net = new MLP([3, 4, 2], ['tanh', 'identity']).init('xavier', 1, 3);
    const c = net.forward([0.1, -0.4, 0.8]), g1 = new Float64Array(net.nParams), g2 = new Float64Array(net.nParams);
    net.backward(c, [1, -2], g1);
    net.backward(c, [1, -2], g2); net.backward(c, [1, -2], g2);
    for (let i = 0; i < net.nParams; i++) close(g2[i], 2 * g1[i], 1e-12, `grad[${i}]`);
  },

  'SGD and Adam fit a small regression'() {
    const rng = createRandom(11), data = [];
    for (let i = 0; i < 64; i++) { const x = [rng.range(-1, 1), rng.range(-1, 1)]; data.push({ x, t: [Math.sin(2 * x[0]) * x[1]] }); }
    for (const [opt, lr, steps] of [['sgd', 0.05, 3000], ['adam', 0.01, 1500]]) {
      const net = new MLP([2, 16, 16, 1], ['tanh', 'tanh', 'identity']).init('xavier', 1, 5);
      const mean = () => data.reduce((s, d) => s + loss(net, d.x, d.t), 0) / data.length;
      const before = mean(), o = createOptimizer(opt, net.nParams, lr);
      for (let k = 0; k < steps; k++) {
        const g = new Float64Array(net.nParams);
        for (const d of data) { const c = net.forward(d.x); net.backward(c, [c.as.at(-1)[0] - d.t[0]], g); }
        for (let i = 0; i < g.length; i++) g[i] /= data.length;
        o.step(net.theta, g);
      }
      const after = mean();
      ok(after < 0.1 * before && after < 2e-3, `${opt}: loss ${before.toFixed(4)} → ${after.toFixed(5)}`);
    }
  },

  'gradient clipping caps the norm and keeps the direction'() {
    const g = Float64Array.from([3, 4]);
    close(clipGradient(g, 1), 5, 1e-12, 'norm before');
    close(g[0], 0.6, 1e-12, 'g0'); close(g[1], 0.8, 1e-12, 'g1');
    const small = Float64Array.from([0.1, 0.1]); clipGradient(small, 1);
    ok(small[0] === 0.1, 'a small gradient must stay as it is');
  },
};
