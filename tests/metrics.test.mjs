import { stepMetrics } from '../src/modes/tuning-lab.js';
import { close, ok } from './assert.mjs';

const sample = (f, T = 8, dt = 0.004) => { const s = []; for (let t = 0; t <= T; t += dt) s.push({ t, y: f(t) }); return s; };

export const tests = {
  'step metrics of a first-order response: rise ln 9 · τ, settling ln 50 · τ, no overshoot'() {
    const m = stepMetrics(sample((t) => 1 - Math.exp(-t)), 0, 0, 1);
    close(m.rise, Math.log(9), 0.01, 'rise'); close(m.settling, Math.log(50), 0.01, 'settling'); close(m.overshoot, 0, 1e-9, 'overshoot');
  },
  'step metrics of an underdamped 2nd-order response: overshoot e^(−πζ/√(1−ζ²))'() {
    const wn = 2, z = 0.4, wd = wn * Math.sqrt(1 - z * z);
    const y = (t) => 1 - Math.exp(-z * wn * t) * (Math.cos(wd * t) + (z / Math.sqrt(1 - z * z)) * Math.sin(wd * t));
    const m = stepMetrics(sample(y, 12), 0, 0, 1);
    close(m.overshoot, 100 * Math.exp((-Math.PI * z) / Math.sqrt(1 - z * z)), 0.1, 'overshoot %');
    ok(m.steadyError < 1e-3, `steady error ${m.steadyError}`);
  },
};
