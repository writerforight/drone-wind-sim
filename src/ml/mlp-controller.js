/**
 * ml/mlp-controller — a controller whose decision is an MLP:
 *
 *   observation x (chosen input groups, scaled) ─▶ MLP (trainable) ─▶ o = [o_T, o_x, o_y, o_z]
 *     ─▶ fixed output layer:  T = m g (1 + o_T),  τ = J ⊙ α_max ⊙ [o_x, o_y, o_z]      (not trained)
 *     ─▶ fixed mixer A⁻¹ (loop/simulation.js, motor/mixer.js): [T, τ] → 4 rotor thrusts  (not trained)
 *
 * The two fixed stages together are one fixed affine layer: o = 0 means "hover thrust, no torque", so an
 * untrained network with small outputs starts near hover instead of near zero thrust. The scaling by m g and by
 * the inertia J keeps the network's outputs O(1) for any airframe.
 *
 * Inputs (all in SI, divided by fixed scales so they are O(1); the boxes in the panel switch groups on / off):
 *   dp     target − position (3), world frame        / 5 m, clipped to ±3
 *   v      velocity (3), world frame                 / 5 m/s
 *   speed  desired speed (1)                         / 5 m/s
 *   down   "down" seen from the body, R(q)ᵀ [0,0,−1] (3): the tilt, without angles
 *   w      body rates (3)                            / 5 rad/s
 */
import { rotateInv } from '../physics/math3.js';
import { MLP } from './mlp.js';

export const INPUT_GROUPS = [
  { id: 'dp', label: 'target − position', n: 3 },
  { id: 'v', label: 'velocity', n: 3 },
  { id: 'speed', label: 'desired speed', n: 1 },
  { id: 'down', label: 'down, in the body frame (tilt)', n: 3 },
  { id: 'w', label: 'body rates', n: 3 },
];
export const OUTPUTS = ['T', 'τx', 'τy', 'τz'];
export const ALPHA_MAX = [30, 30, 10];          // rad/s², angular acceleration for an output of ±1

const clip = (x, a) => Math.max(-a, Math.min(a, x));

/** The default network settings (the panel edits a copy). */
export const defaultNetSettings = () => ({
  hidden: [16, 16], act: 'tanh', optimizer: 'adam', lr: 1e-3, sigma: 0.1, seed: 1,
  inputs: Object.fromEntries(INPUT_GROUPS.map((g) => [g.id, true])),
});

export const inputSize = (inputs) => INPUT_GROUPS.reduce((n, g) => n + (inputs[g.id] ? g.n : 0), 0);

/** The observation vector for a measurement and a reference { position, speed }. */
export function observe(meas, ref, inputs) {
  const x = [];
  if (inputs.dp) for (let k = 0; k < 3; k++) x.push(clip((ref.position[k] - meas.p[k]) / 5, 3));
  if (inputs.v) for (let k = 0; k < 3; k++) x.push(meas.v[k] / 5);
  if (inputs.speed) x.push((ref.speed || 0) / 5);
  if (inputs.down) x.push(...rotateInv(meas.q, [0, 0, -1]));
  if (inputs.w) for (let k = 0; k < 3; k++) x.push(meas.w[k] / 5);
  return x;
}

/** The fixed output layer: network output o (4) → body wrench { thrust, torque }. */
export function outputToWrench(o, airframe) {
  const { mass: m, gravity: g, inertia: J } = airframe;
  return { thrust: Math.max(0, m * g * (1 + o[0])), torque: [0, 1, 2].map((k) => J[k] * ALPHA_MAX[k] * o[k + 1]) };
}

/** Builds the network for the settings: [inputs, …hidden, 4], hidden layers with `act`, a linear last layer. */
export function buildNet(settings) {
  const n = inputSize(settings.inputs);
  if (!n) throw new Error('choose at least one input');
  const dims = [n, ...settings.hidden, OUTPUTS.length];
  const acts = [...settings.hidden.map(() => settings.act), 'identity'];
  const net = new MLP(dims, acts).init('xavier', 1, settings.seed);
  // start near hover: a small last layer makes the initial outputs small (o ≈ 0 means hover thrust)
  const last = net.layout[net.layout.length - 1];
  for (let i = last.w; i < last.b + last.nout; i++) net.theta[i] *= 0.1;
  return net;
}

/**
 * The controller (same interface as the PID: reset(), control(meas, ref, params) → { thrust, torque }).
 * `ctl.noise` (set by the learner) is added to the network output before the fixed layer; `ctl.last` keeps the
 * forward cache, the noise and the output of the latest call, for backprop.
 */
export function createMlpController(airframe, net, inputs) {
  const ctl = {
    name: 'MLP', net, inputs, noise: null, last: null,
    reset() { ctl.last = null; },
    /** What the network would command, without noise and without remembering it. */
    preview(meas, ref) {
      const o = Array.from(net.predict(observe(meas, ref, inputs)));
      return { o, ...outputToWrench(o, airframe) };
    },
    control(meas, ref) {
      const x = observe(meas, ref, inputs), cache = net.forward(x), y = cache.as[cache.as.length - 1];
      const eps = ctl.noise ? ctl.noise() : null;
      const o = Array.from(y, (v, i) => v + (eps ? eps[i] : 0));
      ctl.last = { x, cache, eps, o };
      return outputToWrench(o, airframe);
    },
  };
  return ctl;
}
