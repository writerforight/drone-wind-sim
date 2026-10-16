/**
 * ml/features — what a learned controller sees at one policy step (one "token" of its input sequence),
 * and the window of past tokens a sequence model (LSTM, GRU, TCN, transformer) can be given.
 *
 *   token_t = [ e_t / s_e,  v_t / s_v,  R_t[:, 0], R_t[:, 1],  w_t / s_w,  a_{t−1},  z_t / s_e ]      (20 numbers)
 *
 *   e_t   target − position (world frame): where the target is, seen from the drone. A window of these is the
 *         history of past targets relative to past positions; absolute x, y are left out on purpose, so the
 *         policy does not depend on where in the room it flies (translation invariance).
 *   v_t   velocity (world)              R_t   first two columns of the attitude matrix (continuous, unlike Euler
 *   w_t   body rates                          angles or ±q: the standard choice for learned attitude control)
 *   a_{t−1}  the action applied over the last step: with it, the window holds input → response pairs, which is
 *         what lets a sequence model infer the drone's mass, motor strength or the wind (in-context adaptation)
 *   z_t   height above the floor
 *
 * Every number is divided by its scale (config.ml.obsScale) and clipped to ±5.
 */
import { rotate } from '../physics/math3.js';

export const FEATURES = [
  'ex', 'ey', 'ez', 'vx', 'vy', 'vz', 'r11', 'r21', 'r31', 'r12', 'r22', 'r32',
  'wx', 'wy', 'wz', 'a0_prev', 'a1_prev', 'a2_prev', 'a3_prev', 'z',
];
export const FEATURE_SIZE = FEATURES.length;

const clip5 = (x) => Math.max(-5, Math.min(5, x));

/** One token from a measurement { p, v, q, w }, the target position, the previous action and the scales. */
export function features(meas, target, prevAction, S) {
  const c1 = rotate(meas.q, [1, 0, 0]), c2 = rotate(meas.q, [0, 1, 0]);
  const out = new Float32Array(FEATURE_SIZE);
  let k = 0;
  for (let i = 0; i < 3; i++) out[k++] = clip5((target[i] - meas.p[i]) / S.error);
  for (let i = 0; i < 3; i++) out[k++] = clip5(meas.v[i] / S.velocity);
  for (let i = 0; i < 3; i++) out[k++] = c1[i];
  for (let i = 0; i < 3; i++) out[k++] = c2[i];
  for (let i = 0; i < 3; i++) out[k++] = clip5(meas.w[i] / S.rates);
  for (let i = 0; i < 4; i++) out[k++] = prevAction[i];
  out[k] = clip5(meas.p[2] / S.error);
  return out;
}

/**
 * The last `length` tokens, oldest first. Before the window is full, the missing (oldest) slots are zeros,
 * so the newest token is always the last row: window[(length − 1) · size …].
 */
export function createHistory(length, size) {
  const rows = [];
  return {
    length, size,
    clear() { rows.length = 0; },
    push(token) { rows.push(token); if (rows.length > length) rows.shift(); },
    get filled() { return rows.length; },
    window() {
      const out = new Float32Array(length * size), pad = length - rows.length;
      rows.forEach((r, i) => out.set(r, (pad + i) * size));
      return out;
    },
  };
}
