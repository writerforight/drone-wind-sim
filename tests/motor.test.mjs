import { quadX450 } from '../src/config.js';
import { createMixer } from '../src/motor/mixer.js';
import { createMotors, rotorWrench, stepMotors } from '../src/motor/motors.js';
import { createState, step } from '../src/physics/rigidbody.js';
import { norm, sub } from '../src/physics/math3.js';
import { close, ok } from './assert.mjs';

const M = quadX450.motor, P = quadX450.physics;

export const tests = {
  'first-order lag: after one time constant ω reaches 1 − 1/e of the step'() {
    let m = createMotors(M, 0);
    const dt = P.dt, n = Math.round(M.tau / dt);
    for (let i = 0; i < n; i++) m = stepMotors(m, [800, 800, 800, 800], M, dt);
    close(m.omega[0], 800 * (1 - Math.exp(-1)), 1e-9, 'ω(τ)');
  },

  'commands above the limit saturate at ω_max'() {
    let m = createMotors(M, 0);
    for (let i = 0; i < 2000; i++) m = stepMotors(m, [5000, -50, 600, 600], M, P.dt);
    close(m.omega[0], M.omegaMax, 1e-6, 'ω_max'); close(m.omega[1], M.omegaMin, 1e-6, 'ω_min');
  },

  'mixer inverts the allocation: produced wrench = requested wrench (no saturation)'() {
    const mixer = createMixer(M);
    for (const wish of [{ thrust: 12, torque: [0.2, -0.1, 0.03] }, { thrust: 8, torque: [-0.3, 0.25, -0.05] }]) {
      const r = mixer.mix(wish);
      ok(!r.saturated, 'unexpected saturation');
      const got = rotorWrench(r.omegaCmd, M);
      close(got.force[2], wish.thrust, 1e-9, 'thrust');
      ok(norm(sub(got.torque, wish.torque)) < 1e-9, `torque ${got.torque} vs ${wish.torque}`);
    }
  },

  'hover: equal speeds ω = √(m g / 4 k_T), and a pure yaw torque leaves thrust unchanged'() {
    const mixer = createMixer(M), W = P.mass * P.gravity;
    const h = mixer.mix({ thrust: W, torque: [0, 0, 0] });
    h.omegaCmd.forEach((w) => close(w, Math.sqrt(W / (4 * M.kThrust)), 1e-9, 'hover ω'));
    const y = mixer.mix({ thrust: W, torque: [0, 0, 0.05] });
    close(y.thrusts.reduce((a, b) => a + b), W, 1e-9, 'Σ T under yaw');
    const T = y.thrusts;                               // diagonals (same spin) move together, the other pair opposite
    ok(Math.abs(T[0] - T[1]) < 1e-12 && Math.abs(T[2] - T[3]) < 1e-12 && Math.abs(T[0] - T[2]) > 1e-3, `yaw split ${T}`);
  },

  'saturation is reported when a request exceeds the rotors'() {
    const r = createMixer(M).mix({ thrust: 40, torque: [0, 0, 0] });
    ok(r.saturated, 'not reported'); r.omegaCmd.forEach((w) => close(w, M.omegaMax, 1e-9, 'clipped ω'));
  },

  'open loop: rotors spinning at hover speed hold a level drone (motors + physics together)'() {
    const mixer = createMixer(M), cmd = mixer.mix({ thrust: P.mass * P.gravity, torque: [0, 0, 0] }).omegaCmd;
    let m = createMotors(M, cmd[0]), s = createState({ position: [0, 0, 3] });
    for (let i = 0; i < 1000; i++) { m = stepMotors(m, cmd, M, P.dt); s = step(s, rotorWrench(m.omega, M), P); }
    ok(norm(sub(s.p, [0, 0, 3])) < 1e-9 && norm(s.w) < 1e-9, `drifted: p=${s.p} w=${s.w}`);
  },
};
