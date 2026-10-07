import { quadX450 } from '../src/config.js';
import { createCascadedPID } from '../src/controller/cascaded-pid.js';
import { runScenario } from '../src/loop/simulation.js';
import { fromEuler } from '../src/physics/math3.js';
import { close, ok } from './assert.mjs';

const clone = (o) => JSON.parse(JSON.stringify(o));
const pid = (cfg) => createCascadedPID(cfg.physics);
const deg = Math.PI / 180;

/** Unit step response of ÿ + 2ζω ẏ + ω² y = ω² (y(0) = ẏ(0) = 0), underdamped or critical/overdamped. */
function secondOrderStep(t, wn, zeta) {
  if (zeta < 1) {
    const wd = wn * Math.sqrt(1 - zeta * zeta);
    return 1 - Math.exp(-zeta * wn * t) * (Math.cos(wd * t) + (zeta / Math.sqrt(1 - zeta * zeta)) * Math.sin(wd * t));
  }
  if (zeta === 1) return 1 - Math.exp(-wn * t) * (1 + wn * t);
  const r = Math.sqrt(zeta * zeta - 1), s1 = -wn * (zeta - r), s2 = -wn * (zeta + r);
  return 1 + (s2 * Math.exp(s1 * t) - s1 * Math.exp(s2 * t)) / (s1 - s2);
}

export const tests = {
  'hover is stable: from a 0.5 m offset and 10° tilt back to the setpoint'() {
    const cfg = quadX450;
    const { log } = runScenario(cfg, pid(cfg), {
      seconds: 10, initial: { position: [0.5, -0.3, 1.6], attitude: fromEuler(10 * deg, -8 * deg, 0.4) },
      refAt: () => ({ position: [0, 0, 2], yaw: 0 }),
    });
    const end = log[log.length - 1];
    const err = Math.hypot(end.p[0], end.p[1], end.p[2] - 2);
    ok(err < 0.01, `position error after 10 s: ${err.toFixed(4)} m`);
    ok(Math.abs(end.euler[0]) < 0.5 * deg && Math.abs(end.euler[1]) < 0.5 * deg && Math.abs(end.euler[2]) < 0.5 * deg, `attitude ${end.euler}`);
    ok(!end.saturated, 'motors saturated at rest');
  },

  'hover holds for 30 s with no drift (no disturbance)'() {
    const cfg = quadX450;
    const { log } = runScenario(cfg, pid(cfg), { seconds: 30, initial: { position: [0, 0, 2] }, refAt: () => ({ position: [0, 0, 2] }) });
    const worst = Math.max(...log.map((r) => Math.hypot(r.p[0], r.p[1], r.p[2] - 2)));
    ok(worst < 1e-6, `largest deviation ${worst}`);
  },

  'altitude step matches the analytic 2nd-order response (fast motors, no drag, no integral)'() {
    // With ideal motors and no drag the vertical loop is z̈ = Kp (z_ref − z) − Kd ż  →  ω_n = √Kp, ζ = Kd / (2√Kp).
    const cfg = clone(quadX450);
    cfg.motor.tau = 1e-5; cfg.physics.linearDrag = 0; cfg.controller.position.ki = [0, 0, 0];
    const { kp, kd } = cfg.controller.position, wn = Math.sqrt(kp[2]), zeta = kd[2] / (2 * wn);
    const { log } = runScenario(cfg, pid(cfg), { seconds: 6, initial: { position: [0, 0, 1] }, refAt: () => ({ position: [0, 0, 2] }) });
    let worst = 0;
    for (const r of log) worst = Math.max(worst, Math.abs((r.p[2] - 1) - secondOrderStep(r.t, wn, zeta)));
    ok(worst < 0.02, `largest gap to the analytic response: ${worst.toFixed(4)} m (ω_n ${wn.toFixed(2)}, ζ ${zeta.toFixed(2)})`);
  },

  'angle mode follows a commanded tilt'() {
    const cfg = quadX450, ctl = pid(cfg);
    const { log } = runScenario(cfg, ctl, {
      seconds: 2, initial: { position: [0, 0, 5] },
      refAt: () => ({ attitude: [10 * deg, -5 * deg], yawRate: 0, thrust: cfg.physics.mass * cfg.physics.gravity }),
    });
    const e = log[log.length - 1].euler;
    close(e[0], 10 * deg, 0.3 * deg, 'roll'); close(e[1], -5 * deg, 0.3 * deg, 'pitch');
  },
};
