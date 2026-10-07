/**
 * loop/ — connects the blocks with negative feedback, independent of any rendering:
 *
 *   reference ─▶(+)─▶ controller ─▶ mixer ─▶ motors ─▶ physics ─▶ state
 *               ▲(−)                                               │
 *               └──────────────── measure (sensors/) ◀─────────────┘
 *
 * Physics always runs at its fixed step (config.physics.dt). The controller runs every
 * round(controller.dt / physics.dt) physics steps and its output is held in between (zero-order hold).
 * A renderer or a UI only reads `sim.state` and calls `sim.advance(seconds)`; it never changes the step.
 */
import { createMixer } from '../motor/mixer.js';
import { createMotors, rotorWrench, stepMotors } from '../motor/motors.js';
import { createState, step } from '../physics/rigidbody.js';
import { toEuler } from '../physics/math3.js';

/** Ideal measurement: the true state (sensors/ replaces this with noise, bias and delay). */
export const perfectSensor = (state) => ({ p: state.p.slice(), v: state.v.slice(), q: state.q.slice(), w: state.w.slice() });

export function createSimulation(cfg, { controller, measure = perfectSensor, initial = {} } = {}) {
  const P = cfg.physics, M = cfg.motor, C = cfg.controller;
  const mixer = createMixer(M);
  const hoverOmega = Math.sqrt((P.mass * P.gravity) / (M.rotors.length * M.kThrust));
  const ratio = Math.max(1, Math.round(C.dt / P.dt));
  const sim = {
    t: 0, state: createState(initial), motors: createMotors(M, hoverOmega),
    reference: { position: (initial.position || [0, 0, 0]).slice(), yaw: 0 },
    disturbance: { extForce: [0, 0, 0], wind: [0, 0, 0] },
    command: mixer.mix({ thrust: P.mass * P.gravity, torque: [0, 0, 0] }),
    last: null, steps: 0,

    /** One controller period: measure, control, mix, then `ratio` physics + motor steps. */
    tick() {
      const meas = measure(sim.state, sim.t);
      const u = controller.control(meas, sim.reference, C);
      sim.command = mixer.mix(u);
      for (let i = 0; i < ratio; i++) {
        sim.motors = stepMotors(sim.motors, sim.command.omegaCmd, M, P.dt);
        const w = rotorWrench(sim.motors.omega, M);
        sim.state = step(sim.state, { ...w, ...sim.disturbance }, P);
        sim.t += P.dt;
      }
      sim.last = { t: sim.t, u, meas };
      sim.steps++;
    },

    /** Advance by (at least) `seconds`, calling onTick(sim) after every controller period. */
    advance(seconds, onTick) {
      const n = Math.max(1, Math.round(seconds / (ratio * P.dt)));
      for (let i = 0; i < n; i++) { sim.tick(); if (onTick) onTick(sim); }
    },
  };
  controller.reset();
  return sim;
}

/** Run a scenario and record a log of rows (for tests and plots). refAt(t) may change the reference. */
export function runScenario(cfg, controller, { seconds, initial, refAt, every = 1 } = {}) {
  const sim = createSimulation(cfg, { controller, initial });
  const log = [];
  sim.advance(seconds, (s) => {
    if (refAt) Object.assign(s.reference, refAt(s.t));
    if (s.steps % every) return;
    log.push({ t: s.t, p: s.state.p.slice(), v: s.state.v.slice(), euler: toEuler(s.state.q), w: s.state.w.slice(),
      ref: s.reference.position.slice(), omega: s.motors.omega.slice(), saturated: s.command.saturated });
  });
  return { sim, log };
}
