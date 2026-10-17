import { quadX450 } from '../src/config.js';
import { createCascadedPID } from '../src/controller/cascaded-pid.js';
import { createSimulation } from '../src/loop/simulation.js';
import { carrot, motorsOff, movePin, PIN_LIMITS } from '../src/modes/mlp-lab.js';
import { close, ok } from './assert.mjs';

const still = { roll: 0, pitch: 0, yaw: 0, throttle: 0 };

export const tests = {
  'the pin moves away from the camera on "forward" and to its right on "right"'() {
    // camera east of its target (yaw 0): forward is west (−x), right is north (+y)
    const f = movePin([0, 0, 2], { ...still, pitch: 1 }, 0, 1, 1);
    close(f[0], -1, 1e-12, 'forward x'); close(f[1], 0, 1e-12, 'forward y');
    const r = movePin([0, 0, 2], { ...still, roll: 1 }, 0, 1, 1);
    close(r[0], 0, 1e-12, 'right x'); close(r[1], 1, 1e-12, 'right y');
    // camera north of its target (yaw 90°): forward is south
    const f2 = movePin([0, 0, 2], { ...still, pitch: 1 }, Math.PI / 2, 1, 1);
    close(f2[1], -1, 1e-12, 'forward y at yaw 90°');
    const up = movePin([0, 0, 2], { ...still, throttle: 1 }, 0.7, 0.5, 2);
    close(up[2], 3, 1e-12, 'up');
  },

  'the pin stays inside its limits'() {
    const low = movePin([0, 0, 0.4], { ...still, throttle: -1 }, 0, 10);
    close(low[2], PIN_LIMITS.zMin, 1e-12, 'floor limit');
    const far = movePin([39, 0, 2], { ...still, pitch: -1 }, 0, 10);
    close(Math.hypot(far[0], far[1]), PIN_LIMITS.r, 1e-9, 'radius limit');
  },

  'the carrot walks at the desired speed and stops on the pin'() {
    const c = carrot([0, 0, 0], [3, 4, 0], 2, 0.5);
    close(Math.hypot(...c.position), 1, 1e-12, 'distance walked');
    close(Math.hypot(...c.velocity), 2, 1e-12, 'feed-forward speed');
    const end = carrot([2.95, 3.95, 0], [3, 4, 0], 2, 0.5);
    ok(end.position[0] === 3 && end.position[1] === 4 && Math.hypot(...end.velocity) === 0, 'did not stop on the pin');
  },

  'with motors off the main drone rests on the floor'() {
    const sim = createSimulation(quadX450, { controller: motorsOff, initial: { position: [0, 0, 0] } });
    sim.advance(2);
    close(sim.state.p[2], 0, 1e-12, 'height');
  },

  'the ghost PID flies from the floor to the pin at about the desired speed'() {
    // a far pin, so there is a cruise phase after the take-off transient (the PID first catches up with the carrot)
    const cfg = quadX450, v = 1.5, pin = [8, 6, 3], dt = cfg.controller.dt;
    const g = createSimulation(cfg, { controller: createCascadedPID(cfg.physics), initial: { position: [0, 0, 0] } });
    g.reference = { position: [0, 0, 0], velocity: [0, 0, 0], yaw: 0 };
    const D = Math.hypot(...pin), cruise = [];
    for (let i = 0; i < Math.round((D / v + 8) / dt); i++) {
      const c = carrot(g.reference.position, pin, v, dt);
      g.reference = { position: c.position, velocity: c.velocity, yaw: 0 };
      g.tick();
      if (g.t > 3 && g.t < D / v - 0.5) cruise.push(Math.hypot(...g.state.v));
    }
    const err = Math.hypot(...pin.map((x, k) => x - g.state.p[k]));
    ok(err < 0.05, `ended ${err.toFixed(3)} m from the pin`);
    const mean = cruise.reduce((a, b) => a + b, 0) / cruise.length;
    close(mean, v, 0.15 * v, 'mean cruise speed');
  },
};
