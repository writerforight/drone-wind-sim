import { quadX450 } from '../src/config.js';
import { createSimulation } from '../src/loop/simulation.js';
import { buildNet, createMlpController, defaultNetSettings } from '../src/ml/mlp-controller.js';
import { MLP } from '../src/ml/mlp.js';
import { crashReason, createOnlineLearner, lossOf, SPAWN } from '../src/ml/online-learner.js';
import { createState } from '../src/physics/rigidbody.js';
import { fromEuler } from '../src/physics/math3.js';
import { close, ok } from './assert.mjs';

const at = (p, v = [0, 0, 0], q = [1, 0, 0, 0]) => ({ p, v, q, w: [0, 0, 0] });

export const tests = {
  'loss: distance plus the mismatch to the desired velocity'() {
    const ref = { position: [10, 0, 2], speed: 2 };
    close(lossOf(at([0, 0, 2], [2, 0, 0]), ref), 10, 1e-12, 'flying at the desired velocity');
    close(lossOf(at([0, 0, 2]), ref), 12, 1e-12, 'standing still far away');
    close(lossOf(at([9.5, 0, 2], [1, 0, 0]), ref), 0.5 + 0.5, 1e-12, 'near the target the desired speed shrinks to the distance');
    close(lossOf(at([10, 0, 2]), ref), 0, 1e-12, 'on the target, still');
  },

  'crash reasons'() {
    const ref = { position: [0, 0, 2] };
    ok(crashReason(at([0, 0, 1]), ref) === null, 'flying is not a crash');
    ok(crashReason(at([0, 0, 0.01]), ref) === 'floor', 'floor');
    ok(crashReason(at([0, 0, 1], [0, 0, 0], fromEuler(Math.PI * 0.6, 0, 0)), ref) === 'flipped', 'flipped');
    ok(crashReason(at([50, 0, 1]), ref) === 'lost', 'lost');
  },

  'the rule (L_t − L_{t−1})·ε/σ² learns a static target (no dynamics): it is a gradient estimate'() {
    // a "controller" whose output itself is scored: L(o) = |o − o*|, o* fixed. Same update code path.
    const net = new MLP([2, 8, 4], ['tanh', 'identity']).init('xavier', 1, 2), x = [0.3, -0.5], target = [0.5, -0.3, 0.2, 0.1];
    const ctl = { net, last: null, noise: null };
    const settings = { sigma: 0.05, lr: 3e-3, optimizer: 'adam', seed: 4 };
    const learner = createOnlineLearner(ctl, settings);
    const score = () => Math.hypot(...Array.from(net.predict(x), (v, i) => v - target[i]));
    const before = score();
    for (let t = 0; t < 6000; t++) {
      const cache = net.forward(x), eps = ctl.noise(), o = Array.from(cache.as.at(-1), (v, i) => v + eps[i]);
      ctl.last = { x, cache, eps, o };
      const L = Math.hypot(...o.map((v, i) => v - target[i]));
      // feed L through a fake state whose loss equals it: position at distance L from the reference, still, flying
      learner.update(at([L, 0, 5]), { position: [0, 0, 5], speed: 0 });
    }
    const after = score();
    ok(after < 0.25 * before, `distance to the target output ${before.toFixed(3)} → ${after.toFixed(3)}`);
  },

  'in the closed loop every step updates the weights, and crashes are counted'() {
    const s = { ...defaultNetSettings(), sigma: 0.2, lr: 1e-2 }, net = buildNet(s);
    const ctl = createMlpController(quadX450.physics, net, s.inputs), learner = createOnlineLearner(ctl, s);
    const sim = createSimulation(quadX450, { controller: ctl, initial: { position: SPAWN } });
    sim.reference = { position: [2, 1, 2], speed: 1.5, yaw: 0 };
    const theta0 = Float64Array.from(net.theta);
    let crashes = 0;
    for (let i = 0; i < 2500; i++) {                         // 10 s of flight
      sim.tick();
      const r = learner.update(sim.state, sim.reference);
      if (r.crashed) { crashes++; sim.state = createState({ position: SPAWN }); ctl.reset(); }
    }
    ok(learner.stats.updates > 2000, `updates ${learner.stats.updates}`);
    ok(learner.stats.crashes === crashes, 'crash count');
    ok(net.theta.some((v, i) => v !== theta0[i]), 'weights did not change');
    ok(net.theta.every(Number.isFinite) && sim.state.p.every(Number.isFinite), 'not finite');
  },
};
