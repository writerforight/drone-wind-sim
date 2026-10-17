import { quadX450 } from '../src/config.js';
import { createSimulation } from '../src/loop/simulation.js';
import { buildNet, createMlpController, defaultNetSettings, inputSize, observe, outputToWrench } from '../src/ml/mlp-controller.js';
import { fromEuler } from '../src/physics/math3.js';
import { close, ok } from './assert.mjs';

const P = quadX450.physics;
const level = { p: [1, 2, 3], v: [0.5, 0, -1], q: [1, 0, 0, 0], w: [0, 0.2, 0] };

export const tests = {
  'the default observation has 13 inputs in the documented order and scale'() {
    const s = defaultNetSettings();
    ok(inputSize(s.inputs) === 13, `size ${inputSize(s.inputs)}`);
    const x = observe(level, { position: [6, 2, 3], speed: 2.5 }, s.inputs);
    ok(x.length === 13, `length ${x.length}`);
    close(x[0], 1, 1e-12, 'dp x / 5'); close(x[3], 0.1, 1e-12, 'v x / 5'); close(x[6], 0.5, 1e-12, 'speed / 5');
    close(x[9], -1, 1e-12, 'down z, level'); close(x[11], 0.04, 1e-12, 'w y / 5');
    const far = observe(level, { position: [100, 2, 3], speed: 0 }, s.inputs);
    close(far[0], 3, 1e-12, 'dp clipped');
  },

  '"down" in the body frame shows the tilt'() {
    const rolled = { ...level, q: fromEuler(Math.PI / 6, 0, 0) };
    const x = observe(rolled, { position: [0, 0, 0] }, { down: true });
    close(Math.hypot(...x), 1, 1e-12, 'unit vector');
    close(x[1], -Math.sin(Math.PI / 6), 1e-9, 'rolled right: down leans to body −y');
  },

  'switching input groups off shrinks the input'() {
    const inputs = { ...defaultNetSettings().inputs, v: false, w: false };
    ok(inputSize(inputs) === 7, `size ${inputSize(inputs)}`);
    ok(buildNet({ ...defaultNetSettings(), inputs }).inDim === 7, 'net input');
  },

  'output 0 is hover thrust and no torque; the fixed layer is affine'() {
    const h = outputToWrench([0, 0, 0, 0], P);
    close(h.thrust, P.mass * P.gravity, 1e-12, 'hover thrust');
    ok(h.torque.every((t) => t === 0), 'no torque');
    const a = outputToWrench([0.2, 1, -1, 0.5], P);
    close(a.thrust, 1.2 * P.mass * P.gravity, 1e-12, 'thrust');
    close(a.torque[0], P.inertia[0] * 30, 1e-12, 'roll torque');
  },

  'the network has the chosen shape and starts with small outputs'() {
    const s = { ...defaultNetSettings(), hidden: [24, 24, 24] };
    const net = buildNet(s);
    ok(net.dims.join() === '13,24,24,24,4', `dims ${net.dims}`);
    ok(net.acts.at(-1) === 'identity', 'linear last layer');
    const o = net.predict(observe(level, { position: [6, 2, 3], speed: 1 }, s.inputs));
    ok(o.every((v) => Math.abs(v) < 0.5), `initial output ${Array.from(o)}`);
  },

  'the controller adds the noise and remembers what backprop needs'() {
    const s = defaultNetSettings(), net = buildNet(s), ctl = createMlpController(P, net, s.inputs);
    ctl.noise = () => [0.1, 0, 0, -0.2];
    const ref = { position: [0, 0, 2], speed: 1 }, u = ctl.control(level, ref);
    const clean = Array.from(net.predict(observe(level, ref, s.inputs)));
    close(ctl.last.o[0], clean[0] + 0.1, 1e-12, 'o_T + ε'); close(ctl.last.o[3], clean[3] - 0.2, 1e-12, 'o_z + ε');
    close(u.thrust, P.mass * P.gravity * (1 + ctl.last.o[0]), 1e-12, 'thrust from the noisy output');
    ok(ctl.last.cache && ctl.last.x.length === 13, 'cache kept');
    close(ctl.preview(level, ref).o[0], clean[0], 1e-12, 'preview has no noise');
  },

  'an untrained network can fly the simulation (no learning): the loop runs and stays finite'() {
    const s = defaultNetSettings(), ctl = createMlpController(P, buildNet(s), s.inputs);
    const sim = createSimulation(quadX450, { controller: ctl, initial: { position: [0, 0, 1] } });
    sim.reference = { position: [1, 0, 2], speed: 1, yaw: 0 };
    sim.advance(1);
    ok(sim.state.p.every(Number.isFinite) && sim.state.q.every(Number.isFinite), 'state not finite');
  },
};
