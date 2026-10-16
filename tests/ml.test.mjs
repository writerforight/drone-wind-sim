import { readFileSync } from 'node:fs';
import { cloneConfig, quadX450 } from '../src/config.js';
import { createCascadedPID } from '../src/controller/cascaded-pid.js';
import { createSimulation } from '../src/loop/simulation.js';
import { createActionMap } from '../src/ml/actions.js';
import { policyRepeat } from '../src/ml/agent.js';
import { createEnv } from '../src/ml/env.js';
import { FEATURES, createHistory, features } from '../src/ml/features.js';
import { loadPolicy } from '../src/ml/nn.js';
import { createPolicyController } from '../src/ml/policy-controller.js';
import { createRecorder } from '../src/ml/recorder.js';
import { fromEuler } from '../src/physics/math3.js';
import { close, ok } from './assert.mjs';

const json = (path) => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const hover = (p = [0, 0, 2]) => ({ p, v: [0, 0, 0], q: [1, 0, 0, 0], w: [0, 0, 0] });

export const tests = {
  'history window: oldest first, zero-padded at the front, newest last'() {
    const h = createHistory(3, 2);
    h.push([1, 1]);
    ok(h.window().join() === '0,0,0,0,1,1', `one token: ${h.window().join()}`);
    h.push([2, 2]); h.push([3, 3]); h.push([4, 4]);
    ok(h.window().join() === '2,2,3,3,4,4', `full: ${h.window().join()}`);
  },

  'features: target error, attitude columns and previous action, scaled'() {
    const S = quadX450.ml.obsScale, tok = features(hover([0, 0, 1]), [2, 0, 1], [0.1, 0, 0, -0.2], S);
    ok(tok.length === FEATURES.length, `length ${tok.length}`);
    close(tok[0], 2 / S.error, 1e-6, 'ex');
    ok([1, 0, 0, 0, 1, 0].every((v, i) => Math.abs(tok[6 + i] - v) < 1e-6), 'level attitude: R columns e_x, e_y');
    close(tok[15], 0.1, 1e-6, 'a0_prev'); close(tok[18], -0.2, 1e-6, 'a3_prev');
    close(tok[19], 1 / S.error, 1e-6, 'height');
    ok(features(hover(), [100, 0, 2], [0, 0, 0, 0], S)[0] === 5, 'clipped at 5');
  },

  'actions: a = 0 is hover with no torque (wrench, rates); motors give the per-rotor thrust'() {
    const cfg = quadX450, W = cfg.physics.mass * cfg.physics.gravity;
    for (const level of ['wrench', 'rates']) {
      const u = createActionMap(cfg, level).command([0, 0, 0, 0], hover());
      close(u.thrust, W, 1e-9, `${level} thrust`);
      ok(u.torque.every((t) => Math.abs(t) < 1e-12), `${level} torque ${u.torque}`);
    }
    const m = createActionMap(cfg, 'motors').command([-1, 1, 0, 0], hover());
    const T = m.omegaCmd.map((w) => cfg.motor.kThrust * w * w), Tmax = cfg.motor.kThrust * cfg.motor.omegaMax ** 2;
    close(T[0], 0, 1e-9, 'a = −1'); close(T[1], Tmax, 1e-6, 'a = +1'); close(T[2], Tmax / 2, 1e-6, 'a = 0');
  },

  'actions: fromExpert inverts command (the imitation labels are exact when nothing clips)'() {
    const cfg = quadX450, pid = createCascadedPID(cfg.physics);
    const meas = { p: [0.3, -0.2, 1.8], v: [0.1, 0, 0], q: fromEuler(0.05, -0.03, 0.2), w: [0.1, -0.1, 0.05] };
    const u = pid.control(meas, { position: [0, 0, 2], yaw: 0 }, cfg.controller);
    for (const level of ['wrench', 'motors']) {
      const map = createActionMap(cfg, level), back = map.command(map.fromExpert(u), meas);
      if (level === 'wrench') {
        close(back.thrust, u.thrust, 1e-9, 'thrust');
        u.torque.forEach((t, k) => close(back.torque[k], t, 1e-12, `torque ${k}`));
      } else {
        const T = back.omegaCmd.map((w) => cfg.motor.kThrust * w * w), mix = map.fromExpert(u);
        ok(T.length === 4 && mix.every((a) => a > -1 && a < 1), `motor actions ${mix}`);
      }
    }
  },

  'policyDt must be a whole number of controller periods'() {
    ok(policyRepeat(0.008, 0.004) === 2, '0.008 / 0.004');
    let threw = false;
    try { policyRepeat(0.01, 0.004); } catch { threw = true; }
    ok(threw, '0.01 / 0.004 must be refused (it would silently become 12 ms)');
  },

  'env: the same seed gives the same episode, another seed a different one'() {
    const run = (seed) => {
      const env = createEnv(quadX450, { seed, expert: true });
      let { obs, info } = env.reset(), sum = 0;
      for (let i = 0; i < 100; i++) { const r = env.step(info.expert); info = r.info; sum += r.reward; obs = r.obs; }
      return [sum, ...obs].join();
    };
    ok(run(7) === run(7), 'not reproducible');
    ok(run(7) !== run(8), 'seed has no effect');
  },

  'env: observation sizes for token and window, the window ends with the newest token'() {
    const e1 = createEnv(quadX450, { obs: 'token' }), e2 = createEnv(quadX450, { obs: 'window', history: 10 });
    const a = e1.reset(3).obs, b = e2.reset(3).obs;
    ok(a.length === 20 && b.length === 200, `sizes ${a.length}, ${b.length}`);
    ok(Array.from(b.slice(180)).join() === Array.from(a).join(), 'newest token is not last in the window');
    ok(b.slice(0, 180).every((x) => x === 0), 'the empty history must be zeros');
  },

  'env: the PID expert, fed back as the action, flies every episode to a fixed target'() {
    const env = createEnv(quadX450, { expert: true, randomize: false, targets: 'fixed', target: [1, -1, 2], seed: 5 });
    for (let ep = 0; ep < 5; ep++) {
      let { info } = env.reset(), r;
      do { r = env.step(info.expert); info = r.info; } while (!r.terminated && !r.truncated);
      ok(r.truncated, `episode ${ep} crashed: ${info.crash}`);
      ok(dist(info.target, info.position) < 0.05, `episode ${ep}: ${dist(info.target, info.position).toFixed(3)} m from the target`);
    }
  },

  'env: domain randomisation stays in its ranges and is reported in info.params'() {
    const env = createEnv(quadX450, { seed: 11 }), R = quadX450.ml.randomize;
    for (let i = 0; i < 20; i++) {
      const p = env.reset().info.params;
      ok(Math.abs(p.mass - 1) <= R.mass && Math.abs(p.tau - 1) <= R.tau && Math.hypot(...p.wind) <= R.wind + 1e-12, JSON.stringify(p));
      close(p.mass_kg, quadX450.physics.mass * p.mass, 1e-12, 'mass_kg');
    }
  },

  'env: setTarget overrides the random targets until clearTarget'() {
    const env = createEnv(quadX450, { seed: 2, expert: true, randomize: false });
    let { info } = env.reset();
    env.setTarget([1, 2, 3]);
    for (let i = 0; i < 1000; i++) info = env.step(info.expert).info;   // 8 s: the random schedule would have moved it
    ok(env.target.join() === '1,2,3' && info.target.join() === '1,2,3', `target ${env.target}`);
    ok(dist(info.position, [1, 2, 3]) < 0.2, `the expert did not follow the hand-set target: ${info.position}`);
    env.clearTarget();
    ok(env.target.join() !== '1,2,3', 'clearTarget did not return to the random targets');
  },

  'browser forward pass matches the numpy / PyTorch equations (LSTM, GRU, dense)'() {
    const fx = json('./fixtures/rnn-reference.json');
    for (const [name, c] of Object.entries(fx.cases)) {
      const spec = { format: 'quadlab-policy/1', layers: c.layers };
      const tok = loadPolicy({ ...spec, obs: 'token' }), win = loadPolicy({ ...spec, obs: 'window' });
      const clip = (v) => Math.max(-1, Math.min(1, v));
      fx.tokens.forEach((x, i) => tok.act(x).forEach((v, k) => close(v, clip(c.outputs[i][k]), 1e-12, `${name} step ${i} out ${k}`)));
      win.act(fx.tokens.flat()).forEach((v, k) => close(v, clip(c.outputs.at(-1)[k]), 1e-12, `${name} window out ${k}`));
    }
  },

  'a policy flies the same in the training env and as a browser controller (same code path)'() {
    const policy = json('../policies/linear-bc.json'), cfg = cloneConfig(quadX450);
    cfg.ml = { ...cfg.ml, ...policy.ml };
    const initial = { position: [0.4, -0.3, 1.5], attitude: fromEuler(0.1, -0.05, 0) }, target = [1, 1, 2];
    const env = createEnv(cfg, { randomize: false, targets: 'fixed', target, initial });
    const p1 = loadPolicy(policy);
    let { obs } = env.reset(1);
    for (let i = 0; i < 300; i++) obs = env.step(p1.act(obs)).obs;
    const sim = createSimulation(cfg, { controller: createPolicyController(cfg, loadPolicy(policy)), initial });
    sim.reference = { position: target };
    sim.advance(300 * cfg.ml.policyDt);
    ok(env.sim.t === sim.t, `time ${env.sim.t} vs ${sim.t}`);
    ok(env.sim.state.p.join() === sim.state.p.join() && env.sim.state.q.join() === sim.state.q.join(), `states differ: ${env.sim.state.p} vs ${sim.state.p}`);
  },

  'the linear DAgger policy (policies/linear-bc.json) flies to targets chosen by hand'() {
    const policy = json('../policies/linear-bc.json');
    for (const target of [[1, 1, 2], [-2, 0.5, 1], [0, 0, 3]]) {
      const cfg = cloneConfig(quadX450), sim = createSimulation(cfg, { controller: createPolicyController(cfg, loadPolicy(policy)), initial: { position: [0, 0, 1.5] } });
      sim.reference = { position: target };
      sim.advance(8);
      ok(dist(target, sim.state.p) < 0.05, `${target}: ${dist(target, sim.state.p).toFixed(3)} m after 8 s`);
    }
  },

  'online adaptation hook: adapt() gets one transition per policy step, with the env reward'() {
    const cfg = cloneConfig(quadX450), seen = [];
    const policy = { name: 'zero', act: () => [0, 0, 0, 0], adapt: (tr) => seen.push(tr) };
    const ctl = createPolicyController(cfg, policy);
    const sim = createSimulation(cfg, { controller: ctl, initial: { position: [0, 0, 2] } });
    sim.reference = { position: [0, 0, 2] };
    sim.advance(50 * cfg.ml.policyDt);
    ok(seen.length === 49, `${seen.length} transitions for 50 policy steps`);
    const r = seen[0];
    ok(r.obs.length === 20 && r.next.length === 20 && r.action.length === 4 && r.crash === null, 'transition fields');
    close(r.reward, cfg.ml.reward.alive, 1e-3, 'reward at hover on the target');
  },

  'motor-level commands bypass the mixer'() {
    const cfg = cloneConfig(quadX450), w = 500;
    const sim = createSimulation(cfg, { controller: { reset() {}, control: () => ({ omegaCmd: [w, w, w, 2000] }) } });
    sim.tick();
    ok(sim.command.omegaCmd.join() === [w, w, w, cfg.motor.omegaMax].join() && sim.command.saturated, `command ${sim.command.omegaCmd}`);
  },

  'recorder: CSV header uses the feature names'() {
    const rec = createRecorder();
    rec.add({ t: 0, token: new Float32Array(20), action: [1, 2, 3, 4] });
    const head = rec.toCSV().split('\n')[0].split(',');
    ok(head[1] === 'token_ex' && head[20] === 'token_z' && head[21] === 'action_0' && head.length === 25, head.join());
    ok(JSON.parse(rec.toJSON()).rows.length === 1, 'json');
  },
};
