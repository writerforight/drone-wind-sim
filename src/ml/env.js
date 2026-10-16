/**
 * ml/env — a Gym-style environment around the same simulation the browser shows (SPEC M5):
 *
 *   const env = createEnv(cfg, { level: 'wrench', obs: 'token', seed: 1 });
 *   let { obs, info } = env.reset();
 *   ({ obs, reward, terminated, truncated, info } = env.step(action));   // action ∈ [−1, 1]⁴
 *   env.setTarget([x, y, z]);                                           // a target chosen by hand
 *
 * One step = config.ml.policyDt of simulated time: the action is held (zero-order hold) over
 * policyDt / controller.dt controller ticks, each of which runs the physics at its own fixed step.
 *
 * Observation (opts.obs):
 *   'token'   the newest token (20 numbers, ml/features.js). For a recurrent model that keeps its own state.
 *   'window'  the last `history` tokens, oldest first, flat (history × 20). For a model that is given the
 *             past sequence explicitly at every step (an LSTM run over the window, a TCN, a transformer, an MLP).
 *
 * Targets (opts.targets):
 *   'random'  a new random point in config.ml.episode.targetBox every targetEvery seconds (default)
 *   'fixed'   opts.target the whole episode
 *   function  (t, rng) → [x, y, z], e.g. a trajectory
 *   setTarget(p) overrides any of these until the next reset (the browser's "go there" button).
 *
 * Each reset draws a new drone (domain randomisation, config.ml.randomize; opts.randomize = false turns it
 * off) and a new start state. info.params holds the drawn values: the "privileged" information that a
 * teacher policy or an adaptation module (RMA-style) can be trained with; the policy itself never sees it.
 *
 * opts.expert = true also returns info.expert: the action at the same level that the cascaded PID (tuned for
 * the nominal drone) would take now — labels for behaviour cloning / DAgger, or a base for residual RL.
 */
import { cloneConfig } from '../config.js';
import { createCascadedPID } from '../controller/cascaded-pid.js';
import { createSimulation, perfectSensor } from '../loop/simulation.js';
import { fromEuler } from '../physics/math3.js';
import { createAgent, mergeMl, policyRepeat } from './agent.js';

/** A small seeded random generator (mulberry32): the same seed gives the same episode. */
export function createRng(seed = 1) {
  let s = seed >>> 0;
  const next = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return { next, uniform: (lo, hi) => lo + (hi - lo) * next() };
}

/** A copy of the nominal config with the physical parameters scaled by random factors, and a random wind. */
export function randomizeConfig(nominal, R, rng) {
  const cfg = cloneConfig(nominal), f = (r) => rng.uniform(1 - r, 1 + r);
  const k = { mass: f(R.mass), inertia: f(R.inertia), kThrust: f(R.kThrust), tau: f(R.tau), linearDrag: f(R.linearDrag) };
  cfg.physics.mass *= k.mass;
  cfg.physics.inertia = cfg.physics.inertia.map((J) => J * k.inertia);
  cfg.physics.linearDrag *= k.linearDrag;
  cfg.motor.kThrust *= k.kThrust;
  cfg.motor.kTorque *= k.kThrust;                  // the torque coefficient scales with the propeller like the thrust one
  cfg.motor.tau *= k.tau;
  const speed = rng.uniform(0, R.wind), dir = rng.uniform(-Math.PI, Math.PI);
  const wind = [speed * Math.cos(dir), speed * Math.sin(dir), 0];
  return { cfg, wind, params: { ...k, mass_kg: cfg.physics.mass, wind } };
}

const NO_RANDOMIZATION = { mass: 0, inertia: 0, kThrust: 0, tau: 0, linearDrag: 0, wind: 0 };

export function createEnv(baseCfg, opts = {}) {
  const nominal = cloneConfig(baseCfg);
  // the settings this env really uses end up in nominal.ml, so a policy file can store them (export_policy)
  nominal.ml = mergeMl(nominal.ml, {
    ...(opts.ml || {}),
    ...(opts.history ? { history: opts.history } : {}),
    ...(opts.level ? { action: { level: opts.level } } : {}),
  });
  const ml = nominal.ml, E = ml.episode, C = nominal.controller;
  const agent = createAgent(nominal);
  const obsMode = opts.obs || 'token';
  if (!['token', 'window'].includes(obsMode)) throw new Error(`obs must be 'token' or 'window', not '${obsMode}'`);
  const repeat = policyRepeat(ml.policyDt, C.dt);
  const measure = opts.measure || perfectSensor;
  const targets = opts.targets || 'random';
  let rng = createRng(opts.seed ?? 1);

  let sim, target, manualTarget, nextTargetAt, params, expert, expertU, expertFresh, done;
  const box = (b) => [0, 1, 2].map((k) => rng.uniform(b[k][0], b[k][1]));

  // The controller the simulation runs: the agent's held action, mapped to the loop's input every tick. When an
  // expert is attached it runs alongside on every tick too, so its integrators see the same history as the drone.
  const passthrough = {
    name: 'agent', reset() {},
    control(meas) {
      if (expert) { if (!expertFresh) expertU = expert.control(meas, { position: target, yaw: 0 }, C); expertFresh = false; }
      return agent.command(meas);
    },
  };

  function updateTarget(t) {
    if (manualTarget) { target = manualTarget; return; }
    if (typeof targets === 'function') { target = targets(t, rng).slice(); return; }
    if (targets === 'fixed') { target = (opts.target || [0, 0, 2]).slice(); return; }
    if (t >= nextTargetAt) { target = box(E.targetBox); nextTargetAt = t + rng.uniform(E.targetEvery[0], E.targetEvery[1]); }
  }

  function observation() {
    const meas = measure(sim.state, sim.t);
    const token = agent.observe(meas, target);
    const info = { t: sim.t, position: sim.state.p.slice(), target: target.slice(), params, saturated: sim.command.saturated };
    if (expert) {
      // the PID's next output for this very measurement; the next tick reuses it instead of calling it twice
      expertU = expert.control(meas, { position: target, yaw: 0 }, C);
      expertFresh = true;
      info.expert = agent.actions.fromExpert(expertU);
    }
    return { obs: obsMode === 'window' ? agent.window() : token, meas, info };
  }

  const env = {
    nominal, agent, repeat,
    level: agent.level, obsMode,
    actionSize: agent.actionSize,
    featureSize: agent.featureSize,
    observationSize: obsMode === 'window' ? agent.historyLength * agent.featureSize : agent.featureSize,
    get sim() { return sim; },
    get target() { return target; },
    get params() { return params; },

    reset(seed) {
      if (seed !== undefined) rng = createRng(seed);
      const drawn = opts.randomize === false
        ? { cfg: cloneConfig(nominal), wind: [0, 0, 0], params: { mass: 1, inertia: 1, kThrust: 1, tau: 1, linearDrag: 1, mass_kg: nominal.physics.mass, wind: [0, 0, 0] } }
        : randomizeConfig(nominal, { ...NO_RANDOMIZATION, ...ml.randomize }, rng);
      params = drawn.params;
      const tilt = (E.startTilt * Math.PI) / 180, yaw = ((E.startYaw ?? 180) * Math.PI) / 180;
      const initial = opts.initial || {
        position: box(E.targetBox),
        velocity: [0, 1, 2].map(() => rng.uniform(-E.startSpeed, E.startSpeed)),
        attitude: fromEuler(rng.uniform(-tilt, tilt), rng.uniform(-tilt, tilt), rng.uniform(-yaw, yaw)),
      };
      agent.reset();
      expert = opts.expert ? createCascadedPID(nominal.physics) : null;
      expertFresh = false;
      sim = createSimulation(drawn.cfg, { controller: passthrough, measure, initial });
      sim.disturbance.wind = drawn.wind;
      manualTarget = null; nextTargetAt = 0; done = false;
      updateTarget(0);
      const { obs, info } = observation();
      return { obs, info };
    },

    step(action) {
      if (!sim) throw new Error('call reset() before step()');
      if (done) throw new Error('the episode has ended: call reset()');
      const aPrev = agent.held;
      agent.act(action);
      for (let i = 0; i < repeat; i++) { sim.tick(); updateTarget(sim.t); }
      const { obs, meas, info } = observation();
      const crash = agent.crashed(meas, target);
      let reward = agent.reward(meas, target, aPrev);
      if (crash) reward -= ml.reward.crash;
      const terminated = !!crash, truncated = !crash && sim.t >= E.seconds - 1e-9;
      done = terminated || truncated;
      info.crash = crash;
      return { obs, reward, terminated, truncated, info };
    },

    /** Fly to a point chosen by hand (stays until reset or clearTarget). */
    setTarget(p) { manualTarget = p.slice(); target = manualTarget; },
    clearTarget() { manualTarget = null; nextTargetAt = 0; if (sim) updateTarget(sim.t); },
  };
  return env;
}
