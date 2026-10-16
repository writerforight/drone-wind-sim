/**
 * ml/agent — the one piece shared by training and deployment: observation, action, reward, crash test.
 *
 * The training environment (ml/env.js) and a policy flying in the browser (ml/policy-controller.js) both
 * go through this object, so a trained policy sees exactly the inputs it was trained on.
 *
 *   const agent = createAgent(cfg);          // cfg = the NOMINAL config (what the controller believes)
 *   agent.observe(meas, target)  → token     (also pushed into the history window)
 *   agent.window()               → the last `history` tokens, oldest first, flat (history × 20)
 *   agent.act(a)                 → the clipped action, held until the next act()
 *   agent.command(meas)          → the loop's input for the held action (called every controller tick)
 *   agent.reward(meas, target, aPrev), agent.crashed(meas, target)
 *
 * Reward per policy step (weights in config.ml.reward):
 *   r = alive − w_p |e| − w_v |v| − w_w |w| − w_a |a|² − w_s |a − a_prev|²        (− crash, once, on a crash)
 */
import { norm, rotate } from '../physics/math3.js';
import { ACTION_SIZE, createActionMap } from './actions.js';
import { FEATURE_SIZE, createHistory, features } from './features.js';

const clip1 = (x) => Math.max(-1, Math.min(1, Number.isFinite(x) ? x : 0));

/** config.ml with some settings replaced; nested groups (episode, reward, …) are merged key by key. */
export function mergeMl(base, over = {}) {
  const out = { ...base };
  for (const [k, v] of Object.entries(over)) {
    out[k] = v && typeof v === 'object' && !Array.isArray(v) && base[k] && typeof base[k] === 'object' ? { ...base[k], ...v } : v;
  }
  return out;
}

/** Controller ticks per policy step; policyDt must be a whole number of controller periods. */
export function policyRepeat(policyDt, controllerDt) {
  const n = Math.round(policyDt / controllerDt);
  if (n < 1 || Math.abs(n * controllerDt - policyDt) > 1e-9) {
    throw new Error(`policyDt ${policyDt} s is not a whole number of controller periods (${controllerDt} s)`);
  }
  return n;
}

export function createAgent(cfg, { level = cfg.ml.action.level, history = cfg.ml.history } = {}) {
  const ml = cfg.ml, R = ml.reward, E = ml.episode;
  const actions = createActionMap(cfg, level);
  const hist = createHistory(history, FEATURE_SIZE);
  const cosMaxTilt = Math.cos((E.maxTilt * Math.PI) / 180);
  let held = new Array(ACTION_SIZE).fill(0);

  return {
    cfg, level, actions,
    featureSize: FEATURE_SIZE, actionSize: ACTION_SIZE, historyLength: history,
    reset() { hist.clear(); held = new Array(ACTION_SIZE).fill(0); actions.reset(); },
    get held() { return held; },

    observe(meas, target) { const tok = features(meas, target, held, ml.obsScale); hist.push(tok); return tok; },
    window() { return hist.window(); },
    act(a) { held = Array.from({ length: ACTION_SIZE }, (_, i) => clip1(a[i])); return held; },
    command(meas) { return actions.command(held, meas); },

    reward(meas, target, aPrev) {
      const e = [0, 1, 2].map((k) => target[k] - meas.p[k]);
      let a2 = 0, da2 = 0;
      for (let i = 0; i < ACTION_SIZE; i++) { a2 += held[i] ** 2; da2 += (held[i] - aPrev[i]) ** 2; }
      return R.alive - R.position * norm(e) - R.velocity * norm(meas.v) - R.rates * norm(meas.w) - R.action * a2 - R.smooth * da2;
    },

    /** null while flying, else why the flight ended: 'tilt' (beyond maxTilt), 'floor' (touched it), 'far'. */
    crashed(meas, target) {
      if (rotate(meas.q, [0, 0, 1])[2] < cosMaxTilt) return 'tilt';
      if (meas.p[2] <= 0) return 'floor';
      if (Math.hypot(target[0] - meas.p[0], target[1] - meas.p[1], target[2] - meas.p[2]) > E.maxDistance) return 'far';
      return null;
    },
  };
}
