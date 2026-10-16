// A batch of quadlab environments behind stdin / stdout, one JSON message per line, for a trainer in another
// language (train/quadlab.py). The physics is the browser's own code: there is no second copy to keep in sync.
//
//   → {"cmd": "make", "n": 16, "options": {...createEnv options..., "preset": "Quad X 450"}}
//   ← {"ok": true, "observation_size": 20, "action_size": 4, "features": [...], "level": "wrench", "ml": {...}}
//   → {"cmd": "reset", "seed": 7}                 env i is seeded with seed + i
//   ← {"obs": [[...] × n], "info": [{...} × n]}
//   → {"cmd": "step", "actions": [[a0, a1, a2, a3] × n]}
//   ← {"obs", "reward", "terminated", "truncated", "info"}
//        An env whose episode ended is reset at once (its next episode continues its own random stream):
//        its obs is then the new episode's first observation, and info.final_obs / info.final_info hold the
//        last observation and info of the episode that ended (the Stable-Baselines3 VecEnv convention).
//   → {"cmd": "set_target", "i": 0, "target": [x, y, z]}    → {"cmd": "clear_target", "i": 0}
//   → {"cmd": "close"}
//
// Errors come back as {"error": "..."}; the server keeps running.
import { createInterface } from 'node:readline';
import { cloneConfig, presets } from '../src/config.js';
import { FEATURES } from '../src/ml/features.js';
import { createEnv } from '../src/ml/env.js';

let envs = [];
const out = (msg) => process.stdout.write(JSON.stringify(msg) + '\n');
const round = (x) => Math.round(x * 1e6) / 1e6;
const arr = (a) => Array.from(a, round);
const slim = (info) => ({
  t: round(info.t), position: arr(info.position), target: arr(info.target), crash: info.crash || null,
  saturated: info.saturated, ...(info.expert ? { expert: arr(info.expert) } : {}),
});

const handlers = {
  make({ n = 1, options = {} }) {
    const { preset, ...opts } = options;
    const base = preset ? presets.find((p) => p.name === preset) : presets[0];
    if (!base) throw new Error(`unknown preset ${preset} (${presets.map((p) => p.name).join(', ')})`);
    envs = Array.from({ length: n }, () => createEnv(cloneConfig(base), opts));
    const e = envs[0];
    return { ok: true, n, observation_size: e.observationSize, action_size: e.actionSize, features: FEATURES, level: e.level, obs: e.obsMode, repeat: e.repeat, ml: e.nominal.ml, preset: base.name };
  },
  reset({ seed = 0 }) {
    const rs = envs.map((e, i) => e.reset(seed + i));
    return { obs: rs.map((r) => arr(r.obs)), info: rs.map((r) => ({ ...slim(r.info), params: r.info.params })) };
  },
  step({ actions }) {
    if (!actions || actions.length !== envs.length) throw new Error(`need ${envs.length} actions, got ${actions && actions.length}`);
    const res = { obs: [], reward: [], terminated: [], truncated: [], info: [] };
    envs.forEach((e, i) => {
      const r = e.step(actions[i]);
      let obs = arr(r.obs), info = slim(r.info);
      if (r.terminated || r.truncated) {
        const next = e.reset();
        info = { ...slim(next.info), params: next.info.params, final_obs: obs, final_info: info };
        obs = arr(next.obs);
      }
      res.obs.push(obs); res.reward.push(round(r.reward)); res.terminated.push(r.terminated); res.truncated.push(r.truncated); res.info.push(info);
    });
    return res;
  },
  set_target({ i, target }) { envs[i].setTarget(target); return { ok: true }; },
  clear_target({ i }) { envs[i].clearTarget(); return { ok: true }; },
  close() { setImmediate(() => process.exit(0)); return { ok: true }; },
};

createInterface({ input: process.stdin }).on('line', (line) => {
  if (!line.trim()) return;
  try {
    const msg = JSON.parse(line), h = handlers[msg.cmd];
    if (!h) throw new Error(`unknown cmd ${msg.cmd}`);
    out(h(msg));
  } catch (err) { out({ error: String(err.message || err) }); }
});
