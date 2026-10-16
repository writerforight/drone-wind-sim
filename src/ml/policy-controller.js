/**
 * ml/policy-controller — a learned policy behind the same controller interface as the PID, so the loop, the
 * modes and the 3D view need not know which one is flying:
 *
 *   const ctl = createPolicyController(cfg, loadPolicy(json));
 *   createSimulation(cfg, { controller: ctl });       // reference.position is the target
 *
 * Every policyDt it builds the observation from the measurement and the target (ml/agent.js, the same code
 * the training environment uses), asks the policy for an action and holds it; on every controller tick in
 * between, the held action is turned into the loop's input (wrench, rates through the rate PID, or motors).
 *
 * Online adaptation hook: if the policy has adapt(transition), it is called once per policy step with
 *   { obs, action, reward, next, crash }
 * (obs and action of the previous step, the reward they earned, the observation that followed). A policy
 * can use it to update itself while it flies (online fine-tuning, an adaptive layer, a running estimate of
 * the drone's parameters). Without adapt(), a recurrent policy still adapts in context, through its state.
 */
import { createAgent, mergeMl, policyRepeat } from './agent.js';

export function createPolicyController(cfg, policy, opts = {}) {
  const ml = mergeMl(cfg.ml, policy.ml);
  const known = { ...cfg, ml };
  const level = opts.level || policy.level || ml.action.level;
  const obsMode = opts.obs || policy.obs || 'token';
  const agent = createAgent(known, { level, history: ml.history });
  const repeat = policyRepeat(ml.policyDt, cfg.controller.dt);
  let ticks = 0, last = null;

  const ctl = {
    name: policy.name || 'learned policy', agent, level, repeat,
    reset() { agent.reset(); if (policy.reset) policy.reset(); ticks = 0; last = null; },

    control(meas, ref) {
      if (ticks++ % repeat === 0) {
        const target = ref.position || meas.p;
        const aPrev = agent.held;
        const token = agent.observe(meas, target);
        const obs = obsMode === 'window' ? agent.window() : token;
        if (last && policy.adapt) {
          const crash = agent.crashed(meas, target);
          policy.adapt({ obs: last.obs, action: last.action, reward: agent.reward(meas, target, last.aPrev), next: obs, crash });
        }
        const action = agent.act(policy.act(obs));
        last = { obs, action, aPrev };
      }
      return agent.command(meas);
    },
  };
  ctl.reset();
  return ctl;
}
