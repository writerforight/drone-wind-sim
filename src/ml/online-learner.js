/**
 * ml/online-learner — the owner's "pure" online rule: no physics model, no teacher. Every control step:
 *
 *   1. the controller adds exploration noise ε ~ N(0, σ² I) to the network output o (ctl.noise),
 *   2. the physics advances one controller period with that action,
 *   3. the loss of the NEW state is measured:   L_t = |target − p| + |v − v_des|
 *        v_des = û · min(speed, |target − p| · 1/s)   (toward the target at the desired speed, slowing near it)
 *   4. the change against the previous step is the only learning signal:
 *        ∂L/∂o ≈ g = (L_t − L_{t−1}) · ε / σ²
 *      g is backpropagated through the MLP (∂L/∂θ = Jᵀ g), clipped, and the optimizer takes one step.
 *
 * Why this is a gradient estimate: for a fixed state, L(o + ε) − L(o) ≈ ∇L·ε, and E[ε (∇L·ε)] / σ² = ∇L
 * (perturbation / "node perturbation" learning). L_{t−1} is the baseline that removes the constant part. Here the
 * state is NOT fixed: the drone moved, the target may have moved and the action's effect arrives with a delay
 * (motor lag, then two integrations to position) — the experiment is to see what that does to monotonicity.
 *
 * A crash (floor, tilt beyond 90°, far away, runaway rates) puts the drone back at the spawn point and is counted;
 * the network keeps its weights (it learns across episodes).
 */
import { createOptimizer, clipGradient } from './mlp.js';
import { createRandom } from '../util/random.js';

export const SPAWN = [0, 0, 1.5];
export const CLIP = 5;                 // gradient norm limit per step (the estimate is very noisy)

/** The loss of a state for a reference { position, speed }: distance to the target + speed mismatch. */
export function lossOf(state, ref) {
  const d = [0, 1, 2].map((k) => ref.position[k] - state.p[k]), dist = Math.hypot(...d);
  const s = dist > 1e-9 ? Math.min(ref.speed || 0, dist) / dist : 0;
  const vErr = Math.hypot(...[0, 1, 2].map((k) => state.v[k] - d[k] * s));
  return dist + vErr;
}

/** Why a state counts as a crash (or null). `tiltZ` = body z axis · world z (1 level, < 0 upside down). */
export function crashReason(state, ref) {
  const q = state.q, tiltZ = 1 - 2 * (q[1] * q[1] + q[2] * q[2]);
  if (state.p[2] <= 0.02) return 'floor';
  if (tiltZ < 0) return 'flipped';
  if (Math.hypot(...[0, 1, 2].map((k) => ref.position[k] - state.p[k])) > 40) return 'lost';
  if (Math.hypot(...state.w) > 30) return 'spin';
  return null;
}

/**
 * createOnlineLearner(ctl, settings) → { update(state, ref) → info, stats }
 * settings: { sigma, lr, optimizer, seed }. Reads ctl.last (x, cache, ε, o) written by ctl.control().
 */
export function createOnlineLearner(ctl, settings) {
  const net = ctl.net, rng = createRandom((settings.seed || 1) + 101);
  const opt = createOptimizer(settings.optimizer, net.nParams, settings.lr);
  const grad = new Float64Array(net.nParams);
  const stats = { steps: 0, updates: 0, crashes: 0, lastCrash: null, L: null, prevL: null, gradNorm: 0, episodeSteps: 0 };
  ctl.noise = () => Array.from({ length: net.outDim }, () => settings.sigma * rng.gauss());

  return {
    stats,
    /** Call after every controller period. Returns { L, dL, crashed }. */
    update(state, ref) {
      const L = lossOf(state, ref), prev = stats.L;
      stats.prevL = prev; stats.L = L; stats.steps++; stats.episodeSteps++;
      const last = ctl.last;
      let dL = null;
      if (prev !== null && last && last.eps) {
        dL = L - prev;
        const s2 = settings.sigma * settings.sigma, gOut = last.eps.map((e) => (dL * e) / s2);
        grad.fill(0);
        net.backward(last.cache, gOut, grad);
        stats.gradNorm = clipGradient(grad, CLIP);
        opt.lr = settings.lr;                       // the panel may change it while learning
        opt.step(net.theta, grad);
        stats.updates++;
      }
      const why = crashReason(state, ref);
      if (why) { stats.crashes++; stats.lastCrash = why; stats.L = null; stats.episodeSteps = 0; }
      return { L, dL, crashed: why };
    },
  };
}
