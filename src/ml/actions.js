/**
 * ml/actions — what a policy's four numbers a ∈ [−1, 1]⁴ command, at three levels (config.ml.action.level):
 *
 *   'wrench'  thrust T = m g (1 + a0),  torque τ = J ⊙ angAccel ⊙ a1..3          → mixer → motors
 *   'rates'   thrust as above,           body-rate reference w_ref = maxRate ⊙ a1..3 → rate PID → mixer → motors
 *   'motors'  rotor thrust T_i = T_min + (T_max − T_min)(a_i + 1)/2  →  ω_i = √(T_i / k_T)   (no mixer)
 *
 * a = 0 is "hover, no turning" for 'wrench' and 'rates' (a0 = 0 → T = m g), so an untrained policy starts near
 * a sensible point. The scales use the NOMINAL airframe (cfg given here): if the real drone is heavier, the same
 * a0 lifts less, and the policy has to notice that — this is exactly what domain randomisation trains.
 *
 * fromExpert(u) maps a cascaded PID output back to an action at the same level: labels for imitation learning
 * (behaviour cloning, DAgger) or a base action for residual RL.
 */
import { createRateLoop } from '../controller/cascaded-pid.js';
import { createMixer } from '../motor/mixer.js';

export const ACTION_LEVELS = ['wrench', 'rates', 'motors'];
export const ACTION_SIZE = 4;

const clip1 = (x) => Math.max(-1, Math.min(1, x));

export function createActionMap(cfg, level = cfg.ml.action.level) {
  if (!ACTION_LEVELS.includes(level)) throw new Error(`unknown action level '${level}' (${ACTION_LEVELS.join(', ')})`);
  const P = cfg.physics, M = cfg.motor, C = cfg.controller, A = cfg.ml.action;
  const weight = P.mass * P.gravity, J = P.inertia;
  const mixer = createMixer(M), rateLoop = createRateLoop(P);
  const { Tmin, Tmax } = mixer;
  const thrustOf = (a0) => Math.max(0, weight * (1 + a0));

  return {
    level, size: ACTION_SIZE,
    reset() { rateLoop.reset(); },

    /** a (already in [−1, 1]) and the current measurement → the loop's input ({thrust, torque} or {omegaCmd}). */
    command(a, meas) {
      if (level === 'wrench') return { thrust: thrustOf(a[0]), torque: [0, 1, 2].map((k) => J[k] * A.angAccel[k] * a[k + 1]) };
      if (level === 'rates') {
        const wRef = [0, 1, 2].map((k) => C.maxRate[k] * a[k + 1]);
        return { thrust: thrustOf(a[0]), torque: rateLoop.torque(wRef, meas.w, C) };
      }
      return { omegaCmd: a.map((ai) => Math.sqrt((Tmin + ((Tmax - Tmin) * (ai + 1)) / 2) / M.kThrust)) };
    },

    /** Cascaded PID output { thrust, torque, debug: { wRef } } → the action that commands the same, clipped. */
    fromExpert(u) {
      const a0 = u.thrust / weight - 1;
      if (level === 'wrench') return [a0, ...[0, 1, 2].map((k) => u.torque[k] / (J[k] * A.angAccel[k]))].map(clip1);
      if (level === 'rates') return [a0, ...[0, 1, 2].map((k) => u.debug.wRef[k] / C.maxRate[k])].map(clip1);
      return mixer.mix(u).thrusts.map((T) => clip1((2 * (T - Tmin)) / (Tmax - Tmin) - 1));
    },
  };
}
