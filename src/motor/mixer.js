/**
 * motor/mixer — from the controller's wish (total thrust T, body torques τ) to four rotor speed commands.
 *
 * Allocation (what the rotors produce):   [T, τx, τy, τz]ᵀ = A [T1, T2, T3, T4]ᵀ
 *   row 1: 1                 (thrusts add up)
 *   row 2: y_i               (roll torque from the arm)
 *   row 3: −x_i              (pitch torque from the arm)
 *   row 4: −s_i k_Q / k_T    (yaw torque from rotor drag, Q_i = (k_Q/k_T) T_i)
 * The mixer is A⁻¹: T_i = A⁻¹ [T, τ]. Each T_i is then clipped to what a rotor can give, [k_T ω_min², k_T ω_max²],
 * and turned into a speed command ω_i = √(T_i / k_T).
 *
 * Saturation here is a plain clip per rotor (the simplest choice to explain); when it clips, the produced
 * wrench differs from the requested one and `saturated` says so. Smarter allocation (keep attitude, give up
 * thrust) is a drop-in replacement of mix().
 */
import { rotorGeometry } from './motors.js';

export function allocationMatrix(M) {
  const geo = rotorGeometry(M), c = M.kTorque / M.kThrust;
  return [geo.map(() => 1), geo.map((g) => g.y), geo.map((g) => -g.x), geo.map((g) => -g.spin * c)];
}

/** Inverse of a small square matrix (Gauss–Jordan with partial pivoting). */
export function invert(A) {
  const n = A.length, M = A.map((r, i) => [...r, ...A.map((_, j) => (i === j ? 1 : 0))]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (Math.abs(M[p][c]) < 1e-12) throw new Error('allocation matrix is singular');
    [M[c], M[p]] = [M[p], M[c]];
    const d = M[c][c];
    for (let k = 0; k < 2 * n; k++) M[c][k] /= d;
    for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c]; for (let k = 0; k < 2 * n; k++) M[r][k] -= f * M[c][k]; }
  }
  return M.map((r) => r.slice(n));
}

/** A mixer for one airframe: precomputes A⁻¹ once. */
export function createMixer(M) {
  const A = allocationMatrix(M), Ainv = invert(A);
  const Tmin = M.kThrust * M.omegaMin ** 2, Tmax = M.kThrust * M.omegaMax ** 2;
  return {
    A, Ainv, Tmin, Tmax,
    /** wish = { thrust, torque: [τx, τy, τz] } → { omegaCmd, thrusts, saturated } */
    mix({ thrust, torque }) {
      const w = [thrust, torque[0], torque[1], torque[2]];
      const raw = Ainv.map((row) => row.reduce((acc, a, j) => acc + a * w[j], 0));
      const thrusts = raw.map((t) => Math.min(Tmax, Math.max(Tmin, t)));
      const saturated = raw.some((t, i) => t !== thrusts[i]);
      return { omegaCmd: thrusts.map((t) => Math.sqrt(t / M.kThrust)), thrusts, saturated };
    },
  };
}
