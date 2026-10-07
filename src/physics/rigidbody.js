/**
 * physics/ — 6-DoF rigid-body dynamics of the airframe, integrated with fixed-step RK4.
 *
 * State   s = { p, v, q, w }   position p (W), velocity v (W), attitude q (body→world), body rates w (B)
 * Input   u = { force, torque, extForce, wind }
 *           force    total rotor force in the BODY frame, normally [0, 0, ΣT]       (from motor/)
 *           torque   total rotor torque in the BODY frame [τx, τy, τz]              (from motor/)
 *           extForce an extra WORLD-frame force (a push, a disturbance), optional
 *           wind     WORLD-frame air velocity, optional; drag acts on v − wind
 *
 * Newton–Euler (derivation in docs/derivation.md):
 *   ṗ = v
 *   v̇ = −g e_z + (1/m) [ R(q) force − c_d (v − wind) + extForce ]
 *   q̇ = ½ q ⊗ (0, w)
 *   ẇ = J⁻¹ ( torque − w × J w )                       J = diag(Jx, Jy, Jz)
 *
 * The interface is deliberately small: step(state, input, params) → new state. Nothing here knows about
 * motors, sensors or controllers, so any of those can be replaced without touching this file.
 */
import { add, cross, qmul, qnormalize, rotate, scale, sub } from './math3.js';

export function createState({ position = [0, 0, 0], velocity = [0, 0, 0], attitude = [1, 0, 0, 0], rates = [0, 0, 0] } = {}) {
  return { p: position.slice(), v: velocity.slice(), q: attitude.slice(), w: rates.slice() };
}

/** Time derivative of the state (the right-hand side of the Newton–Euler equations). */
export function derivative(s, u, P) {
  const m = P.mass, J = P.inertia;
  const wind = u.wind || [0, 0, 0];
  const fWorld = rotate(s.q, u.force || [0, 0, 0]);
  const drag = scale(sub(s.v, wind), -P.linearDrag);
  const ext = u.extForce || [0, 0, 0];
  const acc = add(scale(add(add(fWorld, drag), ext), 1 / m), [0, 0, -P.gravity]);
  const qdot = scale4(qmul(s.q, [0, s.w[0], s.w[1], s.w[2]]), 0.5);
  const Jw = [J[0] * s.w[0], J[1] * s.w[1], J[2] * s.w[2]];
  const tau = sub(u.torque || [0, 0, 0], cross(s.w, Jw));
  const wdot = [tau[0] / J[0], tau[1] / J[1], tau[2] / J[2]];
  return { p: s.v, v: acc, q: qdot, w: wdot };
}

const scale4 = (q, k) => [q[0] * k, q[1] * k, q[2] * k, q[3] * k];
const axpy = (s, d, h) => ({
  p: [s.p[0] + h * d.p[0], s.p[1] + h * d.p[1], s.p[2] + h * d.p[2]],
  v: [s.v[0] + h * d.v[0], s.v[1] + h * d.v[1], s.v[2] + h * d.v[2]],
  q: [s.q[0] + h * d.q[0], s.q[1] + h * d.q[1], s.q[2] + h * d.q[2], s.q[3] + h * d.q[3]],
  w: [s.w[0] + h * d.w[0], s.w[1] + h * d.w[1], s.w[2] + h * d.w[2]],
});

/**
 * One RK4 step of length P.dt (or dt if given) with the input held constant over the step (zero-order
 * hold, as from a digital controller). The quaternion is renormalised afterwards.
 */
export function step(s, u, P, dt = P.dt) {
  const k1 = derivative(s, u, P);
  const k2 = derivative(axpy(s, k1, dt / 2), u, P);
  const k3 = derivative(axpy(s, k2, dt / 2), u, P);
  const k4 = derivative(axpy(s, k3, dt), u, P);
  const out = {};
  for (const key of ['p', 'v', 'q', 'w']) {
    out[key] = s[key].map((x, i) => x + (dt / 6) * (k1[key][i] + 2 * k2[key][i] + 2 * k3[key][i] + k4[key][i]));
  }
  out.q = qnormalize(out.q);
  return out;
}

/** Mechanical energy ½ m |v|² + m g z + ½ wᵀ J w (conserved without drag, motors and external forces). */
export function energy(s, P) {
  const J = P.inertia;
  const kin = 0.5 * P.mass * (s.v[0] ** 2 + s.v[1] ** 2 + s.v[2] ** 2);
  const pot = P.mass * P.gravity * s.p[2];
  const rot = 0.5 * (J[0] * s.w[0] ** 2 + J[1] * s.w[1] ** 2 + J[2] * s.w[2] ** 2);
  return kin + pot + rot;
}

/** Angular momentum in the world frame, R(q) J w (conserved without external torques). */
export function angularMomentumWorld(s, P) {
  const J = P.inertia;
  return rotate(s.q, [J[0] * s.w[0], J[1] * s.w[1], J[2] * s.w[2]]);
}
