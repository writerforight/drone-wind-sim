/**
 * motor/ — four rotors: speed dynamics, saturation, and the force / torque they put on the body.
 *
 *   speed      τ ω̇ = sat(ω_cmd) − ω          first-order lag (ESC + motor + propeller inertia)
 *   thrust     T_i = k_T ω_i²                   along +z_B
 *   torque     r_i × (0, 0, T_i)  −  s_i k_Q ω_i² e_z      arm torque + rotor reaction torque
 *
 * The lag is stepped with its exact discretisation for a command held over dt:
 *   ω ← ω + (ω_cmd − ω)(1 − e^(−dt/τ))
 * so the motor response does not depend on the step size.
 */

/** Rotor positions (body x, y) and spin signs from the config's layout. */
export function rotorGeometry(M) {
  return M.rotors.map((r) => {
    const a = (r.angleDeg * Math.PI) / 180;
    return { x: M.arm * Math.cos(a), y: M.arm * Math.sin(a), spin: r.spin, name: r.name };
  });
}

export function createMotors(M, omega0 = 0) {
  return { omega: M.rotors.map(() => omega0) };
}

const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

/** Advance the rotor speeds by dt towards the (saturated) commands. */
export function stepMotors(state, omegaCmd, M, dt) {
  const a = 1 - Math.exp(-dt / M.tau);
  return { omega: state.omega.map((w, i) => w + (clamp(omegaCmd[i], M.omegaMin, M.omegaMax) - w) * a) };
}

/** Total body-frame force and torque produced by rotor speeds ω (the input physics/ expects). */
export function rotorWrench(omega, M) {
  const geo = rotorGeometry(M);
  let T = 0, tx = 0, ty = 0, tz = 0;
  omega.forEach((w, i) => {
    const Ti = M.kThrust * w * w, Qi = M.kTorque * w * w, g = geo[i];
    T += Ti;
    tx += g.y * Ti;                  // (r × T e_z)_x = y T
    ty += -g.x * Ti;                 // (r × T e_z)_y = −x T
    tz += -g.spin * Qi;              // reaction torque opposes the rotor's spin
  });
  return { force: [0, 0, T], torque: [tx, ty, tz] };
}
