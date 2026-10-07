/**
 * All drone and controller parameters, in one place (SPEC: "one config object/file").
 *
 * Units are SI throughout: m, kg, s, N, N·m, rad, rad/s.
 * Frames:
 *   world  W: x east, y north, z up (gravity is −z).
 *   body   B: x forward, y left, z up (thrust is +z_B).
 * The attitude quaternion q = [w, x, y, z] rotates body vectors into the world frame: v_W = R(q) v_B.
 *
 * Nothing in physics/, motor/, controller/ or loop/ has a hidden constant: every number comes from here,
 * so a preset is just a different copy of this object.
 */

export const GRAVITY = 9.81;

/** A ~450 mm class quadrotor in X configuration (numbers of the right order, not a specific product). */
export const quadX450 = {
  name: 'Quad X 450',
  physics: {
    mass: 1.2,                               // kg
    inertia: [0.0123, 0.0123, 0.0224],       // kg·m², principal moments about body x, y, z
    gravity: GRAVITY,                        // m/s²
    linearDrag: 0.25,                        // N per (m/s) of air-relative speed, lumped body drag
    dt: 0.002,                               // s, fixed physics step (500 Hz), RK4
  },
  motor: {
    arm: 0.225,                              // m, centre to motor
    kThrust: 6.0e-6,                         // N per (rad/s)²:      T = kThrust · ω²
    kTorque: 1.0e-7,                         // N·m per (rad/s)²:    Q = kTorque · ω²  (reaction torque)
    tau: 0.03,                               // s, first-order lag of the rotor speed
    omegaMin: 0,                             // rad/s
    omegaMax: 1100,                          // rad/s  (≈ 10 500 rpm)
  },
};

export const config = quadX450;
