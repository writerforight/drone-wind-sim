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
    ground: true,                            // a flat floor at z = 0 (needed for take-off and landing)
  },
  motor: {
    arm: 0.225,                              // m, centre to motor
    kThrust: 6.0e-6,                         // N per (rad/s)²:      T = kThrust · ω²
    kTorque: 1.0e-7,                         // N·m per (rad/s)²:    Q = kTorque · ω²  (reaction torque)
    tau: 0.03,                               // s, first-order lag of the rotor speed
    omegaMin: 0,                             // rad/s
    omegaMax: 1100,                          // rad/s  (≈ 10 500 rpm)
    // X layout seen from above, body x forward / y left. spin +1 = rotor turns counter-clockwise seen from
    // above (its reaction torque on the body is −z), −1 = clockwise. Diagonal rotors share a direction.
    rotors: [
      { name: 'front-right', angleDeg: -45, spin: +1 },
      { name: 'back-left', angleDeg: 135, spin: +1 },
      { name: 'front-left', angleDeg: 45, spin: -1 },
      { name: 'back-right', angleDeg: -135, spin: -1 },
    ],
  },
  // Cascaded PID: position → attitude → body rate. Gains are per axis [x, y, z] (or [roll, pitch, yaw]).
  // The rate loop's output is multiplied by the inertia J, so its gains are in rad/s² per unit error and do
  // not change when the airframe gets heavier or lighter.
  controller: {
    dt: 0.004,                               // s, controller period (250 Hz); physics runs 2 steps per call
    // Each axis is a double integrator under PID: s³ + Kd s² + Kp s + Ki = 0. Ki is placed so the slowest
    // (integral) pole is not sluggish: x, y poles −1.15, −0.43 ± 0.58j (τ_slow 2.4 s, ζ 0.59);
    // z poles −0.58, −1.21 ± 1.07j (τ_slow 1.7 s, ζ 0.75). With Ki 0.2 / 0.6 the integral settled in ~6 s.
    // Anti-windup: an axis's error is integrated only while |e| < iZone (conditional integration), so a large
    // initial error cannot wind the integral up; iLimit additionally caps it.
    position: { kp: [1.5, 1.5, 4.0], ki: [0.6, 0.6, 1.5], kd: [2.0, 2.0, 3.0], iLimit: 2.0, iZone: 0.3 },   // → acceleration, m/s²
    maxTilt: 30,                             // deg, the position loop never asks for more tilt than this
    attitude: { kp: [7.0, 7.0, 3.0] },       // angle error (rad) → rate reference (rad/s)
    rate: { kp: [18, 18, 8], ki: [4, 4, 2], kd: [0.25, 0.25, 0], iLimit: 3.0 },                // → J · (…)
    maxRate: [4, 4, 2],                      // rad/s, limit on the rate references
  },
};


/** A light 5-inch racer: small inertia, fast motors, high thrust-to-weight. */
export const racer5 = {
  name: 'Racer 5"',
  physics: { ...quadX450.physics, mass: 0.65, inertia: [0.0021, 0.0021, 0.0036], linearDrag: 0.15 },
  motor: { ...quadX450.motor, arm: 0.11, kThrust: 9.4e-7, kTorque: 1.4e-8, tau: 0.015, omegaMax: 2800 },
  controller: quadX450.controller,
};

/** A heavy-lift frame: large props, slow motors (bigger lag), low thrust-to-weight. */
export const heavy = {
  name: 'Heavy lift',
  physics: { ...quadX450.physics, mass: 3.0, inertia: [0.08, 0.08, 0.14], linearDrag: 0.5 },
  motor: { ...quadX450.motor, arm: 0.35, kThrust: 2.9e-5, kTorque: 6.0e-7, tau: 0.06, omegaMax: 800 },
  // slower motors: a lower rate-loop bandwidth keeps a phase margin against the 60 ms lag
  controller: { ...quadX450.controller, rate: { ...quadX450.controller.rate, kp: [10, 10, 6], kd: [0.1, 0.1, 0] }, attitude: { kp: [5, 5, 2.5] } },
};

export const presets = [quadX450, racer5, heavy];
export const config = quadX450;

/** A deep copy, so a user's edits (design panel, gain sliders) never change the presets themselves. */
export const cloneConfig = (c) => JSON.parse(JSON.stringify(c));
