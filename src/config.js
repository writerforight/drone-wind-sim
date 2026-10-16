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

/**
 * Learning-based control (ml/): what a policy sees, what it commands, how it is scored, and how the training
 * drones differ from the nominal one. The same numbers are used by the training environment (ml/env.js) and
 * by a policy flying in the browser (ml/policy-controller.js), so a trained policy meets the inputs it was
 * trained on. A trained policy file stores the ml settings it was trained with.
 */
export const mlDefaults = {
  policyDt: 0.008,                           // s, the policy acts every 8 ms (125 Hz, a whole number of controller
                                             //    periods); the action is held in between
  history: 40,                               // steps of history in the observation window (0.32 s at 125 Hz)
  // What the policy's four numbers a ∈ [−1, 1]⁴ mean:
  //   'wrench'  a0 → thrust m g (1 + a0),  a1..3 → body torque J ⊙ angAccel ⊙ a   (through the mixer)
  //   'rates'   a0 → thrust as above,       a1..3 → body-rate reference maxRate ⊙ a  (through the rate PID)
  //   'motors'  a_i → rotor thrust T_max (a_i + 1) / 2                              (no mixer at all)
  action: { level: 'wrench', angAccel: [60, 60, 15] },   // rad/s², the torque at |a| = 1 per unit inertia
  obsScale: { error: 2, velocity: 3, rates: 6 },          // m, m/s, rad/s: features are divided by these, clipped to ±5
  reward: {
    alive: 1.0,                              // per step, for staying in the air
    position: 1.0,                           // per m of distance to the target
    velocity: 0.05,                          // per m/s of speed
    rates: 0.05,                             // per rad/s of body rate
    action: 0.02,                            // per |a|²
    smooth: 0.1,                             // per |a − a_prev|²  (against chattering torques)
    crash: 50,                               // once, when the episode ends in a crash
  },
  episode: {
    seconds: 10,
    targetBox: [[-3, 3], [-3, 3], [0.5, 4]], // m, random targets and start points are drawn in this box
    targetEvery: [2, 5],                     // s, a new random target after a time drawn from this range
    startTilt: 20,                           // deg, largest initial roll / pitch
    startYaw: 180,                           // deg, largest initial heading (180 = any heading)
    startSpeed: 1,                           // m/s, largest initial speed per axis
    maxTilt: 85,                             // deg, crash when the drone tilts beyond this
    maxDistance: 8,                          // m, crash when this far from the target
  },
  // Domain randomisation, drawn once per episode: a factor in [1 − r, 1 + r] for the physical parameters,
  // a constant wind of up to `wind` m/s. A sequence model that sees its own past actions and their effect can
  // infer these online (in-context adaptation); info.params exposes the true values for privileged training.
  randomize: { mass: 0.2, inertia: 0.2, kThrust: 0.1, tau: 0.3, linearDrag: 0.5, wind: 2 },
};

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
  ml: mlDefaults,
};


/** A light 5-inch racer: small inertia, fast motors, high thrust-to-weight. */
export const racer5 = {
  name: 'Racer 5"',
  physics: { ...quadX450.physics, mass: 0.65, inertia: [0.0021, 0.0021, 0.0036], linearDrag: 0.15 },
  motor: { ...quadX450.motor, arm: 0.11, kThrust: 9.4e-7, kTorque: 1.4e-8, tau: 0.015, omegaMax: 2800 },
  controller: quadX450.controller,
  ml: mlDefaults,
};

/** A heavy-lift frame: large props, slow motors (bigger lag), low thrust-to-weight. */
export const heavy = {
  name: 'Heavy lift',
  physics: { ...quadX450.physics, mass: 3.0, inertia: [0.08, 0.08, 0.14], linearDrag: 0.5 },
  motor: { ...quadX450.motor, arm: 0.35, kThrust: 2.9e-5, kTorque: 6.0e-7, tau: 0.06, omegaMax: 800 },
  // slower motors: a lower rate-loop bandwidth keeps a phase margin against the 60 ms lag
  controller: { ...quadX450.controller, rate: { ...quadX450.controller.rate, kp: [10, 10, 6], kd: [0.1, 0.1, 0] }, attitude: { kp: [5, 5, 2.5] } },
  ml: mlDefaults,
};

export const presets = [quadX450, racer5, heavy];
export const config = quadX450;

/** A deep copy, so a user's edits (design panel, gain sliders) never change the presets themselves. */
export const cloneConfig = (c) => JSON.parse(JSON.stringify(c));
