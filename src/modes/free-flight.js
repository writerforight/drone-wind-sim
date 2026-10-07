/**
 * Free flight — manual flight in angle mode.
 *
 *   right stick  → desired roll / pitch angle (± maxTilt): the attitude loop holds it
 *   left stick   → yaw rate (± 2 rad/s) and climb rate (± 3 m/s); centre throttle holds the height
 *
 * Throttle as a climb-rate command (T = m (g + K_v (v_z,cmd − v_z)) / (cos φ cos θ)) instead of raw thrust:
 * with raw thrust a short burst of throttle leaves a vertical speed that never goes away, which is hard to
 * fly with a keyboard. The division by cos φ cos θ keeps the height when the drone tilts.
 */
const CLIMB = 3.0;      // m/s at full stick
const KV = 4.0;         // 1/s, vertical speed loop gain

export const freeFlight = {
  id: 'free',
  name: 'Free flight',
  description: 'Fly it yourself in angle mode: the sticks set the tilt and the climb rate, the controller holds them.',
  camera: 'chase',
  panels: ['plots'],
  start(app) { app.resetSim({ position: [0, 0, 0] }); },
  reference(app, sticks) {
    const C = app.cfg.controller, P = app.cfg.physics, tilt = (C.maxTilt * Math.PI) / 180;
    const roll = sticks.roll * tilt, pitch = sticks.pitch * tilt;
    const meas = app.sim.last ? app.sim.last.meas : app.sim.state;
    const vzCmd = CLIMB * sticks.throttle;
    const thrust = Math.max(0, (P.mass * (P.gravity + KV * (vzCmd - meas.v[2]))) / (Math.cos(roll) * Math.cos(pitch)));
    return { attitude: [roll, pitch], yawRate: 2 * sticks.yaw, thrust };
  },
};
