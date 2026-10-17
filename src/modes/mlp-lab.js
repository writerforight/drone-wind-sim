/**
 * MLP lab (part of the tuning lab) — a target pin you move around; the main drone is to fly there with a
 * learned controller (an MLP, later phases). A semi-transparent "ghost" drone flown by the cascaded PID goes to
 * the same pin at the same desired speed, only for comparison: it has its own simulation, never touches the
 * main drone and never takes part in the learning.
 *
 *   keyboard   W/S/A/D move the pin forward / back / left / right, seen from the camera · ↑/↓ up / down
 *   gamepad, touch: the same sticks (right stick horizontal, left stick vertical)
 *
 * How the ghost goes "at a speed": a carrot. Its PID reference walks from where it is toward the pin at the
 * desired speed v (and carries v as a velocity feed-forward), so the PID follows a moving point instead of
 * jumping to a far setpoint.
 */
import { createCascadedPID } from '../controller/cascaded-pid.js';
import { createSimulation } from '../loop/simulation.js';
import { buildNet, createMlpController, defaultNetSettings } from '../ml/mlp-controller.js';

export const PIN_SPEED = 2.0;                       // m/s, how fast the sticks move the pin
export const PIN_LIMITS = { r: 40, zMin: 0.3, zMax: 15 };
export const DEFAULT_PIN = [3, 2, 2];

/** The controller of the main drone until a network flies it: motors off, so it rests on the floor. */
export const motorsOff = { name: 'none', reset() {}, control: () => ({ thrust: 0, torque: [0, 0, 0] }) };

/**
 * Moves the pin with the sticks, relative to the camera: `camYaw` is the orbit angle of the camera around its
 * target (the camera sits at (cos yaw, sin yaw) from it in world x, y), so "forward" points away from it.
 */
export function movePin(pin, sticks, camYaw, dt, speed = PIN_SPEED, lim = PIN_LIMITS) {
  const fx = -Math.cos(camYaw), fy = -Math.sin(camYaw);          // forward, horizontal
  const rx = fy, ry = -fx;                                        // right = forward turned −90°
  const p = [
    pin[0] + speed * dt * (sticks.pitch * fx + sticks.roll * rx),
    pin[1] + speed * dt * (sticks.pitch * fy + sticks.roll * ry),
    pin[2] + speed * dt * sticks.throttle,
  ];
  const r = Math.hypot(p[0], p[1]);
  if (r > lim.r) { p[0] *= lim.r / r; p[1] *= lim.r / r; }
  p[2] = Math.min(lim.zMax, Math.max(lim.zMin, p[2]));
  return p;
}

/** One step of the carrot: walk `ref` toward `pin` by at most v·dt; returns { position, velocity }. */
export function carrot(ref, pin, v, dt) {
  const d = [pin[0] - ref[0], pin[1] - ref[1], pin[2] - ref[2]], n = Math.hypot(...d);
  if (n <= v * dt || n < 1e-9) return { position: pin.slice(), velocity: [0, 0, 0] };
  const u = d.map((x) => x / n);
  return { position: ref.map((x, k) => x + u[k] * v * dt), velocity: u.map((x) => x * v) };
}

export const mlpLab = {
  id: 'mlp',
  name: 'Tuning lab · MLP',
  description: 'Move a pin anywhere; a neural network is to fly the drone there at the speed you set. A ghost PID drone flies the same task for comparison.',
  camera: 'orbit',
  panels: ['mlp', 'plots'],
  start(app) {
    // a restart keeps the pin and the network settings
    const L = (app.lab ||= { pin: DEFAULT_PIN.slice(), speed: 1.5, ghostOn: true, net: defaultNetSettings() });
    rebuildNetwork(app);
    app.resetSim({ position: [0, 0, 0] }, { controller: motorsOff });
    app.sim.reference = { position: L.pin.slice(), speed: L.speed, yaw: 0 };
    L.ghost = createSimulation(app.cfg, { controller: createCascadedPID(app.cfg.physics), initial: { position: [0, 0, 0] } });
    L.ghost.reference = { position: [0, 0, 0], velocity: [0, 0, 0], yaw: 0 };
  },
  reference(app, sticks) {
    const L = app.lab;
    L.pin = movePin(L.pin, sticks, app.cameraYaw(), app.cfg.controller.dt);
    return { position: L.pin.slice(), speed: L.speed, yaw: 0 };
  },
  /** After the main drone's tick: the ghost runs its own controller period. */
  afterTick(app) {
    const L = app.lab, g = L.ghost;
    // what the (not yet flying) network would command right now
    L.preview = L.controller.preview(app.sim.last ? app.sim.last.meas : app.sim.state, app.sim.reference);
    if (!L.ghostOn) return;
    const c = carrot(g.reference.position, L.pin, L.speed, app.cfg.controller.dt);
    g.reference = { position: c.position, velocity: c.velocity, yaw: 0 };
    g.tick();
  },
};

/** (Re)creates the network and its controller from the lab's settings (after a structural change in the panel). */
export function rebuildNetwork(app) {
  const L = app.lab;
  L.netObj = buildNet(L.net);
  L.controller = createMlpController(app.cfg.physics, L.netObj, L.net.inputs);
  L.preview = null;
}
