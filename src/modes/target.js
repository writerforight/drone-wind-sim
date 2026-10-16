/**
 * Target — give the drone a point to fly to and choose who flies it: the cascaded PID or a learned policy.
 *
 *   type x, y, z and press Go · double-click the floor (the height stays the typed z) · Random
 *
 * The drone in this mode is a copy of the configured one with an unknown payload and a wind added from the
 * panel: the controller is NOT told about them (it was built for the nominal drone), so you can watch how a
 * controller copes — and whether a sequence policy adapts — when the drone is not the one it expects.
 */
export const targetMode = {
  id: 'target',
  name: 'Target',
  description: 'Send the drone to a point (typed, clicked or random) with the PID or a learned policy; add payload and wind it does not know about.',
  camera: 'orbit',
  panels: ['target', 'plots'],
  usesPolicy: true,
  start(app) {
    app.resetSim({ position: [0, 0, 1.5] });
    app.sim.reference = { position: [0, 0, 1.5], yaw: 0 };
  },
  reference(app) { return app.sim.reference; },
};
