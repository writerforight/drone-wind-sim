/**
 * Tuning lab — the drone holds a position (cascaded PID, position mode). You give steps in x or z, change
 * the gains live and read the step response: rise time, overshoot, settling time and steady-state error.
 * For z the panel also prints the simplified analytic prediction ω_n = √Kp, ζ = Kd / (2√Kp)
 * (fast motors, no drag, no integral — docs/derivation.md).
 */
export const tuningLab = {
  id: 'lab',
  name: 'Tuning lab',
  description: 'Hover on a setpoint, apply steps, tune the gains live and read the step response.',
  camera: 'orbit',
  panels: ['gains', 'plots', 'steps'],
  start(app) {
    app.resetSim({ position: [0, 0, 1.5] });
    app.sim.reference = { position: [0, 0, 1.5], yaw: 0 };
  },
  reference(app) { return app.sim.reference; },
};

/** Step-response metrics of y(t) for a step from y0 to y1 applied at t0 (samples [{t, y}]). */
export function stepMetrics(samples, t0, y0, y1) {
  const A = y1 - y0;
  if (Math.abs(A) < 1e-9 || samples.length < 3) return null;
  const n = (y) => (y - y0) / A;                     // normalised: 0 → 1
  let t10 = null, t90 = null, peak = -Infinity, settle = null;
  for (const s of samples) {
    const v = n(s.y);
    if (t10 === null && v >= 0.1) t10 = s.t;
    if (t90 === null && v >= 0.9) t90 = s.t;
    peak = Math.max(peak, v);
  }
  for (let i = samples.length - 1; i >= 0; i--) {
    if (Math.abs(n(samples[i].y) - 1) > 0.02) { settle = i + 1 < samples.length ? samples[i + 1].t - t0 : null; break; }
    if (i === 0) settle = 0;
  }
  return {
    rise: t10 !== null && t90 !== null ? t90 - t10 : null,
    overshoot: Math.max(0, (peak - 1) * 100),
    settling: settle,
    steadyError: Math.abs(samples[samples.length - 1].y - y1),
  };
}
