/**
 * controller/ — cascaded PID: position → attitude → body rate.
 *
 * Interface shared by every controller (PID now; LQR, MPC or a neural policy later):
 *
 *     const ctl = createCascadedPID(airframe);   // airframe = { mass, inertia, gravity }
 *     ctl.reset();
 *     const u = ctl.control(measurement, reference, params);   // → { thrust, torque: [τx, τy, τz] }
 *
 *   measurement  { p, v, q, w }        what the controller is allowed to know (from sensors/)
 *   reference    { position, velocity?, yaw? }                  position mode (automatic)
 *            or  { attitude: [roll, pitch], yawRate, thrust }   angle mode (manual flight)
 *   params       config.controller (gains, limits, dt) — read on every call, so gains can change live
 *
 * Every loop computes error = reference − measurement (negative feedback).
 *
 *   1. position (outer, PID):  a = Kp e_p + Ki ∫e_p + Kd (v_ref − v) + g e_z
 *                              (∫ only while |e_p| < iZone per axis: anti-windup by conditional integration)
 *   2. thrust and tilt:        T = m a·z_B ;  pitch_d = atan2(a_fwd, a_z), roll_d = atan2(−a_left, √(a_fwd² + a_z²))
 *                              (a_fwd, a_left: a turned into the yaw frame; tilt limited to maxTilt)
 *   3. attitude (P):           w_ref = Kp_att (angle_d − angle)        (yaw error wrapped to ±π)
 *   4. rate (inner, PID):      τ = J ⊙ (Kp e_w + Ki ∫e_w + Kd ė_w)
 */
import { dot, rotate, toEuler } from '../physics/math3.js';

const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

export function createCascadedPID(airframe) {
  let iPos, iRate, prevRateErr;
  const ctl = {
    name: 'cascaded PID',
    reset() { iPos = [0, 0, 0]; iRate = [0, 0, 0]; prevRateErr = null; },

    control(meas, ref, P) {
      const dt = P.dt, m = airframe.mass, g = airframe.gravity, J = airframe.inertia;
      const [roll, pitch, yaw] = toEuler(meas.q);
      let rollD, pitchD, yawRateRef, thrust;

      if (ref.attitude) {                                   // angle mode: the pilot sets tilt, yaw rate and thrust
        [rollD, pitchD] = ref.attitude;
        yawRateRef = ref.yawRate || 0;
        thrust = ref.thrust;
      } else {                                              // position mode
        const pos = P.position, vRef = ref.velocity || [0, 0, 0], a = [0, 0, 0];
        for (let k = 0; k < 3; k++) {
          const e = ref.position[k] - meas.p[k];
          // anti-windup (conditional integration): only integrate near the setpoint
          if (Math.abs(e) < pos.iZone) iPos[k] = clamp(iPos[k] + e * dt, -pos.iLimit, pos.iLimit);
          a[k] = pos.kp[k] * e + pos.ki[k] * iPos[k] + pos.kd[k] * (vRef[k] - meas.v[k]);
        }
        a[2] += g;                                          // gravity feed-forward
        const c = Math.cos(yaw), s = Math.sin(yaw);
        const aFwd = c * a[0] + s * a[1], aLeft = -s * a[0] + c * a[1];
        const maxT = (P.maxTilt * Math.PI) / 180;
        pitchD = clamp(Math.atan2(aFwd, a[2]), -maxT, maxT);
        rollD = clamp(Math.atan2(-aLeft, Math.hypot(aFwd, a[2])), -maxT, maxT);
        thrust = Math.max(0, m * dot(a, rotate(meas.q, [0, 0, 1])));   // only the part along the current z_B
        yawRateRef = P.attitude.kp[2] * wrap((ref.yaw || 0) - yaw);
      }

      const att = P.attitude.kp, mr = P.maxRate;
      const wRef = [
        clamp(att[0] * (rollD - roll), -mr[0], mr[0]),
        clamp(att[1] * (pitchD - pitch), -mr[1], mr[1]),
        clamp(yawRateRef, -mr[2], mr[2]),
      ];
      const R = P.rate, torque = [0, 0, 0], e = [0, 0, 0];
      for (let k = 0; k < 3; k++) {
        e[k] = wRef[k] - meas.w[k];
        iRate[k] = clamp(iRate[k] + e[k] * dt, -R.iLimit, R.iLimit);
        const de = prevRateErr ? (e[k] - prevRateErr[k]) / dt : 0;
        torque[k] = J[k] * (R.kp[k] * e[k] + R.ki[k] * iRate[k] + R.kd[k] * de);
      }
      prevRateErr = e;
      return { thrust, torque, debug: { rollD, pitchD, wRef } };
    },
  };
  ctl.reset();
  return ctl;
}
