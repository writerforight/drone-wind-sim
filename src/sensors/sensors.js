/**
 * sensors/ — what the controller is allowed to know: noisy, biased, delayed measurements of the state.
 *
 *   const measure = createSensors(params, airframe);   // params = SENSOR_LEVELS[...] (+ seed)
 *   meas = measure(state, t, { acc });                 // { p, v, q, w, accel } — the controller's input
 *                                                      // acc: the true acceleration v̇ (for the accelerometer)
 *
 *   gyro           w_m = w + b_g + n           b_g: constant bias per axis (±gyroBias), n ~ N(0, gyroNoise²)
 *   accelerometer  a_m = R(q)ᵀ (v̇ + g e_z) + n   specific force in the body frame (what an IMU feels)
 *   attitude       q_m = q ⊗ δq(n)              a small random rotation: the output of the IMU's attitude
 *                                               estimator, modelled directly (no filter is simulated)
 *   position       p_m = p + n, v_m = v + n    from a position system (GNSS / motion capture), sampled at
 *                                               positionRate and held in between
 *   delay          every measurement is `delay` seconds old (a ring buffer of past measurements)
 *
 * Simplification (stated in the docs): the attitude estimate is not computed from gyro + accelerometer by a
 * filter; its noise is given directly. That keeps the block small and replaceable by a real estimator.
 */
import { qmul, qnormalize, rotateInv } from '../physics/math3.js';
import { createRandom } from '../util/random.js';

export function createSensors(S, airframe, seed = 1) {
  const rnd = createRandom(seed);
  const n = (sd) => (sd ? sd * rnd.gauss() : 0);
  const bias = [0, 1, 2].map(() => (S.gyroBias ? rnd.range(-S.gyroBias, S.gyroBias) : 0));
  const buffer = [];                    // [{ t, meas }] — the delay line
  let lastPos = null, lastPosT = -Infinity, lastV = null;
  const g = airframe.gravity;

  return function measure(state, t, { acc = [0, 0, 0] } = {}) {
    // position system: a new sample every 1 / positionRate seconds
    if (t - lastPosT >= 1 / S.positionRate - 1e-9 || !lastPos) {
      lastPos = state.p.map((x) => x + n(S.positionNoise));
      lastV = state.v.map((x) => x + n(S.velocityNoise));
      lastPosT = t;
    }
    const half = (a) => a / 2;
    const dq = qnormalize([1, half(n(S.attitudeNoise)), half(n(S.attitudeNoise)), half(n(S.attitudeNoise))]);
    // specific force: the acceleration the airframe feels, without gravity (finite difference of v)
    const accel = rotateInv(state.q, [acc[0], acc[1], acc[2] + g]).map((x) => x + n(S.accelNoise));
    const meas = {
      p: lastPos.slice(), v: lastV.slice(), q: qmul(state.q, dq),
      w: state.w.map((x, k) => x + bias[k] + n(S.gyroNoise)), accel,
    };
    buffer.push({ t, meas });
    // hand out the newest measurement that is at least `delay` old
    while (buffer.length > 1 && buffer[1].t <= t - S.delay + 1e-9) buffer.shift();
    return buffer[0].meas;
  };
}
