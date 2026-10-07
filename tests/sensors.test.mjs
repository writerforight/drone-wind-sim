import { SENSOR_LEVELS, cloneConfig, quadX450 } from '../src/config.js';
import { createCascadedPID } from '../src/controller/cascaded-pid.js';
import { runScenario, createSimulation } from '../src/loop/simulation.js';
import { createState } from '../src/physics/rigidbody.js';
import { createSensors } from '../src/sensors/sensors.js';
import { close, ok } from './assert.mjs';

const P = quadX450.physics;
const withSensors = (level, seed = 1) => { const c = cloneConfig(quadX450); c.sensors = { level, seed }; return c; };

export const tests = {
  'delay: a measurement is exactly `delay` seconds old'() {
    const S = { ...SENSOR_LEVELS.ideal, delay: 0.04 }, measure = createSensors(S, P);
    let got = null;
    for (let i = 0; i <= 100; i++) {                    // a state moving at 1 m/s, sampled every 2 ms
      const t = i * 0.002, s = createState({ position: [t, 0, 0] });
      got = measure(s, t);
    }
    close(got.p[0], 0.2 - 0.04, 1e-9, 'delayed x');
  },

  'gyro noise and bias have the configured statistics'() {
    const S = SENSOR_LEVELS.poor, measure = createSensors(S, P, 7), s = createState();
    const N = 20000, sum = [0, 0, 0], sq = [0, 0, 0];
    for (let i = 0; i < N; i++) { const m = measure(s, i * 0.002); for (let k = 0; k < 3; k++) { sum[k] += m.w[k]; sq[k] += m.w[k] ** 2; } }
    for (let k = 0; k < 3; k++) {
      const mean = sum[k] / N, sd = Math.sqrt(sq[k] / N - mean * mean);
      ok(Math.abs(mean) <= S.gyroBias + 0.002, `bias ${mean}`);
      close(sd, S.gyroNoise, 0.002, 'gyro σ');
    }
  },

  'position is sampled at its own rate and held in between'() {
    const S = { ...SENSOR_LEVELS.ideal, positionRate: 50 }, measure = createSensors(S, P);
    const xs = [];
    for (let i = 0; i < 50; i++) xs.push(measure(createState({ position: [i * 0.002, 0, 0] }), i * 0.002).p[0]);
    ok(new Set(xs).size === 5, `${new Set(xs).size} distinct samples in 0.1 s at 50 Hz`);
  },

  'accelerometer reads +g up at hover (specific force), with the drone level'() {
    const cfg = cloneConfig(quadX450);
    const sim = createSimulation(cfg, { controller: createCascadedPID(cfg.physics), measure: createSensors(SENSOR_LEVELS.ideal, P), initial: { position: [0, 0, 2] } });
    sim.advance(1);
    close(sim.last.meas.accel[2], P.gravity, 1e-6, 'a_z'); close(sim.last.meas.accel[0], 0, 1e-6, 'a_x');
  },

  'hover with realistic and poor sensors stays within a few centimetres'() {
    for (const [level, lim] of [['realistic', 0.03], ['poor', 0.12]]) {
      const cfg = withSensors(level);
      const { log } = runScenario(cfg, createCascadedPID(cfg.physics), { seconds: 20, initial: { position: [0, 0, 2] }, refAt: () => ({ position: [0, 0, 2] }) });
      const tail = log.filter((r) => r.t > 5);
      const rms = Math.sqrt(tail.reduce((a, r) => a + r.p[0] ** 2 + r.p[1] ** 2 + (r.p[2] - 2) ** 2, 0) / tail.length);
      ok(rms < lim, `${level}: RMS position error ${rms.toFixed(4)} m (limit ${lim})`);
    }
  },
};
