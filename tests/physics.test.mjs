import { quadX450 } from '../src/config.js';
import { fromEuler, norm, rotate, rotateInv, sub, toEuler } from '../src/physics/math3.js';
import { angularMomentumWorld, createState, energy, step } from '../src/physics/rigidbody.js';
import { close, ok } from './assert.mjs';

const P = { ...quadX450.physics };
const noDrag = { ...P, linearDrag: 0 };

export const tests = {
  'quaternion rotate / inverse / Euler round trip'() {
    const q = fromEuler(0.3, -0.5, 1.2), v = [0.4, -1.1, 2.0];
    const back = rotateInv(q, rotate(q, v));
    ok(norm(sub(back, v)) < 1e-12, 'R(q)ᵀ R(q) v ≠ v');
    const e = toEuler(q);
    close(e[0], 0.3, 1e-12, 'roll'); close(e[1], -0.5, 1e-12, 'pitch'); close(e[2], 1.2, 1e-12, 'yaw');
    // yaw by 90°: body x (forward) points to world y (north)
    const f = rotate(fromEuler(0, 0, Math.PI / 2), [1, 0, 0]);
    ok(norm(sub(f, [0, 1, 0])) < 1e-12, 'yaw convention');
  },

  'free fall matches z = −½ g t² (RK4 is exact for it)'() {
    let s = createState({ position: [0, 0, 100] });
    const n = 500;                                   // 1 s
    for (let i = 0; i < n; i++) s = step(s, {}, noDrag);
    close(s.p[2], 100 - 0.5 * P.gravity, 1e-9, 'z after 1 s');
    close(s.v[2], -P.gravity, 1e-9, 'vz after 1 s');
  },

  'thrust = weight keeps a level drone at rest'() {
    let s = createState({ position: [0, 0, 5] });
    const u = { force: [0, 0, P.mass * P.gravity], torque: [0, 0, 0] };
    for (let i = 0; i < 5000; i++) s = step(s, u, P);   // 10 s
    ok(norm(sub(s.p, [0, 0, 5])) < 1e-9 && norm(s.v) < 1e-9, `drifted to ${s.p}`);
  },

  'energy and angular momentum are conserved without drag and motors'() {
    // a tumbling, thrown body: rotation about all three (unequal) axes, gravity on, no drag, no motors
    let s = createState({ position: [0, 0, 10], velocity: [3, -1, 4], rates: [4, -2, 6] });
    const E0 = energy(s, noDrag), L0 = angularMomentumWorld(s, noDrag);
    let worst = 0;
    for (let i = 0; i < 5000; i++) {                   // 10 s
      s = step(s, {}, noDrag);
      worst = Math.max(worst, Math.abs(energy(s, noDrag) - E0) / Math.abs(E0));
    }
    ok(worst < 1e-8, `relative energy drift ${worst.toExponential(2)}`);
    const dL = norm(sub(angularMomentumWorld(s, noDrag), L0)) / norm(L0);
    ok(dL < 1e-8, `relative angular-momentum drift ${dL.toExponential(2)}`);
  },

  'drag gives the terminal speed m g / c_d'() {
    let s = createState({ position: [0, 0, 1000] });
    for (let i = 0; i < 50000; i++) s = step(s, {}, P);  // 100 s ≫ m / c_d = 4.8 s
    close(-s.v[2], (P.mass * P.gravity) / P.linearDrag, 1e-6, 'terminal speed');
  },
};
