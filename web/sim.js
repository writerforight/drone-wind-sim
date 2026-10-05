/*
 * dronesim in JavaScript: a line-by-line port of the Python package (dronesim/) for the browser.
 * Same models, same conventions (world ENU z-up, body FLU, quaternion w x y z body->world, rev/s).
 * tests: node web/parity_test.js compares it against the Python simulator.
 */
(function (root) {
  'use strict';

  // ---------------------------------------------------------------------------------------------
  // small vector helpers (arrays of length 3)
  // ---------------------------------------------------------------------------------------------
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const scl = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = (a) => Math.hypot(a[0], a[1], a[2]);
  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
  const matvec = (M, v) => [dot(M[0], v), dot(M[1], v), dot(M[2], v)];
  const matTvec = (M, v) => [M[0][0] * v[0] + M[1][0] * v[1] + M[2][0] * v[2],
    M[0][1] * v[0] + M[1][1] * v[1] + M[2][1] * v[2], M[0][2] * v[0] + M[1][2] * v[1] + M[2][2] * v[2]];
  const diagMul = (d, v) => [d[0] * v[0], d[1] * v[1], d[2] * v[2]];
  const RAD = Math.PI / 180;

  // seeded random numbers (mulberry32 + Box-Muller)
  function makeRng(seed) {
    let s = seed >>> 0, spare = null;
    const uniform = () => {
      s = (s + 0x6D2B79F5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const normal = () => {
      if (spare !== null) { const v = spare; spare = null; return v; }
      let u = 0, v = 0;
      while (u === 0) u = uniform();
      v = uniform();
      const r = Math.sqrt(-2 * Math.log(u));
      spare = r * Math.sin(2 * Math.PI * v);
      return r * Math.cos(2 * Math.PI * v);
    };
    return { uniform, normal };
  }

  // ---------------------------------------------------------------------------------------------
  // quaternions
  // ---------------------------------------------------------------------------------------------
  function quatMul(a, b) {
    return [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
      a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
      a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
      a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]];
  }
  function quatToRot(q) {
    const [w, x, y, z] = q;
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]];
  }
  function eulerToQuat(r, p, y) {
    const cr = Math.cos(r / 2), sr = Math.sin(r / 2), cp = Math.cos(p / 2), sp = Math.sin(p / 2);
    const cy = Math.cos(y / 2), sy = Math.sin(y / 2);
    return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy];
  }
  function quatToEuler(q) {
    const [w, x, y, z] = q;
    return [Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
      Math.asin(clamp(2 * (w * y - z * x), -1, 1)),
      Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))];
  }

  // ---------------------------------------------------------------------------------------------
  // atmosphere (ISA troposphere)
  // ---------------------------------------------------------------------------------------------
  const R_AIR = 287.05287, G0 = 9.80665, LAPSE = 0.0065, T_SL = 288.15, P_SL = 101325.0;
  class Atmosphere {
    constructor(temperatureC = 15, groundAltitude = 0) {
      this.temperatureC = temperatureC; this.groundAltitude = groundAltitude;
      const tIsaSite = T_SL - LAPSE * groundAltitude;
      this.T0 = temperatureC + 273.15;
      this.p0 = P_SL * Math.pow(tIsaSite / T_SL, G0 / (R_AIR * LAPSE));
    }
    temperature(h) { return this.T0 - LAPSE * h; }
    pressure(h) { return this.p0 * Math.pow(this.temperature(h) / this.T0, G0 / (R_AIR * LAPSE)); }
    density(h) { return this.pressure(h) / (R_AIR * this.temperature(h)); }
  }

  // ---------------------------------------------------------------------------------------------
  // wind: log profile + Dryden turbulence + 1-cos gusts
  // ---------------------------------------------------------------------------------------------
  const FT = 0.3048, KT = 0.514444;
  const SEVERITY_W20 = { none: 0, light: 15 * KT, moderate: 30 * KT, severe: 45 * KT };
  function windVector(speed, fromDeg, vertical = 0) {
    const a = fromDeg * RAD;
    return [-speed * Math.sin(a), -speed * Math.cos(a), vertical];
  }
  class Gust {
    constructor(t0, duration, speed, fromDeg = 0, vertical = 0) { Object.assign(this, { t0, duration, speed, fromDeg, vertical }); }
    at(t) {
      const s = (t - this.t0) / this.duration;
      if (s <= 0 || s >= 1) return [0, 0, 0];
      return scl(windVector(this.speed, this.fromDeg, this.vertical), 0.5 * (1 - Math.cos(2 * Math.PI * s)));
    }
  }
  function drydenParams(h, w20) {
    const hf = clamp(h / FT, 10, 1000), k = 0.177 + 0.000823 * hf;
    const Lw = hf * FT, Lu = hf / Math.pow(k, 1.2) * FT, sw = 0.1 * w20, su = sw / Math.pow(k, 0.4);
    return { L: [Lu, Lu, Lw], sigma: [su, su, sw] };
  }
  class DrydenTurbulence {
    constructor(w20, rng) { this.w20 = w20; this.rng = rng; this.xu = 0; this.zv = [0, 0]; this.zw = [0, 0]; }
    reset(h) {
      const { sigma } = drydenParams(h, this.w20), r = this.rng;
      // stationary covariance of (z1, z2)/sigma^2 = [[1/2, 1/4], [1/4, 1/4]]; Cholesky factor:
      const c11 = Math.SQRT1_2, c21 = 0.25 / c11, c22 = Math.sqrt(0.25 - c21 * c21);
      this.xu = sigma[0] * r.normal();
      for (const [i, z] of [[1, this.zv], [2, this.zw]]) {
        const a = r.normal(), b = r.normal();
        z[0] = sigma[i] * c11 * a; z[1] = sigma[i] * (c21 * a + c22 * b);
      }
    }
    output() {
      const S3 = Math.sqrt(3);
      return [this.xu, S3 * this.zv[0] + (1 - S3) * this.zv[1], S3 * this.zw[0] + (1 - S3) * this.zw[1]];
    }
    step(dt, h, V) {
      if (this.w20 <= 0) return [0, 0, 0];
      const { L, sigma } = drydenParams(h, this.w20), Vc = Math.max(V, 0.5);
      const tau = [L[0] / Vc, L[1] / Vc, L[2] / Vc];
      const n = Math.max(1, Math.ceil(dt / (0.05 * Math.min(...tau)))), hh = dt / n, r = this.rng;
      for (let k = 0; k < n; k++) {
        this.xu += -this.xu / tau[0] * hh + sigma[0] * Math.sqrt(2 * hh / tau[0]) * r.normal();
        for (const [i, z] of [[1, this.zv], [2, this.zw]]) {
          const t = tau[i], z1 = z[0];
          z[0] += -z1 / t * hh + sigma[i] * Math.sqrt(hh / t) * r.normal();
          z[1] += (z1 - z[1]) / t * hh;
        }
      }
      return this.output();
    }
  }
  class WindField {
    constructor(o = {}) {
      this.meanSpeed = o.meanSpeed ?? 0; this.directionDeg = o.directionDeg ?? 0;
      this.refHeight = o.refHeight ?? 10; this.roughness = o.roughness ?? 0.03;
      this.vertical = o.vertical ?? 0; this.gusts = o.gusts ?? []; this.seed = o.seed ?? 0;
      this.setTurbulence(o.turbulence ?? 'none');
      this.turbWorld = [0, 0, 0];
    }
    setTurbulence(t) {
      this.turbulence = t;
      this.w20 = typeof t === 'string' ? SEVERITY_W20[t] : +t;
      if (this.dryden) { this.dryden.w20 = this.w20; } else { this.rng = makeRng(this.seed); this.dryden = new DrydenTurbulence(this.w20, this.rng); }
    }
    mean(h) {
      h = Math.max(h, 2 * this.roughness);
      const U = this.meanSpeed * Math.log(h / this.roughness) / Math.log(this.refHeight / this.roughness);
      return windVector(Math.max(U, 0), this.directionDeg, this.vertical);
    }
    gust(t) { let g = [0, 0, 0]; for (const gu of this.gusts) g = add(g, gu.at(t)); return g; }
    axes(rel) {
      let hor = [rel[0], rel[1], 0], n = norm(hor);
      if (n < 1e-3) { hor = windVector(1, this.directionDeg); hor[2] = 0; n = norm(hor); }
      const eu = scl(hor, 1 / n), ew = [0, 0, 1], ev = cross(ew, eu);
      return (v) => add(add(scl(eu, v[0]), scl(ev, v[1])), scl(ew, v[2]));
    }
    reset(p, v = [0, 0, 0]) {
      this.rng = makeRng(this.seed); this.dryden = new DrydenTurbulence(this.w20, this.rng);
      const rel = sub(this.mean(p[2]), v);
      this.dryden.reset(p[2]);
      this.turbWorld = this.w20 ? this.axes(rel)(this.dryden.output()) : [0, 0, 0];
      return this.value(0, p);
    }
    step(t, dt, p, v) {
      const rel = sub(this.mean(p[2]), v);
      this.turbWorld = this.axes(rel)(this.dryden.step(dt, p[2], norm(rel)));
      return this.value(t + dt, p);
    }
    value(t, p) { return add(add(this.mean(p[2]), this.turbWorld), this.gust(t)); }
  }

  // ---------------------------------------------------------------------------------------------
  // motors and airframes
  // ---------------------------------------------------------------------------------------------
  class MotorModel {
    constructor(o = {}) {
      this.diameter = o.diameter ?? 0.254; this.cT = o.cT ?? 0.11; this.cQ = o.cQ ?? 0.0045;
      this.tau = o.tau ?? 0.04; this.tauDown = o.tauDown ?? null; this.nMin = o.nMin ?? 10; this.nMax = o.nMax ?? 135;
      this.efficiency = o.efficiency ?? 0.75; this.health = o.health ?? 1;
    }
    copy(ch = {}) { return new MotorModel(Object.assign({}, this, ch)); }
    thrust(n, rho) { return this.health * this.cT * rho * n * n * this.diameter ** 4; }
    torque(n, rho) { return this.health * this.cQ * rho * n * n * this.diameter ** 5; }
    power(n, rho) { return 2 * Math.PI * n * this.torque(n, rho) / this.efficiency; }
    speedForThrust(T, rho) { return Math.sqrt(Math.max(T, 0) / (Math.max(this.health, 1e-6) * this.cT * rho * this.diameter ** 4)); }
    maxThrust(rho) { return this.thrust(this.nMax, rho); }
    torquePerThrust() { return this.cQ / this.cT * this.diameter; }
  }
  class Airframe {
    constructor(name, mass, rotors, o = {}) {
      this.name = name; this.mass = mass; this.rotors = rotors;
      this.dragArea = o.dragArea ?? [0.012, 0.012, 0.03]; this.rotorDrag = o.rotorDrag ?? 1.0e-3;
      this.bodyMassFraction = o.bodyMassFraction ?? 0.6; this.bodySize = o.bodySize ?? [0.15, 0.10, 0.06];
      this.inertia = o.inertia ?? this.estimateInertia();          // diagonal [Jx, Jy, Jz]
      this.angDamping = o.angDamping ?? this.inertia.map((j) => 0.07 * j);
      this.label = o.label ?? name; this.source = o.source ?? '';
    }
    estimateInertia() {
      // diagonal of the full estimate (presets are symmetric enough that off-diagonals are ~0)
      const mb = this.bodyMassFraction * this.mass, [a, b, c] = this.bodySize;
      const J = [mb / 12 * (b * b + c * c), mb / 12 * (a * a + c * c), mb / 12 * (a * a + b * b)];
      const mr = (1 - this.bodyMassFraction) * this.mass / this.rotors.length;
      for (const r of this.rotors) {
        const [x, y, z] = r.position;
        J[0] += mr * (y * y + z * z); J[1] += mr * (x * x + z * z); J[2] += mr * (x * x + y * y);
      }
      return J;
    }
    get n() { return this.rotors.length; }
    allocation() { return this.rotors.map((r) => [1, r.position[1], -r.position[0], -r.spin * r.motor.torquePerThrust()]); } // columns
    hoverThrust() { return this.mass * G0; }
    thrustToWeight(rho = 1.225) { return this.rotors.reduce((s, r) => s + r.motor.maxThrust(rho), 0) / (this.mass * G0); }
  }
  const rotor = (x, y, z, spin, motor) => ({ position: [x, y, z], spin, motor });
  function ring(n, arm, startDeg, spins, motor) {
    const out = [];
    for (let i = 0; i < n; i++) {
      const a = (startDeg + 360 * i / n) * RAD;
      out.push(rotor(arm * Math.cos(a), arm * Math.sin(a), 0, spins[i % spins.length], motor.copy()));
    }
    return out;
  }
  const RHO0 = 1.225, TWO_PI = 2 * Math.PI;
  function fromGazebo(kMotor, kMoment, D, wMax, tauUp, tauDown, idle = 0.08) {
    const cT = kMotor * TWO_PI ** 2 / (RHO0 * D ** 4), nMax = wMax / TWO_PI;
    return new MotorModel({ diameter: D, cT, cQ: kMoment * cT / D, tau: tauUp, tauDown, nMin: idle * nMax, nMax });
  }
  const PRESETS = {
    iris: () => {
      const m = fromGazebo(5.84e-06, 0.06, 0.256, 1100, 0.0125, 0.025);
      const rs = [[0.13, -0.22, 1], [-0.13, 0.20, 1], [0.13, 0.22, -1], [-0.13, -0.20, -1]].map(([x, y, s]) => rotor(x, y, 0.023, s, m.copy()));
      return new Airframe('iris', 1.535, rs, { inertia: [0.029125, 0.029125, 0.055225], rotorDrag: TWO_PI * 0.000175,
        label: 'Iris (PX4)', source: 'PX4 Gazebo iris.sdf' });
    },
    x500: () => {
      const m = fromGazebo(8.54858e-06, 0.016, 0.2792, 1000, 0.0125, 0.025);
      const rs = [[0.174, -0.174, 1], [-0.174, 0.174, 1], [0.174, 0.174, -1], [-0.174, -0.174, -1]].map(([x, y, s]) => rotor(x, y, 0.06, s, m.copy()));
      return new Airframe('x500', 2.0 + 4 * 0.016076923, rs, { inertia: [0.0216667, 0.0216667, 0.04], rotorDrag: TWO_PI * 8.06428e-05,
        dragArea: [0.015, 0.015, 0.04], label: 'X500 (PX4)', source: 'PX4 Gazebo x500 model.sdf' });
    },
    crazyflie: () => {
      const mass = 0.027, kf = 3.16e-10, km = 7.94e-12, D = 2 * 2.31348e-2;
      const nMax = Math.sqrt(2.25 * 9.8 * mass / (4 * kf)) / 60;
      const m = new MotorModel({ diameter: D, cT: kf * 3600 / (RHO0 * D ** 4), cQ: km * 3600 / (RHO0 * D ** 5), tau: 0.02, nMin: 0.08 * nMax, nMax });
      const rs = [[0.028, -0.028, 1], [-0.028, -0.028, -1], [-0.028, 0.028, 1], [0.028, 0.028, -1]].map(([x, y, s]) => rotor(x, y, 0, s, m.copy()));
      return new Airframe('crazyflie', mass, rs, { inertia: [1.4e-5, 1.4e-5, 2.17e-5], rotorDrag: TWO_PI * 9.1785e-7,
        dragArea: [2e-3, 2e-3, 4e-3], label: 'Crazyflie 2.X', source: 'gym-pybullet-drones cf2x.urdf' });
    },
    quad_x: () => new Airframe('quad_x', 1.5, ring(4, 0.225, 45, [-1, 1], new MotorModel()), { label: 'Quad X (generic)' }),
    hexa_x: () => new Airframe('hexa_x', 2.2, ring(6, 0.275, 30, [-1, 1], new MotorModel()), { label: 'Hexa X (generic)' }),
    octo_x: () => new Airframe('octo_x', 3.2, ring(8, 0.35, 22.5, [-1, 1], new MotorModel()), { label: 'Octo X (generic)' }),
    quad_asymmetric: () => {
      const m = new MotorModel(), com = 0.04;
      const rs = [[0.20, -0.26, -1], [0.20, 0.26, 1], [-0.22, 0.17, -1], [-0.22, -0.17, 1]].map(([x, y, s]) => rotor(x - com, y, 0, s, m.copy()));
      return new Airframe('quad_asymmetric', 1.6, rs, { label: 'Quad, asymmetric arms' });
    },
  };

  // ---------------------------------------------------------------------------------------------
  // rigid-body dynamics (RK4)
  // ---------------------------------------------------------------------------------------------
  class Dynamics {
    constructor(af, atm) {
      this.af = af; this.atm = atm; this.refresh();
    }
    refresh() {
      const rs = this.af.rotors;
      this.kT = rs.map((r) => r.motor.health * r.motor.cT * r.motor.diameter ** 4);
      this.kQ = rs.map((r) => r.motor.health * r.motor.cQ * r.motor.diameter ** 5);
      this.tauUp = rs.map((r) => r.motor.tau);
      this.tauDown = rs.map((r) => r.motor.tauDown ?? r.motor.tau);
      this.nMin = rs.map((r) => r.motor.nMin); this.nMax = rs.map((r) => r.motor.nMax);
    }
    setMotorHealth(i, h) { this.af.rotors[i].motor.health = h; this.refresh(); }
    wrench(s, wind, rho) {
      const af = this.af, R = quatToRot(s.q), vrel = matTvec(R, sub(s.v, wind)), sp = norm(vrel);
      let Tsum = 0, tx = 0, ty = 0, tz = 0, nsum = 0;
      const T = new Array(af.n);
      for (let i = 0; i < af.n; i++) {
        const n2 = s.n[i] * s.n[i] * rho, Ti = this.kT[i] * n2, Qi = this.kQ[i] * n2, r = af.rotors[i];
        T[i] = Ti; Tsum += Ti; nsum += s.n[i];
        tx += r.position[1] * Ti; ty -= r.position[0] * Ti; tz -= r.spin * Qi;
      }
      const F = [0, 0, Tsum];
      for (let k = 0; k < 3; k++) F[k] += -0.5 * rho * af.dragArea[k] * sp * vrel[k];
      F[0] += -af.rotorDrag * nsum * vrel[0]; F[1] += -af.rotorDrag * nsum * vrel[1];
      const tau = [tx - af.angDamping[0] * s.w[0], ty - af.angDamping[1] * s.w[1], tz - af.angDamping[2] * s.w[2]];
      return { F, tau, T, R };
    }
    deriv(s, nCmd, wind, rho) {
      const af = this.af, { F, tau, R } = this.wrench(s, wind, rho);
      const acc = add([0, 0, -G0], scl(matvec(R, F), 1 / af.mass));
      const qd = quatMul(s.q, [0, s.w[0], s.w[1], s.w[2]]).map((x) => 0.5 * x);
      const J = af.inertia, Jw = diagMul(J, s.w), c = cross(s.w, Jw);
      const wd = [(tau[0] - c[0]) / J[0], (tau[1] - c[1]) / J[1], (tau[2] - c[2]) / J[2]];
      const nd = s.n.map((n, i) => { const dn = clamp(nCmd[i], this.nMin[i], this.nMax[i]) - n; return dn / (dn >= 0 ? this.tauUp[i] : this.tauDown[i]); });
      return { p: s.v, v: acc, q: qd, w: wd, n: nd };
    }
    step(s, nCmd, wind, dt) {
      const rho = this.atm.density(Math.max(s.p[2], 0));
      const mv = (a, k, h) => ({ p: add(a.p, scl(k.p, h)), v: add(a.v, scl(k.v, h)), q: a.q.map((x, i) => x + h * k.q[i]),
        w: add(a.w, scl(k.w, h)), n: a.n.map((x, i) => x + h * k.n[i]) });
      const k1 = this.deriv(s, nCmd, wind, rho), k2 = this.deriv(mv(s, k1, dt / 2), nCmd, wind, rho);
      const k3 = this.deriv(mv(s, k2, dt / 2), nCmd, wind, rho), k4 = this.deriv(mv(s, k3, dt), nCmd, wind, rho);
      const comb = (f) => (a, i) => a + dt / 6 * (k1[f][i] + 2 * k2[f][i] + 2 * k3[f][i] + k4[f][i]);
      const out = { p: s.p.map(comb('p')), v: s.v.map(comb('v')), q: s.q.map(comb('q')), w: s.w.map(comb('w')), n: s.n.map(comb('n')) };
      const qn = Math.hypot(...out.q); out.q = out.q.map((x) => x / qn);
      out.n = out.n.map((x, i) => clamp(x, 0, this.nMax[i]));
      if (out.p[2] < 0) {
        out.p[2] = 0; if (out.v[2] < 0) out.v[2] = 0;
        out.v[0] *= 0.5; out.v[1] *= 0.5; out.w = scl(out.w, 0.5);
      }
      return out;
    }
  }

  // ---------------------------------------------------------------------------------------------
  // sensors (estimate error model)
  // ---------------------------------------------------------------------------------------------
  const SENSOR_PRESETS = {
    ideal: { posStd: 0, velStd: 0, attStdDeg: 0, gyroStd: 0, gyroBiasWalk: 0, gnssRate: 1e9 },
    good: { posStd: 0.05, velStd: 0.05, attStdDeg: 0.3, gyroStd: 0.005, gyroBiasWalk: 0.0005, gnssRate: 50 },
    gps: { posStd: 0.4, velStd: 0.1, attStdDeg: 0.5, gyroStd: 0.01, gyroBiasWalk: 0.0005, gnssRate: 10 },
    poor: { posStd: 1.0, velStd: 0.25, attStdDeg: 1.5, gyroStd: 0.03, gyroBiasWalk: 0.003, gnssRate: 5 },
  };
  class Sensors {
    constructor(preset = 'good', seed = 1) { this.cfg = SENSOR_PRESETS[preset]; this.seed = seed; }
    reset(s) { this.rng = makeRng(this.seed); this.bias = [0, 0, 0]; this.tG = -1e9; this.p = s.p.slice(); this.v = s.v.slice(); }
    measure(t, dt, s) {
      const c = this.cfg, r = this.rng, nv = () => [r.normal(), r.normal(), r.normal()];
      if (t - this.tG >= 1 / c.gnssRate - 1e-9) { this.tG = t; this.p = add(s.p, scl(nv(), c.posStd)); this.v = add(s.v, scl(nv(), c.velStd)); }
      this.bias = add(this.bias, scl(nv(), c.gyroBiasWalk * Math.sqrt(dt)));
      const e = scl(nv(), c.attStdDeg * RAD);
      return { p: this.p.slice(), v: this.v.slice(), q: quatMul(s.q, eulerToQuat(e[0], e[1], e[2])), w: add(add(s.w, this.bias), scl(nv(), c.gyroStd)) };
    }
  }

  // ---------------------------------------------------------------------------------------------
  // control: mixer + cascaded PID
  // ---------------------------------------------------------------------------------------------
  function pinv4(cols) {
    // pseudo-inverse of B (4 x n, given as n columns): B^+ = B^T (B B^T)^-1   (B has full row rank)
    const n = cols.length, BBt = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
    for (const c of cols) for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) BBt[i][j] += c[i] * c[j];
    const inv = invert4(BBt);
    return cols.map((c) => [0, 1, 2, 3].map((j) => c[0] * inv[0][j] + c[1] * inv[1][j] + c[2] * inv[2][j] + c[3] * inv[3][j])); // n x 4
  }
  function invert4(M) {
    const A = M.map((r, i) => [...r, ...[0, 1, 2, 3].map((j) => (i === j ? 1 : 0))]);
    for (let c = 0; c < 4; c++) {
      let p = c; for (let r = c + 1; r < 4; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
      [A[c], A[p]] = [A[p], A[c]];
      const d = A[c][c]; for (let j = 0; j < 8; j++) A[c][j] /= d;
      for (let r = 0; r < 4; r++) if (r !== c) { const f = A[r][c]; for (let j = 0; j < 8; j++) A[r][j] -= f * A[c][j]; }
    }
    return A.map((r) => r.slice(4));
  }
  class Mixer {
    constructor(af, rhoAssumed = 1.225) {
      this.af = af; this.rho = rhoAssumed;
      this.Bp = pinv4(af.allocation());
      this.motors = af.rotors.map((r) => r.motor.copy({ health: 1 }));
      this.tMin = this.motors.map((m) => m.thrust(m.nMin, rhoAssumed));
      this.tMax = this.motors.map((m) => m.thrust(m.nMax, rhoAssumed));
    }
    fits(T) { return T.every((t, i) => t >= this.tMin[i] - 1e-9 && t <= this.tMax[i] + 1e-9); }
    part(u) { return this.Bp.map((row) => row[0] * u[0] + row[1] * u[1] + row[2] * u[2] + row[3] * u[3]); }
    thrusts(u) {
      const base = this.part([u[0], 0, 0, 0]), rp = this.part([0, u[1], u[2], 0]), yaw = this.part([0, 0, 0, u[3]]);
      const T = base.map((b, i) => b + rp[i] + yaw[i]);
      if (this.fits(T)) return T;
      for (const [part, rest] of [[yaw, base.map((b, i) => b + rp[i])], [rp, base]]) {
        if (this.fits(rest)) {
          let lo = 0, hi = 1;
          for (let k = 0; k < 20; k++) { const mid = 0.5 * (lo + hi); if (this.fits(rest.map((r, i) => r + mid * part[i]))) lo = mid; else hi = mid; }
          return rest.map((r, i) => r + lo * part[i]);
        }
      }
      return base.map((b, i) => clamp(b, this.tMin[i], this.tMax[i]));
    }
    speeds(u) { return this.thrusts(u).map((t, i) => this.motors[i].speedForThrust(clamp(t, this.tMin[i], this.tMax[i]), this.rho)); }
    maxCollective() { return this.tMax.reduce((a, b) => a + b, 0); }
  }
  const GAINS = {
    untuned: { posKp: 0.6, velKp: 1.2, velKi: 0, velKd: 0, attKp: [3, 3, 1], rateKp: [6, 6, 2], rateKi: [0, 0, 0], rateKd: [0, 0, 0] },
    tuned: { posKp: 1.4, velKp: 3.2, velKi: 1.4, velKd: 0.15, attKp: [9, 9, 3], rateKp: [22, 22, 8], rateKi: [6, 6, 2], rateKd: [0.25, 0.25, 0] },
  };
  const LIMITS = { vXY: 8, vZ: 3, aXY: 8, aZ: 6, tiltDeg: 40, rate: [220 * RAD, 220 * RAD, 120 * RAD], velInt: 8, rateInt: 3 };
  class PID {
    constructor(kp, ki, kd, intLimit, dTau = 0.02) {
      const v = (k) => (Array.isArray(k) ? k.slice() : [k, k, k]);
      this.kp = v(kp); this.ki = v(ki); this.kd = v(kd); this.intLimit = intLimit; this.dTau = dTau; this.reset();
    }
    reset() { this.i = [0, 0, 0]; this.prev = null; this.d = [0, 0, 0]; }
    update(err, meas, dt) {
      this.i = this.i.map((x, k) => clamp(x + this.ki[k] * err[k] * dt, -this.intLimit, this.intLimit));
      if (this.prev === null) this.prev = meas.slice();
      const raw = meas.map((m, k) => -(m - this.prev[k]) / dt);
      this.prev = meas.slice();
      const a = dt / (this.dTau + dt);
      this.d = this.d.map((d, k) => d + a * (raw[k] - d));
      return err.map((e, k) => this.kp[k] * e + this.i[k] + this.kd[k] * this.d[k]);
    }
  }
  class CascadedPID {
    constructor(af, gains = 'tuned', rhoAssumed = 1.225) {
      this.g = typeof gains === 'string' ? GAINS[gains] : Object.assign({}, GAINS.tuned, gains);
      this.lim = Object.assign({}, LIMITS); this.af = af; this.mass = af.mass; this.J = af.inertia;
      this.mixer = new Mixer(af, rhoAssumed);
      this.velPid = new PID(this.g.velKp, this.g.velKi, this.g.velKd, this.lim.velInt);
      this.ratePid = new PID(this.g.rateKp, this.g.rateKi, this.g.rateKd, this.lim.rateInt);
      this.last = {};
    }
    reset() { this.velPid.reset(); this.ratePid.reset(); }
    accelerationCommand(est, ref, dt) {
      const L = this.lim;
      const vsp = add(scl(sub(ref.p, est.p), this.g.posKp), ref.v || [0, 0, 0]);
      const h = Math.hypot(vsp[0], vsp[1]); if (h > L.vXY) { vsp[0] *= L.vXY / h; vsp[1] *= L.vXY / h; }
      vsp[2] = clamp(vsp[2], -L.vZ, L.vZ);
      return this.velocityToAccel(est, vsp, ref.a || [0, 0, 0], dt);
    }
    velocityToAccel(est, vsp, aff, dt) {
      const L = this.lim, a = add(this.velPid.update(sub(vsp, est.v), est.v, dt), aff);
      const h = Math.hypot(a[0], a[1]); if (h > L.aXY) { a[0] *= L.aXY / h; a[1] *= L.aXY / h; }
      a[2] = clamp(a[2], -L.aZ, L.aZ);
      this.last.vSp = vsp; this.last.aCmd = a;
      return a;
    }
    attitudeAndThrust(est, a, yaw) {
      const f = scl(add(a, [0, 0, G0]), this.mass);
      f[2] = Math.max(f[2], 0.1 * this.mass * G0);
      const tilt = Math.atan2(Math.hypot(f[0], f[1]), f[2]), tmax = this.lim.tiltDeg * RAD;
      if (tilt > tmax) { const k = Math.tan(tmax) * f[2] / Math.hypot(f[0], f[1]); f[0] *= k; f[1] *= k; }
      const zd = scl(f, 1 / norm(f)), xc = [Math.cos(yaw), Math.sin(yaw), 0];
      let yd = cross(zd, xc); yd = scl(yd, 1 / norm(yd));
      const xd = cross(yd, zd), Rd = [[xd[0], yd[0], zd[0]], [xd[1], yd[1], zd[1]], [xd[2], yd[2], zd[2]]];
      const R = quatToRot(est.q), collective = f[0] * R[0][2] + f[1] * R[1][2] + f[2] * R[2][2];
      return { Rd, F: Math.max(collective, 0) };
    }
    rateCommand(est, Rd) {
      const R = quatToRot(est.q);
      // e_R = 1/2 vee(Rd^T R - R^T Rd)
      const M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
        let a = 0, b = 0; for (let k = 0; k < 3; k++) { a += Rd[k][i] * R[k][j]; b += R[k][i] * Rd[k][j]; }
        M[i][j] = a - b;
      }
      const eR = [0.5 * M[2][1], 0.5 * M[0][2], 0.5 * M[1][0]];
      return eR.map((e, k) => clamp(-this.g.attKp[k] * e, -this.lim.rate[k], this.lim.rate[k]));
    }
    torqueCommand(est, wsp, dt) {
      const alpha = this.ratePid.update(sub(wsp, est.w), est.w, dt);
      return add(diagMul(this.J, alpha), cross(est.w, diagMul(this.J, est.w)));
    }
    fromAccel(est, a, yaw, dt) {
      const { Rd, F } = this.attitudeAndThrust(est, a, yaw);
      const wsp = this.rateCommand(est, Rd), tau = this.torqueCommand(est, wsp, dt);
      this.last.wrench = [F, ...tau];
      return this.mixer.speeds([F, ...tau]);
    }
    update(est, ref, dt) { return this.fromAccel(est, this.accelerationCommand(est, ref, dt), ref.yaw || 0, dt); }
    /** manual 'angle' mode: desired roll / pitch [rad] in the heading frame, yaw rate, climb rate */
    updateAngle(est, roll, pitch, yaw, climb, dt) {
      const az = clamp(this.g.velKp * (climb - est.v[2]), -this.lim.aZ, this.lim.aZ);
      const Rd = quatToRot(eulerToQuat(roll, pitch, yaw));
      const R = quatToRot(est.q), cz = Math.max(R[2][2], 0.3);
      const F = this.mass * (G0 + az) / cz;
      const wsp = this.rateCommand(est, Rd), tau = this.torqueCommand(est, wsp, dt);
      this.last.wrench = [F, ...tau];
      return this.mixer.speeds([F, ...tau]);
    }
  }

  // ---------------------------------------------------------------------------------------------
  // guidance
  // ---------------------------------------------------------------------------------------------
  class PathFollower {
    constructor(points, speed = 4, accel = 2) {
      const seg = []; let s = [0];
      for (let i = 1; i < points.length; i++) { seg.push(norm(sub(points[i], points[i - 1]))); s.push(s[i - 1] + seg[i - 1]); }
      const total = s[s.length - 1], n = Math.max(2, Math.floor(total * 20));
      this.pts = []; this.s = [];
      let j = 0;
      for (let k = 0; k < n; k++) {
        const si = total * k / (n - 1);
        while (j < s.length - 2 && s[j + 1] < si) j++;
        const f = (si - s[j]) / Math.max(s[j + 1] - s[j], 1e-9);
        this.pts.push(add(points[j], scl(sub(points[j + 1], points[j]), f))); this.s.push(si);
      }
      const ds = total / (n - 1);
      const grad = (arr, i) => (i === 0 ? sub(arr[1], arr[0]) : i === arr.length - 1 ? sub(arr[i], arr[i - 1]) : scl(sub(arr[i + 1], arr[i - 1]), 0.5));
      const t = this.pts.map((_, i) => grad(this.pts, i));
      this.tan = t.map((v) => scl(v, 1 / Math.max(norm(v), 1e-9)));
      this.kappa = this.tan.map((_, i) => scl(grad(this.tan, i), 1 / ds));
      this.length = total; this.speed = speed; this.brake = speed * speed / 2 + 0.5; this.accel = accel;
      this.k = 0; this.t0 = null; this.done = false; this.crossTrack = 0;
    }
    reference(t, p) {
      let best = Infinity, bi = this.k;
      for (let i = this.k; i < Math.min(this.pts.length, this.k + 200); i++) { const d = norm(sub(this.pts[i], p)); if (d < best) { best = d; bi = i; } }
      this.k = bi; this.crossTrack = best;
      const remaining = this.length - this.s[this.k];
      if (this.t0 === null) this.t0 = t;
      const ramp = Math.min(1, this.accel * (t - this.t0) / this.speed), sp = this.speed * Math.min(ramp, remaining / this.brake);
      const ahead = Math.min(this.pts.length - 1, this.k + 3);
      if (remaining < 0.2 && norm(sub(p, this.pts[this.pts.length - 1])) < 0.5) this.done = true;
      const tg = this.tan[this.k];
      return { p: this.pts[ahead].slice(), v: scl(tg, sp), a: scl(this.kappa[this.k], sp * sp), yaw: Math.atan2(tg[1], tg[0]) };
    }
  }
  function figureEight(c = [0, 0, 10], a = 15, b = 8, n = 400) {
    const out = [];
    for (let i = 0; i < n; i++) { const t = 2 * Math.PI * i / (n - 1); out.push([c[0] + a * Math.sin(t), c[1] + b * Math.sin(2 * t), c[2]]); }
    return out;
  }

  // ---------------------------------------------------------------------------------------------
  // simulation
  // ---------------------------------------------------------------------------------------------
  class Simulation {
    constructor(af, atm, wind, sensors, dt = 0.002, ctrlDt = 0.01) {
      this.af = af; this.atm = atm || new Atmosphere(); this.wind = wind || new WindField(); this.sensors = sensors || new Sensors('good');
      this.dt = dt; this.sub = Math.max(1, Math.round(ctrlDt / dt)); this.ctrlDt = this.sub * dt;
      this.dyn = new Dynamics(af, this.atm);
    }
    reset(position = [0, 0, 10], yaw = 0) {
      const rho = this.atm.density(position[2]);
      const Bp = pinv4(this.af.allocation()), T = Bp.map((r) => r[0] * this.af.hoverThrust());
      const n = this.af.rotors.map((r, i) => r.motor.speedForThrust(T[i], rho));
      this.state = { p: position.slice(), v: [0, 0, 0], q: eulerToQuat(0, 0, yaw), w: [0, 0, 0], n };
      this.t = 0; this.windNow = this.wind.reset(this.state.p, this.state.v); this.sensors.reset(this.state); this.crashed = false;
      return this.state;
    }
    step(nCmd) {
      for (let k = 0; k < this.sub; k++) {
        this.state = this.dyn.step(this.state, nCmd, this.windNow, this.dt);
        this.windNow = this.wind.step(this.t, this.dt, this.state.p, this.state.v);
        this.t += this.dt;
      }
      const R = quatToRot(this.state.q);
      if ((this.state.p[2] <= 0 && this.t > 0.5) || R[2][2] < -0.2) this.crashed = true;     // same rule as Python
      return this.state;
    }
    estimate() { return this.sensors.measure(this.t, this.ctrlDt, this.state); }
    density() { return this.atm.density(Math.max(this.state.p[2], 0)); }
  }

  const api = { Atmosphere, WindField, Gust, DrydenTurbulence, drydenParams, MotorModel, Airframe, PRESETS, Dynamics, Sensors,
    Mixer, CascadedPID, PID, GAINS, LIMITS, PathFollower, figureEight, Simulation, quatToRot, quatToEuler, eulerToQuat, makeRng,
    vec: { add, sub, scl, dot, cross, norm, clamp } };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.DroneSim = api;
})(typeof window !== 'undefined' ? window : globalThis);
