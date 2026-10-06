/* Drone Wind Sim — browser game on top of web/sim.js (the JS port of the Python simulator). */
(function () {
  'use strict';
  const S = window.DroneSim, V = S.vec, NN = window.DroneLearned;
  const $ = (id) => document.getElementById(id);
  const RAD = Math.PI / 180;
  const store = (k, v) => { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } };
  const load = (k) => { try { return localStorage.getItem(k); } catch (e) { return null; } };

  // ===========================================================================================
  // state
  // ===========================================================================================
  const st = {
    mode: 'assist', course: 'rings', gains: 'tuned', airframe: 'iris', paused: false,
    camMode: 0, camYaw: 0, camPitch: 0.25, camDist: 1, dragging: false,
    pHold: [0, 0, 3], yawRef: 0, time: 0, raceStart: null, raceEnd: null, ring: 0, crashed: false, crashReason: '',
    history: [], trail: [], nextGust: 6, lastP: null,
  };
  let sim, ctrl, anglePid, guide, af, wind, rings = [], eight = null;
  const nets = {};                            // airframe -> trained network (web/models, see learn/export_web.py)

  async function loadNets() {
    try {
      const idx = await (await fetch('web/models/index.json')).json();
      for (const [name, e] of Object.entries(idx)) {
        const net = new NN.LearnedNet(await (await fetch(`web/models/${e.file}`)).json());
        const diff = net.selfCheck();
        if (diff < 1e-5) nets[name] = net; else console.warn(`model ${e.file}: self-check failed (${diff})`);
      }
    } catch (err) { console.warn('no trained networks', err); }
    if (st.gains === 'learned') { ctrl = makeCtrl(); updateInfo(); }
  }
  function makeCtrl() {
    if (st.gains === 'learned' && nets[st.airframe]) return new NN.LearnedController(nets[st.airframe]);
    return new S.CascadedPID(S.PRESETS[st.airframe](), st.gains === 'learned' ? 'tuned' : st.gains);
  }
  function ctrlLabel() {
    if (st.gains !== 'learned') return `${st.gains} PID`;
    return nets[st.airframe] ? `learned NN (${nets[st.airframe].m.control})` : 'tuned PID (no network for this airframe)';
  }

  function readEnv() {
    return { speed: +$('windSpeed').value, dir: +$('windDir').value, turb: $('turb').value, gust: +$('gustSpeed').value,
      temp: +$('temp').value, alt: +$('alt').value, sensors: $('sensors').value };
  }

  // ===========================================================================================
  // world setup
  // ===========================================================================================
  function droneScale() {                     // characteristic size: half the rotor span
    return Math.max(...af.rotors.map((r) => Math.hypot(r.position[0], r.position[1]))) + af.rotors[0].motor.diameter / 2;
  }
  function makeRings() {
    const rng = S.makeRng(42), out = [];
    let p = [0, 0, 3], heading = 0;
    for (let i = 0; i < 9; i++) {
      heading += (rng.uniform() - 0.5) * 2.0;
      const d = 14 + rng.uniform() * 8, z = 2.5 + rng.uniform() * 7;
      const np = [p[0] + d * Math.cos(heading), p[1] + d * Math.sin(heading), z];
      const dir = V.sub(np, p), n = V.scl(dir, 1 / V.norm(dir));
      out.push({ c: np, n: [n[0], n[1], 0].map((x) => x / Math.hypot(n[0], n[1])), r: 1.6 });
      p = np;
    }
    return out;
  }

  function build() {
    const env = readEnv();
    af = S.PRESETS[st.airframe]();
    wind = new S.WindField({ meanSpeed: env.speed, directionDeg: env.dir, turbulence: env.turb, seed: (Math.random() * 1e9) | 0 });
    const atm = new S.Atmosphere(env.temp, env.alt);
    sim = new S.Simulation(af, atm, wind, new S.Sensors(env.sensors, (Math.random() * 1e9) | 0));
    const start = [0, 0, 2.5];
    sim.reset(start, 0);
    ctrl = makeCtrl();
    anglePid = new S.CascadedPID(S.PRESETS[st.airframe](), 'tuned');      // angle mode is always the PID's attitude loop
    rings = makeRings();
    eight = S.figureEight([0, 18, 6], 16, 9);
    guide = st.course === 'eight' ? new S.PathFollower(eight, 4) : st.course === 'rings'
      ? new S.PathFollower([start, ...rings.map((r) => r.c)], 4) : null;
    Object.assign(st, { pHold: start.slice(), yawRef: 0, time: 0, raceStart: null, raceEnd: null, ring: 0, crashed: false,
      crashReason: '', history: [], trail: [], nextGust: 5 + Math.random() * 6, lastP: start.slice() });
    st.camDist = Math.max(0.6, droneScale() * 9);
    scene3d.rebuildDrone();
    scene3d.rebuildCourse();
    buildMotorBars();
    updateInfo();
    showMsg('', '');
  }

  // ===========================================================================================
  // input
  // ===========================================================================================
  const keys = new Set();
  function stick() {
    const k = (c) => (keys.has(c) ? 1 : 0);
    let fwd = k('KeyW') + k('ArrowUp') - k('KeyS') - k('ArrowDown');
    let right = k('KeyD') - k('KeyA');
    let up = k('Space') - k('ShiftLeft') - k('ShiftRight');
    let yaw = k('KeyQ') + k('ArrowLeft') - k('KeyE') - k('ArrowRight');
    const pads = navigator.getGamepads ? navigator.getGamepads() : [];
    for (const g of pads) {
      if (!g || g.axes.length < 4) continue;
      const dz = (x) => (Math.abs(x) < 0.08 ? 0 : x);
      up += -dz(g.axes[1]); yaw += -dz(g.axes[0]); right += dz(g.axes[2]); fwd += -dz(g.axes[3]);
    }
    const c = (x) => Math.max(-1, Math.min(1, x));
    return { fwd: c(fwd), right: c(right), up: c(up), yaw: c(yaw) };
  }

  // ===========================================================================================
  // control step (100 Hz)
  // ===========================================================================================
  function controlStep(dt) {
    const est = sim.estimate(), s = stick();
    let n;
    const yawNow = S.quatToEuler(sim.state.q)[2];
    if (st.mode === 'auto' && guide) {
      n = ctrl.update(est, guide.reference(sim.t, est.p), dt);
    } else if (st.mode === 'angle') {
      st.yawRef += s.yaw * 1.6 * dt;
      const tilt = 30 * RAD;
      n = (ctrl.updateAngle ? ctrl : anglePid).updateAngle(est, s.right * tilt, s.fwd * tilt, st.yawRef, s.up * 2.5, dt);
      st.pHold = sim.state.p.slice();
    } else {
      // assist: the sticks move a hold point; released, the PID holds it against the wind
      const vmax = st.airframe === 'crazyflie' ? 3 : ctrl instanceof NN.LearnedController ? 4 : 6, c = Math.cos(yawNow), si = Math.sin(yawNow);
      const vb = [s.fwd * vmax, -s.right * vmax];
      const vcmd = [c * vb[0] - si * vb[1], si * vb[0] + c * vb[1], s.up * 2.5];
      st.yawRef += s.yaw * 1.6 * dt;
      st.pHold = V.add(st.pHold, V.scl(vcmd, dt));
      st.pHold[2] = Math.max(st.pHold[2], 0.3);
      const lag = V.sub(st.pHold, sim.state.p), L = V.norm(lag);
      if (L > 4) st.pHold = V.add(sim.state.p, V.scl(lag, 4 / L));    // no wind-up when the wind wins
      n = ctrl.update(est, { p: st.pHold, v: vcmd, yaw: st.yawRef }, dt);
    }
    st.lastN = n;
    const before = sim.state.p.slice(), vz = sim.state.v[2];
    sim.step(n);
    checkGround(vz);
    checkRings(before, sim.state.p);
  }

  function checkGround(vzBefore) {
    const s = sim.state, R = S.quatToRot(s.q);
    if (R[2][2] < -0.2) return crash('flipped over');
    if (s.p[2] <= 0.0001 && vzBefore < -3.0) return crash(`hit the ground at ${(-vzBefore).toFixed(1)} m/s`);
    if (s.p[2] <= 0.0001 && R[2][2] < 0.5) return crash('tipped over on the ground');
  }
  function crash(reason) {
    if (st.crashed) return;
    st.crashed = true; st.crashReason = reason;
    showMsg('💥 Crashed', `${reason} — press R to try again`);
  }
  function checkRings(a, b) {
    if (st.course !== 'rings' || st.ring >= rings.length) return;
    const r = rings[st.ring];
    const da = V.dot(V.sub(a, r.c), r.n), db = V.dot(V.sub(b, r.c), r.n);
    if (st.raceStart === null && V.norm(V.sub(b, [0, 0, 2.5])) > 0.5) st.raceStart = st.time;
    if (da < 0 && db >= 0) {
      const f = da / (da - db), x = V.add(a, V.scl(V.sub(b, a), f));
      if (V.norm(V.sub(x, r.c)) <= r.r) {
        st.ring++;
        toast(st.ring < rings.length ? `Ring ${st.ring} / ${rings.length}` : 'Finish!');
        if (st.ring === rings.length) finishRace();
      }
    }
  }
  function bestKey() { return `dws.best.${st.airframe}.${st.mode}.${$('windSpeed').value}.${$('turb').value}`; }
  function finishRace() {
    st.raceEnd = st.time;
    const t = st.raceEnd - (st.raceStart ?? 0), prev = parseFloat(load(bestKey()));
    const isBest = !(prev <= t);
    if (isBest) store(bestKey(), t.toFixed(2));
    showMsg(`🏁 ${t.toFixed(2)} s`, (isBest ? 'New best for these settings! ' : `Best: ${prev.toFixed(2)} s. `) + 'Press R to fly again.');
    updateBest();
  }

  // ===========================================================================================
  // 3D scene (three.js, y-up; ENU (x, y, z) -> three (x, z, -y))
  // ===========================================================================================
  const T = (p) => new THREE.Vector3(p[0], p[2], -p[1]);
  const scene3d = (() => {
    const el = $('view');
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0e1520);
    scene.fog = new THREE.Fog(0x0e1520, 60, 220);
    const camera = new THREE.PerspectiveCamera(60, 1, 0.02, 600);
    scene.add(new THREE.HemisphereLight(0xbcd3ff, 0x1a2a1a, 0.9));
    const sun = new THREE.DirectionalLight(0xffffff, 0.8); sun.position.set(30, 60, 20); scene.add(sun);
    // ground
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(600, 600), new THREE.MeshLambertMaterial({ color: 0x1b2a1f }));
    ground.rotation.x = -Math.PI / 2; scene.add(ground);
    const grid = new THREE.GridHelper(400, 200, 0x2c3d30, 0x22312a); grid.position.y = 0.01; scene.add(grid);
    const pad = new THREE.Mesh(new THREE.CircleGeometry(1.2, 40), new THREE.MeshBasicMaterial({ color: 0x2f81f7, transparent: true, opacity: 0.35 }));
    pad.rotation.x = -Math.PI / 2; pad.position.y = 0.02; scene.add(pad);
    // trees for a sense of scale and speed
    const rng = S.makeRng(7), treeGeo = new THREE.ConeGeometry(1.2, 4, 7), trunkGeo = new THREE.CylinderGeometry(0.15, 0.2, 1.2, 6);
    const treeMat = new THREE.MeshLambertMaterial({ color: 0x2e5e3a }), trunkMat = new THREE.MeshLambertMaterial({ color: 0x5b4632 });
    // keep the trees away from the take-off pad and both courses so they never block the view of the course
    const keepOut = [[0, 0, 2.5], ...makeRings().map((r) => r.c), ...S.figureEight([0, 18, 6], 16, 9).filter((_, i) => i % 8 === 0)];
    const nearCourse = (x, y) => keepOut.some((c) => Math.hypot(c[0] - x, c[1] - y) < 14);
    for (let i = 0, placed = 0; placed < 170 && i < 2000; i++) {
      const r = 15 + rng.uniform() * 160, a = rng.uniform() * 2 * Math.PI, s = 0.7 + rng.uniform() * 1.2;
      if (nearCourse(r * Math.cos(a), -r * Math.sin(a))) continue;
      placed++;
      const g = new THREE.Group();
      const c = new THREE.Mesh(treeGeo, treeMat); c.position.y = 3.2; g.add(c);
      const t = new THREE.Mesh(trunkGeo, trunkMat); t.position.y = 0.6; g.add(t);
      g.scale.setScalar(s); g.position.set(r * Math.cos(a), 0, r * Math.sin(a)); scene.add(g);
    }
    // wind particles
    // each particle is drawn as a short streak along the local wind: direction and speed are visible
    const NP = 1200, ppos = new Float32Array(NP * 3), spos = new Float32Array(NP * 6), pgeo = new THREE.BufferGeometry();
    pgeo.setAttribute('position', new THREE.BufferAttribute(spos, 3));
    const pts = new THREE.LineSegments(pgeo, new THREE.LineBasicMaterial({ color: 0xcfe3ff, transparent: true, opacity: 0.45 }));
    pts.frustumCulled = false;
    scene.add(pts);
    let streak = 0.25;
    let box = 30, pInit = false;
    // drone, course, trail
    let drone = null, props = [], courseGroup = new THREE.Group(), refMarker = null;
    scene.add(courseGroup);
    const trailGeo = new THREE.BufferGeometry(), trailPos = new Float32Array(3 * 1500);
    trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPos, 3));
    const trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color: 0x79c0ff, transparent: true, opacity: 0.6 }));
    trail.frustumCulled = false; scene.add(trail);

    function rebuildDrone() {
      if (drone) scene.remove(drone);
      drone = new THREE.Group(); props = [];
      const span = droneScale(), arm = 0.06 * span + 0.004;
      const body = new THREE.Mesh(new THREE.BoxGeometry(span * 0.55, span * 0.18, span * 0.4), new THREE.MeshLambertMaterial({ color: 0xe6edf3 }));
      drone.add(body);
      const nose = new THREE.Mesh(new THREE.BoxGeometry(span * 0.12, span * 0.1, span * 0.12), new THREE.MeshBasicMaterial({ color: 0x2f81f7 }));
      nose.position.set(span * 0.3, 0, 0); drone.add(nose);
      for (const r of af.rotors) {
        const [x, y, z] = r.position, len = Math.hypot(x, y);
        const a = new THREE.Mesh(new THREE.BoxGeometry(len, arm, arm), new THREE.MeshLambertMaterial({ color: 0x8b949e }));
        a.position.set(x / 2, z, -y / 2); a.rotation.y = Math.atan2(y, x); drone.add(a);
        const D = r.motor.diameter, g = new THREE.Group(); g.position.set(x, z + arm, -y);
        const disc = new THREE.Mesh(new THREE.CircleGeometry(D / 2, 32), new THREE.MeshBasicMaterial({
          color: r.spin > 0 ? 0xe3b341 : 0xbc8cff, transparent: true, opacity: 0.18, side: THREE.DoubleSide }));
        disc.rotation.x = -Math.PI / 2; g.add(disc);
        const blade = new THREE.Mesh(new THREE.BoxGeometry(D, arm * 0.25, D * 0.08), new THREE.MeshLambertMaterial({ color: r.spin > 0 ? 0xe3b341 : 0xbc8cff }));
        g.add(blade); drone.add(g);
        props.push({ g, blade, disc, spin: r.spin });
      }
      scene.add(drone);
      streak = Math.max(0.05, Math.min(0.3, span * 1.0));
      box = Math.max(6, Math.min(36, span * 90)); pInit = false;
    }
    function rebuildCourse() {
      scene.remove(courseGroup); courseGroup = new THREE.Group(); scene.add(courseGroup);
      if (st.course === 'rings') {
        rings.forEach((r, i) => {
          const m = new THREE.Mesh(new THREE.TorusGeometry(r.r, 0.08, 10, 48), new THREE.MeshBasicMaterial({ color: 0x3fb950 }));
          m.position.copy(T(r.c));
          m.lookAt(T(V.add(r.c, r.n)));
          m.userData.i = i; courseGroup.add(m);
          const post = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, r.c[2] - r.r, 6), new THREE.MeshLambertMaterial({ color: 0x30363d }));
          post.position.set(r.c[0], (r.c[2] - r.r) / 2, -r.c[1]); courseGroup.add(post);
        });
      }
      const pathPts = st.course === 'eight' ? eight : st.course === 'rings' ? [[0, 0, 2.5], ...rings.map((r) => r.c)] : null;
      if (pathPts) {
        const g = new THREE.BufferGeometry().setFromPoints(pathPts.map(T));
        courseGroup.add(new THREE.Line(g, new THREE.LineDashedMaterial({ color: 0xffffff, dashSize: 0.6, gapSize: 0.4, transparent: true, opacity: 0.35 })).computeLineDistances());
      }
      refMarker = new THREE.Mesh(new THREE.SphereGeometry(0.12, 12, 12), new THREE.MeshBasicMaterial({ color: 0x2f81f7, transparent: true, opacity: 0.7 }));
      courseGroup.add(refMarker);
    }
    const P = new THREE.Matrix4();
    function update(dtReal) {
      const s = sim.state, R = S.quatToRot(s.q);
      // body -> world rotation in three's axes: M = A R A^T, A maps ENU to (x, z, -y)
      const A = [[1, 0, 0], [0, 0, 1], [0, -1, 0]], M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
        let v = 0; for (let k = 0; k < 3; k++) for (let l = 0; l < 3; l++) v += A[i][k] * R[k][l] * A[j][l]; M[i][j] = v;
      }
      P.set(M[0][0], M[0][1], M[0][2], 0, M[1][0], M[1][1], M[1][2], 0, M[2][0], M[2][1], M[2][2], 0, 0, 0, 0, 1);
      drone.position.copy(T(s.p)); drone.quaternion.setFromRotationMatrix(P);
      props.forEach((pr, i) => {
        const n = s.n[i], h = af.rotors[i].motor.health;
        pr.blade.rotation.y += pr.spin * Math.min(n, 40) * dtReal * 2 * Math.PI * 0.35;
        pr.disc.material.opacity = h <= 0 ? 0.04 : 0.1 + 0.25 * n / af.rotors[i].motor.nMax;
        pr.blade.visible = h > 0 || n > 1;
      });
      // rings colour
      courseGroup.children.forEach((m) => {
        if (m.userData.i === undefined) return;
        m.material.color.setHex(m.userData.i < st.ring ? 0x30363d : m.userData.i === st.ring ? 0xe3b341 : 0x3fb950);
      });
      if (refMarker) {
        const showRef = st.mode !== 'angle';
        refMarker.visible = showRef;
        const target = st.mode === 'auto' && guide ? guide.pts[Math.min(guide.k + 3, guide.pts.length - 1)] : st.pHold;
        refMarker.position.copy(T(target));
        refMarker.scale.setScalar(Math.max(0.3, droneScale() * 1.2));
      }
      // trail
      const tr = st.trail, n = Math.min(tr.length, 500);
      for (let i = 0; i < n; i++) { const v = tr[tr.length - n + i]; trailPos[3 * i] = v[0]; trailPos[3 * i + 1] = v[2]; trailPos[3 * i + 2] = -v[1]; }
      trailGeo.setDrawRange(0, n); trailGeo.attributes.position.needsUpdate = true;
      // wind particles drift with the local wind (mean at their height + the turbulence the drone feels)
      const c = s.p, turb = V.sub(sim.windNow, wind.mean(Math.max(c[2], 0.1)));
      if (!pInit) {
        for (let i = 0; i < NP; i++) {
          ppos[3 * i] = c[0] + (Math.random() - 0.5) * 2 * box; ppos[3 * i + 1] = Math.random() * Math.max(12, c[2] + box * 0.6);
          ppos[3 * i + 2] = -c[1] + (Math.random() - 0.5) * 2 * box;
        }
        pInit = true;
      }
      const dt = st.paused ? 0 : Math.min(dtReal, 0.05);
      for (let i = 0; i < NP; i++) {
        const h = ppos[3 * i + 1], w = V.add(wind.mean(Math.max(h, 0.1)), turb);
        ppos[3 * i] += w[0] * dt; ppos[3 * i + 1] += w[2] * dt; ppos[3 * i + 2] += -w[1] * dt;
        // wrap inside a box that follows the drone
        if (ppos[3 * i] - c[0] > box) ppos[3 * i] -= 2 * box; else if (ppos[3 * i] - c[0] < -box) ppos[3 * i] += 2 * box;
        if (ppos[3 * i + 2] + c[1] > box) ppos[3 * i + 2] -= 2 * box; else if (ppos[3 * i + 2] + c[1] < -box) ppos[3 * i + 2] += 2 * box;
        const top = Math.max(12, c[2] + box * 0.6);
        if (ppos[3 * i + 1] < 0) ppos[3 * i + 1] += top; else if (ppos[3 * i + 1] > top) ppos[3 * i + 1] -= top;
        spos[6 * i] = ppos[3 * i]; spos[6 * i + 1] = ppos[3 * i + 1]; spos[6 * i + 2] = ppos[3 * i + 2];
        spos[6 * i + 3] = ppos[3 * i] - w[0] * streak; spos[6 * i + 4] = ppos[3 * i + 1] - w[2] * streak; spos[6 * i + 5] = ppos[3 * i + 2] + w[1] * streak;
      }
      pgeo.attributes.position.needsUpdate = true;
      // camera
      const yaw = S.quatToEuler(s.q)[2];
      const tgt = T(s.p);
      let cy, cp = st.camPitch, cd = st.camDist;
      if (st.camMode === 0) cy = yaw + Math.PI + st.camYaw;            // chase: behind the drone
      else if (st.camMode === 1) cy = st.camYaw;                        // orbit: fixed in the world
      else { cy = st.camYaw; cp = 1.45; cd = st.camDist * 2.2; }        // top
      const off = new THREE.Vector3(Math.cos(cy) * Math.cos(cp), Math.sin(cp), -Math.sin(cy) * Math.cos(cp)).multiplyScalar(cd);
      const want = tgt.clone().add(off); want.y = Math.max(want.y, 0.15);
      camera.position.lerp(want, st.camMode === 0 ? 0.12 : 0.3);
      camera.lookAt(tgt);
      renderer.render(scene, camera);
    }
    function resize() {
      const w = el.clientWidth, h = el.clientHeight;
      renderer.setSize(w, h); camera.aspect = w / Math.max(h, 1); camera.updateProjectionMatrix();
    }
    window.addEventListener('resize', resize);
    if (window.ResizeObserver) new ResizeObserver(resize).observe(el);
    resize();
    // mouse look
    el.addEventListener('pointerdown', (e) => { st.dragging = { x: e.clientX, y: e.clientY, yaw: st.camYaw, pitch: st.camPitch }; el.setPointerCapture(e.pointerId); });
    el.addEventListener('pointermove', (e) => {
      if (!st.dragging) return;
      st.camYaw = st.dragging.yaw - (e.clientX - st.dragging.x) * 0.006;
      st.camPitch = Math.max(-0.1, Math.min(1.4, st.dragging.pitch + (e.clientY - st.dragging.y) * 0.005));
    });
    el.addEventListener('pointerup', () => { st.dragging = false; });
    el.addEventListener('wheel', (e) => { e.preventDefault(); st.camDist = Math.max(0.3, Math.min(80, st.camDist * Math.exp(Math.sign(e.deltaY) * 0.1))); }, { passive: false });
    return { rebuildDrone, rebuildCourse, update };
  })();

  // ===========================================================================================
  // HUD
  // ===========================================================================================
  function buildMotorBars() {
    const box = $('motors'); box.innerHTML = '';
    af.rotors.forEach((r, i) => {
      const d = document.createElement('div'); d.className = 'motor';
      d.title = `Motor ${i + 1} (${r.spin > 0 ? 'counter-clockwise' : 'clockwise'}) — click to damage / repair`;
      d.innerHTML = `<div class="bar"><div class="fill"></div></div><div>${i + 1}</div>`;
      d.onclick = () => {
        const h = r.motor.health, nh = h > 0.75 ? 0.5 : h > 0.25 ? 0 : 1;
        sim.dyn.setMotorHealth(i, nh);
        toast(nh === 1 ? `Motor ${i + 1} repaired` : nh === 0 ? `Motor ${i + 1} FAILED — can the controller cope?` : `Motor ${i + 1} at 50 % thrust`);
      };
      box.appendChild(d);
    });
  }
  function updateHud() {
    const s = sim.state, e = S.quatToEuler(s.q), w = sim.windNow, rho = sim.density();
    const gs = Math.hypot(s.v[0], s.v[1]), as = Math.hypot(s.v[0] - w[0], s.v[1] - w[1]);
    const tilt = Math.acos(Math.max(-1, Math.min(1, S.quatToRot(s.q)[2][2]))) / RAD;
    let line3 = '';
    if (st.course === 'rings') {
      const t = st.raceEnd !== null ? st.raceEnd - st.raceStart : st.raceStart !== null ? st.time - st.raceStart : 0;
      line3 = `<div class="big">${t.toFixed(1)} s</div><div>rings passed <b>${st.ring} / ${rings.length}</b> · fly through the yellow one</div>`;
    } else if (guide && st.course === 'eight') {
      line3 = `<div>off path <b>${guide.crossTrack.toFixed(2)} m</b></div>`;
      if (st.mode !== 'auto') guide.reference(sim.t, s.p);
    }
    const sat = st.lastN ? st.lastN.filter((n, i) => n >= 0.98 * af.rotors[i].motor.nMax).length : 0;
    $('hudL').innerHTML = `${line3}
      <div>altitude <b>${s.p[2].toFixed(1)} m</b> · ground speed <b>${gs.toFixed(1)}</b> m/s</div>
      <div>airspeed ${as.toFixed(1)} m/s · tilt <b class="${tilt > 35 ? 'warn' : ''}">${tilt.toFixed(0)}°</b></div>
      <div>air density ${rho.toFixed(3)} kg/m³${sat ? ` · <span class="warn">${sat} motor${sat > 1 ? 's' : ''} at max</span>` : ''}</div>
      <div style="color:var(--muted)">${st.mode === 'assist' ? 'Assist (GPS hold)' : st.mode === 'angle' ? 'Angle (manual)' : 'Autopilot'} · ${ctrlLabel()}${ctrl.memoryNorm ? ` · memory |h| ${ctrl.memoryNorm().toFixed(2)}` : ''} · cam ${['chase', 'orbit', 'top'][st.camMode]}</div>`;
    // motor bars
    const bars = $('motors').children;
    af.rotors.forEach((r, i) => {
      const f = Math.min(1, s.n[i] / r.motor.nMax) ** 2, b = bars[i];
      b.classList.toggle('dead', r.motor.health <= 0);
      const fill = b.querySelector('.fill');
      fill.style.height = `${100 * f}%`;
      fill.style.background = r.motor.health < 1 ? '#f0883e' : r.spin > 0 ? '#e3b341' : '#bc8cff';
    });
    drawCompass(e[2], w);
    drawSpark();
  }
  function drawCompass(yaw, w) {
    const c = $('compass'), x = c.getContext('2d'), R = 50, cx = 60, cy = 60;
    x.clearRect(0, 0, 120, 120);
    x.strokeStyle = '#30363d'; x.lineWidth = 1.5; x.beginPath(); x.arc(cx, cy, R, 0, 2 * Math.PI); x.stroke();
    // screen up = the direction the camera looks; the camera sits at ENU angle cy and looks back across the drone
    const camA = st.camMode === 0 ? yaw + Math.PI + st.camYaw : st.camYaw, viewYaw = camA + Math.PI;
    const rot = (vx, vy) => {                            // world ENU vector -> screen (up = view direction, left = left)
      const b = Math.atan2(vy, vx) - viewYaw, m = Math.hypot(vx, vy);
      return [-m * Math.sin(b), -m * Math.cos(b)];
    };
    x.fillStyle = '#8b949e'; x.font = '10px system-ui'; x.textAlign = 'center';
    [['N', 0, 1], ['E', 1, 0], ['S', 0, -1], ['W', -1, 0]].forEach(([l, vx, vy]) => { const [sx, sy] = rot(vx, vy); x.fillText(l, cx + sx * (R - 8), cy + sy * (R - 8) + 3); });
    // drone heading
    const [hx, hy] = rot(Math.cos(yaw), Math.sin(yaw));
    x.strokeStyle = '#2f81f7'; x.lineWidth = 2; x.beginPath(); x.moveTo(cx, cy); x.lineTo(cx + hx * 22, cy + hy * 22); x.stroke();
    // wind arrow (where it blows to), length ~ speed
    const sp = Math.hypot(w[0], w[1]);
    if (sp > 0.05) {
      const [wx, wy] = rot(w[0] / sp, w[1] / sp), L = Math.min(R - 4, 10 + sp * 3);
      const x0 = cx - wx * L / 2, y0 = cy - wy * L / 2, x1 = cx + wx * L / 2, y1 = cy + wy * L / 2;
      x.strokeStyle = '#e6edf3'; x.lineWidth = 3; x.beginPath(); x.moveTo(x0, y0); x.lineTo(x1, y1); x.stroke();
      const a = Math.atan2(y1 - y0, x1 - x0);
      x.beginPath(); x.moveTo(x1, y1); x.lineTo(x1 - 8 * Math.cos(a - 0.5), y1 - 8 * Math.sin(a - 0.5)); x.lineTo(x1 - 8 * Math.cos(a + 0.5), y1 - 8 * Math.sin(a + 0.5)); x.closePath(); x.fillStyle = '#e6edf3'; x.fill();
    }
    $('windTxt').innerHTML = `wind <b>${sp.toFixed(1)}</b> m/s${Math.abs(w[2]) > 0.3 ? ` · ${w[2] > 0 ? '↑' : '↓'}${Math.abs(w[2]).toFixed(1)}` : ''}`;
  }
  function drawSpark() {
    const c = $('spark'), x = c.getContext('2d'), h = st.history, W = c.width, H = c.height;
    x.clearRect(0, 0, W, H);
    if (h.length < 2) return;
    const t1 = h[h.length - 1].t, t0 = t1 - 20, X = (t) => ((t - t0) / 20) * W;
    const maxW = Math.max(6, ...h.map((r) => r.w)), Y = (v, m) => H - 4 - (v / m) * (H - 14);
    const line = (key, m, col) => {
      x.strokeStyle = col; x.lineWidth = 1.3; x.beginPath();
      h.forEach((r, i) => { const px = X(r.t), py = Y(r[key], m); i ? x.lineTo(px, py) : x.moveTo(px, py); }); x.stroke();
    };
    line('w', maxW, '#e6edf3'); line('tilt', 45, '#2f81f7');
    x.font = '10px system-ui'; x.fillStyle = '#e6edf3'; x.fillText(`wind (max ${maxW.toFixed(0)} m/s)`, 4, 10);
    x.fillStyle = '#2f81f7'; x.fillText('tilt (0–45°)', 130, 10);
  }
  function updateInfo() {
    const env = readEnv(), atm = new S.Atmosphere(env.temp, env.alt), rho = atm.density(0);
    $('windSpeedO').textContent = `${env.speed}`; $('windDirO').textContent = `${env.dir}°`; $('gustSpeedO').textContent = `${env.gust}`;
    $('tempO').textContent = `${env.temp}°`; $('altO').textContent = `${env.alt}`;
    $('droneInfo').innerHTML = `<span>mass</span><span>${af.mass < 0.1 ? (af.mass * 1000).toFixed(0) + ' g' : af.mass.toFixed(2) + ' kg'}</span>
      <span>rotors</span><span>${af.n} · ${(af.rotors[0].motor.diameter * 39.37).toFixed(1)} in props</span>
      <span>thrust / weight</span><span>${af.thrustToWeight(rho).toFixed(2)} here (${af.thrustToWeight(1.225).toFixed(2)} at sea level, 15 °C)</span>
      ${af.source ? `<span>parameters</span><span>${af.source}</span>` : ''}`;
    $('envInfo').innerHTML = `<span>air density</span><span>${rho.toFixed(3)} kg/m³ (${(100 * rho / 1.225).toFixed(0)} % of standard)</span>`;
    $('sumDrone').textContent = af.label;
    $('sumEnv').textContent = `${env.speed} m/s · ${env.turb} · ${env.temp} °C`;
    $('sumFlight').textContent = `${{ assist: 'Assist', angle: 'Angle', auto: 'Autopilot' }[st.mode]} · ${{ rings: 'rings', eight: 'figure-eight', free: 'free' }[st.course]}`;
    $('modeNote').textContent = {
      assist: 'Fly with WASD, Space / Shift, Q / E. Let go and the drone holds its position against the wind (the blue dot is where it wants to be).',
      angle: 'The keys set the tilt directly. Nothing holds your position: the wind blows you away unless you lean into it.',
      auto: st.course === 'free' ? 'Autopilot needs a course: choose rings or figure-eight.' : 'The PID flies the course by itself. Try the untuned gains, more wind or a failed motor.',
    }[st.mode];
    if (st.gains === 'learned') {
      const net = nets[st.airframe];
      $('modeNote').textContent += net
        ? ` Learned NN: a recurrent MLP trained by backpropagation through the simulator (learn/). It only sees sensors and its own memory — not the wind, mass or air density.${st.mode === 'angle' ? ' In Angle mode you fly, so the PID attitude loop is used.' : ''}`
        : ` No network has been trained for this airframe yet; trained: ${Object.keys(nets).join(', ') || 'loading…'}.`;
    }
    updateBest();
  }
  function updateBest() {
    const b = load(bestKey());
    $('best').textContent = st.course === 'rings' && b ? `Best time with these settings: ${(+b).toFixed(2)} s` : '';
  }
  let toastTimer = null;
  function toast(t) { const el = $('toast'); el.textContent = t; el.classList.add('show'); clearTimeout(toastTimer); toastTimer = setTimeout(() => el.classList.remove('show'), 1800); }
  function showMsg(a, b) { $('msg').textContent = a; $('msg2').textContent = b; }

  // ===========================================================================================
  // UI wiring
  // ===========================================================================================
  function seg(id, key, after) {
    $(id).querySelectorAll('button').forEach((b) => {
      b.onclick = () => {
        $(id).querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b));
        st[key] = b.dataset.v; store(`dws.${key}`, st[key]); after(); b.blur();
      };
    });
  }
  function setSeg(id, v) { $(id).querySelectorAll('button').forEach((x) => x.classList.toggle('on', x.dataset.v === v)); }
  function setMode(m) {
    st.mode = m; setSeg('modeSeg', m); store('dws.mode', m);
    st.pHold = sim.state.p.slice(); st.yawRef = S.quatToEuler(sim.state.q)[2]; ctrl.reset();
    if (m === 'auto' && st.course !== 'free') build();
    updateInfo();
  }
  function wire() {
    seg('modeSeg', 'mode', () => setMode(st.mode));
    seg('gainSeg', 'gains', () => {
      const alt = Object.keys(nets)[0];
      if (st.gains === 'learned' && !nets[st.airframe] && alt) {
        st.airframe = alt; $('airframe').value = alt; store('dws.airframe', alt); build();
        toast(`The network is trained for ${af.label}: switched airframe`);
        return;
      }
      ctrl = makeCtrl(); updateInfo(); toast(ctrlLabel());
    });
    $('course').onchange = (e) => { st.course = e.target.value; store('dws.course', st.course); build(); e.target.blur(); };
    $('airframe').onchange = (e) => { st.airframe = e.target.value; store('dws.airframe', st.airframe); build(); e.target.blur(); };
    $('windSpeed').oninput = (e) => { wind.meanSpeed = +e.target.value; updateInfo(); };
    $('windDir').oninput = (e) => { wind.directionDeg = +e.target.value; updateInfo(); };
    $('turb').onchange = (e) => { wind.setTurbulence(e.target.value); wind.dryden.reset(sim.state.p[2]); updateInfo(); e.target.blur(); };
    $('gustSpeed').oninput = updateInfo;
    $('gustBtn').onclick = (e) => { gustNow(); e.target.blur(); };
    $('temp').oninput = $('alt').oninput = () => {
      const env = readEnv();
      sim.atm = new S.Atmosphere(env.temp, env.alt); sim.dyn.atm = sim.atm; updateInfo();
    };
    $('sensors').onchange = (e) => { const s = new S.Sensors(e.target.value, (Math.random() * 1e9) | 0); s.reset(sim.state); sim.sensors = s; e.target.blur(); };
    $('resetBtn').onclick = (e) => { build(); e.target.blur(); };
    $('pauseBtn').onclick = (e) => { togglePause(); e.target.blur(); };
    document.querySelectorAll('details[data-sec]').forEach((d) => {
      const k = `dws.sec.${d.dataset.sec}`, v = load(k);
      if (v !== null) d.open = v === '1';
      d.addEventListener('toggle', () => store(k, d.open ? '1' : '0'));
    });
    window.addEventListener('keydown', (e) => {
      const tag = (e.target.tagName || '').toLowerCase();
      if (tag === 'input' && e.target.type !== 'range' && e.target.type !== 'checkbox') return;
      if (tag === 'select') return;
      if (['Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.code)) e.preventDefault();
      keys.add(e.code);
      if (e.repeat) return;
      if (e.code === 'KeyR') build();
      else if (e.code === 'KeyP') togglePause();
      else if (e.code === 'KeyG') gustNow();
      else if (e.code === 'KeyC') { st.camMode = (st.camMode + 1) % 3; st.camYaw = st.camMode === 0 ? 0 : S.quatToEuler(sim.state.q)[2] + Math.PI; toast(`camera: ${['chase', 'orbit', 'top'][st.camMode]}`); }
      else if (e.code === 'Digit1') setMode('assist');
      else if (e.code === 'Digit2') setMode('angle');
      else if (e.code === 'Digit3') setMode('auto');
    });
    window.addEventListener('keyup', (e) => keys.delete(e.code));
    window.addEventListener('blur', () => keys.clear());
  }
  function togglePause() { st.paused = !st.paused; $('pauseBtn').textContent = st.paused ? '▶ Resume' : '⏸ Pause'; $('pauseBtn').innerHTML += ' <kbd>P</kbd>'; }
  function gustNow() {
    const env = readEnv(), from = env.dir + (Math.random() - 0.5) * 120;
    wind.gusts = wind.gusts.filter((g) => g.t0 + g.duration > sim.t);
    wind.gusts.push(new S.Gust(sim.t + 0.05, 1.5 + Math.random() * 2, env.gust, from));
    toast(`💨 gust ${env.gust} m/s from ${((from % 360) + 360) % 360 | 0}°`);
  }

  // ===========================================================================================
  // main loop
  // ===========================================================================================
  let last = performance.now(), acc = 0;
  function frame(now) {
    const dtReal = Math.min(0.1, (now - last) / 1000); last = now;
    if (!st.paused && !st.crashed && st.raceEnd === null) {
      acc += dtReal;
      while (acc >= sim.ctrlDt) {
        controlStep(sim.ctrlDt); st.time += sim.ctrlDt; acc -= sim.ctrlDt;
        if ($('autoGust').checked && st.time > st.nextGust) { gustNow(); st.nextGust = st.time + 8 + Math.random() * 10; }
        if (st.crashed) break;
      }
      const s = sim.state;
      if (!st.trail.length || V.norm(V.sub(st.trail[st.trail.length - 1], s.p)) > 0.15) { st.trail.push(s.p.slice()); if (st.trail.length > 1500) st.trail.shift(); }
      const w = sim.windNow;
      st.history.push({ t: st.time, w: Math.hypot(w[0], w[1]), tilt: Math.acos(Math.max(-1, Math.min(1, S.quatToRot(s.q)[2][2]))) / RAD });
      while (st.history.length && st.history[0].t < st.time - 20) st.history.shift();
      if (V.norm(V.sub(s.p, [0, 0, 0])) > 250) crash('blown out of the area');
    } else if (st.raceEnd !== null && !st.paused) {
      // after the finish: keep hovering in the background
      acc += dtReal;
      while (acc >= sim.ctrlDt) { const est = sim.estimate(); sim.step(ctrl.update(est, { p: st.pHold, yaw: st.yawRef }, sim.ctrlDt)); acc -= sim.ctrlDt; }
    }
    scene3d.update(dtReal);
    updateHud();
    requestAnimationFrame(frame);
  }

  // restore preferences
  for (const [k, sel] of [['airframe', 'airframe'], ['course', 'course']]) { const v = load(`dws.${k}`); if (v && $(sel).querySelector(`option[value="${v}"]`)) { $(sel).value = v; st[k] = v; } }
  for (const [k, id] of [['mode', 'modeSeg'], ['gains', 'gainSeg']]) { const v = load(`dws.${k}`); if (v) { st[k] = v; setSeg(id, v); } }
  wire();
  build();
  loadNets();
  requestAnimationFrame(frame);
  window.__dws = { st, get sim() { return sim; }, get ctrl() { return ctrl; }, build, keys, setMode };
})();
