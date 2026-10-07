/**
 * render/ — the 3D view (Three.js). It only READS the simulation state; it never steps physics.
 *
 *   const view = createView(container, cfg);
 *   view.update(sim, { reference, trail: true });   // once per animation frame
 *   view.rebuild(cfg);                              // after the airframe was edited (arm, rotors)
 *
 * Frames: the simulation is z-up (x east, y north); Three.js is y-up. A position (x, y, z) is drawn at
 * (x, z, −y) — a rotation of −90° about x — and a quaternion (w, x, y, z) becomes (w, x, z, −y).
 */
import * as THREE from 'three';

const toThree = (p) => new THREE.Vector3(p[0], p[2], -p[1]);
const quatToThree = (q) => new THREE.Quaternion(q[1], q[3], -q[2], q[0]);   // THREE.Quaternion(x, y, z, w)

/** The drone mesh, built from the airframe config so a design change shows up in the model. */
function buildDrone(cfg) {
  const M = cfg.motor, root = new THREE.Group();
  const s = M.arm / 0.225;                                    // scale details with the frame size
  const dark = new THREE.MeshStandardMaterial({ color: 0x23272e, roughness: 0.6, metalness: 0.3 });
  const carbon = new THREE.MeshStandardMaterial({ color: 0x15171b, roughness: 0.5, metalness: 0.4 });
  const metal = new THREE.MeshStandardMaterial({ color: 0x9aa4b1, roughness: 0.3, metalness: 0.9 });
  const accent = new THREE.MeshStandardMaterial({ color: 0x1f6feb, roughness: 0.4, metalness: 0.2 });

  // body: a centre plate, a canopy and the battery
  const plate = new THREE.Mesh(new THREE.BoxGeometry(0.11 * s, 0.012 * s, 0.11 * s), carbon);
  const canopy = new THREE.Mesh(new THREE.BoxGeometry(0.085 * s, 0.04 * s, 0.06 * s), accent);
  canopy.position.y = 0.026 * s;
  const battery = new THREE.Mesh(new THREE.BoxGeometry(0.1 * s, 0.03 * s, 0.045 * s), dark);
  battery.position.y = -0.022 * s;
  root.add(plate, canopy, battery);
  [plate, canopy, battery].forEach((m) => { m.castShadow = true; });

  // arms, motors and propellers, from the configured rotor layout (body x forward, y left)
  const props = [];
  for (const r of M.rotors) {
    const a = (r.angleDeg * Math.PI) / 180, x = M.arm * Math.cos(a), y = M.arm * Math.sin(a);
    const pos = toThree([x, y, 0]);
    const arm = new THREE.Mesh(new THREE.BoxGeometry(M.arm, 0.012 * s, 0.018 * s), carbon);
    arm.position.copy(pos.clone().multiplyScalar(0.5));
    arm.rotation.y = Math.atan2(-pos.z, pos.x);
    arm.castShadow = true;
    const motor = new THREE.Mesh(new THREE.CylinderGeometry(0.018 * s, 0.018 * s, 0.03 * s, 20), metal);
    motor.position.set(pos.x, 0.012 * s, pos.z);
    motor.castShadow = true;
    const prop = new THREE.Group();
    const bladeMat = new THREE.MeshStandardMaterial({ color: r.spin > 0 ? 0xe6edf3 : 0xb9c2cc, transparent: true, opacity: 0.85, roughness: 0.4 });
    const radius = 0.55 * M.arm;
    for (let k = 0; k < 2; k++) {
      const blade = new THREE.Mesh(new THREE.BoxGeometry(2 * radius, 0.003 * s, 0.022 * s), bladeMat);
      blade.rotation.y = (k * Math.PI) / 2 + 0.3;
      blade.rotation.x = 0.08 * (k ? 1 : -1);
      prop.add(blade);
    }
    const disc = new THREE.Mesh(new THREE.CircleGeometry(radius, 32), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false }));
    disc.rotation.x = -Math.PI / 2;
    prop.add(disc);
    prop.position.set(pos.x, 0.03 * s, pos.z);
    prop.userData = { spin: r.spin, disc };
    props.push(prop);
    // landing leg under each motor
    const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.004 * s, 0.004 * s, 0.05 * s, 8), dark);
    leg.position.set(pos.x * 0.85, -0.03 * s, pos.z * 0.85);
    root.add(arm, motor, prop, leg);
  }

  // navigation lights: green at the front, red at the back (shows the heading)
  const light = (color, x) => {
    const m = new THREE.Mesh(new THREE.SphereGeometry(0.008 * s, 12, 8), new THREE.MeshBasicMaterial({ color }));
    m.position.copy(toThree([x, 0, 0.01 * s]));
    return m;
  };
  root.add(light(0x3fb950, 0.06 * s), light(0xff5555, -0.06 * s));
  root.userData = { props };
  return root;
}

export function createView(container, cfg) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d1117);
  scene.fog = new THREE.Fog(0x0d1117, 25, 70);
  const camera = new THREE.PerspectiveCamera(55, 1, 0.02, 200);

  scene.add(new THREE.HemisphereLight(0xbcd7ff, 0x1a1f26, 0.9));
  const sun = new THREE.DirectionalLight(0xffffff, 1.4);
  sun.position.set(6, 12, 4);
  sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  Object.assign(sun.shadow.camera, { left: -6, right: 6, top: 6, bottom: -6, near: 1, far: 40 });
  scene.add(sun, sun.target);

  // the floor: a soft plane with a 1 m grid (scale reference for heights and distances)
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), new THREE.MeshStandardMaterial({ color: 0x161b22, roughness: 0.95 }));
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  const grid = new THREE.GridHelper(200, 200, 0x30363d, 0x21262d);
  grid.position.y = 0.001;
  scene.add(floor, grid);

  // the reference (setpoint) marker and the flown path
  const marker = new THREE.Mesh(new THREE.TorusGeometry(0.18, 0.012, 8, 40), new THREE.MeshBasicMaterial({ color: 0xffd666 }));
  marker.rotation.x = Math.PI / 2;
  scene.add(marker);
  const trailN = 600, trailPos = new Float32Array(trailN * 3);
  const trailGeo = new THREE.BufferGeometry();
  trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPos, 3));
  const trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color: 0x58a6ff, transparent: true, opacity: 0.6 }));
  trail.frustumCulled = false;
  scene.add(trail);
  let trailLen = 0;

  let drone = buildDrone(cfg);
  scene.add(drone);

  // camera: 'chase' follows behind the heading; 'orbit' is dragged around the drone
  const cam = { mode: 'chase', yaw: 0.6, pitch: 0.35, dist: 3.2 };
  let drag = null;
  const el = renderer.domElement;
  el.style.touchAction = 'none';
  el.addEventListener('pointerdown', (e) => { if (cam.mode === 'orbit') { drag = { x: e.clientX, y: e.clientY }; el.setPointerCapture(e.pointerId); } });
  el.addEventListener('pointermove', (e) => {
    if (!drag) return;
    cam.yaw -= (e.clientX - drag.x) * 0.006; cam.pitch = Math.min(1.4, Math.max(-0.1, cam.pitch + (e.clientY - drag.y) * 0.006));
    drag = { x: e.clientX, y: e.clientY };
  });
  el.addEventListener('pointerup', () => { drag = null; });
  el.addEventListener('wheel', (e) => { e.preventDefault(); cam.dist = Math.min(30, Math.max(0.8, cam.dist * Math.exp(e.deltaY * 0.001))); }, { passive: false });

  function resize() {
    const w = container.clientWidth, h = container.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(container);
  resize();

  return {
    camera: cam,
    rebuild(newCfg) { scene.remove(drone); drone = buildDrone(newCfg); scene.add(drone); },
    clearTrail() { trailLen = 0; },

    update(sim, { reference, showTrail = true } = {}) {
      const s = sim.state, pos = toThree(s.p);
      drone.position.copy(pos);
      drone.quaternion.copy(quatToThree(s.q));
      // propellers: spin at the real rotor speed (slowed for the eye), a faint disc when they are fast
      drone.userData.props.forEach((prop, i) => {
        const w = sim.motors.omega[i] || 0;
        prop.rotation.y += prop.userData.spin * Math.min(w * 0.004, 1.2);
        prop.userData.disc.material.opacity = Math.min(0.18, w / 4000);
      });
      // reference marker
      const ref = reference && reference.position;
      marker.visible = !!ref;
      if (ref) marker.position.copy(toThree(ref));
      // trail
      trail.visible = showTrail;
      if (showTrail && sim.steps % 3 === 0) {
        if (trailLen === trailN) { trailPos.copyWithin(0, 3); trailLen--; }
        trailPos.set([pos.x, pos.y, pos.z], trailLen * 3); trailLen++;
        trailGeo.setDrawRange(0, trailLen);
        trailGeo.attributes.position.needsUpdate = true;
      }
      // camera
      const yaw = cam.mode === 'chase' ? Math.atan2(2 * (s.q[0] * s.q[3] + s.q[1] * s.q[2]), 1 - 2 * (s.q[2] ** 2 + s.q[3] ** 2)) + Math.PI : cam.yaw;
      const target = pos.clone().add(new THREE.Vector3(0, 0.15, 0));
      const off = new THREE.Vector3(Math.cos(yaw) * Math.cos(cam.pitch), Math.sin(cam.pitch), -Math.sin(yaw) * Math.cos(cam.pitch)).multiplyScalar(cam.dist);
      camera.position.lerp(target.clone().add(off), cam.mode === 'chase' ? 0.12 : 1);
      if (camera.position.y < 0.15) camera.position.y = 0.15;
      camera.lookAt(target);
      sun.position.set(pos.x + 6, 12, pos.z + 4); sun.target.position.copy(pos);
      renderer.render(scene, camera);
    },
  };
}
