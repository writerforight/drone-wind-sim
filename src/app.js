/**
 * The app: wires the blocks together and runs them. Rendering is decoupled from physics:
 * every animation frame adds the real elapsed time to an accumulator and spends it in whole controller
 * periods (each = a fixed number of RK4 physics steps), so the simulation does not depend on the frame rate.
 */
import { cloneConfig, presets } from './config.js';
import { createCascadedPID } from './controller/cascaded-pid.js';
import { attachTouchSticks, createInput } from './input/input.js';
import { createSimulation } from './loop/simulation.js';
import { modes } from './modes/index.js';
import { stepMetrics } from './modes/tuning-lab.js';
import { toEuler } from './physics/math3.js';
import { createView } from './render/scene.js';
import { createChart } from './ui/plots.js';
import { renderDesign, renderGains } from './ui/panels.js';

const $ = (id) => document.getElementById(id);
const deg = 180 / Math.PI;

const app = {
  cfg: cloneConfig(presets[0]),
  presetIndex: 0,
  mode: modes[0],
  sim: null,
  controller: null,
  paused: true,
  step: null,                 // the running step test in the lab: { axis, t0, y0, y1, samples }
};
window.app = app;             // for the browser checks and for poking around in the console

const input = createInput();
const view = createView($('stage'), app.cfg);
const charts = {
  error: createChart($('plotError'), { title: 'tracking error', unit: 'm', series: [{ name: 'x', color: '#ff6b6b' }, { name: 'y', color: '#51cf66' }, { name: 'z', color: '#4dabf7' }] }),
  attitude: createChart($('plotAttitude'), { title: 'attitude', unit: '°', series: [{ name: 'roll', color: '#ff6b6b' }, { name: 'pitch', color: '#51cf66' }, { name: 'yaw', color: '#4dabf7' }] }),
  motors: createChart($('plotMotors'), { title: 'motors', unit: '% of ω_max', fixed: [0, 100], series: [{ name: 'FR', color: '#e3b341' }, { name: 'BL', color: '#e3b341' }, { name: 'FL', color: '#bc8cff' }, { name: 'BR', color: '#bc8cff' }] }),
};

app.resetSim = (initial) => {
  app.controller = createCascadedPID(app.cfg.physics);
  app.sim = createSimulation(app.cfg, { controller: app.controller, initial });
  app.step = null;
  Object.values(charts).forEach((c) => c.clear());
  view.clearTrail();
};

/** After the airframe was edited: rebuild the blocks that precompute from it, keep the flight going. */
function applyDesign() {
  const keep = app.sim ? { position: app.sim.state.p, velocity: app.sim.state.v, attitude: app.sim.state.q, rates: app.sim.state.w } : undefined;
  const ref = app.sim && app.sim.reference;
  app.resetSim(keep);
  if (ref) app.sim.reference = ref;
  view.rebuild(app.cfg);
}

// ---- menu ------------------------------------------------------------------------------------------------
function renderMenu() {
  $('modeCards').innerHTML = modes.map((m, i) => `<button class="card${m === app.mode ? ' on' : ''}" data-mode="${i}">
    <b>${m.name}</b><span>${m.description}</span></button>`).join('');
  $('presetSel').innerHTML = presets.map((p, i) => `<option value="${i}"${i === app.presetIndex ? ' selected' : ''}>${p.name}</option>`).join('');
  renderDesign($('designBox'), app.cfg, () => { if (app.sim) applyDesign(); });
}
$('modeCards').onclick = (e) => { const c = e.target.closest('[data-mode]'); if (c) { app.mode = modes[+c.dataset.mode]; renderMenu(); } };
$('presetSel').onchange = (e) => { app.presetIndex = +e.target.value; app.cfg = cloneConfig(presets[app.presetIndex]); view.rebuild(app.cfg); renderMenu(); };
$('flyBtn').onclick = () => startMode();
$('menuBtn').onclick = () => openMenu();
$('resetBtn').onclick = () => startMode();

function openMenu() { app.paused = true; renderMenu(); $('menu').classList.remove('hidden'); }
function startMode() {
  $('menu').classList.add('hidden');
  const m = app.mode;
  m.start(app);
  view.camera.mode = m.camera;
  $('modeName').textContent = `${m.name} · ${app.cfg.name}`;
  document.body.dataset.mode = m.id;
  $('labPanel').classList.toggle('hidden', !m.panels.includes('gains'));
  renderGains($('gainsBox'), app.cfg, () => {});
  renderSteps();
  app.paused = false;
  last = performance.now();
}

// ---- tuning lab: step tests ------------------------------------------------------------------------------
function renderSteps() {
  $('stepResult').innerHTML = '';
  const C = app.cfg.controller.position, wn = Math.sqrt(C.kp[2]), zeta = C.kd[2] / (2 * wn);
  $('analytic').textContent = `z analytic (fast motors, no drag, no Ki): ω_n = √Kp = ${wn.toFixed(2)} rad/s · ζ = Kd/(2√Kp) = ${zeta.toFixed(2)}`;
}
function startStep(axis, size) {
  const ref = app.sim.reference.position, y0 = ref[axis];
  ref[axis] = y0 + size;
  app.step = { axis, t0: app.sim.t, y0: app.sim.state.p[axis], y1: ref[axis], samples: [] };
  $('stepResult').innerHTML = '<span class="muted">measuring…</span>';
}
$('stepZ').onclick = () => startStep(2, app.sim.reference.position[2] > 2 ? -1 : 1);
$('stepX').onclick = () => startStep(0, app.sim.reference.position[0] > 0.5 ? -1 : 1);
$('gainsBox').addEventListener('input', renderSteps);

function finishStep() {
  const s = app.step, m = stepMetrics(s.samples, s.t0, s.y0, s.y1);
  app.step = null;
  if (!m) return;
  const f = (v, u, d = 2) => (v === null ? '—' : `${v.toFixed(d)} ${u}`);
  $('stepResult').innerHTML = `<b>${'xyz'[s.axis]} step ${(s.y1 - s.y0).toFixed(1)} m</b>: rise ${f(m.rise, 's')} · overshoot ${f(m.overshoot, '%', 1)} · settling (2 %) ${f(m.settling, 's')} · steady error ${f(m.steadyError * 100, 'cm', 1)}`;
}

// ---- the loop --------------------------------------------------------------------------------------------
let last = performance.now(), acc = 0, frames = 0;
function frame(now) {
  requestAnimationFrame(frame);
  const dtReal = Math.min(0.05, (now - last) / 1000);   // a long pause (tab switch) does not fast-forward
  last = now;
  if (!app.sim) return;
  if (!app.paused && !document.hidden) {
    acc += dtReal;
    const Tc = app.cfg.controller.dt;
    while (acc >= Tc) {
      acc -= Tc;
      const sticks = input.read();
      app.sim.reference = app.mode.reference(app, sticks) || app.sim.reference;
      app.sim.tick();
      const s = app.sim.state, ref = app.sim.reference;
      if (app.sim.steps % 5 === 0) {
        const e = ref.position ? [0, 1, 2].map((k) => ref.position[k] - s.p[k]) : [0, 0, 0];
        charts.error.push(app.sim.t, e);
        charts.attitude.push(app.sim.t, toEuler(s.q).map((a) => a * deg));
        charts.motors.push(app.sim.t, app.sim.motors.omega.map((w) => (100 * w) / app.cfg.motor.omegaMax));
      }
      if (app.step) {
        app.step.samples.push({ t: app.sim.t, y: s.p[app.step.axis] });
        if (app.sim.t - app.step.t0 > 8) finishStep();
      }
    }
  }
  view.update(app.sim, { reference: app.sim.reference.position ? app.sim.reference : null });
  if (++frames % 2 === 0) Object.values(charts).forEach((c) => c.draw());
  hud();
}

function hud() {
  const s = app.sim.state, e = toEuler(s.q).map((a) => a * deg);
  const sat = app.sim.command.saturated;
  $('hud').innerHTML = `alt <b>${s.p[2].toFixed(2)}</b> m · speed <b>${Math.hypot(...s.v).toFixed(1)}</b> m/s · roll ${e[0].toFixed(0)}° pitch ${e[1].toFixed(0)}° yaw ${e[2].toFixed(0)}°${sat ? ' · <span class="warn">motors saturated</span>' : ''}`;
}

// ---- keyboard shortcuts and touch sticks -----------------------------------------------------------------
window.addEventListener('keydown', (e) => {
  if (/input|select|textarea/i.test(e.target.tagName)) return;
  if (e.code === 'KeyR') startMode();
  if (e.code === 'Escape') { if ($('menu').classList.contains('hidden')) openMenu(); else startMode(); }
  if (e.code === 'KeyC') view.camera.mode = view.camera.mode === 'chase' ? 'orbit' : 'chase';
});
attachTouchSticks($('sticks'), input);
$('camBtn').onclick = () => { view.camera.mode = view.camera.mode === 'chase' ? 'orbit' : 'chase'; };
$('plotsBtn').onclick = () => document.body.classList.toggle('no-plots');

app.resetSim({ position: [0, 0, 0] });
renderMenu();
requestAnimationFrame(frame);
