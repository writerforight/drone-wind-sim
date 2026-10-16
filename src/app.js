/**
 * The app: wires the blocks together and runs them. Rendering is decoupled from physics:
 * every animation frame adds the real elapsed time to an accumulator and spends it in whole controller
 * periods (each = a fixed number of RK4 physics steps), so the simulation does not depend on the frame rate.
 */
import { cloneConfig, presets } from './config.js';
import { createCascadedPID } from './controller/cascaded-pid.js';
import { attachTouchSticks, createInput } from './input/input.js';
import { createSimulation } from './loop/simulation.js';
import { createAgent } from './ml/agent.js';
import { FEATURE_SIZE } from './ml/features.js';
import { loadPolicy } from './ml/nn.js';
import { createPolicyController } from './ml/policy-controller.js';
import { createRecorder } from './ml/recorder.js';
import { modes } from './modes/index.js';
import { stepMetrics } from './modes/tuning-lab.js';
import { toEuler } from './physics/math3.js';
import { createView } from './render/scene.js';
import { createChart } from './ui/plots.js';
import { renderDesign, renderGains } from './ui/panels.js';
import { drawSequence } from './ui/sequence.js';

const $ = (id) => document.getElementById(id);
const esc = (t) => String(t).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);   // names from loaded files
const deg = 180 / Math.PI;

const app = {
  cfg: cloneConfig(presets[0]),
  presetIndex: 0,
  mode: modes[0],
  sim: null,
  controller: null,
  paused: true,
  step: null,                 // the running step test in the lab: { axis, t0, y0, y1, samples }
  // who flies in the Target mode: the PID (policy null) or a learned policy (bundled or loaded from a file)
  policies: [{ name: 'Cascaded PID', policy: null }],
  policyIndex: 0,
  payload: 0,                 // kg added to the simulated drone, unknown to the controller (Target mode)
  wind: 0,                    // m/s along +x, unknown to the controller (Target mode)
  observer: null,             // builds the policy-input window for the heat map and the recorder, whoever flies
  recorder: null,             // non-null while recording
};
window.app = app;             // for the browser checks and for poking around in the console

const input = createInput();
const view = createView($('stage'), app.cfg);
const charts = {
  error: createChart($('plotError'), { title: 'tracking error', unit: 'm', series: [{ name: 'x', color: '#ff6b6b' }, { name: 'y', color: '#51cf66' }, { name: 'z', color: '#4dabf7' }] }),
  attitude: createChart($('plotAttitude'), { title: 'attitude', unit: '°', series: [{ name: 'roll', color: '#ff6b6b' }, { name: 'pitch', color: '#51cf66' }, { name: 'yaw', color: '#4dabf7' }] }),
  motors: createChart($('plotMotors'), { title: 'motors', unit: '% of ω_max', fixed: [0, 100], series: [{ name: 'FR', color: '#e3b341' }, { name: 'BL', color: '#e3b341' }, { name: 'FL', color: '#bc8cff' }, { name: 'BR', color: '#bc8cff' }] }),
};

/** The controller for the current mode: the chosen learned policy where the mode allows one, else the PID. */
function makeController() {
  const choice = app.policies[app.policyIndex];
  if (app.mode.usesPolicy && choice.policy) return createPolicyController(app.cfg, choice.policy);
  return createCascadedPID(app.cfg.physics);
}

app.resetSim = (initial) => {
  app.controller = makeController();
  // The simulated drone is the configured one plus the payload; controllers keep the nominal app.cfg. The gains
  // and the motors stay shared objects, so the live sliders act on the running simulation.
  const trueCfg = { ...app.cfg, physics: { ...app.cfg.physics, mass: app.cfg.physics.mass + (app.mode.usesPolicy ? app.payload : 0) } };
  app.sim = createSimulation(trueCfg, { controller: app.controller, initial });
  if (app.mode.usesPolicy) app.sim.disturbance.wind = [app.wind, 0, 0];
  app.observer = createAgent(app.cfg);
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
  $('targetPanel').classList.toggle('hidden', !m.panels.includes('target'));
  renderTargetPanel();
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

// ---- target mode: target, controller choice, payload and wind -------------------------------------------
const targetInputs = ['tx', 'ty', 'tz'].map((id) => $(id));
function setTarget(p) {
  const box = 20, pos = [Math.max(-box, Math.min(box, p[0])), Math.max(-box, Math.min(box, p[1])), Math.max(0.2, Math.min(15, p[2]))];
  app.sim.reference = { position: pos, yaw: 0 };
  pos.forEach((v, k) => { targetInputs[k].value = v.toFixed(2); });
}
function renderTargetPanel() {
  $('ctlSel').innerHTML = app.policies.map((c, i) => `<option value="${i}"${i === app.policyIndex ? ' selected' : ''}>${esc(c.name)}</option>`).join('');
  if (app.sim && app.sim.reference.position) app.sim.reference.position.forEach((v, k) => { targetInputs[k].value = v.toFixed(2); });
  $('payloadOut').textContent = `${app.payload.toFixed(2)} kg`;
  $('windOut').textContent = `${app.wind.toFixed(1)} m/s`;
  const c = app.controller;
  $('ctlInfo').textContent = c && c.agent
    ? `${c.level} level · acts every ${(c.repeat * app.cfg.controller.dt * 1000).toFixed(0)} ms · sees the last ${c.agent.historyLength} steps`
    : 'cascaded PID: position → attitude → rate';
}
function switchController() {
  app.controller = makeController();
  app.sim.setController(app.controller);
  app.observer = createAgent(app.cfg);
  renderTargetPanel();
}
$('ctlSel').onchange = (e) => { app.policyIndex = +e.target.value; switchController(); };
$('policyFile').onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    const policy = loadPolicy(await file.text());
    app.policies.push({ name: policy.name, policy });
    app.policyIndex = app.policies.length - 1;
    switchController();
  } catch (err) { $('ctlInfo').textContent = `could not load ${file.name}: ${err.message}`; }
  e.target.value = '';
};
$('goBtn').onclick = () => setTarget(targetInputs.map((el) => +el.value || 0));
$('randBtn').onclick = () => {
  const b = app.cfg.ml.episode.targetBox;
  setTarget(b.map(([lo, hi]) => lo + (hi - lo) * Math.random()));
};
targetInputs.forEach((el) => el.addEventListener('keydown', (e) => { if (e.key === 'Enter') $('goBtn').onclick(); }));
view.canvas.addEventListener('dblclick', (e) => {
  if (app.mode.id !== 'target') return;
  const xy = view.pickFloor(e.clientX, e.clientY);
  if (xy) setTarget([xy[0], xy[1], +targetInputs[2].value || 1.5]);
});
$('payload').oninput = (e) => { app.payload = +e.target.value; app.sim.physics.mass = app.cfg.physics.mass + app.payload; renderTargetPanel(); };
$('wind').oninput = (e) => { app.wind = +e.target.value; app.sim.disturbance.wind = [app.wind, 0, 0]; renderTargetPanel(); };

// ---- recording (any mode): the policy-input tokens and the actions, one row per policy step ---------------
$('recBtn').onclick = () => {
  if (!app.recorder) {
    app.recorder = createRecorder({ meta: { drone: app.cfg.name, mode: app.mode.id, controller: app.controller.name, ml: app.cfg.ml } });
    $('recBtn').classList.add('rec');
    $('recBtn').innerHTML = '■<span class="lbl"> Stop</span>';
    return;
  }
  const rec = app.recorder;
  app.recorder = null;
  $('recBtn').classList.remove('rec');
  $('recBtn').innerHTML = '●<span class="lbl"> Rec</span>';
  if (!rec.length) return;
  const url = URL.createObjectURL(new Blob([rec.toJSON()], { type: 'application/json' }));
  const a = Object.assign(document.createElement('a'), { href: url, download: `quadlab-${app.mode.id}-${Date.now()}.json` });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};

/**
 * Once per policy step: the observer builds the same token a policy would see (for the heat map and the
 * recorder). With a learned policy flying, its own agent already does this; with the PID, the PID's output
 * is mapped to an action at the configured level (the label imitation learning would use).
 */
function observeStep() {
  const s = app.sim, ctl = app.controller, ag = ctl.agent || app.observer;
  const repeat = ctl.repeat || Math.round(app.cfg.ml.policyDt / app.cfg.controller.dt);
  if ((s.steps - 1) % repeat !== 0 || !s.last) return;
  const meas = s.last.meas, target = s.reference.position || meas.p;
  if (!ctl.agent) { app.observer.observe(meas, target); app.observer.act(app.observer.actions.fromExpert(s.last.u)); }
  if (app.recorder) {
    const win = ag.window();
    app.recorder.add({ t: s.t, target, position: s.state.p, token: win.slice(win.length - FEATURE_SIZE), action: ag.held });
  }
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
      observeStep();
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
  if (app.mode.id === 'target' && frames % 4 === 0) {
    const ag = app.controller.agent || app.observer;
    drawSequence($('seqCanvas'), ag.window(), ag.historyLength, FEATURE_SIZE);
  }
  hud();
}

function hud() {
  const s = app.sim.state, e = toEuler(s.q).map((a) => a * deg);
  const sat = app.sim.command.saturated;
  const ref = app.sim.reference.position;
  const toTarget = app.mode.id === 'target' && ref ? ` · to target <b>${Math.hypot(ref[0] - s.p[0], ref[1] - s.p[1], ref[2] - s.p[2]).toFixed(2)}</b> m` : '';
  $('hud').innerHTML = `${app.mode.id === 'target' ? `<b>${esc(app.controller.name)}</b> · ` : ''}alt <b>${s.p[2].toFixed(2)}</b> m · speed <b>${Math.hypot(...s.v).toFixed(1)}</b> m/s · roll ${e[0].toFixed(0)}° pitch ${e[1].toFixed(0)}° yaw ${e[2].toFixed(0)}°${toTarget}${sat ? ' · <span class="warn">motors saturated</span>' : ''}`;
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
// the bundled learned policy (works when served over http, e.g. GitHub Pages; not from file://)
fetch('policies/linear-bc.json').then((r) => (r.ok ? r.json() : null)).then((json) => {
  if (json) app.policies.push({ name: json.name, policy: loadPolicy(json) });
}).catch(() => {});
requestAnimationFrame(frame);
