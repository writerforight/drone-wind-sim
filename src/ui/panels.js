/**
 * ui/panels — the live gain sliders and the airframe design form. Both edit the config object directly:
 * the controller reads its params on every call, so a slider takes effect on the next control tick.
 */

/** Gain sliders grouped by loop. `onChange()` is called after every edit. */
export function renderGains(box, cfg, onChange) {
  const C = cfg.controller;
  const rows = [
    ['Position x, y', [['Kp', C.position.kp, [0, 1], 0, 6], ['Ki', C.position.ki, [0, 1], 0, 3], ['Kd', C.position.kd, [0, 1], 0, 6]]],
    ['Position z', [['Kp', C.position.kp, [2], 0, 12], ['Ki', C.position.ki, [2], 0, 5], ['Kd', C.position.kd, [2], 0, 8]]],
    ['Attitude roll, pitch', [['Kp', C.attitude.kp, [0, 1], 0, 20]]],
    ['Rate roll, pitch', [['Kp', C.rate.kp, [0, 1], 0, 40], ['Ki', C.rate.ki, [0, 1], 0, 20], ['Kd', C.rate.kd, [0, 1], 0, 1]]],
    ['Yaw (attitude, rate)', [['Kp att', C.attitude.kp, [2], 0, 10], ['Kp rate', C.rate.kp, [2], 0, 20]]],
  ];
  box.innerHTML = rows.map(([title, sliders], g) => `<div class="group"><div class="group-title">${title}</div>${
    sliders.map(([lab, arr, idx, lo, hi], k) => `<label class="slider"><span>${lab}</span>
      <input type="range" min="${lo}" max="${hi}" step="${(hi - lo) / 200}" value="${arr[idx[0]]}" data-g="${g}" data-k="${k}">
      <output>${(+arr[idx[0]]).toFixed(2)}</output></label>`).join('')}</div>`).join('');
  box.oninput = (e) => {
    const el = e.target;
    if (el.type !== 'range') return;
    const [, arr, idx] = rows[+el.dataset.g][1][+el.dataset.k];
    for (const i of idx) arr[i] = +el.value;
    el.nextElementSibling.textContent = (+el.value).toFixed(2);
    onChange();
  };
}

/** Airframe design: mass, arm, motor constants. Shows what the numbers mean (hover speed, thrust-to-weight). */
export function renderDesign(box, cfg, onChange) {
  const P = cfg.physics, M = cfg.motor;
  const fields = [
    ['Mass', P, 'mass', 'kg', 0.2, 6, 0.05],
    ['Arm length', M, 'arm', 'm', 0.05, 0.6, 0.005],
    ['Motor lag τ', M, 'tau', 's', 0.005, 0.15, 0.005],
    ['Max rotor speed', M, 'omegaMax', 'rad/s', 300, 3500, 10],
    ['Thrust coeff. k_T', M, 'kThrust', 'N/(rad/s)²', 2e-7, 5e-5, 1e-7],
    ['Body drag', P, 'linearDrag', 'N·s/m', 0, 1.5, 0.01],
  ];
  const derived = () => {
    const W = P.mass * P.gravity, Tmax = 4 * M.kThrust * M.omegaMax ** 2, wh = Math.sqrt(W / (4 * M.kThrust));
    return `hover: ${wh.toFixed(0)} rad/s (${(wh * 60 / (2 * Math.PI)).toFixed(0)} rpm) · thrust / weight ${(Tmax / W).toFixed(2)}${Tmax / W < 1.3 ? ' — too low to climb well' : ''}`;
  };
  box.innerHTML = fields.map(([lab, obj, key, unit, lo, hi, st], k) => `<label class="slider"><span>${lab}</span>
      <input type="range" min="${lo}" max="${hi}" step="${st}" value="${obj[key]}" data-k="${k}">
      <output>${fmtNum(obj[key])} ${unit}</output></label>`).join('') + `<div class="derived">${derived()}</div>`;
  box.oninput = (e) => {
    const el = e.target;
    if (el.type !== 'range') return;
    const [, obj, key, unit] = fields[+el.dataset.k];
    obj[key] = +el.value;
    el.nextElementSibling.textContent = `${fmtNum(obj[key])} ${unit}`;
    box.querySelector('.derived').textContent = derived();
    onChange();
  };
}
const fmtNum = (v) => (Math.abs(v) < 0.01 && v !== 0 ? v.toExponential(1) : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(3).replace(/0+$/, '').replace(/\.$/, ''));

/**
 * MLP lab: desired speed, ghost switch, pin help, and the network: layers, width, activation, optimizer,
 * learning rate, exploration noise σ and the input groups. Edits `lab` directly; a structural change (layers,
 * width, activation, inputs) calls onRebuild() so the network is built anew.
 */
export function renderMlpLab(box, lab, { onRebuild = () => {}, onFly = () => {}, onLand = () => {}, groups = [], activations = [] } = {}) {
  const N = lab.net, lrExp = Math.log10(N.lr);
  const sel = (id, opts, v) => `<select id="${id}">${opts.map((o) => `<option${o === v ? ' selected' : ''}>${o}</option>`).join('')}</select>`;
  box.innerHTML = `<label class="slider"><span>Speed</span>
      <input id="mlpSpeed" type="range" min="0.2" max="5" step="0.1" value="${lab.speed}">
      <output>${lab.speed.toFixed(1)} m/s</output></label>
    <label class="check"><input id="ghostOn" type="checkbox"${lab.ghostOn ? ' checked' : ''}> Ghost PID drone (comparison only)</label>
    <div class="muted small">Move the pin: <b>W/A/S/D</b> horizontal (as seen from the camera) · <b>↑/↓</b> up / down ·
      Shift finer · drag the view to turn the camera, wheel to zoom. Touch: right stick horizontal, left stick up / down.</div>
    <div class="btns learn-btns"><button id="flyBtnMlp" class="primary">${lab.flying ? '■ Land' : '▶ Fly + learn'}</button>
      <button id="learnBtn"${lab.flying ? '' : ' disabled'}>${lab.learnOn === false ? 'Learning off' : 'Learning on'}</button>
      <button id="netResetBtn" title="New random weights">New weights</button></div>
    <div id="learnStats" class="mono small"></div>
    <div class="panel-title">Network</div>
    <div id="netShape" class="netshape"></div>
    <label class="slider"><span>Layers</span><input id="netLayers" type="range" min="1" max="4" step="1" value="${N.hidden.length}"><output>${N.hidden.length}</output></label>
    <label class="slider"><span>Width</span><input id="netWidth" type="range" min="2" max="64" step="1" value="${N.hidden[0]}"><output>${N.hidden[0]}</output></label>
    <label class="row small">Activation ${sel('netAct', activations, N.act)}</label>
    <label class="row small">Optimizer ${sel('netOpt', ['adam', 'sgd'], N.optimizer)}</label>
    <label class="slider"><span>Learn rate</span><input id="netLr" type="range" min="-5" max="-1" step="0.1" value="${lrExp}"><output>${N.lr.toExponential(0)}</output></label>
    <label class="slider"><span>Noise σ</span><input id="netSigma" type="range" min="0.01" max="0.5" step="0.01" value="${N.sigma}"><output>${N.sigma.toFixed(2)}</output></label>
    <div class="group-title">Inputs</div>
    ${groups.map((g) => `<label class="check small"><input type="checkbox" data-in="${g.id}"${N.inputs[g.id] ? ' checked' : ''}> ${g.label} (${g.n})</label>`).join('')}
    <div id="mlpOut" class="mono small"></div>
    <div class="muted small">Every control step (250 Hz): noise ε ~ N(0, σ²) on the output → one step of physics →
      L = |target − p| + |v − v_des| → gradient (L<sub>t</sub> − L<sub>t−1</sub>)·ε/σ² → backprop → one optimizer step.
      No physics model, no teacher. A crash puts the drone back at the start; the weights stay.</div>`;
  const shape = () => {
    const nIn = groups.reduce((n, g) => n + (N.inputs[g.id] ? g.n : 0), 0);
    box.querySelector('#netShape').innerHTML = `<b>${nIn}</b> → ${N.hidden.map((w) => `<b>${w}</b>`).join(' → ')} → <b>4</b> [T, τ] → <span class="fixed">fixed A⁻¹</span> → 4 motors`;
  };
  shape();
  box.onclick = (e) => {
    const id = e.target.id;
    if (id === 'flyBtnMlp') {
      if (lab.flying) onLand(); else onFly();
      e.target.textContent = lab.flying ? '■ Land' : '▶ Fly + learn';
      box.querySelector('#learnBtn').disabled = !lab.flying;
    }
    if (id === 'learnBtn') { lab.learnOn = !lab.learnOn; e.target.textContent = lab.learnOn ? 'Learning on' : 'Learning off'; }
    if (id === 'netResetBtn') { N.seed = (N.seed || 1) + 1; onRebuild(); }
  };
  box.oninput = box.onchange = (e) => {
    const el = e.target, out = el.nextElementSibling;
    let rebuild = false;
    if (el.id === 'mlpSpeed') { lab.speed = +el.value; out.textContent = `${lab.speed.toFixed(1)} m/s`; }
    if (el.id === 'ghostOn') lab.ghostOn = el.checked;
    if (el.id === 'netLayers' || el.id === 'netWidth') {
      const n = +box.querySelector('#netLayers').value, w = +box.querySelector('#netWidth').value;
      out.textContent = el.value;
      if (n !== N.hidden.length || w !== N.hidden[0]) { N.hidden = Array(n).fill(w); rebuild = true; }
    }
    if (el.id === 'netAct' && el.value !== N.act) { N.act = el.value; rebuild = true; }
    if (el.id === 'netOpt') N.optimizer = el.value;
    if (el.id === 'netLr') { N.lr = 10 ** +el.value; out.textContent = N.lr.toExponential(0); }
    if (el.id === 'netSigma') { N.sigma = +el.value; out.textContent = N.sigma.toFixed(2); }
    if (el.dataset.in && e.type === 'change') {
      N.inputs[el.dataset.in] = el.checked;
      if (!Object.values(N.inputs).some(Boolean)) { N.inputs[el.dataset.in] = true; el.checked = true; return; }   // keep ≥ 1 input
      rebuild = true;
    }
    if (rebuild && (e.type === 'change' || el.type === 'range')) { shape(); onRebuild(); }
  };
}
