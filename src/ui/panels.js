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
