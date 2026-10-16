/**
 * ui/sequence — the observation window a sequence policy is given, drawn as a heat map: time runs left → right
 * (oldest → newest token), one row per feature, blue negative, red positive (white 0). This is literally what
 * an LSTM in 'window' mode reads at every policy step.
 */
const GROUPS = [['e', 0, 3], ['v', 3, 6], ['R', 6, 12], ['w', 12, 15], ['a', 15, 19], ['z', 19, 20]];

export function drawSequence(canvas, window, steps, size) {
  const w = canvas.clientWidth, h = canvas.clientHeight, dpr = globalThis.devicePixelRatio || 1;
  if (!w || !h) return;
  if (canvas.width !== Math.round(w * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const left = 16, cw = (w - left) / steps, ch = h / size;
  for (let t = 0; t < steps; t++) {
    for (let f = 0; f < size; f++) {
      const v = Math.max(-1, Math.min(1, window[t * size + f] / 1.5));
      const a = Math.round(255 * (1 - Math.abs(v)));
      ctx.fillStyle = v >= 0 ? `rgb(255,${a},${a})` : `rgb(${a},${a},255)`;
      ctx.fillRect(left + t * cw, f * ch, Math.ceil(cw), Math.ceil(ch));
    }
  }
  ctx.fillStyle = '#8b949e';
  ctx.font = '10px ui-monospace, monospace';
  ctx.textBaseline = 'middle';
  for (const [name, a, b] of GROUPS) {
    ctx.fillText(name, 3, ((a + b) / 2) * ch);
    ctx.fillRect(left, b * ch - 0.5, w - left, 1);
  }
}
