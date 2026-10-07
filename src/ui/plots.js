/**
 * ui/plots — small live strip charts on a canvas (no chart library).
 *
 *   const chart = createChart(canvas, { title, series: [{ name, color }], span: 8, unit, fixed?: [lo, hi] });
 *   chart.push(t, [v1, v2, ...]);  chart.draw();
 */
export function createChart(canvas, { title, series, span = 8, unit = '', fixed = null }) {
  const data = [];                                       // [t, values[]]
  const ctx = canvas.getContext('2d');
  return {
    clear() { data.length = 0; },
    push(t, values) {
      data.push([t, values]);
      while (data.length && data[0][0] < t - span) data.shift();
    },
    draw() {
      const w = canvas.clientWidth, h = canvas.clientHeight, dpr = window.devicePixelRatio || 1;
      if (!w || !h) return;
      if (canvas.width !== Math.round(w * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      const L = 34, R = 6, T = 16, B = 14;
      let lo = Infinity, hi = -Infinity;
      if (fixed) [lo, hi] = fixed;
      else for (const [, v] of data) for (const x of v) { if (x < lo) lo = x; if (x > hi) hi = x; }
      if (!Number.isFinite(lo)) { lo = -1; hi = 1; }
      if (hi - lo < 1e-6) { lo -= 0.5; hi += 0.5; }
      const pad = (hi - lo) * 0.08; lo -= fixed ? 0 : pad; hi += fixed ? 0 : pad;
      const t1 = data.length ? data[data.length - 1][0] : span, t0 = t1 - span;
      const X = (t) => L + ((t - t0) / span) * (w - L - R), Y = (v) => T + (1 - (v - lo) / (hi - lo)) * (h - T - B);
      ctx.font = '10px ui-monospace, monospace';
      ctx.fillStyle = '#8b949e';
      ctx.fillText(title, L, 11);
      ctx.textAlign = 'right';
      ctx.fillText(fmt(hi), L - 4, T + 8); ctx.fillText(fmt(lo), L - 4, h - B);
      ctx.textAlign = 'left';
      if (lo < 0 && hi > 0) { ctx.strokeStyle = '#30363d'; ctx.beginPath(); ctx.moveTo(L, Y(0)); ctx.lineTo(w - R, Y(0)); ctx.stroke(); }
      ctx.strokeStyle = '#30363d'; ctx.strokeRect(L, T, w - L - R, h - T - B);
      series.forEach((s, k) => {
        ctx.strokeStyle = s.color; ctx.lineWidth = 1.5; ctx.beginPath();
        data.forEach(([t, v], i) => { const x = X(t), y = Y(v[k]); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
        ctx.stroke();
      });
      let lx = L + ctx.measureText(title).width + 10;
      series.forEach((s) => { ctx.fillStyle = s.color; ctx.fillText(s.name, lx, 11); lx += ctx.measureText(s.name).width + 8; });
      if (unit) { ctx.fillStyle = '#8b949e'; ctx.textAlign = 'right'; ctx.fillText(unit, w - R, 11); ctx.textAlign = 'left'; }
    },
  };
}
const fmt = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));
