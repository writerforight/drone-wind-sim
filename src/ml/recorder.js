/**
 * ml/recorder — logs what a controller saw and did, one row per policy step, exportable as JSON or CSV
 * (SPEC M5: data for imitation learning, from the PID or from your own manual flights, and for plots).
 *
 *   const rec = createRecorder();
 *   rec.add({ t, target, position, token, action, reward });   // any flat numbers or arrays of numbers
 *   rec.toJSON(); rec.toCSV();
 *
 * CSV columns are the row keys, arrays are expanded (token → token_ex, token_ey, … with the feature names,
 * other arrays → key_0, key_1, …).
 */
import { FEATURES } from './features.js';

export function createRecorder({ meta = {} } = {}) {
  const rows = [];
  const columns = (row) => Object.entries(row).flatMap(([k, v]) => {
    if (!v || typeof v !== 'object') return [k];
    return Array.from(v, (_, i) => (k === 'token' && v.length === FEATURES.length ? `token_${FEATURES[i]}` : `${k}_${i}`));
  });
  return {
    rows,
    get length() { return rows.length; },
    clear() { rows.length = 0; },
    add(row) {
      const r = {};
      for (const [k, v] of Object.entries(row)) r[k] = v && typeof v === 'object' ? Array.from(v) : v;
      rows.push(r);
    },
    toJSON() { return JSON.stringify({ format: 'quadlab-log/1', features: FEATURES, meta, rows }); },
    toCSV() {
      if (!rows.length) return '';
      const head = columns(rows[0]);
      const line = (r) => Object.values(r).flatMap((v) => (v && typeof v === 'object' ? v : [v])).map((x) => (x === null || x === undefined ? '' : x)).join(',');
      return [head.join(','), ...rows.map(line)].join('\n');
    },
  };
}
