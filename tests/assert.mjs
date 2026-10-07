/** Tiny assertions with readable messages. */
export function close(actual, expected, tol, what = 'value') {
  if (!(Math.abs(actual - expected) <= tol)) throw new Error(`${what}: ${actual} differs from ${expected} by more than ${tol}`);
}
export function ok(cond, msg) { if (!cond) throw new Error(msg); }
