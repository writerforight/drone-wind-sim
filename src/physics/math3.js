/**
 * Small vector / quaternion helpers on plain arrays (no classes, no allocation tricks).
 * Quaternions are [w, x, y, z] (Hamilton convention, unit length for rotations).
 */

export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const norm = (a) => Math.hypot(a[0], a[1], a[2]);

/** Hamilton product p ⊗ q. */
export function qmul(p, q) {
  return [
    p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
    p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
    p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
    p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0],
  ];
}

export function qnormalize(q) {
  const n = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

/** Rotate a body vector into the world frame: R(q) v. */
export function rotate(q, v) {
  const [w, x, y, z] = q;
  // v + 2 r × (r × v + w v), r = (x, y, z) — the standard expansion of q ⊗ v ⊗ q*
  const r = [x, y, z];
  const t = add(cross(r, v), scale(v, w));
  return add(v, scale(cross(r, t), 2));
}

/** Rotate a world vector into the body frame: R(q)ᵀ v. */
export const rotateInv = (q, v) => rotate([q[0], -q[1], -q[2], -q[3]], v);

/** Quaternion from ZYX Euler angles (roll φ about x, pitch θ about y, yaw ψ about z). */
export function fromEuler(roll, pitch, yaw) {
  const cr = Math.cos(roll / 2), sr = Math.sin(roll / 2);
  const cp = Math.cos(pitch / 2), sp = Math.sin(pitch / 2);
  const cy = Math.cos(yaw / 2), sy = Math.sin(yaw / 2);
  return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy];
}

/** ZYX Euler angles [roll, pitch, yaw] of a unit quaternion. */
export function toEuler(q) {
  const [w, x, y, z] = q;
  const roll = Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
  const pitch = Math.asin(Math.max(-1, Math.min(1, 2 * (w * y - z * x))));
  const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  return [roll, pitch, yaw];
}
