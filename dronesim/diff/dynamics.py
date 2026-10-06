"""Differentiable, batched copy of the 6-DOF dynamics (PyTorch).

Same equations as ``dronesim/dynamics.py`` (see there), so gradients of any loss on the trajectory can
flow back through the physics into a controller:

    p' = v,   v' = g + R F_b / m,   q' = 1/2 q (x) (0, w),   w' = J^-1 (tau - w x J w),   n' = (n_cmd - n) / tau_m

Every tensor has a leading batch dimension B: B drones fly at once, each with its own mass, air and
wind.  Differences to the NumPy version:
    - wind is an input held constant over a step (no gradient flows into it); the NumPy version updates
      it every physics step from the full wind model
    - no ground plane (training starts high up; the clamp would cut the gradient)
``tests/test_diff_dynamics.py`` checks that both versions agree to ~1e-12 m with the same inputs.
"""
import numpy as np
import torch

from ..atmosphere import G0, LAPSE, R_AIR


def quat_to_rot(q):
    """(B, 4) quaternions (w, x, y, z) -> (B, 3, 3) rotation matrices body -> world."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).view(-1, 3, 3)


def quat_mul_w(q, w):
    """q (x) (0, w) for (B, 4) q and (B, 3) w."""
    qw, qx, qy, qz = q.unbind(-1)
    wx, wy, wz = w.unbind(-1)
    return torch.stack([-qx * wx - qy * wy - qz * wz,
                        qw * wx + qy * wz - qz * wy,
                        qw * wy - qx * wz + qz * wx,
                        qw * wz + qx * wy - qy * wx], -1)


class DiffState:
    """p (B,3), v (B,3), q (B,4), w (B,3), n (B,n_rotors)."""
    __slots__ = ("p", "v", "q", "w", "n")

    def __init__(self, p, v, q, w, n):
        self.p, self.v, self.q, self.w, self.n = p, v, q, w, n

    def detach(self):
        return DiffState(*(t.detach() for t in (self.p, self.v, self.q, self.w, self.n)))

    def axpy(self, k, h):
        return DiffState(self.p + h * k.p, self.v + h * k.v, self.q + h * k.q, self.w + h * k.w, self.n + h * k.n)


class DiffDynamics:
    """Batched rigid-body + motor dynamics for one airframe type.

    mass_scale (B,)  payload: mass, inertia and rotational damping scaled together (as in DroneEnv)
    T0, p0 (B,)      ground temperature [K] and pressure [Pa] of each drone's air (ISA law above it)
    """

    def __init__(self, airframe, mass_scale, T0, p0, dtype=torch.float32, device="cpu"):
        kw = dict(dtype=dtype, device=device)
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=float), **kw)   # noqa: E731
        af = airframe
        motors = [r.motor for r in af.rotors]
        self.n_rot = af.n
        self.mass = t(af.mass) * mass_scale                                         # (B,)
        J = t(af.inertia)
        self.J = J[None] * mass_scale[:, None, None]                                # (B,3,3)
        self.J_inv = torch.linalg.inv(J)[None] / mass_scale[:, None, None]
        self.ang_damping = t(af.ang_damping)[None] * mass_scale[:, None]            # (B,3)
        self.drag_area = t(af.drag_area)
        self.rotor_drag = float(af.rotor_drag)
        self.kT = t([m.health * m.c_t * m.diameter ** 4 for m in motors])
        self.kQ = t([m.health * m.c_q * m.diameter ** 5 for m in motors])
        pos = np.array([r.position for r in af.rotors], dtype=float)
        self.rx, self.ry = t(pos[:, 0]), t(pos[:, 1])
        self.spin = t([r.spin for r in af.rotors])
        self.tau_up = t([m.tau for m in motors])
        self.tau_down = t([m.tau if m.tau_down is None else m.tau_down for m in motors])
        self.n_min = t([m.n_min for m in motors])
        self.n_max = t([m.n_max for m in motors])
        self.T0, self.p0 = T0, p0
        self.gravity = t([0.0, 0.0, -G0])

    def density(self, h):
        """ISA density at height h (B,) above each drone's ground."""
        T = self.T0 - LAPSE * h.clamp(min=0.0)
        return self.p0 * (T / self.T0) ** (G0 / (R_AIR * LAPSE)) / (R_AIR * T)

    def derivative(self, s, n_cmd, wind, rho):
        n2 = s.n * s.n * rho[:, None]
        T = self.kT * n2                                                            # (B,n)
        Q = self.kQ * n2
        R = quat_to_rot(s.q)
        v_rel_b = torch.einsum("bji,bj->bi", R, s.v - wind)                         # R^T (v - wind)
        speed = v_rel_b.norm(dim=-1, keepdim=True)
        F = -0.5 * rho[:, None] * self.drag_area * speed * v_rel_b
        F_xy = F[:, :2] - self.rotor_drag * s.n.sum(-1, keepdim=True) * v_rel_b[:, :2]
        F = torch.cat([F_xy, F[:, 2:] + T.sum(-1, keepdim=True)], -1)
        tau = torch.stack([T @ self.ry, -(T @ self.rx), -(Q @ self.spin)], -1) - self.ang_damping * s.w
        acc = self.gravity + torch.einsum("bij,bj->bi", R, F) / self.mass[:, None]
        Jw = torch.einsum("bij,bj->bi", self.J, s.w)
        wd = torch.einsum("bij,bj->bi", self.J_inv, tau - torch.cross(s.w, Jw, dim=-1))
        dn = torch.minimum(torch.maximum(n_cmd, self.n_min), self.n_max) - s.n
        nd = dn / torch.where(dn >= 0.0, self.tau_up, self.tau_down)
        return DiffState(s.v, acc, 0.5 * quat_mul_w(s.q, s.w), wd, nd)

    def step(self, s, n_cmd, wind, dt, integrator="rk4"):
        """One physics step; motor commands and wind held constant over it."""
        rho = self.density(s.p[:, 2])
        k1 = self.derivative(s, n_cmd, wind, rho)
        if integrator == "euler":
            out = s.axpy(k1, dt)
        else:
            k2 = self.derivative(s.axpy(k1, dt / 2), n_cmd, wind, rho)
            k3 = self.derivative(s.axpy(k2, dt / 2), n_cmd, wind, rho)
            k4 = self.derivative(s.axpy(k3, dt), n_cmd, wind, rho)
            out = DiffState(*(a + dt / 6 * (b1 + 2 * b2 + 2 * b3 + b4) for a, b1, b2, b3, b4 in
                              zip((s.p, s.v, s.q, s.w, s.n), (k1.p, k1.v, k1.q, k1.w, k1.n),
                                  (k2.p, k2.v, k2.q, k2.w, k2.n), (k3.p, k3.v, k3.q, k3.w, k3.n),
                                  (k4.p, k4.v, k4.q, k4.w, k4.n))))
        out.q = out.q / out.q.norm(dim=-1, keepdim=True)
        out.n = torch.minimum(out.n.clamp(min=0.0), self.n_max)
        return out
