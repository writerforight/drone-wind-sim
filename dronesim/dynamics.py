"""6-DOF rigid-body dynamics of a multirotor, integrated with RK4.

State: position p and velocity v (world, ENU), attitude quaternion q (body -> world), body rates w,
rotor speeds n (rev/s).

    p' = v
    v' = g + (R F_b) / m                    F_b = rotor thrust + body drag + rotor drag (in body axes)
    q' = 1/2 q (x) (0, w)
    w' = J^-1 (tau - w x J w)               tau = rotor torques (allocation) + aerodynamic damping
    n' = (n_cmd - n) / tau_motor            per motor, n_cmd clipped to [n_min, n_max]

Aerodynamics use the velocity relative to the air, v_rel = v - wind:
    body drag   F_i = -1/2 rho (C_d A)_i |v_rel| v_rel,i         (per body axis)
    rotor drag  F_xy = -k_r (sum n_i) v_rel,xy                     (blade flapping / induced drag;
                                                                    linear in airspeed, grows with rotor speed)
The wind enters only through these two terms, exactly as in reality.  The ground is a plane at z = 0.
"""
import numpy as np

from .atmosphere import G0
from .rotation import quat_mul, quat_to_rot

GRAVITY = np.array([0.0, 0.0, -G0])


def cross(a, b):
    """3-vector cross product (np.cross is slow for single vectors)."""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


class State:
    __slots__ = ("p", "v", "q", "w", "n")

    def __init__(self, p, v, q, w, n):
        self.p, self.v, self.q, self.w, self.n = p, v, q, w, n

    def copy(self):
        return State(self.p.copy(), self.v.copy(), self.q.copy(), self.w.copy(), self.n.copy())

    @property
    def R(self):
        return quat_to_rot(self.q)


class RigidBodyDynamics:
    def __init__(self, airframe, atmosphere):
        self.af = airframe
        self.atm = atmosphere
        self.B = airframe.allocation()
        self.motors = [r.motor for r in airframe.rotors]
        self.tau_m = np.array([m.tau for m in self.motors])
        self.tau_down = np.array([m.tau if m.tau_down is None else m.tau_down for m in self.motors])
        self.n_min = np.array([m.n_min for m in self.motors])
        self.n_max = np.array([m.n_max for m in self.motors])
        self.ang_damping = np.asarray(airframe.ang_damping, dtype=float)   # aerodynamic rotational damping
        # per-rotor constants as arrays (vectorised forces; same formulas as MotorModel)
        self.kT = np.array([m.health * m.c_t * m.diameter ** 4 for m in self.motors])
        self.kQ = np.array([m.health * m.c_q * m.diameter ** 5 for m in self.motors])
        pos = np.array([r.position for r in airframe.rotors], dtype=float)
        self.rx, self.ry = pos[:, 0], pos[:, 1]
        self.spin = np.array([r.spin for r in airframe.rotors], dtype=float)

    def set_motor_health(self, i, health):
        """Change the health of rotor i mid-flight (e.g. 0.0 = motor failure)."""
        m = self.af.rotors[i].motor = self.af.rotors[i].motor.copy(health=health)
        self.motors[i] = m
        self.kT[i] = m.health * m.c_t * m.diameter ** 4
        self.kQ[i] = m.health * m.c_q * m.diameter ** 5

    # ---------------------------------------------------------------------------------------------
    def rotor_forces(self, n, rho):
        n2 = n * n * rho
        return self.kT * n2, self.kQ * n2

    def wrench(self, s, wind, rho):
        """Total body-frame force and torque, plus the individual rotor thrusts."""
        af = self.af
        T, Q = self.rotor_forces(s.n, rho)
        R = quat_to_rot(s.q)
        v_rel_b = R.T @ (s.v - wind)
        speed = np.linalg.norm(v_rel_b)
        F = np.array([0.0, 0.0, T.sum()])
        F += -0.5 * rho * af.drag_area * speed * v_rel_b
        F[:2] += -af.rotor_drag * s.n.sum() * v_rel_b[:2]
        # r_i x (0, 0, T_i) = (y_i T_i, -x_i T_i, 0); reaction torque -s_i Q_i about z
        tau = np.array([self.ry @ T, -(self.rx @ T), -(self.spin @ Q)]) - self.ang_damping * s.w
        return F, tau, T

    def derivative(self, s, n_cmd, wind, rho, ext_force=None):
        af = self.af
        F, tau, _ = self.wrench(s, wind, rho)
        R = quat_to_rot(s.q)
        acc = GRAVITY + R @ F / af.mass
        if ext_force is not None:
            acc = acc + ext_force / af.mass
        qd = 0.5 * quat_mul(s.q, np.array([0.0, *s.w]))
        wd = af.inertia_inv @ (tau - cross(s.w, af.inertia @ s.w))
        dn = np.clip(n_cmd, self.n_min, self.n_max) - s.n
        nd = dn / np.where(dn >= 0.0, self.tau_m, self.tau_down)
        return State(s.v, acc, qd, wd, nd)

    def step(self, s, n_cmd, wind, dt, ext_force=None):
        """One RK4 step; wind and commands are held constant over the step."""
        rho = float(self.atm.density(max(s.p[2], 0.0)))

        def add(a, k, h):
            return State(a.p + h * k.p, a.v + h * k.v, a.q + h * k.q, a.w + h * k.w, a.n + h * k.n)

        k1 = self.derivative(s, n_cmd, wind, rho, ext_force)
        k2 = self.derivative(add(s, k1, dt / 2), n_cmd, wind, rho, ext_force)
        k3 = self.derivative(add(s, k2, dt / 2), n_cmd, wind, rho, ext_force)
        k4 = self.derivative(add(s, k3, dt), n_cmd, wind, rho, ext_force)
        out = State(s.p + dt / 6 * (k1.p + 2 * k2.p + 2 * k3.p + k4.p),
                    s.v + dt / 6 * (k1.v + 2 * k2.v + 2 * k3.v + k4.v),
                    s.q + dt / 6 * (k1.q + 2 * k2.q + 2 * k3.q + k4.q),
                    s.w + dt / 6 * (k1.w + 2 * k2.w + 2 * k3.w + k4.w),
                    s.n + dt / 6 * (k1.n + 2 * k2.n + 2 * k3.n + k4.n))
        out.q /= np.linalg.norm(out.q)
        out.n = np.clip(out.n, 0.0, self.n_max)
        # ground plane
        if out.p[2] < 0.0:
            out.p[2] = 0.0
            if out.v[2] < 0.0:
                out.v[2] = 0.0
            out.v[:2] *= 0.5
            out.w *= 0.5
        return out
