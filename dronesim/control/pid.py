"""Cascaded PID controller (the structure used by PX4 / ArduPilot).

    position --P--> velocity setpoint --PID--> acceleration --> thrust vector + yaw
             --> attitude (geometric, on SO(3)) --P--> body-rate setpoint --PID--> torque --> mixer

Rate and acceleration loops are normalised by inertia / mass, so the same gains behave similarly on
a quad, a hexa or an octo.  The controller only knows the *nominal* mass, inertia and air density.

Gain sets:
    'untuned'  what a first try often looks like: position P only (no integral), soft loops
               -> steady drift in wind, slow and wobbly
    'tuned'    hand-tuned cascade with integral action in velocity and rate loops
    Any gain can be overridden: CascadedPID(airframe, gains={'vel_ki': 2.0}).
"""
import numpy as np

from ..atmosphere import G0
from ..dynamics import cross
from ..rotation import quat_to_rot, vee
from .mixer import Mixer

GAINS = {
    "untuned": dict(pos_kp=0.6, vel_kp=1.2, vel_ki=0.0, vel_kd=0.0,
                    att_kp=(3.0, 3.0, 1.0), rate_kp=(6.0, 6.0, 2.0), rate_ki=(0.0, 0.0, 0.0), rate_kd=(0.0, 0.0, 0.0)),
    "tuned": dict(pos_kp=1.4, vel_kp=3.2, vel_ki=1.4, vel_kd=0.15,
                  att_kp=(9.0, 9.0, 3.0), rate_kp=(22.0, 22.0, 8.0), rate_ki=(6.0, 6.0, 2.0), rate_kd=(0.25, 0.25, 0.0)),
}
LIMITS = dict(v_xy=8.0, v_z=3.0, a_xy=8.0, a_z=6.0, tilt_deg=40.0, rate=np.radians([220, 220, 120]),
              vel_int=8.0, rate_int=3.0)      # integrators may supply up to the acceleration limit


class PID:
    """Vector PID with integral clamping and derivative on measurement (low-pass filtered)."""

    def __init__(self, kp, ki, kd, int_limit, d_tau=0.02):
        self.kp, self.ki, self.kd = (np.asarray(k, dtype=float) for k in (kp, ki, kd))
        self.int_limit = int_limit
        self.d_tau = d_tau
        self.reset()

    def reset(self):
        self.i = 0.0
        self.prev = None
        self.d = 0.0

    def update(self, err, meas, dt):
        self.i = np.clip(self.i + self.ki * err * dt, -self.int_limit, self.int_limit)
        if self.prev is None:
            self.prev = meas
        raw = -(meas - self.prev) / dt
        self.prev = meas
        a = dt / (self.d_tau + dt)
        self.d = self.d + a * (raw - self.d)
        return self.kp * err + self.i + self.kd * self.d


class CascadedPID:
    def __init__(self, airframe, gains="tuned", rho_assumed=1.225, limits=None):
        g = dict(GAINS[gains]) if isinstance(gains, str) else dict(GAINS["tuned"], **gains)
        self.g = g
        self.lim = dict(LIMITS, **(limits or {}))
        self.af = airframe
        self.mass = airframe.mass
        self.J = airframe.inertia
        self.mixer = Mixer(airframe, rho_assumed)
        self.vel_pid = PID(g["vel_kp"], g["vel_ki"], g["vel_kd"], self.lim["vel_int"])
        self.rate_pid = PID(g["rate_kp"], g["rate_ki"], g["rate_kd"], self.lim["rate_int"])
        self.last = {}

    def reset(self):
        self.vel_pid.reset()
        self.rate_pid.reset()

    # ---------------------------------------------------------------------------------------------
    def acceleration_command(self, est, ref, dt):
        """Outer loops: returns the desired acceleration (world)."""
        lim = self.lim
        v_sp = self.g["pos_kp"] * (ref["p"] - est.p) + ref.get("v", 0.0)
        hor = np.linalg.norm(v_sp[:2])
        if hor > lim["v_xy"]:
            v_sp[:2] *= lim["v_xy"] / hor
        v_sp[2] = np.clip(v_sp[2], -lim["v_z"], lim["v_z"])
        a = self.vel_pid.update(v_sp - est.v, est.v, dt) + ref.get("a", 0.0)
        hor = np.linalg.norm(a[:2])
        if hor > lim["a_xy"]:
            a[:2] *= lim["a_xy"] / hor
        a[2] = np.clip(a[2], -lim["a_z"], lim["a_z"])
        self.last["v_sp"] = v_sp
        return a

    def attitude_and_thrust(self, est, a_cmd, yaw):
        """Thrust vector f = m (a + g e_z), tilt limited; desired rotation from its direction and yaw."""
        f = self.mass * (a_cmd + np.array([0.0, 0.0, G0]))
        f[2] = max(f[2], 0.1 * self.mass * G0)
        tilt = np.arctan2(np.linalg.norm(f[:2]), f[2])
        tmax = np.radians(self.lim["tilt_deg"])
        if tilt > tmax:
            f[:2] *= np.tan(tmax) * f[2] / np.linalg.norm(f[:2])
        z_d = f / np.linalg.norm(f)
        x_c = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        y_d = np.cross(z_d, x_c)
        y_d /= np.linalg.norm(y_d)
        R_d = np.column_stack([np.cross(y_d, z_d), y_d, z_d])
        R = quat_to_rot(est.q)
        collective = f @ R[:, 2]
        return R_d, max(collective, 0.0)

    def rate_command(self, est, R_d):
        R = quat_to_rot(est.q)
        e_R = 0.5 * vee(R_d.T @ R - R.T @ R_d)
        w_sp = -np.asarray(self.g["att_kp"]) * e_R
        return np.clip(w_sp, -self.lim["rate"], self.lim["rate"])

    def torque_command(self, est, w_sp, dt):
        alpha = self.rate_pid.update(w_sp - est.w, est.w, dt)
        return self.J @ alpha + cross(est.w, self.J @ est.w)

    def update(self, est, ref, dt):
        """Full cascade -> motor speed commands [rev/s].  ref: {'p', optional 'v', 'a', 'yaw'}."""
        a = self.acceleration_command(est, ref, dt)
        R_d, F = self.attitude_and_thrust(est, a, ref.get("yaw", 0.0))
        w_sp = self.rate_command(est, R_d)
        tau = self.torque_command(est, w_sp, dt)
        u = np.array([F, *tau])
        self.last.update(a_cmd=a, w_sp=w_sp, wrench=u)
        return self.mixer.speeds(u)

    def update_rates(self, est, collective, w_sp, dt):
        """Inner loop only: used when an outer policy commands collective thrust + body rates."""
        tau = self.torque_command(est, w_sp, dt)
        u = np.array([collective, *tau])
        self.last.update(w_sp=w_sp, wrench=u)
        return self.mixer.speeds(u)
