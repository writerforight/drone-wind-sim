"""Control allocation ("mixer") for any rotor geometry.

Desired wrench u = (F_z, tau_x, tau_y, tau_z) -> rotor thrusts T = B^+ u, then motor speed commands.
When a motor would saturate, the command is degraded in the order a real flight controller does it:
first give up yaw torque, then scale roll / pitch torque, keeping total thrust last (staying in the
air matters most).

The mixer works with *assumed* parameters (nominal air density, healthy motors): it does not know the
real temperature or that a propeller is damaged.  That mismatch is exactly what an adaptive controller
has to absorb.
"""
import numpy as np


class Mixer:
    def __init__(self, airframe, rho_assumed=1.225):
        self.af = airframe
        self.B = airframe.allocation()
        self.Bp = np.linalg.pinv(self.B)
        self.rho = rho_assumed
        self.motors = [r.motor.copy(health=1.0) for r in airframe.rotors]
        self.t_min = np.array([m.thrust(m.n_min, rho_assumed) for m in self.motors])
        self.t_max = np.array([m.thrust(m.n_max, rho_assumed) for m in self.motors])

    def _fits(self, T):
        return np.all(T >= self.t_min - 1e-9) and np.all(T <= self.t_max + 1e-9)

    def thrusts(self, u):
        u = np.asarray(u, dtype=float)
        base = self.Bp @ np.array([u[0], 0.0, 0.0, 0.0])
        rp = self.Bp @ np.array([0.0, u[1], u[2], 0.0])
        yaw = self.Bp @ np.array([0.0, 0.0, 0.0, u[3]])
        T = base + rp + yaw
        if self._fits(T):
            return T
        # 1) drop yaw as much as needed, 2) then roll / pitch (bisection on the scale factor)
        for part, rest in ((yaw, base + rp), (rp, base)):
            if self._fits(rest):
                lo, hi = 0.0, 1.0
                for _ in range(20):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if self._fits(rest + mid * part) else (lo, mid)
                return rest + lo * part
        return np.clip(base, self.t_min, self.t_max)

    def speeds(self, u):
        T = np.clip(self.thrusts(u), self.t_min, self.t_max)
        return np.array([m.speed_for_thrust(t, self.rho) for m, t in zip(self.motors, T)])

    def max_collective(self):
        return self.t_max.sum()
