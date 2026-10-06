"""What the network's output u in [-1, 1]^k means for the motors.

Like the PID's mixer, this only knows the *nominal* drone: nominal mass and inertia, healthy motors,
standard air (rho = 1.225).  The real payload, air and wind are for the memory to figure out.

    'motors'        k = number of rotors.  u_i = 0 is the nominal hover thrust of rotor i,
                    u_i = +1 / -1 is hover thrust + / - the largest symmetric margin the motor allows
    'thrust_rates'  k = 4.  u_0: collective thrust (0 = nominal weight, +1 = 2 x weight),
                    u_1..3: body-rate setpoints (+1 = max rate); a fixed P rate loop + allocation turns
                    them into motor speeds.  Easier to learn: the attitude is already stabilised.
"""
import numpy as np
import torch

from dronesim.atmosphere import G0

RHO_NOMINAL = 1.225
RATE_MAX = np.radians([220.0, 220.0, 120.0])     # same limits as the cascaded PID
RATE_KP = np.array([22.0, 22.0, 8.0])            # P gains of the inner rate loop [1/s]


class Actuator:
    def __init__(self, airframe, mode="motors", dtype=torch.float32, device="cpu"):
        if mode not in ("motors", "thrust_rates"):
            raise ValueError(f"unknown control mode {mode!r}")
        kw = dict(dtype=dtype, device=device)
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=float), **kw)   # noqa: E731
        self.mode = mode
        self.act_dim = airframe.n if mode == "motors" else 4
        motors = [r.motor.copy(health=1.0) for r in airframe.rotors]
        B = airframe.allocation()
        self.Bp = t(np.linalg.pinv(B))                                       # (n, 4)
        self.mass = float(airframe.mass)
        self.J = t(airframe.inertia)
        self.k = t([m.c_t * RHO_NOMINAL * m.diameter ** 4 for m in motors])  # T = k n^2 at nominal air
        self.t_min = t([m.thrust(m.n_min, RHO_NOMINAL) for m in motors])
        self.t_max = t([m.thrust(m.n_max, RHO_NOMINAL) for m in motors])
        self.t_hover = self.Bp[:, 0] * self.mass * G0
        self.margin = torch.minimum(self.t_hover - self.t_min, self.t_max - self.t_hover)
        self.rate_max, self.rate_kp = t(RATE_MAX), t(RATE_KP)

    def thrusts(self, u, w):
        """u (B, k) in [-1, 1], measured body rates w (B, 3) -> rotor thrusts (B, n) the controller asks for."""
        if self.mode == "motors":
            return self.t_hover + u * self.margin
        F = self.mass * G0 * (1.0 + u[:, :1])
        w_sp = u[:, 1:] * self.rate_max
        Jw = w @ self.J.T
        tau = (self.rate_kp * (w_sp - w)) @ self.J.T + torch.cross(w, Jw, dim=-1)
        T = torch.cat([F, tau], -1) @ self.Bp.T
        # keep inside what the motors can do (gradient still flows for rotors that are not saturated)
        return torch.minimum(torch.maximum(T, self.t_min), self.t_max)

    def speeds(self, u, w):
        """-> motor speed commands n_cmd [rev/s] (B, n)."""
        return torch.sqrt(self.thrusts(u, w).clamp(min=1e-6) / self.k)
