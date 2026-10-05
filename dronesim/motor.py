"""Motor + propeller model.

    thrust  T = C_T * rho * n^2 * D^4          n = rotor speed [rev/s], D = propeller diameter [m]
    torque  Q = C_Q * rho * n^2 * D^5          (reaction torque on the frame, opposite to the spin)
    power   P = 2 pi n Q / eta                 electrical power drawn (eta = motor + ESC efficiency)
    speed   tau * dn/dt = n_cmd - n            first-order lag of motor + ESC, n limited to [n_min, n_max];
                                               tau_down (if set) is used while slowing down (PX4 / Gazebo
                                               motors spin up faster than they spin down)

C_T, C_Q are the standard non-dimensional propeller coefficients (UIUC propeller database convention),
so thrust drops automatically with air density (temperature, altitude).  ``health`` scales the thrust
and torque of a damaged motor or propeller (1.0 = healthy).
"""
from dataclasses import dataclass, replace

import numpy as np


@dataclass
class MotorModel:
    diameter: float = 0.254         # 10 inch propeller [m]
    c_t: float = 0.11               # thrust coefficient
    c_q: float = 0.0045             # torque coefficient
    tau: float = 0.04               # time constant of the speed response [s] (spin-up)
    tau_down: float = None          # time constant while slowing down [s]; None = same as tau
    n_min: float = 10.0             # idle speed [rev/s]
    n_max: float = 135.0            # top speed [rev/s] (~8100 rpm)
    efficiency: float = 0.75        # electrical -> mechanical
    health: float = 1.0             # 1 = healthy, 0.7 = 30 % thrust loss, 0 = failed

    def thrust(self, n, rho):
        return self.health * self.c_t * rho * n ** 2 * self.diameter ** 4

    def torque(self, n, rho):
        return self.health * self.c_q * rho * n ** 2 * self.diameter ** 5

    def power(self, n, rho):
        return 2 * np.pi * n * self.torque(n, rho) / self.efficiency

    def speed_for_thrust(self, T, rho):
        """Inverse of ``thrust`` (used by the mixer to turn a thrust command into a speed command)."""
        k = max(self.health, 1e-6) * self.c_t * rho * self.diameter ** 4
        return np.sqrt(np.maximum(T, 0.0) / k)

    def max_thrust(self, rho):
        return self.thrust(self.n_max, rho)

    def torque_per_thrust(self):
        """Q / T = (C_Q / C_T) D, independent of speed and density."""
        return self.c_q / self.c_t * self.diameter

    def copy(self, **changes):
        return replace(self, **changes)
