"""Sensor / state-estimate noise.

The controller never sees the true state; it sees what a flight controller's estimator would output:

    position, velocity   GNSS-like: updated at ``gnss_rate`` Hz (held in between), white noise
    attitude             small white noise on roll / pitch / yaw (an attitude filter's residual error)
    body rates           gyro: white noise + a bias that drifts as a random walk

This is a measurement-error model of the estimate, not a full EKF; it is enough to show how noise and
update rate limit a controller (and it is a clean place to plug in a real estimator later).
"""
from dataclasses import dataclass

import numpy as np

from .rotation import euler_to_quat, quat_mul


@dataclass
class SensorModel:
    pos_std: float = 0.05            # [m]
    vel_std: float = 0.05            # [m/s]
    att_std_deg: float = 0.3         # [deg]
    gyro_std: float = 0.005          # [rad/s]
    gyro_bias_walk: float = 0.0005   # [rad/s/sqrt(s)]
    gnss_rate: float = 50.0          # [Hz]
    seed: int = 1

    @classmethod
    def ideal(cls):
        return cls(0, 0, 0, 0, 0, 1e9)

    @classmethod
    def preset(cls, name):
        return {"ideal": cls.ideal(),
                "good": cls(),
                "gps": cls(pos_std=0.4, vel_std=0.1, att_std_deg=0.5, gyro_std=0.01, gnss_rate=10.0),
                "poor": cls(pos_std=1.0, vel_std=0.25, att_std_deg=1.5, gyro_std=0.03, gyro_bias_walk=0.003,
                            gnss_rate=5.0)}[name]

    def reset(self, state):
        self.rng = np.random.default_rng(self.seed)
        self.bias = np.zeros(3)
        self.t_gnss = -1e9
        self._p = state.p.copy()
        self._v = state.v.copy()

    def measure(self, t, dt, s):
        r = self.rng
        if t - self.t_gnss >= 1.0 / self.gnss_rate - 1e-9:
            self.t_gnss = t
            self._p = s.p + self.pos_std * r.standard_normal(3)
            self._v = s.v + self.vel_std * r.standard_normal(3)
        self.bias += self.gyro_bias_walk * np.sqrt(dt) * r.standard_normal(3)
        dq = euler_to_quat(*(np.radians(self.att_std_deg) * r.standard_normal(3)))
        return Estimate(p=self._p.copy(), v=self._v.copy(), q=quat_mul(s.q, dq),
                        w=s.w + self.bias + self.gyro_std * r.standard_normal(3))


@dataclass
class Estimate:
    p: np.ndarray
    v: np.ndarray
    q: np.ndarray
    w: np.ndarray
