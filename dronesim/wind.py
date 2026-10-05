"""Wind = mean wind (log profile) + Dryden turbulence + discrete gusts.

World frame is ENU: x = east, y = north, z = up.  All speeds in m/s.

Mean wind
    U(h) = U_ref * ln(h / z0) / ln(h_ref / z0)
    The usual neutral-atmosphere log law: wind grows with height above the ground.  z0 is the surface
    roughness length (0.0002 open water, 0.03 open grass, 0.1 crops, 0.5-1 suburbs / forest).
    ``direction_deg`` follows the weather-report convention: the direction the wind blows FROM
    (0 = from the north, 90 = from the east).

Dryden turbulence (MIL-F-8785C / MIL-HDBK-1797, low altitude model)
    Frozen-turbulence assumption: the vehicle flies through a random velocity field at relative
    airspeed V, so a spatial correlation length L becomes a time constant tau = L / V.
        longitudinal  H_u(s) ∝ 1 / (1 + tau_u s)
        lateral       H_v(s) ∝ (1 + sqrt(3) tau_v s) / (1 + tau_v s)^2      (vertical: same with w)
    Low altitude (h in feet):  L_w = h,  L_u = L_v = h / (0.177 + 0.000823 h)^1.2
                               sigma_w = 0.1 W20,  sigma_u = sigma_v = sigma_w / (0.177 + 0.000823 h)^0.4
    W20 is the wind speed at 20 ft (6.1 m).  The standard severities are light (W20 = 15 kt),
    moderate (30 kt) and severe (45 kt).  Each filter is driven by white noise whose intensity is
    chosen so that the stationary standard deviation is exactly sigma (checked in tests/).

Gusts
    Discrete "1 - cos" gust (also from MIL-F-8785C): v(t) = A/2 (1 - cos(2 pi (t - t0) / T)) on [t0, t0+T].
"""
from dataclasses import dataclass, field

import numpy as np

FT = 0.3048
KT = 0.514444
SEVERITY_W20 = {"none": 0.0, "light": 15 * KT, "moderate": 30 * KT, "severe": 45 * KT}


def wind_vector(speed, from_deg, vertical=0.0):
    """ENU vector of a wind of ``speed`` blowing from ``from_deg`` (0 = north, 90 = east)."""
    a = np.radians(from_deg)
    return np.array([-speed * np.sin(a), -speed * np.cos(a), vertical])


@dataclass
class Gust:
    """1 - cos gust: peak ``speed`` [m/s] blowing from ``from_deg``, starting at t0, lasting ``duration``."""
    t0: float
    duration: float
    speed: float
    from_deg: float = 0.0
    vertical: float = 0.0

    def __call__(self, t):
        s = (t - self.t0) / self.duration
        if s <= 0.0 or s >= 1.0:
            return np.zeros(3)
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * s)) * wind_vector(self.speed, self.from_deg, self.vertical)


def dryden_params(h, w20):
    """(L_u, L_v, L_w) [m] and (sigma_u, sigma_v, sigma_w) [m/s] of the low-altitude Dryden model."""
    hf = np.clip(h / FT, 10.0, 1000.0)                     # the model is defined for 10 ft .. 1000 ft
    k = 0.177 + 0.000823 * hf
    L_w = hf * FT
    L_u = hf / k ** 1.2 * FT
    s_w = 0.1 * w20
    s_u = s_w / k ** 0.4
    return np.array([L_u, L_u, L_w]), np.array([s_u, s_u, s_w])


class DrydenTurbulence:
    """Three shaping filters driven by white noise; output in the (longitudinal, lateral, vertical) axes.

    Realisation chosen so every state has a variance that does not depend on tau = L / V.  When the
    vehicle speeds up or turns, tau changes from one step to the next; with these states the output
    stays continuous (a plain controllable-canonical realisation would jump by a factor (tau_old/tau_new)^2).

        u:     du  = -u / tau dt + sigma sqrt(2 / tau) dW                       (Ornstein-Uhlenbeck)
        v, w:  (1 + sqrt3 tau s) / (1 + tau s)^2 = F (1 + sqrt3 tau s) F,  F = 1 / (1 + tau s)
               z1 = F n,  z2 = F z1,  output = sqrt3 z1 + (1 - sqrt3) z2
               dz1 = -z1 / tau dt + sigma / sqrt(tau) dW,   dz2 = (z1 - z2) / tau dt
               (noise intensity sigma^2 tau makes the output variance exactly sigma^2)
    """

    S3 = np.sqrt(3.0)
    # stationary covariance of (z1, z2) / sigma^2: var z1 = 1/2, var z2 = 1/4, cov = 1/4
    P_UNIT = np.array([[0.5, 0.25], [0.25, 0.25]])
    CHOL = np.linalg.cholesky(P_UNIT)

    def __init__(self, w20, rng):
        self.w20 = float(w20)
        self.rng = rng
        self.xu = 0.0                 # longitudinal state
        self.zv = np.zeros(2)         # lateral states (z1, z2)
        self.zw = np.zeros(2)         # vertical states
        self.L, self.sigma = dryden_params(10.0, self.w20)

    def reset(self, h, V):
        """Start in the stationary distribution, so there is no warm-up transient."""
        self.L, self.sigma = dryden_params(h, self.w20)
        r = self.rng
        self.xu = self.sigma[0] * r.standard_normal()
        self.zv[:] = self.sigma[1] * (self.CHOL @ r.standard_normal(2))
        self.zw[:] = self.sigma[2] * (self.CHOL @ r.standard_normal(2))

    def output(self):
        return np.array([self.xu, self.S3 * self.zv[0] + (1 - self.S3) * self.zv[1],
                         self.S3 * self.zw[0] + (1 - self.S3) * self.zw[1]])

    def step(self, dt, h, V):
        """Advance by dt at height h with airspeed V; returns (u, v, w)."""
        if self.w20 <= 0.0:
            return np.zeros(3)
        self.L, self.sigma = dryden_params(h, self.w20)
        tau = self.L / max(V, 0.5)
        n = max(1, int(np.ceil(dt / (0.05 * tau.min()))))      # keep Euler-Maruyama well inside stability
        h_ = dt / n
        r = self.rng
        for _ in range(n):
            self.xu += -self.xu / tau[0] * h_ + self.sigma[0] * np.sqrt(2.0 * h_ / tau[0]) * r.standard_normal()
            for i, z in ((1, self.zv), (2, self.zw)):
                t = tau[i]
                z1 = z[0]
                z[0] += -z1 / t * h_ + self.sigma[i] * np.sqrt(h_ / t) * r.standard_normal()
                z[1] += (z1 - z[1]) / t * h_
        return self.output()


@dataclass
class WindField:
    """Mean wind + turbulence + gusts.  Call ``reset`` once, then ``step`` every simulation step.

    mean_speed      mean horizontal wind at ``ref_height`` [m/s]
    direction_deg   where the wind blows from (0 = north, 90 = east)
    turbulence      'none' | 'light' | 'moderate' | 'severe' | W20 in m/s
    vertical        constant vertical wind, e.g. a thermal updraft [m/s]
    """
    mean_speed: float = 0.0
    direction_deg: float = 0.0
    ref_height: float = 10.0
    roughness: float = 0.03
    turbulence: object = "none"
    vertical: float = 0.0
    gusts: list = field(default_factory=list)
    seed: int = 0

    def __post_init__(self):
        self.w20 = SEVERITY_W20[self.turbulence] if isinstance(self.turbulence, str) else float(self.turbulence)
        self.rng = np.random.default_rng(self.seed)
        self.dryden = DrydenTurbulence(self.w20, self.rng)
        self.turb_body = np.zeros(3)
        self.turb_world = np.zeros(3)

    # -- mean wind -------------------------------------------------------------------------------
    def mean(self, h):
        h = max(float(h), 2.0 * self.roughness)
        U = self.mean_speed * np.log(h / self.roughness) / np.log(self.ref_height / self.roughness)
        return wind_vector(max(U, 0.0), self.direction_deg, self.vertical)

    def gust(self, t):
        g = np.zeros(3)
        for gu in self.gusts:
            g += gu(t)
        return g

    # -- turbulence axes: longitudinal along the relative horizontal airflow ---------------------
    def _axes(self, rel):
        hor = np.array([rel[0], rel[1], 0.0])
        n = np.linalg.norm(hor)
        if n < 1e-3:
            hor = wind_vector(1.0, self.direction_deg)
            hor[2] = 0.0
            n = np.linalg.norm(hor)
        e_u = hor / n
        e_w = np.array([0.0, 0.0, 1.0])
        e_v = np.cross(e_w, e_u)
        return np.column_stack([e_u, e_v, e_w])

    def reset(self, position, velocity=np.zeros(3)):
        self.rng = np.random.default_rng(self.seed)
        self.dryden.rng = self.rng
        rel = self.mean(position[2]) - velocity
        self.dryden.reset(position[2], np.linalg.norm(rel))
        self.turb_world = self._axes(rel) @ self.dryden.output() if self.w20 else np.zeros(3)
        return self.value(0.0, position)

    def step(self, t, dt, position, velocity):
        """Advance the turbulence by dt along the vehicle's path; returns the wind vector at t + dt."""
        h = position[2]
        rel = self.mean(h) - velocity
        V = np.linalg.norm(rel)
        self.turb_world = self._axes(rel) @ self.dryden.step(dt, h, V)
        return self.value(t + dt, position)

    def value(self, t, position):
        return self.mean(position[2]) + self.turb_world + self.gust(t)
