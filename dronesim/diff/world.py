"""A batch of randomised worlds for training: payload, air (temperature, site altitude) and wind.

Wind = constant mean (random speed and direction, horizontal) + turbulence as a first-order
Gauss-Markov process per axis (a simple stand-in for Dryden, cheap enough to pre-sample for a whole
batch).  The full wind model (log profile, Dryden, gusts) stays in the NumPy simulator, which is where a
trained controller is evaluated.
"""
import numpy as np
import torch

from ..atmosphere import G0, LAPSE, P_SL, R_AIR, T_SL
from .dynamics import DiffDynamics, DiffState


def _uniform(gen, rng, B, **kw):
    lo, hi = rng
    return lo + (hi - lo) * torch.rand(B, generator=gen, **kw)


class WorldBatch:
    """B random worlds for one airframe.  ``cfg`` is the "world" part of a training config."""

    def __init__(self, airframe, B, cfg, gen, dtype=torch.float32, device="cpu"):
        kw = dict(dtype=dtype, device=device)
        self.B, self.gen, self.kw = B, gen, kw
        self.mass_scale = _uniform(gen, cfg.get("mass_scale", (1.0, 1.0)), B, **kw)
        temp_c = _uniform(gen, cfg.get("temperature_c", (15.0, 15.0)), B, **kw)
        site = _uniform(gen, cfg.get("site_altitude", (0.0, 0.0)), B, **kw)
        T0 = temp_c + 273.15
        p0 = P_SL * ((T_SL - LAPSE * site) / T_SL) ** (G0 / (R_AIR * LAPSE))
        self.dyn = DiffDynamics(airframe, self.mass_scale, T0, p0, dtype, device)
        speed = _uniform(gen, cfg.get("wind_speed", (0.0, 0.0)), B, **kw)
        ang = _uniform(gen, (0.0, 2 * np.pi), B, **kw)
        self.wind_mean = torch.stack([speed * torch.cos(ang), speed * torch.sin(ang), torch.zeros_like(speed)], -1)
        self.turb_sigma = _uniform(gen, cfg.get("turbulence_sigma", (0.0, 0.0)), B, **kw)[:, None]
        self.turb_tau = float(cfg.get("turbulence_tau", 1.0))
        self.turb = self.turb_sigma * torch.randn(B, 3, generator=gen, **kw)

    def wind_step(self, dt):
        """Advance the turbulence by dt and return the wind (B, 3).  No gradient flows into the wind."""
        a = np.exp(-dt / self.turb_tau)
        self.turb = a * self.turb + np.sqrt(1 - a * a) * self.turb_sigma * torch.randn(self.B, 3, generator=self.gen,
                                                                                     **self.kw)
        return self.wind_mean + self.turb


def allocation(dyn):
    """4 x n allocation matrix (thrust, roll, pitch, yaw rows), as Airframe.allocation()."""
    return torch.stack([torch.ones_like(dyn.rx), dyn.ry, -dyn.rx, -dyn.spin * dyn.kQ / dyn.kT], 0)


def hover_state(dyn, p, v=None, att_std=0.0, gen=None):
    """Start state: level (plus optional random tilt), rotors at the speed that carries the actual mass."""
    B = p.shape[0]
    kw = dict(dtype=p.dtype, device=p.device)
    q = torch.zeros(B, 4, **kw)
    q[:, 0] = 1.0
    if att_std > 0:
        q[:, 1:3] = 0.5 * att_std * torch.randn(B, 2, generator=gen, **kw)
        q = q / q.norm(dim=-1, keepdim=True)
    rho = dyn.density(p[:, 2])
    share = torch.linalg.pinv(allocation(dyn))[:, 0]                                # (n,) thrust split for hover
    T = share[None] * (dyn.mass * G0)[:, None]
    n = torch.sqrt(T / (dyn.kT * rho[:, None]))
    return DiffState(p, torch.zeros_like(p) if v is None else v, q, torch.zeros(B, 3, **kw), n)
