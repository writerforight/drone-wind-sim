"""The outdoor environment on its own: wind severities, height profile, turbulence spectrum, and how
temperature / altitude change the thrust a motor can make.   ->  docs/environment.png"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import welch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dronesim import Atmosphere, Gust, WindField, airframe  # noqa: E402
from dronesim.wind import DrydenTurbulence, dryden_params  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "environment.png")
plt.style.use("dark_background")
fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))

# 1) wind seen by a hovering drone at 10 m, mean 6 m/s from the west, three severities + a gust
dt, T = 0.01, 60.0
t = np.arange(0, T, dt)
p, v = np.array([0, 0, 10.0]), np.zeros(3)
for sev, c in (("light", "#79c0ff"), ("moderate", "#e3b341"), ("severe", "#f778ba")):
    w = WindField(mean_speed=6, direction_deg=270, turbulence=sev, seed=4,
                  gusts=[Gust(t0=40, duration=3, speed=6, from_deg=270)])
    w.reset(p, v)
    east = [w.step(tt, dt, p, v)[0] for tt in t]
    ax[0, 0].plot(t, east, color=c, lw=0.9, label=sev)
ax[0, 0].axvspan(40, 43, color="white", alpha=0.08)
ax[0, 0].text(40.2, ax[0, 0].get_ylim()[1] * 0.92, "1-cos gust", fontsize=9)
ax[0, 0].set(title="Wind along the mean direction at 10 m (mean 6 m/s)", xlabel="time [s]", ylabel="m/s")
ax[0, 0].legend(loc="lower left")

# 2) mean wind profile for different ground roughness
h = np.linspace(0.5, 60, 200)
for z0, lab in ((0.0002, "water z0=0.0002"), (0.03, "grass z0=0.03"), (0.5, "suburb z0=0.5")):
    w = WindField(mean_speed=6, direction_deg=270, roughness=z0)
    ax[0, 1].plot([w.mean(hh)[0] for hh in h], h, label=lab)
ax[0, 1].axhline(10, color="grey", lw=0.6, ls=":")
ax[0, 1].set(title="Mean wind vs height (log law, 6 m/s at 10 m)", xlabel="wind speed [m/s]", ylabel="height [m]")
ax[0, 1].legend()

# 3) turbulence spectrum vs Dryden theory
h0, V = 10.0, 6.0
d = DrydenTurbulence(30 * 0.514444, np.random.default_rng(1))
d.reset(h0, V)
x = np.array([d.step(dt, h0, V) for _ in range(300_000)])
L, s = dryden_params(h0, d.w20)
for k, (lab, c) in enumerate((("u (longitudinal)", "#79c0ff"), ("w (vertical)", "#3fb9a0"))):
    kk = 0 if k == 0 else 2
    f, P = welch(x[:, kk], fs=1 / dt, nperseg=2 ** 14)
    om = 2 * np.pi * f
    tau = L[kk] / V
    if kk == 0:
        th = 2 * s[kk] ** 2 * tau / (1 + (tau * om) ** 2)
    else:
        th = s[kk] ** 2 * tau * (1 + 3 * (tau * om) ** 2) / (1 + (tau * om) ** 2) ** 2
    ax[1, 0].loglog(f[1:], P[1:], color=c, lw=0.8, alpha=0.8, label=f"simulated {lab}")
    ax[1, 0].loglog(f[1:], 2 * th[1:], color="white", lw=1.2, ls="--")       # one-sided PSD in Hz
ax[1, 0].plot([], [], color="white", ls="--", label="Dryden theory")
ax[1, 0].set(title="Turbulence spectrum (moderate, h = 10 m, V = 6 m/s)", xlabel="frequency [Hz]",
             ylabel="PSD [(m/s)²/Hz]", xlim=(1e-3, 10))
ax[1, 0].legend(fontsize=8)

# 4) maximum thrust / weight vs temperature at three site altitudes
af = airframe.quad_x()
temps = np.linspace(-20, 45, 100)
for alt, c in ((0, "#79c0ff"), (1500, "#e3b341"), (3000, "#f778ba")):
    tw = []
    for tc in temps:
        # ISA-consistent: a site at altitude is colder on a standard day; here we vary temperature freely
        rho = float(Atmosphere(temperature_c=tc, ground_altitude=alt).density(0))
        tw.append(af.thrust_to_weight(rho))
    ax[1, 1].plot(temps, tw, color=c, label=f"site at {alt} m")
ax[1, 1].axhline(1, color="grey", lw=0.6, ls=":")
ax[1, 1].set(title="Quad-X: max thrust / weight (thrust ∝ air density)", xlabel="air temperature [°C]",
             ylabel="thrust / weight")
ax[1, 1].legend()

fig.tight_layout()
fig.savefig(OUT, dpi=110)
print("saved", os.path.abspath(OUT))
