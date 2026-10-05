"""Untuned vs tuned cascaded PID on a figure-eight path in moderate turbulence with a strong gust.
->  docs/pid_comparison.png"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dronesim import (Atmosphere, CascadedPID, Gust, PathFollower, SensorModel, Simulation, WindField,  # noqa: E402
                      airframe, figure_eight, metrics)

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "pid_comparison.png")
path = figure_eight(center=(0, 0, 10), a=15, b=8)
af = airframe.quad_x()


def world():
    return dict(atmosphere=Atmosphere(temperature_c=32),
                wind=WindField(mean_speed=6, direction_deg=250, turbulence="moderate", seed=7,
                               gusts=[Gust(t0=14, duration=2.5, speed=8, from_deg=180)]),
                sensors=SensorModel.preset("good"))


logs = {}
for gains in ("untuned", "tuned"):
    sim = Simulation(af, **world())
    sim.reset(path[0])
    logs[gains] = sim.run(CascadedPID(af, gains), PathFollower(path, speed=4.0), duration=40, stop_when_done=True)
    m = metrics(logs[gains])
    print(f"{gains:8s} " + "  ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in m.items()))

plt.style.use("dark_background")
fig = plt.figure(figsize=(13, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1])
ax = fig.add_subplot(gs[0, :])
ax.plot(path[:, 0], path[:, 1], color="white", lw=1, ls="--", label="path")
cols = {"untuned": "#f778ba", "tuned": "#3fb9a0"}
for g, log in logs.items():
    m = metrics(log)
    ax.plot(log["p"][:, 0], log["p"][:, 1], color=cols[g], lw=1.6, label=f"{g} PID  (RMS {m['rmse_m']:.2f} m, max {m['max_err_m']:.2f} m)")
w = logs["tuned"]["wind"].mean(axis=0)
u = w[:2] / np.linalg.norm(w[:2])
ax.annotate("", xy=(0.06 + 0.06 * u[0], 0.85 + 0.1 * u[1]), xytext=(0.06, 0.85), xycoords="axes fraction",
            arrowprops=dict(color="#e3b341", width=2))
ax.text(0.02, 0.93, "mean wind", color="#e3b341", transform=ax.transAxes)
ax.set(aspect="equal", title="Figure-eight at 4 m/s · wind 6 m/s + moderate turbulence + 8 m/s gust at t = 14 s · 32 °C",
       xlabel="east [m]", ylabel="north [m]")
ax.legend(loc="lower right", fontsize=9)
ax2 = fig.add_subplot(gs[1, 0])
for g, log in logs.items():
    ax2.plot(log["t"], log["track_err"], color=cols[g], label=g)
ax2.axvspan(14, 16.5, color="white", alpha=0.08)
ax2.set(title="Distance from the path", xlabel="time [s]", ylabel="m")
ax2.legend()
ax3 = fig.add_subplot(gs[1, 1])
lw = logs["tuned"]
ax3.plot(lw["t"], np.linalg.norm(lw["wind"][:, :2], axis=1), color="#e3b341", lw=0.8, label="horizontal wind speed")
ax3.plot(lw["t"], np.degrees(np.linalg.norm(lw["euler"][:, :2], axis=1)), color="#79c0ff", lw=0.8, label="tilt [deg] (tuned)")
ax3.set(title="What the drone feels", xlabel="time [s]")
ax3.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT, dpi=110)
print("saved", os.path.abspath(OUT))
