"""Real drones with published parameters (PX4 Iris, PX4 X500, Crazyflie 2.X) in steady wind:
how far must each tilt to hold position, and where does it lose?   ->  docs/real_drones.png"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dronesim import (Atmosphere, CascadedPID, Hover, PathFollower, SensorModel, Simulation, WindField,  # noqa: E402
                      airframe, figure_eight, metrics)

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "real_drones.png")
DRONES = {"crazyflie": "#f778ba", "iris": "#79c0ff", "x500": "#3fb9a0"}
winds = np.arange(0, 19, 1.0)

plt.style.use("dark_background")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for name, c in DRONES.items():
    af = airframe.make(name)
    tilt, held = [], []
    for ws in winds:
        sim = Simulation(af, Atmosphere(), WindField(mean_speed=ws, direction_deg=270), SensorModel.ideal())
        sim.reset((0, 0, 10))
        log = sim.run(CascadedPID(af, "tuned"), Hover((0, 0, 10)), 15)
        ok = np.linalg.norm(log["p"][-1] - [0, 0, 10]) < 0.5
        tilt.append(np.degrees(np.abs(log["euler"][-1, :2]).max()) if ok else np.nan)
        held.append(ok)
    limit = winds[np.argmin(held)] if not all(held) else None
    lab = f"{af.name} ({af.mass:.3g} kg)" + (f" — lost at {limit:.0f} m/s" if limit is not None else "")
    ax[0].plot(winds, tilt, "o-", color=c, ms=4, label=lab)
    print(lab)
ax[0].axhline(40, color="grey", ls=":", lw=0.8)
ax[0].text(0.3, 41, "tilt limit of the controller", fontsize=8, color="grey")
ax[0].set(title="Tilt needed to hold position in steady wind", xlabel="wind speed at 10 m [m/s]", ylabel="tilt [deg]",
          ylim=(0, 45))
ax[0].legend(fontsize=8, loc="upper left")

path = figure_eight(center=(0, 0, 10), a=15, b=8)
levels = [(2, "light"), (4, "light"), (6, "moderate"), (9, "moderate")]
x = np.arange(len(levels))
for k, (name, c) in enumerate(DRONES.items()):
    af = airframe.make(name)
    rms = []
    for ws, turb in levels:
        sim = Simulation(af, Atmosphere(temperature_c=25),
                         WindField(mean_speed=ws, direction_deg=250, turbulence=turb, seed=3), SensorModel.preset("good"))
        sim.reset(path[0])
        rms.append(metrics(sim.run(CascadedPID(af, "tuned"), PathFollower(path, speed=4), 45, stop_when_done=True))["rmse_m"])
    ax[1].bar(x + (k - 1) * 0.27, np.minimum(rms, 5), width=0.25, color=c, label=af.name)
    for xi, r in zip(x, rms):
        if r > 5:
            ax[1].text(xi + (k - 1) * 0.27, 5.05, "lost", ha="center", fontsize=8, color=c)
ax[1].set_xticks(x, [f"{w} m/s\n{t}" for w, t in levels])
ax[1].set(title="Figure-eight at 4 m/s, tuned PID: RMS path error", ylabel="m", ylim=(0, 5.6))
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT, dpi=110)
print("saved", os.path.abspath(OUT))
