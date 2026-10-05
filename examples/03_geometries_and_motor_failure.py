"""Same controller, five airframes, same wind - then a motor fails mid-flight.
->  docs/geometries.png"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dronesim import (Atmosphere, CascadedPID, PathFollower, SensorModel, Simulation, WindField,  # noqa: E402
                      airframe, figure_eight, metrics)

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "geometries.png")
path = figure_eight(center=(0, 0, 10), a=15, b=8)
COL = {"quad_x": "#79c0ff", "quad_plus": "#3fb9a0", "hexa_x": "#e3b341", "octo_x": "#bc8cff", "quad_asymmetric": "#f778ba"}


def fly(name, failures=(), fail_at=10.0, turbulence="moderate", duration=40):
    af = airframe.make(name)
    sim = Simulation(af, Atmosphere(temperature_c=25),
                     WindField(mean_speed=5, direction_deg=250, turbulence=turbulence, seed=11),
                     SensorModel.preset("good"))
    sim.reset(path[0])

    def on_step(s):
        if failures and abs(s.t - fail_at) < s.ctrl_dt / 2 + 1e-9:
            for i, h in failures:
                s.set_motor_health(i, h)
    return sim.run(CascadedPID(af, "tuned"), PathFollower(path, speed=4.0), duration, stop_when_done=True,
                   on_step=on_step)


plt.style.use("dark_background")
fig, ax = plt.subplots(1, 3, figsize=(16, 5.4), gridspec_kw=dict(width_ratios=[1.35, 0.9, 1.25]))

# 1) top view of every airframe: rotor positions, spin direction, centre of mass
for k, name in enumerate(airframe.PRESETS):
    af = airframe.make(name)
    ox = 1.0 * k
    for r in af.rotors:
        x, y = ox + r.position[1] * -1.0, r.position[0]          # draw with forward = up
        ax[0].plot([ox, x], [0, y], color="#8b949e", lw=1.5)
        ax[0].add_patch(plt.Circle((x, y), 0.07, fill=False, lw=1.3,
                                   color="#e3b341" if r.spin > 0 else "#bc8cff"))
    ax[0].plot(ox, 0, "o", color=COL[name], ms=7)
    ax[0].text(ox, -0.68, f"{af.name}\n{af.mass:.1f} kg", ha="center", fontsize=8.5)
ax[0].annotate("", xy=(-0.42, 0.45), xytext=(-0.42, 0.2), arrowprops=dict(color="white", width=1))
ax[0].text(-0.47, 0.5, "front", fontsize=8)
ax[0].plot([], [], "o", mfc="none", color="#e3b341", label="spins counter-clockwise")
ax[0].plot([], [], "o", mfc="none", color="#bc8cff", label="spins clockwise")
ax[0].plot([], [], "o", color="white", label="centre of mass")
ax[0].set(aspect="equal", xlim=(-0.55, 4.5), ylim=(-0.85, 0.75), title="Airframes (top view, same scale)")
ax[0].axis("off")
ax[0].legend(fontsize=8, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02), frameon=False)

# 2) same tuned PID on the same windy figure-eight: tracking error and energy
names, rmses, energy = [], [], []
for name in airframe.PRESETS:
    log = fly(name)
    m = metrics(log)
    print(f"{name:16s} rmse {m['rmse_m']:.2f}  max {m['max_err_m']:.2f}  energy {m['energy_Wh']:.2f} Wh")
    names.append(log["airframe"])
    rmses.append(m["rmse_m"])
    energy.append(m["energy_Wh"])
y = np.arange(len(names))
ax[1].barh(y + 0.2, rmses, height=0.38, color=[COL[n] for n in airframe.PRESETS], label="RMS path error [m]")
ax[1].barh(y - 0.2, np.array(energy) / 10, height=0.38, color="#8b949e", label="energy [Wh / 10]")
ax[1].set_yticks(y, names)
ax[1].set(title="Figure-eight, wind 5 m/s + moderate turbulence")
ax[1].legend(fontsize=8, loc="lower right")

# 3) motor failure mid-flight
cases = [("quad_x", [(0, 0.6)], "quad: motor 1 at 60 %", "#79c0ff"),
         ("quad_x", [(0, 0.0)], "quad: motor 1 fails", "#f778ba"),
         ("hexa_x", [(0, 0.0)], "hexa: motor 1 fails", "#e3b341"),
         ("octo_x", [(0, 0.0)], "octo: motor 1 fails", "#bc8cff")]
for name, fail, label, c in cases:
    log = fly(name, fail, turbulence="light")
    end = "  ✗ crash" if log["crashed"] else "  ✓"
    ax[2].plot(log["t"], log["track_err"], color=c, label=label + end)
    if log["crashed"]:
        ax[2].plot(log["t"][-1], min(log["track_err"][-1], 5.8), "x", color=c, ms=10, mew=2)
    print(f"{label:24s} crashed {log['crashed']}  rmse {metrics(log)['rmse_m']:.2f}")
ax[2].set(title="Motor failure at t = 10 s (the controller is not told)", xlabel="time [s]",
          ylabel="distance from path [m]", ylim=(0, 6))
ax[2].axvline(10, color="white", lw=0.8, ls=":")
ax[2].text(10.3, 5.5, "failure", fontsize=9)
ax[2].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.92))
fig.tight_layout()
fig.savefig(OUT, dpi=110)
print("saved", os.path.abspath(OUT))
