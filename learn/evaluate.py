"""Fly a trained loop in the full NumPy simulator (log wind profile, Dryden turbulence, gusts, sensor
model, ground) - a world it was not trained in - and compare it with the PID on the same flights.

    python -m learn.evaluate runs/default/model.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dronesim import Atmosphere, CascadedPID, Gust, Hover, SensorModel, Simulation, WindField, metrics  # noqa: E402
from dronesim.diff.dynamics import DiffState  # noqa: E402
from learn.loop import observation  # noqa: E402
from learn.train import load  # noqa: E402


class LearnedController:
    """Wraps a trained model as a dronesim controller: reset() / update(estimate, reference, dt)."""

    def __init__(self, model, actuator, ctrl_dt):
        self.model, self.act, self.ctrl_dt = model, actuator, ctrl_dt
        self.reset()

    def reset(self):
        self.dtype = next(self.model.parameters()).dtype
        self.h = self.model.initial_memory(1, torch.zeros(1, dtype=self.dtype))
        self.u = torch.zeros(1, self.act.act_dim, dtype=self.dtype)

    @torch.no_grad()
    def update(self, est, ref, dt):
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=float), dtype=self.dtype)[None]   # noqa: E731
        s = DiffState(t(est.p), t(est.v), t(est.q), t(est.w), None)
        obs = observation(s, t(ref["p"]), t(ref.get("v", np.zeros(3))), self.u, None, None)
        self.u, self.h = self.model(obs, self.h)
        return self.act.speeds(self.u, s.w)[0].double().numpy()


def test_flights(n=8, seed=100):
    """Fixed test worlds: hover / step to a point under random wind, turbulence, gust, air and payload."""
    rng = np.random.default_rng(seed)
    flights = []
    for i in range(n):
        start = np.array([0.0, 0.0, 10.0])
        target = start + (0 if i % 2 == 0 else 1) * np.r_[rng.uniform(-4, 4, 2), rng.uniform(-1.5, 1.5)]
        flights.append(dict(
            start=start, target=target, mass_scale=rng.uniform(0.9, 1.2),
            atm=Atmosphere(temperature_c=rng.uniform(-5, 35), ground_altitude=rng.uniform(0, 1000)),
            wind=dict(mean_speed=rng.uniform(0, 6), direction_deg=rng.uniform(0, 360),
                      turbulence=["none", "light", "moderate"][i % 3], seed=int(rng.integers(1 << 30)),
                      gusts=[Gust(t0=rng.uniform(3, 6), duration=2.0, speed=rng.uniform(2, 5),
                                  from_deg=rng.uniform(0, 360))])))
    return flights


def fly(af_factory, controller, flight, duration, ctrl_dt):
    af = af_factory()
    af.mass *= flight["mass_scale"]
    af.inertia = af.inertia * flight["mass_scale"]
    af.inertia_inv = np.linalg.inv(af.inertia)
    af.ang_damping = af.ang_damping * flight["mass_scale"]
    sim = Simulation(af, flight["atm"], WindField(**flight["wind"]), SensorModel.preset("good"), ctrl_dt=ctrl_dt)
    sim.reset(flight["start"])
    return sim.run(controller, Hover(flight["target"]), duration)


def evaluate(path, n=8, duration=10.0, plot=None):
    cfg, af, act, model = load(path)
    from dronesim import airframe as af_mod
    factory = lambda: af_mod.make(cfg["airframe"])   # noqa: E731
    ctrl_dt = cfg["sim"]["ctrl_dt"]
    results = {"learned loop": [], "tuned PID": []}
    logs = {"learned loop": [], "tuned PID": []}
    for flight in test_flights(n):
        for name, ctrl in (("learned loop", LearnedController(model, act, ctrl_dt)),
                           ("tuned PID", CascadedPID(factory(), "tuned"))):
            log = fly(factory, ctrl, flight, duration, ctrl_dt)
            results[name].append(metrics(log, skip=3.0))
            logs[name].append(log)
    print(f"{n} test flights of {duration:.0f} s in the full NumPy simulator (error measured after 3 s):")
    for name, ms in results.items():
        rmse = [m["rmse_m"] for m in ms]
        print(f"  {name:13s} RMS error mean {np.nanmean(rmse):6.2f} m   median {np.nanmedian(rmse):6.2f} m   "
              f"crashes {sum(m['crashed'] for m in ms)}/{n}")
    if plot:
        _plot(logs, plot)
    return results


def _plot(logs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(logs["learned loop"])
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3.2 * ((n + 1) // 2), 5.6), sharey=True)
    for ax, i in zip(axes.flat, range(n)):
        for name, style in (("learned loop", "-"), ("tuned PID", "--")):
            log = logs[name][i]
            err = np.linalg.norm(log["p"] - log["p_ref"], axis=1)
            ax.plot(log["t"], err, style, lw=1, label=name)
        ax.set_title(f"flight {i}", fontsize=9)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes.flat[0].set_ylabel("distance to target [m]")
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--flights", type=int, default=8)
    ap.add_argument("--duration", type=float, default=10.0)
    a = ap.parse_args()
    evaluate(a.model, a.flights, a.duration, plot=os.path.join(os.path.dirname(a.model), "evaluation.png"))
