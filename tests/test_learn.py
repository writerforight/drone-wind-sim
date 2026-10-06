"""Checks of the learned-loop pipeline.  Run:  python tests/test_learn.py"""
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dronesim import airframe  # noqa: E402
from learn.actuation import Actuator  # noqa: E402
from learn.loss import step_loss  # noqa: E402
from learn.train import build, load, load_config, train  # noqa: E402


def test_untrained_model_commands_nominal_hover():
    """Last layer starts at zero -> u = 0 -> exactly the nominal hover thrust, in both control modes."""
    for mode in ("motors", "thrust_rates"):
        cfg = load_config(overrides=[f"control={mode}", "airframe=quad_asymmetric"])
        af, act, model = build(cfg)
        h = model.initial_memory(1, torch.zeros(1))
        u, h2 = model(torch.randn(1, model.obs_dim), h)
        assert u.abs().max() == 0 and h2.abs().max() == 0
        T = act.thrusts(u, torch.zeros(1, 3))[0].detach().double().numpy()
        wrench = af.allocation() @ T
        assert abs(wrench[0] - af.mass * 9.80665) < 1e-4 and np.abs(wrench[1:]).max() < 1e-4, (mode, wrench)


def test_rate_mode_saturates_inside_motor_limits():
    act = Actuator(airframe.make("quad_x"), "thrust_rates")
    T = act.thrusts(torch.tensor([[1.0, 1.0, -1.0, 1.0]]), torch.zeros(1, 3))
    assert torch.all(T >= act.t_min - 1e-6) and torch.all(T <= act.t_max + 1e-6)


def test_loss_terms_switch_on_and_off():
    z = torch.zeros(2, 3)
    p = torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
    v = torch.ones(2, 3)
    total, terms = step_loss({"error": 1.0}, p, v, z, z, z, z, z, z, z)
    assert set(terms) == {"error"} and abs(float(total) - 12.5) < 1e-6          # mean of 25 and 0
    total, terms = step_loss({"error": 0.0, "error_rate": 2.0}, p, v, z, z, z, z, z, z, z)
    assert set(terms) == {"error", "error_rate"} and abs(float(total) - 6.0) < 1e-6


def test_short_training_runs_and_reloads():
    cfg = load_config(overrides=["train.iters=2", "train.batch=4", "train.episode_s=0.4", "train.chunk_s=0.2",
                                 "target.radius_end=0.5"])
    with tempfile.TemporaryDirectory() as d:
        model, rows = train(cfg, d, quiet=True)
        assert all(np.isfinite(r["loss"]) for r in rows)
        assert any(p.abs().sum() > 0 for p in model.out.parameters())          # it moved away from zero
        cfg2, af, act, model2 = load(os.path.join(d, "model.pt"))
        x = torch.randn(1, model.obs_dim)
        h = model.initial_memory(1, x)
        assert torch.allclose(model(x, h)[0], model2(x, h)[0])


if __name__ == "__main__":
    for f in (test_untrained_model_commands_nominal_hover, test_rate_mode_saturates_inside_motor_limits,
              test_loss_terms_switch_on_and_off, test_short_training_runs_and_reloads):
        f()
        print("ok  ", f.__name__)
