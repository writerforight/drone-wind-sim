"""Train the learned control loop by backpropagation through the differentiable simulator, then fly it in
the full NumPy simulator against the tuned PID.

    [observation_t, memory_{t-1}]  --MLP-->  [motor command_t, memory_t]

A short run (~5 min on one core) so you can see the mechanism work; the full run is
``python -m learn.train`` with learn/configs/default.json.  Change anything in the config - control mode,
hidden size, layers, the loss terms on the error and its derivatives, targets, world randomisation - with
--set on the command line or a copy of the JSON file.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from learn.evaluate import evaluate  # noqa: E402
from learn.train import load_config, train  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")

cfg = load_config(overrides=[
    "control=thrust_rates",            # or "motors": the network drives every rotor directly (slower to learn)
    "train.iters=120",
    "target.curriculum_iters=100",
    "target.radius_end=2.0",
    "loss.error_rate=0.05",            # weight of the error's first derivative (0 = off)
    "loss.error_accel=0.0",            # weight of its second derivative
])
out = os.path.join(ROOT, "runs", "example_07")
train(cfg, out)
evaluate(os.path.join(out, "model.pt"), n=8, duration=10.0, plot=os.path.join(out, "evaluation.png"))
print(f"plots: {os.path.relpath(out)}/loss.png, evaluation.png")
