"""Template: plug your own model (e.g. a neural network you design and train) into the environment
and compare it with the PID baselines on the same unseen worlds.

Replace ``my_policy`` with your model.  It receives the observation vector (see dronesim/env.py) and
returns an action in [-1, 1]^k; with control='residual' the action is an extra acceleration on top of
the PID (k = 3).  Train it however you like - the environment follows the Gymnasium API.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dronesim import DroneEnv  # noqa: E402

ENV_KW = dict(task="hover", episode_time=10.0, airframes=["quad_x", "hexa_x", "octo_x", "quad_asymmetric"],
              wind_speed=(0.0, 9.0))


def my_policy(obs):
    """Your model goes here.  obs[0:3] = position error / 5, obs[3:6] = velocity / 5, ...
    This placeholder is a hand-written proportional-derivative correction, just to show the interface."""
    return np.tanh(4.0 * obs[0:3] - 0.6 * obs[3:6])


def evaluate(policy, seeds, **kw):
    rms, crashes = [], 0
    for seed in seeds:
        env = DroneEnv(**dict(ENV_KW, **kw))
        obs, info = env.reset(seed=int(seed))
        errs = []
        while True:
            action = policy(obs) if policy is not None else np.zeros(env.act_dim)
            obs, reward, terminated, truncated, info = env.step(action)
            errs.append(info["error"])
            if terminated or truncated:
                break
        e = np.array(errs[len(errs) // 4:])
        rms.append(np.sqrt(np.mean(e ** 2)))
        crashes += info["crashed"]
    return np.array(rms), crashes


if __name__ == "__main__":
    seeds = range(10_000, 10_012)                    # fixed test worlds: same for every controller
    for name, policy, kw in (("untuned PID", None, dict(control="pid", gains="untuned")),
                             ("untuned PID + my_policy", my_policy, dict(control="residual", gains="untuned")),
                             ("tuned PID", None, dict(control="pid", gains="tuned"))):
        rms, crashes = evaluate(policy, seeds, **kw)
        print(f"{name:26s} hover RMS error: mean {rms.mean():.2f} m, median {np.median(rms):.2f} m, crashes {crashes}")
