"""Checks of the Python ↔ JavaScript bridge (numpy only):   python3 train/test_quadlab.py"""
import json
import os
import subprocess
import tempfile

import numpy as np

from quadlab import QuadLabVecEnv, export_policy, rnn_forward

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def test_shapes_and_seeds():
    with QuadLabVecEnv(3, obs="window", history=8) as env:
        assert env.observation_size == 8 * 20 and env.action_size == 4 and len(env.features) == 20
        a, _ = env.reset(seed=5)
        b, _ = env.reset(seed=5)
        c, _ = env.reset(seed=6)
        assert a.shape == (3, 160) and a.dtype == np.float32
        assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_autoreset_and_final_obs():
    with QuadLabVecEnv(2, ml={"episode": {"seconds": 0.08}}) as env:   # 10 policy steps per episode
        env.reset(seed=1)
        for k in range(10):
            obs, rew, term, trunc, infos = env.step(np.zeros((2, 4)))
        assert trunc.all() and not term.any(), (term, trunc)
        assert "final_obs" in infos[0] and len(infos[0]["final_obs"]) == 20
        assert abs(infos[0]["t"]) < 1e-9, "obs after an ended episode must be the new episode's first"


def test_expert_and_hand_set_target():
    with QuadLabVecEnv(2, expert=True, randomize=False) as env:
        obs, infos = env.reset(seed=3)
        env.set_target(1, [0.5, 0.5, 2.0])
        for _ in range(750):                                            # 6 s
            obs, rew, term, trunc, infos = env.step(np.array([i["expert"] for i in infos]))
        assert not term.any()
        assert infos[1]["target"] == [0.5, 0.5, 2.0]
        assert np.linalg.norm(np.subtract(infos[1]["position"], [0.5, 0.5, 2.0])) < 0.1, infos[1]["position"]


def test_exported_policy_runs_the_same_in_javascript():
    rng = np.random.default_rng(1)
    H = 5
    layers = [{"type": "lstm", "weight_ih": rng.normal(0, .3, (4 * H, 20)), "weight_hh": rng.normal(0, .3, (4 * H, H)),
               "bias_ih": rng.normal(0, .3, 4 * H), "bias_hh": rng.normal(0, .3, 4 * H)},
              {"type": "dense", "weight": rng.normal(0, .3, (4, H)), "bias": rng.normal(0, .3, 4), "activation": "tanh"}]
    seq = rng.uniform(-1, 1, (4, 20))
    expected, _ = rnn_forward(layers, seq)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "p.json")
        export_policy(path, layers, name="t")
        js = ("import('./src/ml/nn.js').then(({loadPolicy}) => { const fs = require('fs');"
              f"const p = loadPolicy(fs.readFileSync({json.dumps(path)}, 'utf8'));"
              f"console.log(JSON.stringify({json.dumps(seq.tolist())}.map((x) => p.act(x)))); }})")
        out = subprocess.run(["node", "-e", js], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    assert np.allclose(json.loads(out), np.clip(expected, -1, 1), atol=1e-12)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok   ", name)
    print("all python bridge tests passed")
