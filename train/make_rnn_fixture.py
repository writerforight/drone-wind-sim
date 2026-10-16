"""Write tests/fixtures/rnn-reference.json: random LSTM / GRU policies and their outputs from the numpy forward pass
in quadlab.py (PyTorch's equations). tests/ml.test.mjs checks that the browser's forward pass (src/ml/nn.js)
gives the same numbers, so a network trained in PyTorch flies the same in the browser.

    python3 train/make_rnn_fixture.py
"""
import json
import os

import numpy as np

from quadlab import rnn_forward

rng = np.random.default_rng(42)
F, T = 20, 6


def rand(*shape):
    return rng.normal(0, 0.4, shape)


def rnn(kind, inp, H):
    g = 4 if kind == "lstm" else 3
    return {"type": kind, "weight_ih": rand(g * H, inp), "weight_hh": rand(g * H, H), "bias_ih": rand(g * H), "bias_hh": rand(g * H)}


cases = {
    "lstm2_dense_tanh": [rnn("lstm", F, 8), rnn("lstm", 8, 8), {"type": "dense", "weight": rand(4, 8), "bias": rand(4), "activation": "tanh"}],
    "gru_relu_dense": [rnn("gru", F, 6), {"type": "dense", "weight": rand(16, 6), "bias": rand(16), "activation": "relu"},
                       {"type": "dense", "weight": rand(4, 16), "bias": rand(4), "activation": "none"}],
}
seq = rng.uniform(-1, 1, (T, F))
out = {"tokens": seq.tolist(), "cases": {}}
for name, layers in cases.items():
    y, _ = rnn_forward(layers, seq)
    as_lists = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in L.items()} for L in layers]
    out["cases"][name] = {"layers": as_lists, "outputs": y.tolist()}

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "fixtures", "rnn-reference.json")
with open(path, "w") as f:
    json.dump(out, f)
print("wrote", os.path.normpath(path))
