"""
quadlab.py — train controllers in Python on the browser's own simulator.

The physics, the observation, the action mapping and the reward all live in JavaScript (src/). This module
starts `node train/env_server.mjs` and talks to it, so a policy trained here meets exactly the drone it will
fly in the browser. Dependencies: numpy (and torch only if you export a torch model).

    from quadlab import QuadLabVecEnv, export_policy

    env = QuadLabVecEnv(num_envs=16, level="wrench", obs="token", expert=True)
    obs, infos = env.reset(seed=0)                     # obs: (16, 20) float32
    obs, rew, term, trunc, infos = env.step(actions)   # actions: (16, 4) in [-1, 1]
    env.set_target(0, [1.0, 2.0, 1.5])                 # a target chosen by hand, for env 0
    env.close()

Ended episodes are reset automatically: the returned obs is then the first observation of the new episode,
and infos[i]["final_obs"] / ["final_info"] describe the step that ended it (Stable-Baselines3 convention).

Policy files: export_policy() writes the JSON that the browser loads (src/ml/nn.js). Layers use PyTorch's
layouts, so torch modules export with torch_lstm_layers() / torch_gru_layers() / torch_dense_layer().
"""
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(HERE, "env_server.mjs")


class QuadLabVecEnv:
    """A batch of environments (see src/ml/env.js for every option: level, obs, targets, target, randomize,
    expert, history, ml={...config.ml overrides...}, preset="Quad X 450" | 'Racer 5"' | "Heavy lift")."""

    def __init__(self, num_envs=8, node="node", **options):
        self.num_envs = num_envs
        self._proc = subprocess.Popen([node, SERVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        spec = self._call({"cmd": "make", "n": num_envs, "options": options})
        self.observation_size = spec["observation_size"]
        self.action_size = spec["action_size"]
        self.features = spec["features"]
        self.level = spec["level"]
        self.obs_mode = spec["obs"]
        self.ml = spec["ml"]
        self.preset = spec["preset"]
        self._spaces()

    def _spaces(self):
        try:  # gymnasium is optional: spaces only describe the shapes
            from gymnasium import spaces
            self.observation_space = spaces.Box(-np.inf, np.inf, (self.observation_size,), np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, (self.action_size,), np.float32)
        except ImportError:
            self.observation_space = self.action_space = None

    def _call(self, msg):
        self._proc.stdin.write(json.dumps(msg) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("env_server.mjs exited (is node >= 18 on the PATH?)")
        res = json.loads(line)
        if "error" in res:
            raise RuntimeError(res["error"])
        return res

    def reset(self, seed=0):
        r = self._call({"cmd": "reset", "seed": int(seed)})
        return np.asarray(r["obs"], np.float32), r["info"]

    def step(self, actions):
        a = np.clip(np.asarray(actions, np.float64).reshape(self.num_envs, self.action_size), -1, 1)
        r = self._call({"cmd": "step", "actions": a.tolist()})
        return (np.asarray(r["obs"], np.float32), np.asarray(r["reward"], np.float32),
                np.asarray(r["terminated"], bool), np.asarray(r["truncated"], bool), r["info"])

    def set_target(self, i, target):
        self._call({"cmd": "set_target", "i": int(i), "target": [float(x) for x in target]})

    def clear_target(self, i):
        self._call({"cmd": "clear_target", "i": int(i)})

    def close(self):
        if self._proc.poll() is None:
            try:
                self._call({"cmd": "close"})
            except (RuntimeError, BrokenPipeError):
                pass
            self._proc.wait(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---- policy files ---------------------------------------------------------------------------------------

def _np(x):
    """numpy array from a numpy array, a list or a torch tensor."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, np.float64)


def _clean(layer):
    return {k: (_np(v).tolist() if k.startswith(("weight", "bias")) else v) for k, v in layer.items()}


def export_policy(path, layers, level="wrench", obs="token", ml=None, name="policy"):
    """Write a policy the browser can load. layers: dicts like
         {"type": "lstm" | "gru", "weight_ih", "weight_hh", "bias_ih", "bias_hh"}
         {"type": "dense", "weight", "bias", "activation": "tanh" | "relu" | "none"}
    ml: the config.ml settings the policy was trained with (env.ml), so the browser builds the same inputs."""
    spec = {"format": "quadlab-policy/1", "name": name, "level": level, "obs": obs, "ml": ml or {},
            "layers": [_clean(layer) for layer in layers]}
    with open(path, "w") as f:
        json.dump(spec, f)
    return spec


def torch_lstm_layers(lstm):
    """Layers of a torch.nn.LSTM (batch_first or not, one direction)."""
    return [{"type": "lstm", **{k: getattr(lstm, f"{k}_l{i}") for k in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")}}
            for i in range(lstm.num_layers)]


def torch_gru_layers(gru):
    return [{"type": "gru", **{k: getattr(gru, f"{k}_l{i}") for k in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")}}
            for i in range(gru.num_layers)]


def torch_dense_layer(linear, activation="none"):
    return {"type": "dense", "weight": linear.weight, "bias": linear.bias, "activation": activation}


# ---- a numpy forward pass, written from PyTorch's equations (a reference for the JavaScript one) -------------

def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def rnn_forward(layers, seq, state=None):
    """Run the layers over seq (T, input). Returns (outputs (T, out), state). PyTorch gate orders:
    LSTM i, f, g, o; GRU r, z, n (n = tanh(W_in x + b_in + r * (W_hn h + b_hn)))."""
    xs = np.asarray(seq, np.float64)
    state = list(state) if state is not None else [None] * len(layers)
    for li, L in enumerate(layers):
        if L["type"] == "dense":
            y = xs @ _np(L["weight"]).T + _np(L["bias"])
            xs = {"tanh": np.tanh, "relu": lambda v: np.maximum(v, 0), "none": lambda v: v}[L.get("activation", "none")](y)
            continue
        Wi, Wh, bi, bh = (_np(L[k]) for k in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"))
        H = Wh.shape[1]
        h, c = state[li] if state[li] is not None else (np.zeros(H), np.zeros(H))
        out = []
        for x in xs:
            gi, gh = Wi @ x + bi, Wh @ h + bh
            if L["type"] == "lstm":
                i, f, g, o = _sig(gi[:H] + gh[:H]), _sig(gi[H:2*H] + gh[H:2*H]), np.tanh(gi[2*H:3*H] + gh[2*H:3*H]), _sig(gi[3*H:] + gh[3*H:])
                c = f * c + i * g
                h = o * np.tanh(c)
            else:
                r, z = _sig(gi[:H] + gh[:H]), _sig(gi[H:2*H] + gh[H:2*H])
                n = np.tanh(gi[2*H:] + r * gh[2*H:])
                h = (1 - z) * n + z * h
            out.append(h)
        state[li] = (h, c)
        xs = np.stack(out)
    return xs, state
