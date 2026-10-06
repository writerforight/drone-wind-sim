"""Export a trained loop for the browser game (web/learned.js).

    python -m learn.export_web runs/iris_rates/model.pt --name iris_rates

Writes web/models/<name>.json (weights + the actuation constants + a self-check: random inputs with the
outputs PyTorch computes for them, which web/learned.js recomputes when it loads the file) and adds it to
web/models/index.json, which maps an airframe to its model.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from learn.loop import observation  # noqa: E402
from learn.train import load  # noqa: E402

WEB_MODELS = os.path.join(os.path.dirname(__file__), "..", "web", "models")


def _arr(t):
    return np.round(t.detach().double().numpy(), 9).tolist()


def export(path, name, note=""):
    cfg, af, act, model = load(path)
    model = model.double()
    layers, act_name = [], None
    for m in model.body:
        if isinstance(m, nn.Linear):
            layers.append({"W": _arr(m.weight), "b": _arr(m.bias)})
        else:
            act_name = type(m).__name__.lower()
    out = {"name": name, "airframe": cfg["airframe"], "control": cfg["control"], "ctrl_dt": cfg["sim"]["ctrl_dt"],
           "note": note, "obs_dim": model.obs_dim, "act_dim": model.act_dim, "memory": model.H,
           "activation": act_name, "memory_activation": type(model.mem_act).__name__.lower(),
           "layers": layers, "out": {"W": _arr(model.out.weight), "b": _arr(model.out.bias)},
           "actuation": {k: _arr(getattr(act, k)) for k in
                         ("t_hover", "margin", "k", "t_min", "t_max", "Bp", "J", "rate_max", "rate_kp")}}
    out["actuation"]["mass"] = act.mass
    # self-check: a few random states -> the outputs PyTorch gives (checked again in the browser)
    g = torch.Generator().manual_seed(0)
    checks = []
    from dronesim.diff.dynamics import DiffState
    for _ in range(3):
        q = torch.randn(1, 4, generator=g, dtype=torch.float64)
        s = DiffState(10 * torch.randn(1, 3, generator=g, dtype=torch.float64),
                      3 * torch.randn(1, 3, generator=g, dtype=torch.float64), q / q.norm(),
                      torch.randn(1, 3, generator=g, dtype=torch.float64), None)
        ref_p, ref_v = torch.randn(1, 3, generator=g, dtype=torch.float64), torch.randn(1, 3, generator=g, dtype=torch.float64)
        u_prev = torch.rand(1, model.act_dim, generator=g, dtype=torch.float64) * 2 - 1
        h = torch.rand(1, model.H, generator=g, dtype=torch.float64) * 2 - 1
        obs = observation(s, ref_p, ref_v, u_prev, None, None)
        with torch.no_grad():
            u, h2 = model(obs, h)
            act64 = type(act)(af, cfg["control"], dtype=torch.float64)
            n = act64.speeds(u, s.w)
        checks.append({"p": _arr(s.p[0]), "v": _arr(s.v[0]), "q": _arr(s.q[0]), "w": _arr(s.w[0]), "ref_p": _arr(ref_p[0]),
                       "ref_v": _arr(ref_v[0]), "u_prev": _arr(u_prev[0]), "h": _arr(h[0]), "u": _arr(u[0]),
                       "h_next": _arr(h2[0]), "n": _arr(n[0])})
    out["checks"] = checks
    os.makedirs(WEB_MODELS, exist_ok=True)
    with open(os.path.join(WEB_MODELS, f"{name}.json"), "w") as f:
        json.dump(out, f, separators=(",", ":"))
    idx_path = os.path.join(WEB_MODELS, "index.json")
    idx = json.load(open(idx_path)) if os.path.exists(idx_path) else {}
    idx[cfg["airframe"]] = {"file": f"{name}.json", "control": cfg["control"], "note": note}
    with open(idx_path, "w") as f:
        json.dump(idx, f, indent=2)
    size = os.path.getsize(os.path.join(WEB_MODELS, f"{name}.json")) / 1024
    print(f"wrote web/models/{name}.json ({size:.0f} kB) for airframe {cfg['airframe']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--name", required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    export(a.model, a.name, a.note)
