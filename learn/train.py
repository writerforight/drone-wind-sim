"""Train the learned loop by backpropagation through the differentiable simulator.

    python -m learn.train                                   # learn/configs/default.json
    python -m learn.train --config my.json --out runs/test
    python -m learn.train --set train.iters=300 loss.error_rate=0.1 model.hidden_state=64

Every iteration: draw B new worlds (payload, air, wind) and targets, fly them from hover, and after every
``chunk_s`` seconds backpropagate the loss through the physics and the memory (truncated BPTT), clip the
gradient and take an Adam step.  Writes runs/<name>/{config.json, model.pt, log.csv, loss.png}.
"""
import argparse
import copy
import csv
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dronesim import airframe as af_mod  # noqa: E402
from dronesim.diff import WorldBatch  # noqa: E402
from learn.actuation import Actuator  # noqa: E402
from learn.loop import OBS_BASE, Loop  # noqa: E402
from learn.model import build_model  # noqa: E402
from learn.targets import make_targets, radius_at  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "configs", "default.json")


def load_config(path=DEFAULT_CONFIG, overrides=()):
    with open(path) as f:
        cfg = json.load(f)
    for item in overrides:                       # "section.key=value", value parsed as JSON if possible
        key, val = item.split("=", 1)
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            pass
        node = cfg
        *parents, last = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = val
    return cfg


def build(cfg, device="cpu"):
    """Airframe, actuator and model from a config (also used to load a trained model)."""
    af = af_mod.make(cfg["airframe"])
    act = Actuator(af, cfg["control"], device=device)
    model = build_model(cfg["model"], OBS_BASE + act.act_dim, act.act_dim).to(device)
    return af, act, model


def new_episode(cfg, af, act, model, B, radius, gen, device="cpu"):
    world = WorldBatch(af, B, cfg["world"], gen, device=device)
    start = torch.tensor([0.0, 0.0, 10.0], device=device).repeat(B, 1)
    sim = cfg["sim"]
    steps = int(round(cfg["train"]["episode_s"] / sim["ctrl_dt"]))
    refs = make_targets(cfg["target"], start, steps, sim["ctrl_dt"], radius, gen)
    return Loop(model, act, world, start, refs, sim, gen), steps


def train(cfg, out_dir, device="cpu", quiet=False):
    tc = cfg["train"]
    torch.manual_seed(tc.get("seed", 0))
    gen = torch.Generator(device=device).manual_seed(tc.get("seed", 0))
    af, act, model = build(cfg, device)
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    if not quiet:
        print(model.describe())
        print(f"airframe {af.name}, control '{cfg['control']}', target '{cfg['target']['mode']}', "
              f"loss {cfg['loss']}")
    chunk = int(round(tc["chunk_s"] / cfg["sim"]["ctrl_dt"]))
    rows, best = [], math.inf
    t0 = time.time()
    for it in range(tc["iters"]):
        radius = radius_at(cfg["target"], it)
        loop, steps = new_episode(cfg, af, act, model, tc["batch"], radius, gen, device)
        ep_loss, ep_terms, n_chunks = 0.0, {}, 0
        while loop.k < steps:
            loss, terms = loop.run(min(chunk, steps - loop.k), cfg["loss"])
            if not torch.isfinite(loss):
                print(f"iter {it}: non-finite loss, chunk skipped")
                break
            opt.zero_grad()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            loop.detach()
            ep_loss += float(loss.detach())
            n_chunks += 1
            for k, v in terms.items():
                ep_terms[k] = ep_terms.get(k, 0.0) + v
        n_chunks = max(n_chunks, 1)
        final_err = float((loop.state.p - loop.refs["p"][:, loop.k - 1]).norm(dim=-1).mean())
        row = {"iter": it, "loss": ep_loss / n_chunks, "final_error_m": final_err, "radius_m": radius,
               "grad_norm": float(gnorm), "time_s": time.time() - t0,
               **{f"term_{k}": v / n_chunks for k, v in ep_terms.items()}}
        rows.append(row)
        if radius >= cfg["target"].get("radius_end", radius) and row["loss"] < best:
            best = row["loss"]
            save(model, cfg, os.path.join(out_dir, "model_best.pt"))
        if not quiet and (it % tc.get("log_every", 10) == 0 or it == tc["iters"] - 1):
            print(f"iter {it:5d}  loss {row['loss']:8.4f}  final error {final_err:6.3f} m  "
                  f"target radius {radius:4.1f} m  |grad| {row['grad_norm']:7.3f}  {row['time_s']:6.0f} s")
    save(model, cfg, os.path.join(out_dir, "model.pt"))
    write_log(rows, out_dir)
    return model, rows


def save(model, cfg, path):
    torch.save({"config": cfg, "state_dict": model.state_dict()}, path)


def load(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    af, act, model = build(ck["config"], device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return ck["config"], af, act, model


def write_log(rows, out_dir):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(out_dir, "log.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    it = [r["iter"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].semilogy(it, [r["loss"] for r in rows], lw=0.8)
    ax[0].set(xlabel="iteration", ylabel="loss", title="training loss")
    ax[1].plot(it, [r["final_error_m"] for r in rows], lw=0.8, label="mean error at episode end")
    ax[1].plot(it, [r["radius_m"] for r in rows], "--", lw=0.8, label="target radius (curriculum)")
    ax[1].set(xlabel="iteration", ylabel="m", title="distance to target")
    ax[1].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss.png"), dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--out", default=None, help="output folder (default runs/<config name>)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config entries")
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    cfg = load_config(a.config, a.set)
    name = os.path.splitext(os.path.basename(a.config))[0]
    out = a.out or os.path.join(HERE, "..", "runs", name)
    train(copy.deepcopy(cfg), out)
    print(f"saved to {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
