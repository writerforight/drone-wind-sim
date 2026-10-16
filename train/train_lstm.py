"""
Train a sequence controller (LSTM) that is given the recent past — past positions, past targets, past actions
and their effect — and outputs the next action. Then export it for the browser (Target mode → load policy).

    pip install torch numpy
    python3 train/train_lstm.py --level wrench --hidden 32 --iters 8 --out policies/lstm-dagger.json

What the model sees (obs = 'window'): the last `history` tokens (config.ml.history, 40 × 8 ms = 0.32 s), each
    [target − position, velocity, attitude (2 columns of R), body rates, previous action, height]
so the window holds both the history of targets relative to the drone and input → response pairs. Trained
over randomised drones (mass, inertia, motor strength and lag, drag, wind), the LSTM has to infer from that
window which drone it is flying — adaptation in context, without changing its weights while it flies.

How it learns here: DAgger from the cascaded PID (the expert). That gives a working sequence controller
quickly; it cannot be better than the PID. To go beyond it, start reinforcement learning (e.g. PPO with a
recurrent policy) from these weights, with the reward in config.ml.reward — the environment is the same.

NOTE: this script was written against PyTorch's documented API but has not been run in the repository's CI
container (PyTorch could not be installed there). The pieces it relies on are tested: the environment and
bridge (train/test_quadlab.py) and the browser forward pass of exported LSTMs (tests/ml.test.mjs).
"""
import argparse
import os

import numpy as np
import torch
from torch import nn

from quadlab import QuadLabVecEnv, export_policy, torch_dense_layer, torch_lstm_layers

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


class WindowLSTM(nn.Module):
    """LSTM over the history window (from a zero state, every step), last hidden state → tanh head."""

    def __init__(self, features, hidden, actions=4, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(features, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, actions)

    def forward(self, window):                  # window: (batch, history, features)
        out, _ = self.lstm(window)
        return torch.tanh(self.head(out[:, -1]))


def collect(env, model, beta, steps, rng, seed, every=2):
    """Fly with the expert (probability beta per env and step) or the model; label every visited window with the
    expert's action. Keeps every `every`-th step (neighbouring windows are nearly the same)."""
    H, F = env.ml["history"], len(env.features)
    obs, infos = env.reset(seed=seed)
    X, Y, crashes, episodes = [], [], 0, 0
    for k in range(steps):
        expert = np.array([i["expert"] for i in infos], np.float32)
        windows = obs.reshape(env.num_envs, H, F)
        if k % every == 0:
            X.append(windows)
            Y.append(expert)
        if model is None or beta >= 1:
            act = expert
        else:
            with torch.no_grad():
                learner = model(torch.from_numpy(windows)).numpy()
            act = np.where((rng.random(env.num_envs) < beta)[:, None], expert, learner)
        obs, rew, term, trunc, infos = env.step(act)
        crashes += int(term.sum())
        episodes += int((term | trunc).sum())
    return np.concatenate(X), np.concatenate(Y), crashes, episodes


def train(model, X, Y, epochs, batch=512, lr=2e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        total = 0.0
        for i in range(0, len(Xt), batch):
            idx = perm[i:i + batch]
            loss = nn.functional.mse_loss(model(Xt[idx]), Yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
    return total / len(Xt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="wrench", choices=["wrench", "rates", "motors"])
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1250, help="policy steps per env per iteration (1250 = 10 s)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=300_000, help="keep at most this many windows (oldest dropped)")
    ap.add_argument("--out", default=os.path.join(ROOT, "policies", "lstm-dagger.json"))
    args = ap.parse_args()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    with QuadLabVecEnv(args.envs, level=args.level, obs="window", expert=True) as env:
        model = WindowLSTM(len(env.features), args.hidden)
        X, Y, c, e = collect(env, None, 1.0, args.steps, rng, seed=1000)
        print(f"expert: crashes {c}/{e} episodes, {len(X)} windows")
        for it in range(1, args.iters + 1):
            loss = train(model, X, Y, args.epochs)
            beta = 0.5 ** it
            Xi, Yi, c, e = collect(env, model, beta, args.steps, rng, seed=1000 + 100 * it)
            X, Y = np.concatenate([X, Xi])[-args.max_samples:], np.concatenate([Y, Yi])[-args.max_samples:]
            print(f"iter {it}: loss {loss:.5f}, β={beta:.3f}, crashes {c}/{e} episodes, data {len(X)}")
        train(model, X, Y, args.epochs)
        _, _, c, e = collect(env, model, 0.0, args.steps, rng, seed=99999)
        print(f"LSTM alone: crashes {c}/{e} episodes")
        export_policy(args.out, torch_lstm_layers(model.lstm) + [torch_dense_layer(model.head, "tanh")],
                      level=env.level, obs="window", ml=env.ml, name=f"LSTM {args.hidden} (DAgger from PID, {env.level})")
    print("wrote", os.path.normpath(args.out))


if __name__ == "__main__":
    main()
