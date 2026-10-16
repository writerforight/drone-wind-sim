"""
A first learned controller, end to end, with numpy only: imitate the cascaded PID with a LINEAR policy
a = W token + b, trained with DAgger, and export it for the browser (policies/linear-bc.json).

It is deliberately the simplest learner: it proves the pipeline (environment → data → fit → policy file →
browser) and gives a baseline that an LSTM has to beat. A linear map cannot turn a world-frame target error
into a body-frame tilt for every heading, so it is trained and used with a near-zero start heading.

DAgger (Ross et al., 2011): fly with a mix of expert and learner, label every visited state with the
expert's action, refit on everything collected so far. Plain behaviour cloning only sees the expert's own
states and drifts once the learner's small errors take it elsewhere.

    python3 train/fit_linear_bc.py [--iters 6] [--envs 16] [--out policies/linear-bc.json]
"""
import argparse
import os

import numpy as np

from quadlab import QuadLabVecEnv, export_policy

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def rollout(env, policy, beta, steps, rng, seed):
    """Fly `steps` steps per env; act with the expert with probability beta, else the learner. Returns the visited
    observations, their expert labels, and episode statistics of the actions actually taken."""
    obs, infos = env.reset(seed=seed)
    X, Y, crashes, episodes, rewards = [], [], 0, 0, []
    for _ in range(steps):
        expert = np.array([i["expert"] for i in infos])
        X.append(obs)
        Y.append(expert)
        learner = policy(obs) if policy is not None else expert
        use_expert = rng.random(env.num_envs) < beta
        act = np.where(use_expert[:, None], expert, learner)
        obs, rew, term, trunc, infos = env.step(act)
        rewards.append(rew.mean())
        crashes += int(term.sum())
        episodes += int((term | trunc).sum())
    return np.concatenate(X), np.concatenate(Y), {"crashes": crashes, "episodes": episodes, "reward": float(np.mean(rewards))}


def fit(X, Y, ridge=1e-3):
    """Least squares with a small ridge: [W b] = argmin |[X 1][W b]ᵀ − Y|² + ridge |W|²."""
    A = np.hstack([X, np.ones((len(X), 1))])
    reg = ridge * np.eye(A.shape[1])
    reg[-1, -1] = 0
    Wb = np.linalg.solve(A.T @ A + reg, A.T @ Y)
    return Wb[:-1].T, Wb[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1250, help="policy steps per env per iteration (1250 = 10 s)")
    ap.add_argument("--out", default=os.path.join(ROOT, "policies", "linear-bc.json"))
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    ml = {"episode": {"startYaw": 0}, "randomize": {"mass": 0.1, "inertia": 0.1, "kThrust": 0.05, "tau": 0.2, "linearDrag": 0.3, "wind": 1}}
    with QuadLabVecEnv(args.envs, level="wrench", obs="token", expert=True, ml=ml) as env:
        X, Y, stats = rollout(env, None, 1.0, args.steps, rng, seed=1000)
        print(f"expert            : reward/step {stats['reward']:+.3f}, crashes {stats['crashes']}/{stats['episodes']} episodes")
        W = b = None
        for it in range(1, args.iters + 1):
            W, b = fit(X, Y)
            beta = 0.5 ** it                     # hand over to the learner step by step
            Xi, Yi, stats = rollout(env, lambda o: np.clip(o @ W.T + b, -1, 1), beta, args.steps, rng, seed=1000 + 100 * it)
            X, Y = np.concatenate([X, Xi]), np.concatenate([Y, Yi])
            print(f"DAgger iter {it} β={beta:.3f}: reward/step {stats['reward']:+.3f}, crashes {stats['crashes']}/{stats['episodes']}, data {len(X)}")
        W, b = fit(X, Y)
        _, _, stats = rollout(env, lambda o: np.clip(o @ W.T + b, -1, 1), 0.0, args.steps, rng, seed=99999)
        print(f"learner alone     : reward/step {stats['reward']:+.3f}, crashes {stats['crashes']}/{stats['episodes']} episodes")
        export_policy(args.out, [{"type": "dense", "weight": W, "bias": b, "activation": "none"}],
                      level=env.level, obs="token", ml=env.ml, name="Linear (DAgger from PID)")
    print("wrote", os.path.normpath(args.out))


if __name__ == "__main__":
    main()
