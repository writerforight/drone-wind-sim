# train/ — learning controllers on the browser's simulator

The simulator, the observation, the action mapping and the reward live in JavaScript (`src/`). Python only
learns: `quadlab.py` starts `node train/env_server.mjs` and steps a batch of environments through it, so a
policy is trained on exactly the drone it later flies in the browser — there is no second physics to keep in
sync. Measured here: one environment step (8 ms of flight, 2 controller ticks, 4 RK4 steps) costs ~17 µs in
Node (≈ 58 000 steps/s); through the JSON bridge a batch of 64 environments gives ≈ 19 000 steps/s, 16 give
≈ 11 500 (10 M steps ≈ 9 min). Faster if needed: several server processes, or a binary protocol.

```
            Python (numpy / torch)                         Node (the browser's own modules)
  ┌──────────────────────────────────┐   JSON lines   ┌──────────────────────────────────────────┐
  │ QuadLabVecEnv.reset / step       │ ─────────────▶ │ env_server.mjs → N × createEnv (ml/env)  │
  │ DAgger / BC / PPO / your method  │ ◀───────────── │   agent: features · actions · reward     │
  │ export_policy → policies/*.json  │                │   loop · motors · RK4 physics            │
  └──────────────────────────────────┘                └──────────────────────────────────────────┘
                     │ policy JSON
                     ▼
     browser: Target mode → "Load a policy" → ml/nn.js (LSTM / GRU / dense, by hand) → same agent code
```

## The interface (what a controller sees and does)

| | |
|---|---|
| **Target** | random points (`targets: 'random'`), a fixed one, a function `(t, rng) → [x, y, z]` (a trajectory), or set by hand: `env.set_target(i, [x, y, z])` here, the Target mode in the browser |
| **Observation** | per step a token of 20 numbers: target − position, velocity, attitude (2 columns of R), body rates, **previous action**, height. `obs="token"` gives the newest token (a recurrent model keeps its own state); `obs="window"` gives the last `history` tokens (40 × 8 ms), oldest first — the past positions and past targets as a sequence |
| **Action** | 4 numbers in [−1, 1] at a level: `wrench` (thrust + body torques — the policy decides the torques), `rates` (thrust + body-rate references, a rate PID turns them into torque), `motors` (four rotor thrusts, no mixer) |
| **Reward** | alive − distance − speed − body rate − action² − Δaction², minus a penalty on a crash (`config.ml.reward`) |
| **Randomisation** | per episode: mass, inertia, motor strength and lag, drag ±, constant wind (`config.ml.randomize`); true values in `info["params"]` (privileged information, never in the observation) |
| **Expert** | `expert=True` adds `info["expert"]`: the cascaded PID's action at the same level — labels for BC / DAgger, or a base for residual RL |

### Adaptation, two kinds

1. **In context** (no weight change while flying): the window holds action → response pairs; trained over
   randomised drones, a sequence model learns to infer *which* drone it flies (heavier, weaker motors, wind) and
   to act for it. This is what an LSTM gives you here, and what to try first.
2. **Online learning** (weights change while flying): a policy object with `adapt(transition)` gets
   `{obs, action, reward, next, crash}` once per policy step in the browser (`src/ml/policy-controller.js`),
   so it can update itself during the flight.

## Scripts

| | needs | status |
|---|---|---|
| `python3 train/test_quadlab.py` | numpy, node | tested: shapes, seeds, auto-reset, hand-set targets, export → browser forward pass |
| `python3 train/fit_linear_bc.py` | numpy | run: a linear policy by DAgger from the PID → `policies/linear-bc.json` (the bundled one) |
| `python3 train/train_lstm.py` | torch | **not run here** (PyTorch could not be installed in the CI container); LSTM over the window, DAgger from the PID, exported for the browser |
| `python3 train/make_rnn_fixture.py` | numpy | regenerates `tests/fixtures/rnn-reference.json` (LSTM / GRU reference outputs) |

Results of `fit_linear_bc.py` (16 envs, 6 DAgger iterations, ~14 s): the linear policy alone flies 16 randomised
episodes without a crash (reward/step −0.61, PID −0.51) and reaches typed targets within 2 cm. Under a
payload or wind it was not trained for it keeps an offset (no memory, no integral) — the gap a sequence model
should close.

## Next: reinforcement learning

`QuadLabVecEnv` follows the vector-env conventions (auto-reset, `final_obs` in the infos), so PPO / SAC loops
plug in directly; for Stable-Baselines3 wrap it in a `VecEnv` subclass. Start RL from the DAgger weights, keep
the action level fixed per experiment, and compare against the PID on the same seeds.
