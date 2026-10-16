# Quadrotor Lab

A quadrotor simulator that runs in the browser, built for **controller design and tuning**: a 6-DoF
rigid-body model integrated with fixed-step RK4, a motor model with lag and saturation, a mixer, and a
cascaded PID (position → attitude → body rate). You can fly it yourself in angle mode, or hold a
setpoint in the tuning lab, apply steps, change the gains live and read the step response next to
the analytic prediction. Plain JavaScript ES modules, no build step, no ML libraries.

**▶ Live demo: https://writerforight.github.io/drone-wind-sim/**

> Status: milestones M1–M3 of [SPEC.md](SPEC.md) are done (physics, motors, PID, 3D view, manual flight,
> live tuning) and the ML interface (M5, extended: a training bridge to Python, sequence policies, a Target
> mode) are done. Sensors with noise and delay and a robustness study (M4) and the interview documents with
> result plots (M6) are next — see [PROGRESS.md](PROGRESS.md).
> The earlier Python simulator of this repository is kept on the
> [`legacy-python`](https://github.com/writerforight/drone-wind-sim/tree/legacy-python) branch.

## Controls

| | Keyboard | Gamepad | Touch |
|---|---|---|---|
| Tilt (pitch / roll) | W / S, A / D | right stick | right on-screen stick |
| Climb / descend | ↑ / ↓ | left stick up / down | left stick up / down |
| Yaw | ← / → | left stick left / right | left stick left / right |
| Finer input | Shift | — | — |
| Restart · camera · menu | R · C · Esc | — | buttons in the top bar |
| Target (Target mode) | type x, y, z + Go · double-click the floor · Random | — | panel · double-tap the floor |
| Record policy inputs + actions | ● Rec (JSON download when stopped) | — | ● button |

*Free flight* is angle mode: the stick sets the tilt angle and the attitude loop holds it; the throttle is a
climb-rate command, so the centre stick holds the height. *Tuning lab* holds a setpoint: **Step z** / **Step x**
apply a 1 m step and report rise time, overshoot, 2 % settling time and steady-state error. *Target* sends the
drone to a point with the PID or a learned policy (bundled: a linear policy trained by DAgger; or load your own
`.json`), lets you add a payload and wind the controller does not know about, and shows the sequence a policy
reads as a heat map.

## Learned controllers

The environment is built for controllers that see the recent past and decide the next action — e.g. an LSTM
that is given past positions, past targets and its own past actions and outputs the torques:

| | |
|---|---|
| target | random points, a trajectory, or set by hand (`env.setTarget`, Target mode) |
| observation | 20 numbers per step (target − position, velocity, attitude, body rates, previous action, height); the newest token or the last 40 (0.32 s) as a sequence |
| action | 4 numbers in [−1, 1] at a chosen level: thrust + **body torques**, thrust + body rates, or four motors |
| training | `train/quadlab.py` drives the same JavaScript simulator from Python (no second physics); domain randomisation of mass, inertia, motors, drag and wind; PID expert labels for imitation; reward for RL |
| deployment | the policy file loads in the browser; the forward pass (LSTM / GRU / dense) is written by hand and matches the PyTorch equations to 1e-12; a test shows training env and browser give **bit-identical trajectories** |
| adaptation | in context through the sequence (trained over randomised drones), or online through an `adapt(transition)` hook |

See [train/README.md](train/README.md) for the workflow and the measured numbers.

## Architecture

```
            reference ─▶(+)─▶ controller ─▶ mixer ─▶ motors ─▶ physics ─▶ state ─▶ render
                         ▲(−)                                               │
                         └──────────── measurement (sensors) ◀──────────────┘
```

| Module | What it does | Interface |
|---|---|---|
| `src/physics/` | Newton–Euler rigid body: position, velocity, quaternion attitude, body rates; gravity, linear drag, wind / external force; fixed-step RK4 | `step(state, input, params)` |
| `src/motor/` | first-order rotor lag (exact discretisation), speed limits, thrust k_T ω² and reaction torque k_Q ω²; X mixer = inverse of the allocation matrix, with saturation | `stepMotors`, `rotorWrench`, `createMixer().mix({thrust, torque})` |
| `src/controller/` | cascaded PID: position PID (gravity feed-forward, anti-windup) → thrust and tilt → attitude P → rate PID scaled by the inertia; angle mode | `control(measurement, reference, params)` |
| `src/loop/` | connects the blocks with negative feedback; physics at 500 Hz, controller at 250 Hz with zero-order hold; floor contact | `createSimulation(cfg, {controller, measure})` |
| `src/render/` | Three.js view; reads the state, never steps it | `createView(container, cfg).update(sim)` |
| `src/modes/` | each mode is one module in one list (free flight, tuning lab, target) | `{ start(app), reference(app, sticks) }` |
| `src/input/` | keyboard, Gamepad API, touch sticks → one stick state | `createInput().read()` |
| `src/ml/` | observation token and history window, action levels, reward (one agent shared by training and browser); Gym-style env; LSTM / GRU / dense forward pass; policy controller with an adaptation hook; recorder | `createEnv(cfg, opts).reset() / .step(a)`, `createPolicyController(cfg, loadPolicy(json))` |
| `train/` | Node env server + Python client (numpy), policy export, linear DAgger baseline, LSTM DAgger script | `QuadLabVecEnv(n, **opts)` |
| `src/ui/` | menu with presets and airframe design, live gain sliders, live plots, policy-input heat map | |
| `src/config.js` | **every** drone and controller parameter, three presets | |

The simulation is decoupled from rendering: each animation frame adds the elapsed real time to an
accumulator and spends it in whole controller periods, so results do not depend on the frame rate.

## Tests

```bash
node tests/run.mjs              # 35 tests, no dependencies
python3 train/test_quadlab.py   # 4 tests of the Python bridge (numpy)
```

Including: free fall and hover exact; **energy and angular momentum conserved
without drag and motors** (relative drift < 1e-8 over 10 s); mixer inverts the allocation exactly; **the
altitude step matches the analytic second-order response** (ω_n = √Kp, ζ = Kd / 2√Kp, within 2 cm); hover
recovery for every preset; step metrics against the first- and second-order formulas; the browser's LSTM / GRU
forward pass against a numpy implementation of PyTorch's equations; the same policy flying bit-identically in
the training environment and as a browser controller; the bundled linear policy reaching typed targets.

## Limitations (honest)

- Rigid body with lumped linear drag; no blade flapping, no ground effect, no aerodynamic interaction
  between rotors, no battery model. Thrust does not depend on the inflow (a vertical gust does not change it).
- The floor is a simple inelastic contact, only for take-off and landing.
- Until M4 the controller sees the true state (no sensor noise, bias or delay).
- Preset numbers are of the right order for each class of drone, not identified from a real airframe.
- The PID's anti-windup (integrate only within 0.3 m) leaves a steady offset under a payload above ≈ 0.17 kg
  on the 450 frame (+0.3 kg → 58 cm low) — see PROGRESS.md.
- The bundled learned policy is a linear imitation of the PID: it proves the pipeline, it does not beat the
  PID. `train/train_lstm.py` (PyTorch) was not run in the container this was built in.

## License

MIT — see [LICENSE](LICENSE).
