# Quadrotor Lab

A quadrotor simulator that runs in the browser, used as a testbed for model-free control. It contains
a classical cascaded PID and an MLP controller that learns online, without a plant model, while it flies.

**Live demo: https://writerforight.github.io/drone-wind-sim/**

## Motivation

I want to work on model-free control: a controller that is blind to the physics and to the plant model, so
that the same approach can be put into any system without first deriving and identifying that system's
equations.

The idea rests on the MLP as a universal function approximator: if a network can represent the control law,
the question is whether it can find it from experience alone. This repository tests that on a quadrotor, with
an MLP that learns online and gets no plant model.

The goal is a systematic comparison: the learned controller against classical controllers (PID, model-based
control), and the controllers against each other, under the same plant, the same references and the same
random seeds. I could not find much systematic academic literature that compares them this way, so I want
to build the evidence myself. The simulator is the testbed; real hardware demonstrators would be the next step.

## What is in this repository

**The testbed** (plain JavaScript ES modules, no build step, no ML libraries):

- 6-DoF rigid-body physics (position, velocity, quaternion attitude, body rates), gravity, lumped linear drag,
  fixed-step RK4 at 500 Hz.
- Motor model: first-order rotor lag, speed limits, thrust k_T ω² and reaction torque k_Q ω²; an X mixer that
  inverts the allocation matrix, with per-rotor saturation.
- Cascaded PID (position → attitude → body rate) at 250 Hz with zero-order hold, gravity feed-forward and
  anti-windup; an angle mode for manual flight.
- Sensors with noise, bias, delay and their own sample rates (levels `ideal`, `realistic`, `poor`).
- *Free flight*: fly it yourself in angle mode (keyboard, gamepad or touch sticks).
- *Tuning lab*: hold a setpoint, apply 1 m steps in x or z, change the gains live and read rise time,
  overshoot, 2 % settling time and steady-state error next to the analytic prediction for z.
- Three airframe presets (Quad X 450, Racer 5", Heavy lift) and an airframe design form.

**The MLP lab** (*Tuning lab · MLP* in the menu):

- A target pin that you move in 3D: W/A/S/D horizontally as seen from the camera, ↑/↓ up and down
  (gamepad and touch sticks work too); a desired speed slider.
- The main drone is flown **only** by an MLP that learns online while it flies.
- A see-through ghost drone flown by the cascaded PID goes to the same pin at the same desired speed. It is
  there only for comparison: it has its own simulation and takes no part in the learning. Its reference is a
  "carrot" that walks toward the pin at the desired speed, with that speed as a velocity feed-forward.
- A network panel: number of layers (1–4), width (2–64), activation (identity, ReLU, tanh, sigmoid, GELU,
  sin), optimizer (Adam or SGD), learning rate, noise σ, which input groups are used, and live statistics
  (loss, steps, updates, crashes and the reason for the last one, flight time, gradient norm).

## How the MLP controller works

Code: `src/ml/mlp.js` (network, backprop, optimizers), `src/ml/mlp-controller.js` (inputs, fixed output
layer), `src/ml/online-learner.js` (the update rule).

**Inputs** (13 by default, each group can be switched off; fixed scales so the values are O(1)):

| Group | Size | Scaling |
|---|---|---|
| target − position (world frame) | 3 | / 5 m, clipped to ±3 |
| velocity (world frame) | 3 | / 5 m/s |
| desired speed | 1 | / 5 m/s |
| "down" seen from the body, R(q)ᵀ [0, 0, −1] (the tilt, without angles) | 3 | unit vector |
| body rates | 3 | / 5 rad/s |

The inputs come from the measurement the simulation hands to every controller (in the app the sensors are
at the `ideal` level, so this is the true state).

**Network**: `[inputs, hidden…, 4]`, hidden layers with the chosen activation, a linear last layer. The
default is 13 → 16 → 16 → 4 with tanh. The last layer is scaled by 0.1 at initialisation, so the first
outputs are small.

**What is not model-free.** The four outputs o = [o_T, o_x, o_y, o_z] do not drive the motors directly.
They pass through two fixed stages that are not trained:

1. a fixed output layer that uses the airframe's mass and inertia:
   T = m g (1 + o_T), τ = J ⊙ α_max ⊙ [o_x, o_y, o_z] with α_max = [30, 30, 10] rad/s².
   So o = 0 means hover thrust and no torque;
2. the fixed mixer A⁻¹, which turns [T, τ] into four rotor thrusts using the rotor geometry and the
   thrust/torque coefficients.

The controller therefore knows m, g, J and the allocation matrix. It does not know the equations of motion,
the motor lag, the drag or the sensor model, and it has no teacher.

**Update rule** (every controller period, 250 Hz):

1. Exploration noise ε ~ N(0, σ² I) is added to the four network outputs.
2. The physics advances one controller period with that action.
3. The loss of the new state is measured:
   L_t = |target − p| + |v − v_des|, where v_des points to the target with magnitude
   min(speed, distance · 1/s), so the drone is asked to slow down close to the target.
4. The only learning signal is the change against the previous step:
   g = (L_t − L_{t−1}) · ε / σ².
   This is a perturbation estimate of ∂L/∂o: for a fixed state, E[ε (∇L · ε)] / σ² = ∇L, and L_{t−1} acts
   as the baseline.
5. g is backpropagated through the MLP, the parameter gradient is clipped to norm 5, and the optimizer
   takes one step.

**Crashes**: touching the floor (z ≤ 2 cm), tilting past 90°, getting more than 40 m from the target, or
body rates above 30 rad/s count as a crash. The drone is put back at (0, 0, 1.5) m, level and still, and the
crash is counted. The weights are kept, so learning continues across flights. *Learning off* freezes the
weights and lets the current network keep flying; *New weights* starts from a new random initialisation.

## Current results

The current rule does **not** learn to fly the quadrotor yet.

One measurement, run in Node (not part of the test suite): 120 s of simulated flight per setting, one seed,
pin at (3, 2, 2) m, desired speed 1.5 m/s, network 13 → 16 → 16 → 4 with tanh, Adam.

| σ | Learning rate | Crashes in 120 s | Flight time per crash, first → last | Steps where L fell |
|---|---|---|---|---|
| 0.1 | 1e-3 | 343 | 1.44 s → 0.22 s | 46.5 % |
| 0.05 | 1e-4 | 56 | about 2 s, no trend | 33 % |
| 0.2 | 1e-3 | 166 | 1.6 s → 0.4 s | 14 % |

At the higher learning rate the flights get shorter over time: learning makes it worse. At the low rate
nothing changes. In none of the settings does the loss fall monotonically. The app uses σ = 0.05 and a
learning rate of 1e-4 by default, the setting with the longest flights.

What is verified: on a static problem without dynamics (a network whose output itself is scored), the same
update code moves the output toward the target and reduces its distance below 25 % of the initial value.
The rule is a working gradient estimate when the effect of ε is immediate.

**Hypothesis (not a finding).** The noise added at step t reaches the loss only after the motor lag (30 ms
for the Quad X 450, about 7 controller periods) and after the chain attitude → acceleration → velocity →
position. Within one 4 ms step its effect on L is tiny compared with the change of L caused by the drone's
own motion, so a one-step credit assignment has a very low signal-to-noise ratio. The next step is to
test a multi-step eligibility trace or a windowed baseline, which would credit a perturbation with the loss
change over the time its effect needs to arrive.

## Controls

| | Keyboard | Gamepad | Touch |
|---|---|---|---|
| Free flight: tilt (pitch / roll) | W / S, A / D | right stick | right on-screen stick |
| Free flight: climb / descend | ↑ / ↓ | left stick up / down | left stick up / down |
| Free flight: yaw | ← / → | left stick left / right | left stick left / right |
| MLP lab: move the pin | W / A / S / D, ↑ / ↓ | right stick, left stick up / down | the same on-screen sticks |
| Finer input | Shift | — | — |
| Restart · camera · menu | R · C · Esc | — | buttons in the top bar |

In the tuning lab and the MLP lab, drag the view to turn the camera and use the wheel to zoom.

## Architecture

```
            reference ─▶(+)─▶ controller ─▶ mixer ─▶ motors ─▶ physics ─▶ state ─▶ render
                         ▲(−)                                               │
                         └──────────── measurement (sensors) ◀──────────────┘
```

| Module | What it does | Interface |
|---|---|---|
| `src/physics/` | Newton–Euler rigid body: position, velocity, quaternion attitude, body rates; gravity, linear drag, wind / external force input; fixed-step RK4 | `step(state, input, params)` |
| `src/motor/` | first-order rotor lag (exact discretisation), speed limits, thrust k_T ω² and reaction torque k_Q ω²; X mixer = inverse of the allocation matrix, with saturation | `stepMotors`, `rotorWrench`, `createMixer().mix({thrust, torque})` |
| `src/sensors/` | gyro (noise, constant bias), accelerometer (specific force), attitude estimate with noise, position / velocity at their own rate, one common delay; seeded | `createSensors(SENSOR_LEVELS[level], airframe, seed)(state, t)` |
| `src/controller/` | cascaded PID: position PID (gravity feed-forward, anti-windup) → thrust and tilt → attitude P → rate PID scaled by the inertia; angle mode | `control(measurement, reference, params)` |
| `src/ml/` | MLP with backprop, SGD, Adam and gradient clipping; the MLP controller (inputs, fixed output layer); the online learner (update rule, loss, crash detection) | `new MLP(dims, acts)`, `createMlpController(...)`, `createOnlineLearner(ctl, settings)` |
| `src/loop/` | connects the blocks with negative feedback; physics at 500 Hz, controller at 250 Hz with zero-order hold; floor contact | `createSimulation(cfg, {controller, measure})` |
| `src/render/` | Three.js view: drone, ghost drone, target pin; reads the state, never steps it | `createView(container, cfg).update(sim, {pin, ghost})` |
| `src/modes/` | each mode is one module in one list: free flight, tuning lab, MLP lab (`mlp-lab.js`: pin, ghost PID, flying and learning) | `{ start(app), reference(app, sticks), afterTick?(app) }` |
| `src/input/` | keyboard, Gamepad API, touch sticks → one stick state | `createInput().read()` |
| `src/ui/` | menu with presets and airframe design, live gain sliders, MLP lab panel, live plots | |
| `src/config.js` | **every** drone and controller parameter, three presets, sensor levels | |

The simulation is decoupled from rendering: each animation frame adds the elapsed real time to an
accumulator and spends it in whole controller periods, so results do not depend on the frame rate.

## Tests

```bash
node tests/run.mjs
```

45 tests, no dependencies:

- **Physics**: free fall exact; hover at thrust = weight; energy and angular momentum conserved without drag
  and motors (relative drift < 1e-8 over 10 s); terminal speed m g / c_d.
- **Motors and mixer**: lag reaches 1 − 1/e after τ; saturation; the mixer inverts the allocation exactly;
  hover speeds; open-loop hover.
- **Control**: the altitude step matches the analytic second-order response (ω_n = √Kp, ζ = Kd / 2√Kp,
  within 2 cm); hover recovery for every preset; 30 s hover without drift; angle mode; step metrics
  against first- and second-order formulas.
- **Sensors**: exact delay; gyro noise and bias statistics; position sample-and-hold; the accelerometer reads
  +g at hover; hover with realistic and poor sensors stays within a few centimetres.
- **MLP**: activation derivatives and backprop (weights, biases, inputs, every activation) against central
  finite differences (worst relative error < 1e-6); gradient accumulation; SGD and Adam fit a regression;
  gradient clipping.
- **MLP controller and learner**: the 13 inputs in order and scale; the fixed output layer; network shape;
  noise and forward cache; the loss and the crash reasons; the update rule learns a static target; in the
  closed loop an update happens every step and crashes are counted.
- **MLP lab**: camera-relative pin moves and limits; the carrot's speed; the ghost PID reaches a far pin
  with a cruise speed within 15 % of the desired one.

The browser checks used during development (headless Firefox at 1400 and 500 px wide) are not in this
repository.

## Limitations

- Rigid body with lumped linear drag; no blade flapping, no ground effect, no aerodynamic interaction
  between rotors, no battery model. Thrust does not depend on the inflow.
- The floor is a simple inelastic contact, only for take-off and landing.
- The sensor model exists and is tested, but the app runs at the `ideal` level (there is no switch in the
  UI yet), and the attitude estimate is noise added to the true attitude, not the output of a simulated
  filter.
- The MLP's loss is computed from the true state, and the controller uses m, g, J and the mixer (see
  above), so it is not model-free in the strict sense.
- The MLP results above are one seed and one target; there is no multi-seed comparison yet.
- Preset numbers are of the right order for each class of drone, not identified from a real airframe.

## Roadmap

- Metrics in the MLP lab: the loss per step for the MLP and the ghost PID side by side, the fraction of
  steps in which the loss fell, crash counts; then a multi-seed comparison of the controllers under the
  same plant, references and seeds.
- Testing the credit-assignment hypothesis above (eligibility trace, windowed baseline).
- Wind, gusts and a robustness study under randomised mass, delay and wind. A first version is parked on the
  local branch `wip-wind` (not pushed) because of an open conflict between the anti-windup and wind rejection.
- A Gym-style interface (`reset()`, `step(action)`) and a data recorder, so imitation learning and RL can be
  plugged in.

The earlier Python simulator of this repository is kept on the
[`legacy-python`](https://github.com/writerforight/drone-wind-sim/tree/legacy-python) branch.

## License

MIT — see [LICENSE](LICENSE).
