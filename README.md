# Quadrotor Lab

A quadrotor simulator that runs in the browser, built for **controller design and tuning**: a 6-DoF
rigid-body model integrated with fixed-step RK4, a motor model with lag and saturation, a mixer, and a
cascaded PID (position → attitude → body rate). You can fly it yourself in angle mode, or hold a
setpoint in the tuning lab, apply steps, change the gains live and read the step response next to
the analytic prediction. Plain JavaScript ES modules, no build step, no ML libraries.

**▶ Live demo: https://writerforight.github.io/drone-wind-sim/**

> Status: milestones M1–M3 of [SPEC.md](SPEC.md) are done (physics, motors, PID, 3D view, manual flight,
> live tuning). Sensors with noise and delay, wind and a robustness study (M4), the Gym-style ML
> interface (M5) and the interview documents with result plots (M6) are next — see [PROGRESS.md](PROGRESS.md).
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

*Free flight* is angle mode: the stick sets the tilt angle and the attitude loop holds it; the throttle is a
climb-rate command, so the centre stick holds the height. *Tuning lab* holds a setpoint: **Step z** / **Step x**
apply a 1 m step and report rise time, overshoot, 2 % settling time and steady-state error.

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
| `src/modes/` | each mode is one module in one list (free flight, tuning lab) | `{ start(app), reference(app, sticks) }` |
| `src/input/` | keyboard, Gamepad API, touch sticks → one stick state | `createInput().read()` |
| `src/ui/` | menu with presets and airframe design, live gain sliders, live plots | |
| `src/config.js` | **every** drone and controller parameter, three presets | |

The simulation is decoupled from rendering: each animation frame adds the elapsed real time to an
accumulator and spends it in whole controller periods, so results do not depend on the frame rate.

## Tests

```bash
node tests/run.mjs
```

19 tests, no dependencies, including: free fall and hover exact; **energy and angular momentum conserved
without drag and motors** (relative drift < 1e-8 over 10 s); mixer inverts the allocation exactly; **the
altitude step matches the analytic second-order response** (ω_n = √Kp, ζ = Kd / 2√Kp, within 2 cm); hover
recovery for every preset; step metrics against the first- and second-order formulas.

## Limitations (honest)

- Rigid body with lumped linear drag; no blade flapping, no ground effect, no aerodynamic interaction
  between rotors, no battery model. Thrust does not depend on the inflow (a vertical gust does not change it).
- The floor is a simple inelastic contact, only for take-off and landing.
- Until M4 the controller sees the true state (no sensor noise, bias or delay).
- Preset numbers are of the right order for each class of drone, not identified from a real airframe.

## License

MIT — see [LICENSE](LICENSE).
