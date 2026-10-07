# PROGRESS — quadrotor-lab

Log kept by the work loop (see SPEC.md). Newest entry at the bottom of "Log".

## Rules for the loop

- One unit of work per iteration, toward the next unfinished milestone: implement, run the tests, fix failures, commit (clear message), add a log entry here.
- **Review gate:** SPEC.md says "work in milestones and stop after each for review". When a milestone is complete, mark it `[x] — awaiting review`, write its summary below, and **stop the loop**. The next milestone starts only after the owner says so (and unchecks "awaiting review").
- Do not push, force-push, or touch files outside this folder. Do not add features beyond SPEC.md.
- If the same test fails twice in a row, describe the problem here under "Open problems" and stop. Do not guess.

## Milestones

- [x] M1 — physics + motor + PID hover with plots
- [x] M2 — menu + manual flight with keyboard / gamepad / touch (gamepad and touch not tested on real hardware)
- [x] M3 — live gain tuning UI
- [ ] M4 — sensors, delay, wind + robustness demo
- [ ] M5 — ml/ interface + data recording
- [ ] M6 — docs for the interview deliverables, extra modes and graphics

## Open problems

(none — the M1 unit 3 hover-recovery issue is resolved, see the log)

## Log

- 2026-10-07 — Project created: SPEC.md, PROGRESS.md, LICENSE (MIT). No code yet. Tests: none yet.
- 2026-10-07 — **M1 · unit 1/4: physics core.** `src/config.js` (single config: Quad X 450 preset, SI units, frames documented), `src/physics/math3.js` (vector / quaternion helpers), `src/physics/rigidbody.js` (Newton–Euler 6-DoF: p, v, q, body rates; gravity, linear drag on air-relative velocity, wind and external force inputs; fixed-step RK4 with zero-order-hold input, quaternion renormalised). Tests: `node tests/run.mjs` → 5/5 pass (quaternion round trips, free fall exact, hover at thrust = weight, energy + angular momentum conserved without drag/motors with relative drift < 1e-8 over 10 s, terminal speed m g / c_d). Open problems: none. Next: motor model + mixer.
- 2026-10-07 — **M1 · unit 2/4: motor model + mixer.** `src/motor/motors.js` (per-rotor first-order lag with exact discretisation, speed saturation, thrust k_T ω² and reaction torque k_Q ω², body wrench from four rotors), `src/motor/mixer.js` (X allocation matrix A from the config's rotor layout, mixer = A⁻¹ with per-rotor thrust clipping and a `saturated` flag). Rotor layout and spin directions added to `src/config.js`. Tests: 11/11 pass (lag reaches 1 − 1/e after τ, saturation at ω_min/ω_max, mixer inverts the allocation exactly, hover speeds √(mg/4k_T), pure yaw splits the diagonals with ΣT unchanged, saturation reported, motors + physics hold a level hover open loop). One test failed once on an exact float comparison in the test itself; fixed with a tolerance. Open problems: none. Next: cascaded PID + closed loop.
- 2026-10-07 — **M1 · unit 3/4 (stopped): cascaded PID + closed loop.** Implemented `src/controller/cascaded-pid.js` (position PID → thrust and tilt → attitude P → rate PID scaled by J; angle mode for manual flight; interface `control(measurement, reference, params)`), `src/loop/simulation.js` (fixed-step physics, controller at 250 Hz with zero-order hold, pluggable measurement, scenario runner with logs), controller gains in `src/config.js`, `tests/control.test.mjs`. Tests: 14/15 pass — analytic altitude step (ω_n = √Kp, ζ = Kd/2√Kp) matches within 2 cm, 30 s hover without drift, angle mode tracks; **hover recovery from an offset failed twice** (see Open problems). Loop stopped per the rules; nothing committed for this unit.
- 2026-10-07 — **M1 · unit 3/4 resolved: anti-windup (owner's choice).** The position integral now runs only while an axis's error is below `iZone` = 0.3 m (conditional integration), keeping the pole-placed Ki = [0.6, 0.6, 1.5]. Hover recovery from 0.5 m / 10° tilt: 0.9 cm after 10 s (limit 1 cm), largest overshoot 10.7 cm (was ~23 cm without anti-windup). Tests: 15/15 pass. Committed with the controller, loop and tests of unit 3. Open problems: none. Next: plots of hover and step response (last unit of M1).
- 2026-10-07 — **Owner's change of plan:** skip the per-milestone review gates for now, build the graphics and the playable / design side, and publish over the old `drone-wind-sim` repository (old version kept on a `legacy-python` branch).
  Done in this session (outside the one-unit loop, by the owner's request):
  - `src/render/scene.js`: Three.js view, drone mesh built from the config (arms, motors, props spinning at the real rotor speed with their spin direction, nav lights, legs), floor grid, shadows, setpoint marker, trail, chase / orbit camera.
  - `src/input/input.js`: keyboard, Gamepad API and two touch sticks (pointer events), merged into one stick state.
  - `src/modes/`: registry with *Free flight* (angle mode; throttle = climb-rate command so a keyboard can hold height) and *Tuning lab* (position hold, x / z steps, step metrics: rise, overshoot, 2 % settling, steady error, next to the analytic ω_n, ζ of the z loop).
  - `src/ui/`: start menu (mode, preset, airframe design sliders with hover rpm and thrust-to-weight), live gain sliders, three live plots (tracking error, attitude, motors).
  - `src/app.js`, `index.html`, `style.css`: rendering decoupled from the fixed-step simulation (accumulator), responsive layout, touch-sized controls.
  Tests: 19/19 in Node (incl. step metrics against first- and second-order formulas); browser smoke check (headless Firefox, 1400 and 500 px): menu, lab step (z +1 m: rise 1.18 s, overshoot 4.4 %, settling 4.17 s, steady error 0.1 cm), design edit, free-flight climb and forward flight, no JavaScript errors.
  Not verified: a real gamepad and a real touch screen (no hardware here).
