# SPEC — quadrotor-lab

Build a browser-based quadrotor simulator in plain JavaScript (ES modules,
no ML libraries, no build step required), deployable on GitHub Pages and
usable on desktop and phone. Main purpose: controller synthesis and tuning.
Games and graphics are secondary but must be easy to add later.

Secondary purpose: this is a portfolio project for job interviews in drone
control and machine learning (robotics/ML student job). Every design choice
should make it easy for me to explain, defend and demo in a 10-minute
interview. Prefer clarity, correctness and verifiable results over features.

Architecture: separate modules with small, documented interfaces, so each
block can be made more detailed or replaced without touching the others.
- physics/: 6-DoF rigid-body dynamics (position, velocity, quaternion
  attitude, body rates), gravity, drag, wind/disturbances. Fixed-timestep
  RK4 integration.
- motor/: motor model (first-order lag, saturation, thrust/torque from rpm)
  and mixer from thrust+moments to 4 motors.
- sensors/: IMU and position measurement with noise, bias and delay.
- controller/: interface u = control(measurement, reference, params).
  Start with a cascaded PID (position -> attitude -> rate). Make it easy to
  add LQR, MPC or a neural policy.
- loop/: connects the blocks with negative feedback (error = reference -
  measurement), decoupled from rendering (physics stays fixed-step).
- ml/: a Gym-style interface (reset(), step(action) -> observation, reward,
  done), a data recorder (observation/action logs exportable as JSON/CSV),
  and a hook to plug in a learned policy. No training code required yet,
  but the interface must make imitation learning and RL straightforward.
- render/: 3D view (Three.js), kept separate from physics.
- modes/: each game mode (free flight, controller tuning lab, race, ...) is
  its own module registered in one list.
- input/: keyboard, gamepad (Gamepad API) and touch joysticks.
- ui/: start menu (mode select, drone preset, settings), live sliders for
  controller gains, and plots of error, motor outputs and attitude.

Requirements:
- All drone and controller parameters live in one config object/file.
- A manual flight mode (angle mode) and an automatic mode where I tune
  gains live and see the step response.
- Responsive layout, touch-sized controls, pointer events.
- Tests (small, runnable in the browser or Node): hover is stable, step
  response matches the simplified analytic case, energy conserved without
  drag and motors.

Interview readiness (required deliverables):
- README: one-paragraph pitch, live demo link, a short GIF of the demo,
  controls, an architecture diagram, and a "Results" section with plots
  (step response, disturbance rejection, tracking error with and without
  sensor delay/noise).
- docs/derivation.md: the equations of motion and the control loop derived
  from first principles (Newton-Euler, motor model, closed-loop transfer
  function), written so I can reproduce them on a whiteboard.
- docs/design-decisions.md: for each major choice, what I chose, the
  alternatives, and the trade-offs (why RK4, why cascaded PID, why this
  motor model, known limitations).
- docs/interview-notes.md: a one-page summary: the problem, my approach,
  3 key results with numbers, limitations and how I would address them
  (sim-to-real gap, delay, saturation, model mismatch), and how ML would
  plug in (imitation learning from manual flights, RL, learned
  controller).
- A "robustness" demo: run the same controller under randomized mass,
  delay and wind and show the success rate.
- Honest limitations section; do not overclaim realism.
- MIT license, clean commit history with meaningful messages.

Work in milestones and stop after each for review:
M1 physics + motor + PID hover with plots. M2 menu + manual flight with
keyboard/gamepad/touch. M3 live gain tuning UI. M4 sensors, delay, wind +
robustness demo. M5 ml/ interface + data recording. M6 docs for the
interview deliverables, extra modes and graphics.
Explain design choices briefly; do not add features beyond this list.
