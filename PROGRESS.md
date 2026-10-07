# PROGRESS — quadrotor-lab

Log kept by the work loop (see SPEC.md). Newest entry at the bottom of "Log".

## Rules for the loop

- One unit of work per iteration, toward the next unfinished milestone: implement, run the tests, fix failures, commit (clear message), add a log entry here.
- **Review gate:** SPEC.md says "work in milestones and stop after each for review". When a milestone is complete, mark it `[x] — awaiting review`, write its summary below, and **stop the loop**. The next milestone starts only after the owner says so (and unchecks "awaiting review").
- Do not push, force-push, or touch files outside this folder. Do not add features beyond SPEC.md.
- If the same test fails twice in a row, describe the problem here under "Open problems" and stop. Do not guess.

## Milestones

- [ ] M1 — physics + motor + PID hover with plots
- [ ] M2 — menu + manual flight with keyboard / gamepad / touch
- [ ] M3 — live gain tuning UI
- [ ] M4 — sensors, delay, wind + robustness demo
- [ ] M5 — ml/ interface + data recording
- [ ] M6 — docs for the interview deliverables, extra modes and graphics

## Open problems

(none yet)

## Log

- 2026-10-07 — Project created: SPEC.md, PROGRESS.md, LICENSE (MIT). No code yet. Tests: none yet.
- 2026-10-07 — **M1 · unit 1/4: physics core.** `src/config.js` (single config: Quad X 450 preset, SI units, frames documented), `src/physics/math3.js` (vector / quaternion helpers), `src/physics/rigidbody.js` (Newton–Euler 6-DoF: p, v, q, body rates; gravity, linear drag on air-relative velocity, wind and external force inputs; fixed-step RK4 with zero-order-hold input, quaternion renormalised). Tests: `node tests/run.mjs` → 5/5 pass (quaternion round trips, free fall exact, hover at thrust = weight, energy + angular momentum conserved without drag/motors with relative drift < 1e-8 over 10 s, terminal speed m g / c_d). Open problems: none. Next: motor model + mixer.
