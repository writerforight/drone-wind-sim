"""Build your own airframe with per-motor models, and plug in your own controller.

Airframe: a Y6 - three arms, two coaxial rotors per arm spinning in opposite directions.  The lower
rotor works in the upper rotor's downwash, so it gets its own, less efficient motor model.  Nothing
else has to change: the allocation matrix is built from the rotor list.

Controller: anything with ``reset()`` and ``update(estimate, reference, dt) -> motor speeds [rev/s]``.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dronesim import (Airframe, Atmosphere, CascadedPID, Gust, MotorModel, PathFollower, Rotor,  # noqa: E402
                      SensorModel, Simulation, WindField, figure_eight, metrics)

# --- 1) motors: one model for the upper rotors, a weaker one for the lower rotors -------------------
upper = MotorModel(diameter=0.254, c_t=0.11, c_q=0.0045, tau=0.04, n_max=135)
lower = upper.copy(c_t=0.11 * 0.8, c_q=0.0045 * 0.85)        # ~20 % less thrust in the downwash

# --- 2) airframe: three arms at 0°, 120°, 240°, rotors 6 cm above / below the arm --------------------
rotors = []
for k, a in enumerate(np.radians([0, 120, 240])):
    xy = 0.30 * np.array([np.cos(a), np.sin(a)])
    rotors.append(Rotor(np.array([*xy, +0.06]), spin=+1, motor=upper.copy()))
    rotors.append(Rotor(np.array([*xy, -0.06]), spin=-1, motor=lower.copy()))
y6 = Airframe("y6-coaxial", mass=2.4, rotors=rotors)
print(y6.summary())
print("allocation matrix B (rows: thrust, roll, pitch, yaw):")
print(np.round(y6.allocation(), 3))


# --- 3) your own controller: here a PID whose speed limit shrinks in strong wind -----------------
class CautiousPID:
    """Example of a custom controller.  Wraps the cascaded PID but slows down when it has to tilt a lot
    (a crude 'feel the wind' rule).  Replace update() with anything - an LQR, MPC or a neural network."""

    def __init__(self, airframe):
        self.pid = CascadedPID(airframe, "tuned")
        self.scale = 1.0

    def reset(self):
        self.pid.reset()
        self.scale = 1.0

    def update(self, est, ref, dt):
        a = self.pid.last.get("a_cmd")
        tilt_demand = 0.0 if a is None else np.linalg.norm(a[:2]) / 9.81
        target = np.clip(1.4 - tilt_demand, 0.4, 1.0)
        self.scale += (target - self.scale) * min(1.0, dt / 0.5)     # smooth
        ref = dict(ref, v=ref["v"] * self.scale, a=ref.get("a", 0.0) * self.scale ** 2)
        return self.pid.update(est, ref, dt)


path = figure_eight(center=(0, 0, 12), a=15, b=8)
for name, ctrl in (("tuned PID", CascadedPID(y6, "tuned")), ("custom CautiousPID", CautiousPID(y6))):
    sim = Simulation(y6, Atmosphere(temperature_c=20),
                     WindField(mean_speed=7, direction_deg=200, turbulence="moderate", seed=5,
                               gusts=[Gust(12, 2.5, 9, 180)]),
                     SensorModel.preset("gps"))
    sim.reset(path[0])
    m = metrics(sim.run(ctrl, PathFollower(path, speed=5.0), 40, stop_when_done=True))
    print(f"{name:20s} " + "  ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in m.items()))
