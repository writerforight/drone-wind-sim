"""Reference trajectories from the Python simulator for web/parity_test.js."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dronesim import Atmosphere, CascadedPID, Gust, PathFollower, SensorModel, Simulation, WindField, airframe, figure_eight  # noqa: E402

cases = []
for name, gains, wind, temp, health in [("iris", "tuned", 6, 30, None), ("hexa_x", "untuned", 4, 15, None),
                                        ("quad_asymmetric", "tuned", 8, -5, None), ("crazyflie", "tuned", 3, 25, None),
                                        ("octo_x", "tuned", 5, 20, [(0, 0.0)]), ("x500", "tuned", 9, 35, [(2, 0.7)])]:
    af = airframe.make(name)
    for i, h in health or []:
        af.rotors[i].motor = af.rotors[i].motor.copy(health=h)
    sim = Simulation(af, Atmosphere(temperature_c=temp),
                     WindField(mean_speed=wind, direction_deg=250, gusts=[Gust(6, 2.5, 7, 180)]), SensorModel.ideal())
    path = figure_eight(center=(0, 0, 10), a=15, b=8)
    sim.reset(path[0])
    ctrl, guid = CascadedPID(airframe.make(name), gains), PathFollower(path, speed=4.0)
    ps = []
    for _ in range(1500):
        ps.append(sim.state.p.tolist())
        est = sim.estimate()
        sim.step(ctrl.update(est, guid.reference(sim.t, est.p), sim.ctrl_dt))
    cases.append(dict(airframe=name, gains=gains, wind=wind, temp=temp, health=health, start=path[0].tolist(), p=ps))
print(json.dumps({"cases": cases}))
