"""Reference flights of an exported learned loop in the Python simulator, for web/learned_parity_test.js.

    python web/learned_parity_ref.py web/models/<name>.json > /tmp/lref.json
    node web/learned_parity_test.js web/models/<name>.json /tmp/lref.json
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dronesim import Atmosphere, Gust, Hover, SensorModel, Simulation, WindField, airframe  # noqa: E402
from learn.actuation import Actuator  # noqa: E402
from learn.evaluate import LearnedController  # noqa: E402
from learn.model import LoopMLP  # noqa: E402

m = json.load(open(sys.argv[1]))
model = LoopMLP(m["obs_dim"], m["act_dim"], m["memory"], [len(L["b"]) for L in m["layers"]], m["activation"],
                m["memory_activation"]).double()
lin = [x for x in model.body if isinstance(x, torch.nn.Linear)] + [model.out]
with torch.no_grad():
    for L, src in zip(lin, m["layers"] + [m["out"]]):          # the rounded weights the browser uses
        L.weight.copy_(torch.tensor(src["W"]))
        L.bias.copy_(torch.tensor(src["b"]))
act = Actuator(airframe.make(m["airframe"]), m["control"], dtype=torch.float64)
cases = []
for wind_speed, temp, target in ((0.0, 15.0, (0, 0, 10)), (5.0, 30.0, (2.0, -1.5, 11.0)), (3.0, -5.0, (-3, 2, 9))):
    sim = Simulation(airframe.make(m["airframe"]), Atmosphere(temperature_c=temp),
                     WindField(mean_speed=wind_speed, direction_deg=250, gusts=[Gust(3.0, 2.0, 4.0, 180)]),
                     SensorModel.ideal(), ctrl_dt=m["ctrl_dt"])
    sim.reset((0.0, 0.0, 10.0))
    log = sim.run(LearnedController(model, act, m["ctrl_dt"]), Hover(np.array(target, float)), 8.0)
    cases.append({"wind": wind_speed, "temp": temp, "target": list(target), "p": log["p"].tolist()})
print(json.dumps({"cases": cases}))
