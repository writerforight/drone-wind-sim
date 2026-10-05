"""Simulation loop: physics at ``dt`` (default 500 Hz), sensors + controller at ``ctrl_dt`` (100 Hz).

    sim = Simulation(airframe, atmosphere, wind, sensors)
    log = sim.run(controller, guidance, duration=30)
    print(metrics(log))
"""
import numpy as np

from .atmosphere import Atmosphere
from .dynamics import RigidBodyDynamics, State
from .rotation import euler_to_quat, quat_to_euler
from .sensors import SensorModel
from .wind import WindField


class Simulation:
    def __init__(self, airframe, atmosphere=None, wind=None, sensors=None, dt=0.002, ctrl_dt=0.01):
        self.af = airframe
        self.atm = atmosphere or Atmosphere()
        self.wind = wind or WindField()
        self.sensors = sensors or SensorModel()
        self.dt = dt
        self.sub = max(1, int(round(ctrl_dt / dt)))
        self.ctrl_dt = self.sub * dt
        self.dyn = RigidBodyDynamics(airframe, self.atm)

    def reset(self, position=(0.0, 0.0, 10.0), yaw=0.0, velocity=(0.0, 0.0, 0.0)):
        """Start in hover: rotors already spinning at the speed that carries the weight."""
        rho = float(self.atm.density(position[2]))
        B = self.dyn.B
        T = np.linalg.pinv(B) @ np.array([self.af.hover_thrust(), 0, 0, 0])
        n = np.array([r.motor.speed_for_thrust(t, rho) for r, t in zip(self.af.rotors, T)])
        self.state = State(np.array(position, float), np.array(velocity, float), euler_to_quat(0, 0, yaw),
                           np.zeros(3), n)
        self.t = 0.0
        self.wind_now = self.wind.reset(self.state.p, self.state.v)
        self.sensors.reset(self.state)
        self.crashed = False
        return self.state

    def step(self, n_cmd, ext_force=None):
        """Advance one control period with constant motor commands."""
        for _ in range(self.sub):
            self.state = self.dyn.step(self.state, n_cmd, self.wind_now, self.dt, ext_force)
            self.wind_now = self.wind.step(self.t, self.dt, self.state.p, self.state.v)
            self.t += self.dt
        R = self.state.R
        if (self.state.p[2] <= 0.0 and self.t > 0.5) or R[2, 2] < -0.2:
            self.crashed = True
        return self.state

    def set_motor_health(self, i, health):
        self.dyn.set_motor_health(i, health)

    def estimate(self):
        return self.sensors.measure(self.t, self.ctrl_dt, self.state)

    def run(self, controller, guidance, duration, stop_when_done=False, on_step=None):
        if not hasattr(self, "state"):
            self.reset(guidance.reference(0.0, np.zeros(3))["p"])
        controller.reset()
        rho_fn = self.atm.density
        rows = []
        steps = int(round(duration / self.ctrl_dt))
        for _ in range(steps):
            est = self.estimate()
            ref = guidance.reference(self.t, est.p)
            n_cmd = controller.update(est, ref, self.ctrl_dt)
            s = self.state
            rho = float(rho_fn(max(s.p[2], 0.0)))
            power = sum(r.motor.power(ni, rho) for r, ni in zip(self.af.rotors, s.n))
            ct = getattr(guidance, "cross_track", None)
            rows.append((self.t, s.p.copy(), s.v.copy(), quat_to_euler(s.q), s.w.copy(), s.n.copy(), n_cmd.copy(),
                         self.wind_now.copy(), ref["p"].copy(), power,
                         np.linalg.norm(ref["p"] - s.p) if ct is None else ct))
            self.step(n_cmd)
            if on_step:
                on_step(self)
            if self.crashed or (stop_when_done and guidance.done):
                break
        cols = list(zip(*rows))
        names = ["t", "p", "v", "euler", "w", "n", "n_cmd", "wind", "p_ref", "power", "track_err"]
        log = {k: np.array(c) for k, c in zip(names, cols)}
        log["crashed"] = self.crashed
        log["n_max"] = np.array([r.motor.n_max for r in self.af.rotors])
        log["path"] = guidance.path_points()
        log["airframe"] = self.af.name
        return log


def metrics(log, skip=1.0):
    """Tracking error statistics, energy and saturation over the run (after the first ``skip`` s)."""
    m = log["t"] >= skip
    e = log["track_err"][m]
    sat = np.mean(log["n_cmd"][m] >= 0.98 * log["n_max"]) if m.any() else 0.0
    dt = np.diff(log["t"]).mean() if len(log["t"]) > 1 else 0.0
    return {"rmse_m": float(np.sqrt(np.mean(e ** 2))) if e.size else np.nan,
            "max_err_m": float(e.max()) if e.size else np.nan,
            "energy_Wh": float(log["power"].sum() * dt / 3600.0),
            "saturation_%": float(100 * sat),
            "max_tilt_deg": float(np.degrees(np.abs(log["euler"][m, :2]).max())) if m.any() else np.nan,
            "crashed": bool(log["crashed"]),
            "duration_s": float(log["t"][-1]) if len(log["t"]) else 0.0}
