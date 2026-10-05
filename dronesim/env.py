"""Training environment with the Gymnasium API (reset / step), without depending on gymnasium.

    env = DroneEnv(control="residual", task="path", airframes=["quad_x", "hexa_x"])
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

Every episode draws a new world (domain randomisation): airframe, mass, motor health, wind (mean,
direction, turbulence, gusts), temperature and site altitude, plus a random mission.  A policy that
does well across these draws has learned to *adapt*.

Control modes (what the action means):
    'pid'           no action: the cascaded PID flies (baseline; action is ignored)
    'residual'      3 values in [-1, 1] -> extra acceleration (± ``residual_accel`` m/s^2) added to the
                    PID's command.  The PID keeps the drone stable; a learned model only has to predict
                    the disturbance (wind, missing thrust) - the idea behind Neural-Fly-style controllers.
    'thrust_rates'  4 values: collective thrust and body-rate setpoints; the inner rate PID + mixer
                    turn them into motor speeds (how most RL drone controllers are set up)
    'motors'        one value per rotor: normalised thrust of each motor (end-to-end; hardest)

Observation (world frame, scaled to roughly unit size):
    position error to the reference (3), velocity (3), rotation matrix (9), body rates (3),
    reference feed-forward velocity (3), previous action (k)
The true wind, air density and motor health are *not* observed; they are in ``info`` for analysis.
"""
from dataclasses import dataclass, field

import numpy as np

from . import airframe as af_mod
from .atmosphere import G0, Atmosphere
from .control.pid import CascadedPID
from .guidance import Hover, PathFollower, Waypoints
from .rotation import quat_to_rot
from .sensors import SensorModel
from .simulation import Simulation
from .wind import Gust, WindField


class Box:
    """Minimal stand-in for gymnasium.spaces.Box."""

    def __init__(self, low, high, shape):
        self.low = np.full(shape, low, dtype=np.float32)
        self.high = np.full(shape, high, dtype=np.float32)
        self.shape = shape
        self.dtype = np.float32
        self._rng = np.random.default_rng()

    def sample(self):
        return self._rng.uniform(self.low, self.high).astype(np.float32)

    def contains(self, x):
        x = np.asarray(x)
        return x.shape == self.shape and np.all(x >= self.low) and np.all(x <= self.high)

    def __repr__(self):
        return f"Box({self.low.flat[0]}, {self.high.flat[0]}, {self.shape})"


@dataclass
class Randomization:
    """Ranges sampled at every reset.  Set a range to a single value (lo == hi) to fix it."""
    airframes: list = field(default_factory=lambda: ["quad_x"])
    mass_scale: tuple = (0.9, 1.2)            # payload: mass and inertia scaled together
    motor_health: tuple = (1.0, 1.0)          # e.g. (0.7, 1.0) for worn / damaged propellers
    weak_motor_prob: float = 0.0              # chance that one random motor is weakened by motor_health
    wind_speed: tuple = (0.0, 8.0)            # mean wind at 10 m [m/s]
    turbulence: tuple = ("none", "light", "moderate")
    gust_prob: float = 0.5
    gust_speed: tuple = (2.0, 7.0)
    temperature_c: tuple = (-5.0, 38.0)
    site_altitude: tuple = (0.0, 1500.0)


class DroneEnv:
    metadata = {"render_modes": []}

    def __init__(self, control="residual", task="hover", randomization=None, sensors="good",
                 episode_time=20.0, ctrl_dt=0.01, action_repeat=2, residual_accel=4.0, gains="tuned", **rand_kw):
        self.control = control
        self.task = task
        self.rand = randomization or Randomization(**rand_kw)
        self.sensor_preset = sensors
        self.episode_time = episode_time
        self.ctrl_dt = ctrl_dt
        self.repeat = action_repeat
        self.residual_accel = residual_accel
        self.gains = gains
        self.rng = np.random.default_rng()
        n_rot = af_mod.make(self.rand.airframes[0]).n
        if control == "motors":
            counts = {af_mod.make(a).n for a in self.rand.airframes}
            if len(counts) > 1:
                raise ValueError("control='motors' needs airframes with the same number of rotors")
        self.act_dim = {"pid": 1, "residual": 3, "thrust_rates": 4, "motors": n_rot}[control]
        self.action_space = Box(-1.0, 1.0, (self.act_dim,))
        self.obs_dim = 21 + self.act_dim
        self.observation_space = Box(-np.inf, np.inf, (self.obs_dim,))

    # ---------------------------------------------------------------------------------------------
    def _u(self, rng_range):
        lo, hi = rng_range
        return float(self.rng.uniform(lo, hi))

    def _make_world(self, options):
        r, o = self.rand, options
        name = o.get("airframe") or self.rng.choice(r.airframes)
        af = af_mod.make(name)
        k = o.get("mass_scale", self._u(r.mass_scale))
        af.mass *= k
        af.inertia = af.inertia * k
        af.inertia_inv = np.linalg.inv(af.inertia)
        af.ang_damping = af.ang_damping * k
        nominal = af_mod.make(name)                   # what the controller believes (no payload, healthy)
        health = o.get("motor_health", self._u(r.motor_health))
        if "weak_motor" in o or self.rng.random() < r.weak_motor_prob:
            i = o.get("weak_motor", int(self.rng.integers(af.n)))
            af.rotors[i].motor = af.rotors[i].motor.copy(health=health)
        else:
            for rot in af.rotors:
                rot.motor = rot.motor.copy(health=health)
        turb = o.get("turbulence", self.rng.choice(r.turbulence))
        gusts = []
        if o.get("gust", self.rng.random() < r.gust_prob):
            T = self.episode_time
            gusts.append(Gust(t0=self._u((0.1 * T, max(0.1 * T, T - 4.0))), duration=self._u((1.0, 3.0)),
                              speed=self._u(r.gust_speed), from_deg=self._u((0, 360))))
        wind = WindField(mean_speed=o.get("wind_speed", self._u(r.wind_speed)),
                         direction_deg=o.get("wind_dir", self._u((0, 360))), turbulence=turb, gusts=gusts,
                         seed=int(self.rng.integers(1 << 31)))
        atm = Atmosphere(temperature_c=o.get("temperature_c", self._u(r.temperature_c)),
                         ground_altitude=o.get("site_altitude", self._u(r.site_altitude)))
        sens = SensorModel.preset(self.sensor_preset)
        sens.seed = int(self.rng.integers(1 << 31))
        return af, nominal, atm, wind, sens

    def _make_mission(self, start, options):
        if "guidance" in options:
            return options["guidance"]
        if self.task == "hover":
            return Hover(start + self.rng.uniform([-3, -3, -1], [3, 3, 1]))
        pts = [start]
        for _ in range(4):
            pts.append(pts[-1] + np.r_[self.rng.uniform(-12, 12, 2), self.rng.uniform(-2, 2)])
        pts = np.array(pts)
        pts[:, 2] = np.clip(pts[:, 2], 4.0, 25.0)
        if self.task == "waypoints":
            return Waypoints(pts[1:], radius=0.7)
        return PathFollower(pts, speed=self._u((2.0, 6.0)))

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.action_space._rng = np.random.default_rng(seed + 1)
        options = options or {}
        af, nominal, atm, wind, sens = self._make_world(options)
        self.af, self.nominal = af, nominal
        self.sim = Simulation(af, atm, wind, sens, ctrl_dt=self.ctrl_dt)
        start = np.array(options.get("start", (0.0, 0.0, 10.0)))
        self.sim.reset(start)
        self.pid = CascadedPID(nominal, self.gains)            # controller uses nominal parameters
        self.guidance = self._make_mission(start, options)
        self.prev_action = np.zeros(self.act_dim, dtype=np.float32)
        self.steps = 0
        self.max_steps = int(self.episode_time / (self.ctrl_dt * self.repeat))
        self.est = self.sim.estimate()
        self.ref = self.guidance.reference(self.sim.t, self.est.p)
        return self._obs(), self._info()

    # ---------------------------------------------------------------------------------------------
    def _motor_command(self, action):
        est, ref, dt, pid = self.est, self.ref, self.ctrl_dt, self.pid
        if self.control == "pid":
            return pid.update(est, ref, dt)
        if self.control == "residual":
            a = pid.acceleration_command(est, ref, dt) + self.residual_accel * np.asarray(action, float)
            R_d, F = pid.attitude_and_thrust(est, a, ref.get("yaw", 0.0))
            tau = pid.torque_command(est, pid.rate_command(est, R_d), dt)
            return pid.mixer.speeds(np.array([F, *tau]))
        if self.control == "thrust_rates":
            F = (np.asarray(action[0]) + 1.0) / 2.0 * pid.mixer.max_collective()
            w_sp = np.asarray(action[1:]) * pid.lim["rate"]
            return pid.update_rates(est, F, w_sp, dt)
        # motors: normalised thrust per rotor
        T = (np.asarray(action, float) + 1.0) / 2.0 * pid.mixer.t_max
        return np.array([m.speed_for_thrust(t, pid.mixer.rho) for m, t in zip(pid.mixer.motors, T)])

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(self.act_dim), -1.0, 1.0)
        reward = 0.0
        for _ in range(self.repeat):
            n_cmd = self._motor_command(action)
            self.sim.step(n_cmd)
            self.est = self.sim.estimate()
            self.ref = self.guidance.reference(self.sim.t, self.est.p)
            reward += self._reward(action)
            if self.sim.crashed:
                break
        self.steps += 1
        err = np.linalg.norm(self.ref["p"] - self.sim.state.p)
        terminated = bool(self.sim.crashed or err > 25.0)
        if terminated:
            reward -= 50.0
        truncated = self.steps >= self.max_steps
        self.prev_action = action
        return self._obs(), float(reward), terminated, truncated, self._info()

    def _reward(self, action):
        s = self.sim.state
        err = np.linalg.norm(self.ref["p"] - s.p)
        smooth = np.sum((action - self.prev_action) ** 2)
        return self.ctrl_dt * (1.0 - err - 0.05 * np.linalg.norm(s.w) - 0.1 * smooth)

    def _obs(self):
        est, ref = self.est, self.ref
        R = quat_to_rot(est.q)
        o = np.concatenate([np.clip((ref["p"] - est.p) / 5.0, -3, 3), est.v / 5.0, R.ravel(), est.w / 5.0,
                            np.asarray(ref.get("v", np.zeros(3))) / 5.0, self.prev_action])
        return o.astype(np.float32)

    def _info(self):
        s, sim = self.sim.state, self.sim
        return {"t": sim.t, "position": s.p.copy(), "reference": self.ref["p"].copy(),
                "error": float(np.linalg.norm(self.ref["p"] - s.p)), "wind": sim.wind_now.copy(),
                "air_density": float(sim.atm.density(max(s.p[2], 0.0))), "airframe": self.af.name,
                "mass": self.af.mass, "motor_health": [r.motor.health for r in self.af.rotors],
                "crashed": sim.crashed}


def make_gymnasium(**kw):
    """Wrap DroneEnv as a real gymnasium.Env (only if gymnasium is installed)."""
    import gymnasium as gym

    class GymDroneEnv(gym.Env):
        def __init__(self):
            self.env = DroneEnv(**kw)
            a, o = self.env.action_space, self.env.observation_space
            self.action_space = gym.spaces.Box(a.low, a.high, a.shape, np.float32)
            self.observation_space = gym.spaces.Box(o.low, o.high, o.shape, np.float32)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            return self.env.reset(seed=seed, options=options)

        def step(self, action):
            return self.env.step(action)

    return GymDroneEnv()
