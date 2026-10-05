"""Physics checks.  Run:  python tests/test_physics.py   (also works with pytest)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dronesim import (Atmosphere, CascadedPID, DroneEnv, Gust, Hover, SensorModel, Simulation, WindField,  # noqa: E402
                      airframe, metrics)
from dronesim.dynamics import RigidBodyDynamics, State  # noqa: E402
from dronesim.wind import DrydenTurbulence, dryden_params  # noqa: E402


def close(a, b, tol):
    assert abs(a - b) <= tol, f"{a} != {b} (tol {tol})"


# --- atmosphere ---------------------------------------------------------------------------------
def test_isa_values():
    a = Atmosphere()
    close(float(a.density(0)), 1.2250, 0.0005)
    close(float(a.temperature(1000)), 281.65, 0.01)
    close(float(a.pressure(1000)), 89874.6, 5.0)            # ISA table
    close(float(a.density(1000)), 1.1117, 0.0005)
    close(float(Atmosphere(temperature_c=35).density(0)), 101325 / (287.05287 * 308.15), 1e-4)
    # a site at 1500 m starts at the ISA pressure of 1500 m
    close(float(Atmosphere(temperature_c=5.25, ground_altitude=1500).density(0)), 1.0581, 0.0005)


# --- wind ---------------------------------------------------------------------------------------
def test_log_profile_and_gust():
    w = WindField(mean_speed=6.0, direction_deg=270.0)        # from the west -> blows towards +x (east)
    v10 = w.mean(10.0)
    close(v10[0], 6.0, 1e-9)
    close(v10[1], 0.0, 1e-9)
    assert w.mean(30.0)[0] > v10[0] > w.mean(3.0)[0]
    g = Gust(t0=1.0, duration=2.0, speed=5.0, from_deg=0.0)   # from the north -> towards -y
    close(g(2.0)[1], -5.0, 1e-9)
    close(np.linalg.norm(g(0.9)), 0.0, 1e-12)


def test_dryden_variance_and_correlation():
    """Stationary std must equal the model's sigma; the longitudinal autocorrelation at lag tau is e^-1."""
    # V = 20 m/s keeps the correlation times short (tau_u = 5.8 s), so 4 x 1500 s of samples give a
    # statistical error of a few percent; averaged over 4 independent seeds.
    h, V, dt, n = 20.0, 20.0, 0.01, 150_000
    stds, rhos = [], []
    for seed in range(4):
        d = DrydenTurbulence(30 * 0.514444, np.random.default_rng(seed))
        d.reset(h, V)
        out = np.array([d.step(dt, h, V) for _ in range(n)])
        L, sigma = dryden_params(h, d.w20)
        stds.append(out.std(axis=0) / sigma)
        lag = int(round(L[0] / V / dt))
        u = out[:, 0] - out[:, 0].mean()
        rhos.append((u[:-lag] @ u[lag:]) / (u @ u))
    assert np.all(np.abs(np.mean(stds, axis=0) - 1.0) < 0.05), np.mean(stds, axis=0)
    assert abs(np.mean(rhos) - np.exp(-1.0)) < 0.05, np.mean(rhos)


# --- airframes / allocation ---------------------------------------------------------------------
def test_allocation_all_presets():
    for name in airframe.all_names():
        af = airframe.make(name)
        B = af.allocation()
        assert np.linalg.matrix_rank(B) == 4, name
        T = np.linalg.pinv(B) @ np.array([af.hover_thrust(), 0, 0, 0])
        assert np.all(T > 0), name
        close(T.sum(), af.hover_thrust(), 1e-9)
        assert np.allclose(B @ T, [af.hover_thrust(), 0, 0, 0], atol=1e-9), name
        assert 1.5 < af.thrust_to_weight() < 3.5, name


def test_real_presets_reproduce_source_values():
    """At standard density the converted motors give exactly the thrust / torque of the source models."""
    m = airframe.make("iris").rotors[0].motor
    w = 1100.0                                                   # rad/s, maxRotVelocity in iris.sdf
    close(m.thrust(w / (2 * np.pi), 1.225), 5.84e-06 * w ** 2, 1e-9)
    close(m.torque(w / (2 * np.pi), 1.225), 0.06 * 5.84e-06 * w ** 2, 1e-9)
    m = airframe.make("x500").rotors[0].motor
    close(m.thrust(1000.0 / (2 * np.pi), 1.225), 8.54858e-06 * 1000.0 ** 2, 1e-9)
    cf = airframe.make("crazyflie")
    rpm = 15000.0
    close(cf.rotors[0].motor.thrust(rpm / 60, 1.225), 3.16e-10 * rpm ** 2, 1e-12)
    close(cf.rotors[0].motor.torque(rpm / 60, 1.225), 7.94e-12 * rpm ** 2, 1e-14)
    close(cf.thrust_to_weight(1.225, 9.8), 2.25, 1e-9)


def test_thrust_scales_with_density():
    m = airframe.quad_x().rotors[0].motor
    hot, cold = Atmosphere(temperature_c=40).density(0), Atmosphere(temperature_c=-10).density(0)
    close(m.thrust(80, hot) / m.thrust(80, cold), hot / cold, 1e-12)


# --- rigid body ---------------------------------------------------------------------------------
def test_free_fall_and_hover_equilibrium():
    af = airframe.quad_x()
    dyn = RigidBodyDynamics(af, Atmosphere())
    s = State(np.array([0, 0, 50.0]), np.zeros(3), np.array([1.0, 0, 0, 0]), np.zeros(3), np.zeros(4))
    d = dyn.derivative(s, np.zeros(4), np.zeros(3), 1.225)
    assert np.allclose(d.v, [0, 0, -9.80665])
    sim = Simulation(af, wind=WindField(), sensors=SensorModel.ideal())
    s0 = sim.reset((0, 0, 10)).copy()
    n_hover = s0.n.copy()
    for _ in range(100):                      # 1 s with constant hover speeds, no controller
        sim.step(n_hover)
    assert np.linalg.norm(sim.state.p - s0.p) < 0.01, sim.state.p


def test_torque_free_rotation_conserves_energy_and_momentum():
    af = airframe.quad_asymmetric()
    dyn = RigidBodyDynamics(af, Atmosphere())
    dyn.ang_damping = np.zeros(3)
    dyn.kT[:] = 0.0
    dyn.kQ[:] = 0.0
    s = State(np.zeros(3) + [0, 0, 100], np.zeros(3), np.array([1.0, 0, 0, 0]), np.array([3.0, 0.1, 1.0]),
              np.zeros(4))
    J = af.inertia
    E0, L0 = 0.5 * s.w @ J @ s.w, s.R @ (J @ s.w)
    for _ in range(2000):
        s = dyn.step(s, np.zeros(4), np.zeros(3), 0.001)
    close(0.5 * s.w @ J @ s.w, E0, 1e-6 * max(E0, 1))
    assert np.allclose(s.R @ (J @ s.w), L0, atol=1e-6)


def test_wind_pushes_drone_downwind():
    af = airframe.quad_x()
    sim = Simulation(af, wind=WindField(mean_speed=8.0, direction_deg=270.0), sensors=SensorModel.ideal())
    sim.reset((0, 0, 10))
    n = sim.state.n.copy()
    for _ in range(50):
        sim.step(n)
    assert sim.state.v[0] > 0.3, sim.state.v      # blown east


# --- closed loop --------------------------------------------------------------------------------
def test_tuned_pid_holds_position_in_wind_untuned_drifts():
    for name in airframe.all_names():
        af = airframe.make(name)
        res = {}
        for gains in ("untuned", "tuned"):
            sim = Simulation(af, Atmosphere(temperature_c=30),
                             WindField(mean_speed=6, direction_deg=270, turbulence="light", seed=2),
                             SensorModel.preset("good"))
            sim.reset((0, 0, 10))
            res[gains] = metrics(sim.run(CascadedPID(af, gains), Hover((0, 0, 10)), 15), skip=5)
        assert not res["tuned"]["crashed"] and res["tuned"]["rmse_m"] < 0.3, (name, res["tuned"])
        assert res["untuned"]["rmse_m"] > 1.0, (name, res["untuned"])


def test_env_api():
    for mode in ("pid", "residual", "thrust_rates", "motors"):
        env = DroneEnv(control=mode, task="path", episode_time=2.0)
        obs, info = env.reset(seed=1)
        assert obs.shape == env.observation_space.shape and np.all(np.isfinite(obs))
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert obs.shape == env.observation_space.shape and np.isfinite(r)
    env = DroneEnv(control="residual", task="hover", episode_time=3.0)
    env.reset(seed=3)
    done = False
    while not done:
        _, _, term, trunc, info = env.step(np.zeros(3))
        done = term or trunc
    assert not info["crashed"]


if __name__ == "__main__":
    import time
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, fn in tests:
        t0 = time.time()
        fn()
        print(f"ok   {name}  ({time.time() - t0:.1f}s)")
    print(f"all {len(tests)} tests passed")
