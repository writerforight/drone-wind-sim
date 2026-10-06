"""The differentiable PyTorch physics must match the NumPy physics.  Run:  python tests/test_diff_dynamics.py"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dronesim import Atmosphere, airframe  # noqa: E402
from dronesim.diff import DiffDynamics, DiffState, hover_state  # noqa: E402
from dronesim.dynamics import RigidBodyDynamics, State  # noqa: E402
from dronesim.rotation import euler_to_quat  # noqa: E402


def _compare(name, temp_c=31.0, site=900.0, wind=(4.0, -2.5, 0.3), steps=1500, dt=0.002):
    af = airframe.make(name)
    atm = Atmosphere(temperature_c=temp_c, ground_altitude=site)
    ref = RigidBodyDynamics(af, atm)
    one = torch.ones(1, dtype=torch.float64)
    dyn = DiffDynamics(af, one, one * atm.T0, one * atm.p0, dtype=torch.float64)
    # same start: tilted, moving, spinning, rotors not at hover
    q0 = euler_to_quat(0.2, -0.1, 0.7)
    n0 = np.array([0.6 * r.motor.n_max for r in af.rotors])
    s = State(np.array([1.0, -2.0, 200.0]), np.array([1.5, 0.5, -0.4]), q0, np.array([0.3, -0.2, 0.1]), n0.copy())
    t = lambda a: torch.tensor(np.asarray(a, float), dtype=torch.float64)[None]   # noqa: E731
    d = DiffState(t(s.p), t(s.v), t(s.q), t(s.w), t(s.n))
    w = np.array(wind)
    n_max = np.array([r.motor.n_max for r in af.rotors])
    for k in range(steps):
        # varying commands, partly outside [n_min, n_max] so the clipping and both lag constants are used
        n_cmd = n_max * (0.55 + 0.5 * np.sin(0.01 * k + np.arange(af.n)))
        s = ref.step(s, n_cmd, w, dt)
        d = dyn.step(d, t(n_cmd), t(w), dt)
    err = max(np.abs(d.p.numpy()[0] - s.p).max(), np.abs(d.v.numpy()[0] - s.v).max(),
              np.abs(d.q.numpy()[0] - s.q).max(), np.abs(d.w.numpy()[0] - s.w).max())
    assert s.p[2] > 0.0, "test flight hit the ground (NumPy version has a ground plane, the torch one not)"
    return err, s.p


def test_parity_numpy_vs_torch():
    for name in ("quad_x", "quad_asymmetric", "iris", "hexa_x"):
        err, p = _compare(name)
        assert err < 1e-9, f"{name}: max difference {err:.2e}"
        print(f"     {name:16s} 3 s of flight, max |torch - numpy| = {err:.1e}   (final position {np.round(p, 2)})")


def test_hover_state_hovers():
    af = airframe.make("quad_asymmetric")
    B = 3
    ms = torch.tensor([0.9, 1.0, 1.2], dtype=torch.float64)
    atm = Atmosphere()
    dyn = DiffDynamics(af, ms, torch.full((B,), atm.T0, dtype=torch.float64),
                       torch.full((B,), atm.p0, dtype=torch.float64), dtype=torch.float64)
    s = hover_state(dyn, torch.tensor([[0.0, 0.0, 10.0]] * B, dtype=torch.float64))
    k = dyn.derivative(s, s.n, torch.zeros(B, 3, dtype=torch.float64), dyn.density(s.p[:, 2]))
    assert k.v.abs().max() < 1e-9 and k.w.abs().max() < 1e-9, (k.v, k.w)


def test_gradient_flows_through_physics():
    af = airframe.make("quad_x")
    one = torch.ones(1)
    atm = Atmosphere()
    dyn = DiffDynamics(af, one, one * atm.T0, one * atm.p0)
    s = hover_state(dyn, torch.tensor([[0.0, 0.0, 10.0]]))
    gain = torch.zeros(1, requires_grad=True)
    n_cmd = s.n * (1.0 + gain)
    for _ in range(100):
        s = dyn.step(s, n_cmd, torch.zeros(1, 3), 0.005)
    s.p[0, 2].backward()
    assert gain.grad.item() > 0.0          # more speed -> higher


if __name__ == "__main__":
    for f in (test_parity_numpy_vs_torch, test_hover_state_hovers, test_gradient_flows_through_physics):
        f()
        print("ok  ", f.__name__)
