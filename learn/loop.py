"""The closed loop, batched and differentiable:

    sensors -> observation -> model (with memory h) -> u -> actuation -> motors -> physics -> sensors ...

Observation (world frame, scaled to roughly unit size), the same layout as DroneEnv:
    target - position (3), velocity (3), target velocity (3), rotation matrix (9), body rates (3),
    previous output u (k)
Wind, mass and air density are *not* observed.
"""
import torch

from dronesim.diff import hover_state, quat_to_rot

from .loss import step_loss

OBS_BASE = 21


def observation(s, ref_p, ref_v, u_prev, noise, gen):
    p, v, w = s.p, s.v, s.w
    if noise:
        kw = dict(generator=gen, dtype=p.dtype, device=p.device)
        p = p + noise.get("pos", 0.0) * torch.randn(p.shape, **kw)
        v = v + noise.get("vel", 0.0) * torch.randn(v.shape, **kw)
        w = w + noise.get("gyro", 0.0) * torch.randn(w.shape, **kw)
    R = quat_to_rot(s.q).reshape(-1, 9)
    return torch.cat([(ref_p - p) / 5.0, v / 5.0, ref_v / 5.0, R, w / 5.0, u_prev], -1)


class Loop:
    """One batch of episodes.  ``run(k)`` advances k control steps and returns the summed loss, so a
    long episode can be cut into chunks for truncated backpropagation through time (``detach()``)."""

    def __init__(self, model, actuator, world, start, refs, sim_cfg, gen):
        self.model, self.act, self.world, self.refs, self.gen = model, actuator, world, refs, gen
        self.ctrl_dt = sim_cfg.get("ctrl_dt", 0.02)
        self.phys_dt = sim_cfg.get("phys_dt", 0.005)
        self.sub = max(1, round(self.ctrl_dt / self.phys_dt))
        self.integrator = sim_cfg.get("integrator", "rk4")
        self.noise = sim_cfg.get("sensor_noise", {})
        self.state = hover_state(world.dyn, start, att_std=sim_cfg.get("start_tilt", 0.0), gen=gen)
        B = start.shape[0]
        self.h = model.initial_memory(B, start)
        self.u_prev = torch.zeros(B, actuator.act_dim, dtype=start.dtype, device=start.device)
        self.v_prev = self.state.v
        self.k = 0
        self.log = None

    def detach(self):
        self.state = self.state.detach()
        self.h, self.u_prev, self.v_prev = self.h.detach(), self.u_prev.detach(), self.v_prev.detach()

    def record(self):
        """Keep positions, references, wind and outputs of every step (for plots / evaluation)."""
        self.log = {"p": [], "ref": [], "wind": [], "u": [], "n": []}
        return self

    def run(self, steps, loss_cfg=None):
        total, sums = 0.0, {}
        r = self.refs
        for _ in range(steps):
            k = self.k
            s = self.state
            obs = observation(s, r["p"][:, k], r["v"][:, k], self.u_prev, self.noise, self.gen)
            u, self.h = self.model(obs, self.h)
            n_cmd = self.act.speeds(u, s.w)
            wind = self.world.wind_step(self.ctrl_dt)
            for _ in range(self.sub):
                s = self.world.dyn.step(s, n_cmd, wind, self.phys_dt, self.integrator)
            a = (s.v - self.v_prev) / self.ctrl_dt
            if loss_cfg is not None:
                l, terms = step_loss(loss_cfg, s.p, s.v, a, s.w, u, self.u_prev, r["p"][:, k], r["v"][:, k],
                                     r["a"][:, k])
                total = total + l
                for name, val in terms.items():
                    sums[name] = sums.get(name, 0.0) + val / steps
            if self.log is not None:
                for name, val in (("p", s.p), ("ref", r["p"][:, k]), ("wind", wind), ("u", u), ("n", s.n)):
                    self.log[name].append(val.detach())
            self.state, self.u_prev, self.v_prev = s, u, s.v
            self.k += 1
        return total / steps, sums

    def stacked_log(self):
        return {k: torch.stack(v, 1) for k, v in self.log.items()}
