"""Where the drone should be at every control step: p_ref, v_ref, a_ref, each (B, T, 3).

Modes ("target" part of the config):
    'hover'   stay where you start (pure disturbance rejection)
    'point'   fly to a random point and stay there (a step: the error starts at the full distance)
    'moving'  follow a point that moves on a straight line at constant speed
    'mixed'   each drone in the batch draws one of the above

Curriculum: the target distance grows from ``radius_start`` to ``radius_end`` over the first
``curriculum_iters`` training iterations, so the controller first learns to hang in the air, then to
move a little, then to fly further.
"""
import torch

MODES = ("hover", "point", "moving")


def radius_at(cfg, it):
    r0, r1 = cfg.get("radius_start", 1.0), cfg.get("radius_end", 4.0)
    n = max(1, cfg.get("curriculum_iters", 1))
    return r0 + (r1 - r0) * min(1.0, it / n)


def _random_directions(B, gen, kw, vertical_scale):
    d = torch.randn(B, 3, generator=gen, **kw)
    d[:, 2] *= vertical_scale
    return d / d.norm(dim=-1, keepdim=True).clamp(min=1e-9)


def make_targets(cfg, start, T, dt, radius, gen):
    """start (B, 3) -> dict of p, v, a references, each (B, T, 3)."""
    B = start.shape[0]
    kw = dict(dtype=start.dtype, device=start.device)
    mode = cfg.get("mode", "point")
    vs = cfg.get("vertical_scale", 0.4)
    if mode == "mixed":
        pick = torch.randint(len(MODES), (B,), generator=gen, device=start.device)
    else:
        pick = torch.full((B,), MODES.index(mode), device=start.device)
    dist = radius * torch.rand(B, 1, generator=gen, **kw).sqrt()          # uniform over the disc area
    offset = dist * _random_directions(B, gen, kw, vs)
    offset = torch.where((pick == 0)[:, None], torch.zeros_like(offset), offset)
    speed = cfg.get("speed", 1.5) * torch.rand(B, 1, generator=gen, **kw)
    vel = speed * _random_directions(B, gen, kw, vs)
    vel = torch.where((pick == 2)[:, None], vel, torch.zeros_like(vel))
    t = dt * torch.arange(1, T + 1, **kw)[None, :, None]                   # reference for the state after step k
    p = (start + offset)[:, None, :] + vel[:, None, :] * t
    v = vel[:, None, :].expand(B, T, 3)
    return {"p": p, "v": v, "a": torch.zeros_like(p)}
