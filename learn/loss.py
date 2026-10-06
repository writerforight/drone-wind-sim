"""Switchable loss on the tracking error e = p - p_ref and its time derivatives.

    L = w_error  |e|^q  +  w_error_rate  |e'|^2  +  w_error_accel  |e''|^2           (main terms)
      + w_body_rates |omega|^2  +  w_action_smooth |u_t - u_{t-1}|^2                  (optional regularisers)

    e'  = v - v_ref     (first derivative of the error: velocity error)
    e'' = a - a_ref     (second derivative: acceleration error; a from the velocity change over one step)

Every weight comes from the "loss" part of the config; 0 switches a term off.  Note: the derivative
terms alone do not bring the drone to the target - e' = 0 only means "stand still", anywhere.  They act as
damping / smoothness on top of the error term.
"""
import torch

DEFAULTS = {"error": 1.0, "error_power": 2.0, "error_rate": 0.0, "error_accel": 0.0,
            "body_rates": 0.0, "action_smooth": 0.0}


def step_loss(cfg, p, v, a, w, u, u_prev, ref_p, ref_v, ref_a):
    """Per-step loss, averaged over the batch.  Returns (total, dict of the individual terms)."""
    c = {**DEFAULTS, **cfg}
    terms = {}
    e = (p - ref_p).norm(dim=-1)
    terms["error"] = (e ** c["error_power"]).mean()
    if c["error_rate"]:
        terms["error_rate"] = ((v - ref_v) ** 2).sum(-1).mean()
    if c["error_accel"]:
        terms["error_accel"] = ((a - ref_a) ** 2).sum(-1).mean()
    if c["body_rates"]:
        terms["body_rates"] = (w ** 2).sum(-1).mean()
    if c["action_smooth"]:
        terms["action_smooth"] = ((u - u_prev) ** 2).sum(-1).mean()
    total = sum(c[k] * v_ for k, v_ in terms.items())
    return total, {k: float(v_.detach()) for k, v_ in terms.items()}
