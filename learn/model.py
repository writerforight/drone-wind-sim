"""The learned control loop: one MLP chain with a recurrent memory vector.

    [observation_t, h_{t-1}]  --MLP-->  [u_t, h_t]

u_t goes to the motors (through ``actuation.py``), h_t comes back in at the next step.  Nothing else:
no gates, no separate estimator.  The memory h has to learn on its own what to keep about the drone and
its air from the sensor stream.

The architecture is built from the "model" part of the config, so it can be changed (or later drawn in a
diagram editor) without touching code:
    {"hidden_state": 32, "layers": [128, 128], "activation": "tanh", "memory_activation": "tanh"}
"""
import torch
from torch import nn

ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "gelu": nn.GELU, "silu": nn.SiLU,
               "sigmoid": nn.Sigmoid, "identity": nn.Identity}


class LoopMLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_state=32, layers=(128, 128), activation="tanh",
                 memory_activation="tanh"):
        super().__init__()
        self.obs_dim, self.act_dim, self.H = obs_dim, act_dim, hidden_state
        sizes = [obs_dim + hidden_state, *layers]
        mods = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            mods += [nn.Linear(a, b), ACTIVATIONS[activation]()]
        self.body = nn.Sequential(*mods)
        self.out = nn.Linear(sizes[-1], act_dim + hidden_state)
        # last layer starts at zero: u = 0 (= hover, see actuation.py) and h = 0, so training starts from a
        # drone that hangs in the air instead of one that flips in the first 100 ms
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.mem_act = ACTIVATIONS[memory_activation]()

    def initial_memory(self, B, like):
        return torch.zeros(B, self.H, dtype=like.dtype, device=like.device)

    def forward(self, obs, h):
        y = self.out(self.body(torch.cat([obs, h], -1)))
        return torch.tanh(y[:, :self.act_dim]), self.mem_act(y[:, self.act_dim:])

    def describe(self):
        n = sum(p.numel() for p in self.parameters())
        layers = " -> ".join(str(m.out_features) for m in self.body if isinstance(m, nn.Linear))
        return (f"LoopMLP: [obs {self.obs_dim} + memory {self.H}] -> {layers} -> [u {self.act_dim} + memory {self.H}]"
                f", {n} parameters")


MODELS = {"loop_mlp": LoopMLP}


def build_model(cfg, obs_dim, act_dim):
    cfg = dict(cfg)
    kind = cfg.pop("type", "loop_mlp")
    return MODELS[kind](obs_dim, act_dim, **cfg)
