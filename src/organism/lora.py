"""Minimal LoRA (Hu et al. 2022) in plain torch: frozen base Linear + trainable low-rank update.

y = W x + (alpha / r) * B(A(x)),  A: in->r (Kaiming init), B: r->out (zero init, so the adapter starts as identity).
Adapters can be switched off globally to obtain the frozen base model's outputs (used for the KL regulariser),
without a second copy of the weights. Only A and B are trained and saved (tens of MB)."""
from __future__ import annotations
import math
import torch
import torch.nn as nn

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_ENABLED = True


def set_adapters(enabled: bool):
    global _ENABLED
    _ENABLED = enabled


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        y = self.base(x)
        if not _ENABLED:
            return y
        upd = (x.to(torch.float32) @ self.A.t()) @ self.B.t()
        return y + (self.scale * upd).to(y.dtype)


def add_lora(model, r=16, alpha=32.0):
    n = 0
    for _, mod in list(model.named_modules()):
        for name, child in list(mod.named_children()):
            if name in TARGETS and isinstance(child, nn.Linear):
                setattr(mod, name, LoRALinear(child, r, alpha)); n += 1
    # freeze everything except the adapters (embeddings, norms and lm_head stay fixed)
    for pname, p in model.named_parameters():
        p.requires_grad_(pname.endswith(".A") or pname.endswith(".B"))
    return n


def lora_state(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k.endswith(".A") or k.endswith(".B")}


def load_lora_state(model, state):
    missing = model.load_state_dict(state, strict=False)
    return missing
