"""Frozen M2 directions; train only input-dependent coordinates and gates."""
import math
import torch
from torch import nn

class HeadEditor(nn.Module):
    def __init__(self, u, s):
        super().__init__()
        self.register_buffer('U', u.float().clone())
        self.register_buffer('S', s.float().clone())
        self.mapping = nn.Linear(u.shape[0], u.shape[1], bias=False)
        self.gate = nn.Linear(2*u.shape[1], 1)
        with torch.no_grad():
            self.mapping.weight.copy_(u.T)
            self.gate.weight.zero_()
            self.gate.bias.fill_(math.log(.01/.99))

    def forward(self, z, r, force_zero=False):
        current = z.float() @ self.U
        target = self.mapping(r.detach().float().reshape_as(z))
        gate = self.gate(torch.cat((current,target),dim=-1)).sigmoid()
        delta = gate * ((target-current) @ self.U.T)
        if force_zero:
            delta = delta*0
        return (z.float()+delta).to(z.dtype), dict(
            magnitude=delta.square().sum()/z.float().square().sum().detach().clamp_min(1e-8),
            signal_change=(delta @ self.S).square().sum(), gate=gate.mean())

class LearnableM3(nn.Module):
    def __init__(self, frozen, n_heads=32):
        super().__init__()
        self.editors = nn.ModuleDict({str(h):HeadEditor(op['U_N'],op['S']) for h,op in frozen.items()})
        self.heads_by_layer = {}
        for h in frozen:
            self.heads_by_layer.setdefault(int(h)//n_heads,[]).append(int(h))
    def edit(self, head, z, r, force_zero=False):
        return self.editors[str(head)](z,r,force_zero)
