import math
import torch
from torch import nn


class ConditionalIntervention(nn.Module):
    def __init__(self, width, rank=4, variant='full'):
        super().__init__()
        self.variant = variant
        q = torch.linalg.qr(torch.randn(width, rank, dtype=torch.float32), mode='reduced').Q
        self.basis = nn.Parameter(q)
        self.mapping = nn.Linear(width, rank, bias=False)
        with torch.no_grad():
            self.mapping.weight.copy_(q.T)
        self.gate = nn.Linear(2 * rank, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(0.01 / 0.99))

    def forward(self, output, blind=None, force_zero=False):
        original = output.float()
        if force_zero:
            return output, original.sum() * 0, original.sum() * 0
        basis = torch.linalg.qr(self.basis, mode='reduced').Q
        current = original @ basis
        source = original if self.variant in ('ordinary', 'context') else blind.detach().float()
        target = self.mapping(source)
        gate = self.gate(torch.cat([current, target], dim=-1)).sigmoid()
        delta = gate * ((target - current) @ basis.T)
        magnitude = delta.square().sum() / original.square().sum().detach().clamp_min(1e-8)
        return (original + delta).to(output.dtype), magnitude, gate.mean()


def objective(logits, base_logits, target, magnitudes, use_js=True):
    ce = torch.stack([torch.nn.functional.cross_entropy(z[None], target[None]) for z in logits]).mean()
    p, q = [z.softmax(-1) for z in logits]
    middle = (p + q) / 2
    js = sum((v * (v.clamp_min(1e-12).log() - middle.clamp_min(1e-12).log())).sum() for v in (p, q)) / 2
    preserve = []
    for edited, base in zip(logits, base_logits):
        if int(base.argmax()) == int(target):
            pb = base.detach().softmax(-1)
            preserve.append((pb * (pb.clamp_min(1e-12).log() - edited.log_softmax(-1))).sum())
    keep = torch.stack(preserve).mean() if preserve else ce * 0
    magnitude = torch.stack(magnitudes).mean()
    total = ce + (js if use_js else js * 0) + 0.1 * keep + 0.001 * magnitude
    return total, {'ce': float(ce.detach()), 'js': float(js.detach()),
                   'preserve': float(keep.detach()), 'magnitude': float(magnitude.detach())}
