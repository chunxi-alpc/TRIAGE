import contextlib
import sys
from pathlib import Path
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'aster_action_mass_seed0_20261025'))
from action_engine import ActionEngine
from mass_operator import action_scores, group_roles
from sparse_attention import sparse_attention

GEOMETRIES = {
    'midlate4': {'layers': [18, 22, 26, 30], 'heads': [0, 8, 16, 24]},
    'late4': {'layers': [28, 30, 32, 34], 'heads': [0, 8, 16, 24]},
    'midlate2': {'layers': [18, 22, 26, 30], 'heads': [0, 16]},
}

def configurations():
    result = {}
    for geometry in GEOMETRIES:
        result['zero_' + geometry] = dict(geometry=geometry, strength=0.)
        result['dynamic_' + geometry] = dict(geometry=geometry, strength=1.)
    result['dynamic_half'] = dict(geometry='midlate4', strength=.5)
    result['misaligned'] = dict(geometry='midlate4', strength=1., control='misaligned')
    result['reverse'] = dict(geometry='midlate4', strength=-1.)
    result['sign_flip'] = dict(geometry='midlate4', strength=1., control='sign_flip')
    result['static_sparse'] = dict(geometry='midlate4', strength=1., kind='static')
    return result

class SparseEngine(ActionEngine):
    @contextlib.contextmanager
    def intervene(self, roles, phase_start, mode, p=None, lam=1., layers='all', capture=None, **controls):
        if mode != 'sparse':
            with super().intervene(roles, phase_start, mode, p, lam, layers, capture, **controls):
                yield
            return
        geometry = GEOMETRIES[controls['geometry']]
        if max(geometry['layers']) >= self.nl:
            raise ValueError('geometry does not fit the model')
        groups = group_roles(self.current_case, roles)
        _, support = action_scores(self.current_case, p)
        if controls.get('control') == 'misaligned' and support:
            support = support[-1:] + support[:-1]
        strength = controls['strength']
        if controls.get('control') == 'sign_flip':
            strength *= controls['case_sign']
        old = self.mod.eager_attention_forward
        def wrapped(module, q, k, v, attention_mask, scaling, dropout=0., **kw):
            if module.layer_idx in geometry['layers']:
                out = sparse_attention(q, k, v, attention_mask, scaling, groups, support,
                    geometry['heads'], phase_start, strength, 2., controls.get('kind', 'dynamic'))
            else:
                repeat = q.shape[1] // k.shape[1]
                out = F.scaled_dot_product_attention(q, k.repeat_interleave(repeat, 1),
                    v.repeat_interleave(repeat, 1), attn_mask=attention_mask,
                    is_causal=attention_mask is None, dropout_p=0., scale=scaling)
            return out.transpose(1, 2).contiguous(), None
        self.mod.eager_attention_forward = wrapped
        try:
            yield
        finally:
            self.mod.eager_attention_forward = old
