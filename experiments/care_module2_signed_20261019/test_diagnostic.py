"""Numerical tests for intervention coordinates and the sign ambiguity of B4."""
from types import SimpleNamespace
import numpy as np
import torch
from diagnostic import score_geometry,calibrate
from support import contrastive_basis
def test_local_score_delta_matches_explicit_rotated_qk():
    torch.manual_seed(7);n,d=11,128;idx=[2,3,4];h=2
    q=torch.randn(n,32,d);k=torch.randn(n,8,d)
    angles=torch.arange(n)[:,None]*torch.linspace(.001,.2,d)[None,:]
    co,si=angles.cos(),angles.sin()
    model=SimpleNamespace(config=SimpleNamespace(hidden_size=4096),
        model=SimpleNamespace(rotary_emb=lambda x,p:(co[None],si[None])))
    selected,fn=score_geometry(model,{0:dict(q=q,k=k)},idx,(0,h))
    changed=selected+torch.randn_like(selected)*.03
    from transformers.models.qwen3.modeling_qwen3 import rotate_half
    qr=q*co[:,None]+rotate_half(q)*si[:,None]
    kb=selected*co[idx]+rotate_half(selected)*si[idx]
    ka=changed*co[idx]+rotate_half(changed)*si[idx]
    actual=[]
    # Match flattening order: query head, causal query position, action token.
    for head in range(4*h,4*h+4):
        for t in range(n):
            for j,s in enumerate(idx):
                if t>=s:
                    actual.append((torch.dot(qr[t,head],ka[j])-torch.dot(qr[t,head],kb[j]))/(d**.5))
    assert torch.max(torch.abs(fn(changed)-torch.stack(actual)))<2e-6
    direction=torch.randn_like(selected)
    calibrated,audit=calibrate(selected,direction,fn,.05)
    assert abs(float(fn(calibrated).square().mean().sqrt())-.05)<1e-6
    assert audit['relative_error']<1e-5
def test_projector_loses_sign_but_signed_mean_preserves_it():
    rng=np.random.default_rng(8);signal=rng.normal(size=(30,128));background=rng.normal(size=(25,128))
    a=contrastive_basis(signal,background,4);b=contrastive_basis(-signal,background,4)
    P=a@a.T;Q=b@b.T
    assert np.max(abs(P-Q))<1e-9
    assert np.max(abs(P@P-P))<1e-9
    assert np.max(abs(P-P.T))<1e-9
    assert np.allclose((-signal).mean(0),-signal.mean(0))

