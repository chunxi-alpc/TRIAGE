import copy
import pytest
import torch
from care_m2_v2.n3_mean_basis import mean_preserving_basis, DevEngine

def example():
    mean=torch.tensor([2.,0.,0.,0.])
    variation=torch.tensor([[0.,10.,1.,0.],[0.,-10.,1.,0.],[0.,10.,-1.,0.],[0.,-10.,-1.,0.]])
    return mean+variation,mean

def test_mean_is_retained_when_old_centered_pca_discards_it():
    differences,mean=example();original=differences.clone()
    _,_,v=torch.linalg.svd(differences-differences.mean(0),full_matrices=False)
    old=v[:2].T
    assert torch.allclose(old@(old.T@mean),torch.zeros_like(mean))
    basis,values=mean_preserving_basis(differences,mean)
    assert basis.shape==(4,2) and torch.linalg.matrix_rank(basis)==2
    assert torch.allclose(basis.T@basis,torch.eye(2),atol=1e-6)
    assert torch.allclose(basis@(basis.T@mean),mean,atol=1e-6)
    assert torch.allclose(basis@basis.T,torch.diag(torch.tensor([1.,1.,0.,0.])),atol=1e-6)
    assert torch.equal(differences,original)

def test_pair_order_does_not_change_projection():
    differences,mean=example()
    a,_=mean_preserving_basis(differences,mean)
    b,_=mean_preserving_basis(differences[[2,0,3,1]],mean)
    assert torch.allclose(a@a.T,b@b.T,atol=1e-6)

def test_route_at_one_and_mean_at_half_have_identical_control_component():
    differences,mean=example();u,_=mean_preserving_basis(differences,mean);p=u@u.T
    center=torch.tensor([3.,4.,5.,6.]);h=torch.tensor([-1.,2.,8.,9.])
    shared=[]
    for sign in (-1,1):
        target=center+sign*mean/2
        route=p@(target-h);mean_shift=sign*mean*.5
        shared.append(route-mean_shift)
        assert torch.allclose(route-mean_shift,p@(center-h),atol=1e-6)
    assert torch.allclose(shared[0],shared[1],atol=1e-6)

@pytest.mark.parametrize('case',['zero_mean','no_second_direction','nonfinite','inconsistent_mean'])
def test_invalid_training_geometry_is_not_silently_repaired(case):
    d,m=example()
    if case=='zero_mean': m=torch.zeros_like(m);d=d-d.mean(0)
    if case=='no_second_direction': d=m.repeat(4,1)
    if case=='nonfinite': d[0,0]=float('nan')
    if case=='inconsistent_mean': m=m+1
    with pytest.raises(ValueError):mean_preserving_basis(d,m)

def test_development_engine_forbids_new_capture_and_generation():
    engine=object.__new__(DevEngine)
    with pytest.raises(RuntimeError,match='outside'):engine.capture('train')
    with pytest.raises(RuntimeError,match='outside'):engine.generate('test')

def test_development_allowlist_and_forward_cap(monkeypatch):
    import hashlib
    from care_m2_v2.engine import Engine
    from care_m2_v2.candidates import candidate_contract,score_key
    class Tokenizer:
        def encode(self,text,add_special_tokens=False):return [1,3] if text.endswith(' A') else [1]
        def decode(self,ids):return ' A'
    e=object.__new__(DevEngine);e.tokenizer=Tokenizer();e.modelhash='model';e.calls=48;e.cache={}
    prompt='Answer:';options={'A':'one'}
    e.frozen={'allowed_prompt_sha256':[hashlib.sha256(prompt.encode()).hexdigest()],'max_new_forwards':48}
    monkeypatch.setattr(Engine,'score',lambda *args,**kwargs:{'cache_hit':True})
    with pytest.raises(ValueError,match='development prompts'):e.score('Test Answer:',options)
    with pytest.raises(RuntimeError,match='48-forward'):e.score(prompt,options)
    key=score_key(e.modelhash,prompt,options,{'edits':[],'biases':[]},candidate_contract(e.tokenizer,prompt,options))
    e.cache[key]={'prediction':'A'}
    assert e.score(prompt,options)['cache_hit']
    assert e.calls==48
