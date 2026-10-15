"""Integration checks against persisted real model observations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
R=Path(__file__).resolve().parents[1]/'results'
pytestmark=pytest.mark.skipif(not (R/'manifest.json').exists(),reason='real model outputs not present')
def test_complete_real_splits_and_methods():
    m=json.loads((R/'manifest.json').read_text())
    assert m['model']=='Qwen/Qwen3-4B-Base' and not m['smoke']
    assert m['groups']==60 and m['cases']==360 and m['seka_unmodified']
    dat=np.load(R/'obs2/key_embeddings.npz')
    assert dat['keys'].shape==(60,4,36,8,128)
    assert dat['token_keys'].shape[:3]==(60,6,36)
    assert {s:int((dat['split']==s).sum()) for s in ['train','dev','test']}=={'train':36,'dev':12,'test':12}
    assert set(m['projection_training_groups']).isdisjoint(set(dat['group_ids'][dat['split']=='test']))
    df=pd.read_csv(R/'obs1/predictions.csv')
    t=df[(df.split=='test')&(df.variant=='modified')]
    assert t.groupby('method').size().to_dict()=={'baseline':24,'pasta':24,'seka':24}
    assert not t.prior_eligible.any()
def test_real_rope_and_direction_identity():
    df=pd.read_csv(R/'obs3/logits.csv')
    assert len(df)>0 and len(df.group_id.unique())==12
    assert (df.query_head//4==df.kv_head).all()
    assert set(df.projection_variant)=={'positive_only','official_posneg'}
    assert df[df.weight_type=='continuous'].omega.abs().nunique()>1
    assert np.max(abs(df.a_after-df.a_before-df.delta_a))<1e-6
    assert np.max(abs(df.delta_a-df['lambda']*df.omega*df.z))<1e-4
    assert np.max(abs(df.a_after_bf16-df.a_before_bf16-df.delta_a_bf16))<1e-6
    # Reconstruct actual baseline BF16 attention logits from raw post-RoPE tensors.
    case=df.case_id.iloc[0]
    raw=np.load(R/'obs3/full_qk'/f'{case}.npz')
    row=df[(df.case_id==case)&(df.layer==26)].iloc[0]
    score=np.dot(raw['queries_actual_rope'][0,int(row.query_head)].astype(np.float64),
                 raw['keys_actual_rope'][0,int(row.token_idx),int(row.kv_head)].astype(np.float64))/np.sqrt(128)
    assert abs(score-row.a_before_bf16)<1e-4
def test_official_helper_and_pasta_checks():
    k=json.loads((R/'checks/official_key_extraction.json').read_text())
    p=json.loads((R/'checks/pasta_equivalence.json').read_text())
    assert k['max_abs_error']<1e-6
    assert p['softmax_max_error']<1e-6 and p['mask_max_error']<1e-6

def test_exact_group_permutation_includes_identity_at_float32_precision():
    from scripts.analyze_results import paired_permutation
    assert paired_permutation(np.full(12,.0026357265,dtype=np.float32))==pytest.approx(2/4096)
    assert paired_permutation(np.zeros(12,dtype=np.float32))==1.

