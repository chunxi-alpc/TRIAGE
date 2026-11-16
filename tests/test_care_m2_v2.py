import copy
from itertools import product
from care_m2_v2.data import n1,n2,n3,n4,n5,verify,oracle,signature,family
from care_m2_v2.study import correct

def test_oracle_rejects_irrelevant_question():
    ast={'op':'AND','args':['a',{'op':'OR','args':['b','c']}]}
    values={'a':None,'b':True,'c':None}
    possible,attributes=oracle(ast,values)
    assert possible=={False,True}
    assert attributes==['a']

def test_protocol_data_and_exact_join():
    a1,provenance=n5(); rows=n1()+n2()+n3()+n4()+a1
    assert verify(rows)['valid']
    assert len(provenance)==3
    assert all(len(r['signals'])==3 for r in a1)
    assert any(s['signal']==0 for r in a1 for s in r['signals'])

def test_n1_queries_and_one_fact_change():
    rows=n1()
    for i in range(0,len(rows),4):
        a,b,c,d=rows[i:i+4]
        assert a['values']==c['values'] and b['values']==d['values']
        assert sum(a['values'][k]!=b['values'][k] for k in a['values'])==1

def test_no_gold_driven_dq_wording():
    for r in n3():
        assert 'The action is determined' not in r['prompt']
        assert 'The action is not determined' not in r['prompt']
        assert r['options'][r['gold']] in r['prompt']

def test_receiver_scoring_not_donor():
    r=n2()[16]; c={'source':'C'}
    assert correct(r,c,r['gold_by_source']['C'])
    assert not correct(r,c,r['gold_by_source']['P'])
    assert len(set(signature(family(i)) for i in range(24)))==24

def test_dev_never_reads_test_donors(tmp_path):
    import torch
    from care_m2_v2.study import train,select
    from care_m2_inc.io import write_json
    class Tokenizer:
        def encode(self,p): return list(range(len(p.split())))
    class Fake:
        tokenizer=Tokenizer()
        def positions(self,r):return [[1,2] for _ in r['spans']]
        def capture(self,p,positions=None):
            import re
            units=re.findall(r'unit (\d+)',p)
            assert all(int(x)<=16 for x in units),'test donor read during dev selection'
            return torch.arange(8).float()+len(p)%7
        def score(self,p,options,edits=None,biases=None):
            return {'prediction':next(iter(options)),'forward_count':0,'cache_hit':False,'prompt_tokens':len(p.split()),'hook':{}}
    engine=Fake();rows=n2();axes=train(engine,[r for r in rows if r['split']=='train'],tmp_path)
    write_json(tmp_path/'data_manifest.json',{'mock':True})
    select(engine,[r for r in rows if r['split']=='dev'],axes,tmp_path,[r for r in rows if r['split']!='test'])
