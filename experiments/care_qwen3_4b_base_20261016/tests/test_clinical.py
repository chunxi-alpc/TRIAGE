"""Clinical source-rule, grouping, span and saved model-output verification."""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_clinical_pairs import classify
@pytest.mark.parametrize('kind,value,label',[
 ('a1c',5.7,'Prediabetes'),('a1c',6.5,'Diabetes range'),('a1c',5.6,'Normal glycemia'),
 ('fpg',100,'Prediabetes'),('fpg',126,'Diabetes range'),('fpg',99,'Normal glycemia'),
 ('bmi',18.5,'Healthy weight'),('bmi',25,'Overweight'),('bmi',30,'Obesity'),
 ('gfr',45,'G3a'),('gfr',30,'G3b'),('gfr',15,'G4'),('gfr',14,'G5'),
 ('bp_clinic',79,'Below 140/90 mmHg'),('bp_clinic',80,'Below 150/90 mmHg'),
 ('bp_home',79,'Below 135/85 mmHg'),('bp_home',80,'Below 145/85 mmHg'),
 ('pe',4,'PE unlikely'),('pe',4.5,'PE likely'),('dvt',1,'DVT unlikely'),('dvt',2,'DVT likely'),
 ('crb',0,'Low risk'),('crb',1,'Intermediate risk'),('crb',3,'High risk'),
 ('curb',1,'Low risk'),('curb',2,'Intermediate risk'),('curb',3,'High risk'),
 ('centor',2,'Do not offer an antibiotic prescription'),('centor',3,'Consider an immediate or a back-up prescription'),
 ('feverpain',1,'Do not offer an antibiotic prescription'),('feverpain',2,'Consider no antibiotic or a back-up prescription'),
 ('feverpain',4,'Consider an immediate or a back-up prescription'),
 ('af',0,'Do not offer stroke-prevention anticoagulation'),('af',1,'Consider a direct-acting oral anticoagulant'),
 ('af',2,'Offer a direct-acting oral anticoagulant'),('bone',-1,'Normal bone density'),
 ('bone',-2.4,'Low bone mass'),('bone',-2.5,'Osteoporosis range'),
 ('anaemia',120,'No anaemia'),('anaemia',110,'Mild anaemia'),('anaemia',80,'Moderate anaemia'),('anaemia',79,'Severe anaemia'),
 ('copd',80,'Stage 1 (mild)'),('copd',50,'Stage 2 (moderate)'),('copd',30,'Stage 3 (severe)'),('copd',29,'Stage 4 (very severe)'),
 ('hypo',70,'No biochemical hypoglycemia'),('hypo',54,'Level 1 hypoglycemia'),('hypo',53,'Level 2 hypoglycemia')
])
def test_reference_threshold_boundaries(kind,value,label):
    assert classify(kind,value)==label
def test_clinical_pairs_have_fixed_rules_and_options():
    rows=[json.loads(x) for x in (ROOT/'data/clinical/candidates.jsonl').read_text().splitlines()]
    assert len(rows)==384 and len({r['pair_id'] for r in rows})==192
    for pid in {r['pair_id'] for r in rows}:
        a,b=[r for r in rows if r['pair_id']==pid]
        assert a['guidance']==b['guidance'] and a['option_map']==b['option_map']
        assert a['outer_fold']==b['outer_fold'] and a['group_id']==b['group_id']
        assert {a['state'],b['state']}=={'app','non'}
        assert a['gold_answer']!=b['gold_answer']
        for r in [a,b]:
            assert r['option_map'][r['gold_answer']]==classify(r['kind'],r['value'])
            assert r['prompt'][slice(*r['spans']['action'])]==r['target_action']
            assert r['spans']['action'][1]<r['prompt'].index('\nOptions:')
            assert r['source_verified'] and not r['counterfactual_guideline']
def test_all_clinical_target_token_boundaries():
    from transformers import AutoTokenizer
    from run_experiments import tokenize_span
    tok=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-4B-Base')
    rows=[json.loads(x) for x in (ROOT/'data/clinical/candidates.jsonl').read_text().splitlines()]
    for r in rows:
        enc,idx,_=tokenize_span(tok,r['prompt'],r['spans']['action'])
        for prefix in ['',f"Question: {r['query']}\n","Question: What type of patient information is being presented?\n"]:
            text=prefix+'Context: '+r['patient']+'\nClinical guidance: '+r['guidance']
            start=text.index(r['target_action'])
            tokenize_span(tok,text,[start,start+len(r['target_action'])])
@pytest.mark.skipif(not (ROOT/'results/obs1_clinical/manifest.json').exists(),reason='clinical model run pending')
def test_clinical_real_screening_and_family_holdout():
    r=ROOT/'results/obs1_clinical'; s=pd.read_csv(r/'prior_screen.csv')
    assert len(s)==384
    expected=s.source_verified&s.stable&(s.prior_prediction!=s.gold_answer)
    assert np.array_equal(expected,s.eligible)
    m=json.loads((r/'manifest.json').read_text())
    for f in m['folds']:
        assert not set(f['test_families'])&(set(f['train_families'])|set(f['dev_families']))
        assert not set(f['train_families'])&set(f['dev_families'])
    p=pd.read_csv(r/'predictions.csv'); t=p[p.role=='test']
    assert len(t)==1152 and t.groupby(['case_id','method']).size().eq(1).all()
    assert t.groupby('case_id').prior_eligible.nunique().eq(1).all()
    assert set(t.method)=={'baseline','seka','pasta'}

