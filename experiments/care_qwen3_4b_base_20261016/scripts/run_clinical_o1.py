#!/usr/bin/env python3
"""Source-grounded clinical O1: real model, permissive screening, family-held-out folds."""
from __future__ import annotations
import argparse, csv, datetime, hashlib, importlib.metadata, json, os, pathlib, random, subprocess, sys, time
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_experiments import Runner, tokenize_span, causal_inputs, dump
from build_clinical_pairs import classify
from src.model.projection_builder_base import ProjectionBuilderBase
R=ROOT/'results/obs1_clinical'
SEED=261016
def log(s): print(datetime.datetime.now(datetime.timezone.utc).isoformat(),s,flush=True)
class ClinicalBuilder(ProjectionBuilderBase):
    """Only prompt assembly/extraction adapts; official SVD and intervention are unchanged."""
    def __init__(self,model,tok,rows):
        self.model=model; self.tokenizer=tok; self.rows=rows
        self.layers=list(range(26,36)); self.device='cuda'; self.model_path='Qwen3-4B-Base'
        self.top_pct=.9; self.min_diff=0.; self.feature=None
    def iter_examples(self): return iter(self.rows)
    def get_triplets(self,row):
        return [(row['patient']+'\nClinical guidance: '+row['guidance'],row['query'],row['target_action'],
                'What type of patient information is being presented?')]
    def extract(self):
        buffers=[[[[] for _ in range(8)] for _ in self.layers] for _ in range(3)]
        audit=[]
        for row in self.rows:
            ctx,rel,ans,irr=self.get_triplets(row)[0]
            texts=[f'Context: {ctx} ',f'Question: {rel}\nContext: {ctx}',f'Question: {irr}\nContext: {ctx}']
            ids=[]
            for c,text in enumerate(texts):
                start=text.index(ans); enc,idx,_=tokenize_span(self.tokenizer,text,[start,start+len(ans)])
                ids.append(enc['input_ids'][0,idx].tolist())
                keys={}; handles=[]
                for L in self.layers:
                    def hook(mod,inp,out,L=L): keys[L]=out[0,idx].float().cpu().clone()
                    handles.append(self.model.model.layers[L].self_attn.k_norm.register_forward_hook(hook))
                try: self.model(**causal_inputs(enc,self.model.device))
                finally:
                    for h in handles: h.remove()
                for li,L in enumerate(self.layers):
                    for kv in range(8): buffers[c][li][kv].append(keys[L][:,kv,:])
                audit.append({'case_id':row['case_id'],'family':row['group_id'],'condition':c,'text':text,'target_indices':idx,'target_ids':ids[-1]})
            assert ids[0]==ids[1]==ids[2]
        return buffers,audit
    def build(self,out):
        buffers,audit=self.extract()
        self._compute_and_save_traditional(*buffers,len(self.layers),8,str(out))
        for name in ['pos','neg']:
            data=torch.load(out/f'Qwen3-4B-Base_{name}_proj.pt',weights_only=True)
            assert data['layers']==self.layers
            torch.save(data,out/f'last10_{name}.pt')
        torch.save({'H':buffers[0],'H_relevant':buffers[1],'H_irrelevant':buffers[2]},out/'builder_keys.pt')
        dump(out/'span_audit.json',audit)
def letter_probabilities(model,tok,text):
    # Identical controlled candidate likelihood rule as contextual Runner.
    enc=tok(text,add_special_tokens=False,return_tensors='pt')
    logits=model(**causal_inputs(enc,model.device)).logits[0,-1].float()
    ids=[tok.encode(' '+x,add_special_tokens=False)[0] for x in 'ABCD']
    vals=logits[ids]; probs=vals.softmax(-1).cpu().tolist()
    ordered=sorted(probs,reverse=True)
    return {'prediction':'ABCD'[int(vals.argmax())],'candidate_probabilities':probs,
            'margin':ordered[0]-ordered[1],'candidate_mass':float(logits.softmax(-1)[ids].sum()),'n_tokens':enc['input_ids'].shape[1]}
def score_record(row,meta,method,strength,fold,role,prior):
    pred=meta['prediction']; pr=prior[row['case_id']]
    return {'case_id':row['case_id'],'pair_id':row['pair_id'],'group_id':row['group_id'],'rule_id':row['rule_id'],
        'outer_fold':fold,'role':role,'method':method,'strength':strength,'subset':row['subset'],'applicable':row['applicable'],
        'gold_answer':row['gold_answer'],'gold_label':row['gold_label'],'prediction':pred,
        'correct':int(pred==row['gold_answer']),
        'misapplied':int(not row['applicable'] and pred==row['inapplicable_action']),
        'prior_reversion':int(pred==pr['prior_prediction']),'prior_prediction':pr['prior_prediction'],
        'prior_eligible':pr['eligible'],'prior_stable':pr['stable'],'complete_pair_eligible':pr['complete_pair_eligible'],
        'source_verified':True,'source_type':row['source_type'],'patient_type':'simulated','counterfactual_guideline':False,
        'model':'Qwen/Qwen3-4B-Base','seed':SEED,'source_url':row['source_url'],
        'candidate_probabilities':meta['candidate_probabilities'],'full_vocabulary_candidate_mass':meta['full_vocabulary_candidate_mass'],
        'n_tokens':meta['n_tokens'],'target_token_indices':meta['target_token_indices'],'target_token_ids':meta['target_token_ids']}
def metric_h(records):
    d=pd.DataFrame(records)
    a=d[d.subset=='app'].correct.mean(); b=d[d.subset=='non'].correct.mean()
    return float(2*a*b/(a+b)) if a+b else 0.
def main():
    os.chdir(ROOT); R.mkdir(exist_ok=True)
    p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true'); args=p.parse_args()
    if args.smoke:
        global_r=R
        # Smoke files stay separate; no smoke rows enter the main statistics.
        out=R/'smoke'; out.mkdir(exist_ok=True)
    else: out=R
    start=time.time(); random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    all_rows=[json.loads(x) for x in (ROOT/'data/clinical/candidates.jsonl').read_text().splitlines()]
    protocol=json.loads((ROOT/'data/clinical/protocol.json').read_text())
    rows=all_rows
    if args.smoke:
        families=sorted({x['group_id'] for x in rows})[:3]
        rows=[r for r in rows if r['group_id'] in families and r['pair_index']==0]
    model_path=ROOT/'models/Qwen3-4B-Base'
    tok=AutoTokenizer.from_pretrained(model_path)
    model=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    cfg=model.config
    assert (cfg.num_hidden_layers,cfg.num_attention_heads,cfg.num_key_value_heads,cfg.head_dim)==(36,32,8,128)
    log(f'model ready; cases={len(rows)}, smoke={args.smoke}')
    prior={}; screening=[]
    # Screening is frozen before any intervention is evaluated.
    for i,row in enumerate(rows):
        assert classify(row['kind'],row['value'])==row['gold_label']
        a=letter_probabilities(model,tok,row['closedbook_prompt'])
        b=letter_probabilities(model,tok,row['paraphrase_prompt'])
        stable=a['prediction']==b['prediction']; eligible=bool(row['source_verified'] and stable and a['prediction']!=row['gold_answer'])
        rec={'case_id':row['case_id'],'pair_id':row['pair_id'],'group_id':row['group_id'],'rule_id':row['rule_id'],
             'outer_fold':row['outer_fold'],'subset':row['subset'],'gold_answer':row['gold_answer'],'gold_label':row['gold_label'],
             'prior_prediction':a['prediction'],'paraphrase_prediction':b['prediction'],'stable':stable,'eligible':eligible,
             'source_verified':True,'prior_probabilities':a['candidate_probabilities'],'paraphrase_probabilities':b['candidate_probabilities'],
             'prior_margin':a['margin'],'paraphrase_margin':b['margin'],'prior_candidate_mass':a['candidate_mass'],
             'exclusion_reason':'' if eligible else 'unstable closedbook argmax' if not stable else 'closedbook agrees with clinical gold'}
        prior[row['case_id']]=rec; screening.append(rec)
        if i%64==0: log(f'closedbook screening {i+1}/{len(rows)}')
    pair_ok={pid:all(r['eligible'] for r in screening if r['pair_id']==pid) for pid in {r['pair_id'] for r in screening}}
    for rec in screening: rec['complete_pair_eligible']=pair_ok[rec['pair_id']]
    pd.DataFrame(screening).to_csv(out/'prior_screen.csv',index=False)
    dump(out/'screening_frozen.json',{'cases':len(rows),'eligible':sum(x['eligible'] for x in screening),
        'eligible_app':sum(x['eligible'] and x['subset']=='app' for x in screening),
        'eligible_non':sum(x['eligible'] and x['subset']=='non' for x in screening),
        'complete_eligible_pairs':sum(pair_ok.values()),'no_probability_threshold':True,
        'selection_depends_on_intervention_outputs':False,
        'data_sha256':hashlib.sha256((ROOT/'data/clinical/candidates.jsonl').read_bytes()).hexdigest(),
        'screen_sha256':hashlib.sha256((out/'prior_screen.csv').read_bytes()).hexdigest()})
    log(f'screening frozen eligible={sum(r["eligible"] for r in screening)}, both-side pairs={sum(pair_ok.values())}')
    if args.smoke:
        # All three methods plus the actual official builder are exercised on six patients.
        train=rows[:2]; proj=out/'projection'; proj.mkdir(exist_ok=True)
        ClinicalBuilder(model,tok,train).build(proj); runner=Runner(model,tok,proj)
        records=[]
        for row in rows:
            for method in ['baseline','seka','pasta']:
                meta,_=runner.forward(row,method,.5 if method!='baseline' else 0)
                records.append(score_record(row,meta,method,.5 if method!='baseline' else 0,-1,'smoke',prior))
        pd.DataFrame(records).to_csv(out/'predictions.csv',index=False)
        dump(out/'complete.json',{'cases':len(rows),'records':len(records),'seconds':time.time()-start,
             'peak_memory_allocated_GiB':torch.cuda.max_memory_allocated()/2**30})
        log('SMOKE COMPLETE'); return
    records=[]; fold_manifest=[]
    # Cache unmodified contextual predictions once; source family remains held out for every method.
    baseline={}
    runner0=Runner(model,tok,ROOT/'results/projections')
    for row in rows:
        meta,_=runner0.forward(row); baseline[row['case_id']]=meta
    dump(out/'baseline_probabilities.json',baseline)
    for fold in range(3):
        test=[r for r in rows if r['outer_fold']==fold]
        rest=sorted({r['group_id'] for r in rows if r['outer_fold']!=fold})
        rng=random.Random(SEED+fold); rng.shuffle(rest)
        dev_families=set(rest[:2]); train_families=set(rest[2:]); test_families={r['group_id'] for r in test}
        assert not (train_families&test_families or dev_families&test_families or train_families&dev_families)
        dev=[r for r in rows if r['group_id'] in dev_families]
        train=[r for r in rows if r['group_id'] in train_families and r['pair_index']<2]
        proj=out/f'fold{fold}/projections'; proj.mkdir(parents=True,exist_ok=True)
        ClinicalBuilder(model,tok,train).build(proj)
        runner=Runner(model,tok,proj)
        fold_info={'fold':fold,'train_families':sorted(train_families),'dev_families':sorted(dev_families),
            'test_families':sorted(test_families),'projection_case_ids':[r['case_id'] for r in train],
            'dev_case_ids':[r['case_id'] for r in dev],'test_case_ids':[r['case_id'] for r in test],
            'selection':{},'dev_scores':{}}
        for row in dev: records.append(score_record(row,baseline[row['case_id']],'baseline',0.,fold,'dev',prior))
        for row in test: records.append(score_record(row,baseline[row['case_id']],'baseline',0.,fold,'test',prior))
        for method in ['seka','pasta']:
            scores=[]
            for strength in protocol['strengths'][method]:
                batch=[]
                for row in dev:
                    meta,_=runner.forward(row,method,strength)
                    rec=score_record(row,meta,method,strength,fold,'dev',prior); records.append(rec); batch.append(rec)
                score=metric_h(batch); scores.append((score,-strength,strength))
                fold_info['dev_scores'][f'{method}:{strength}']=score
            chosen=max(scores)[2]; fold_info['selection'][method]=chosen
            for row in test:
                meta,_=runner.forward(row,method,chosen)
                records.append(score_record(row,meta,method,chosen,fold,'test',prior))
            log(f'fold {fold} {method} chosen={chosen}; held-out cases={len(test)}')
        dump(out/f'fold{fold}/manifest.json',fold_info); fold_manifest.append(fold_info)
        pd.DataFrame(records).to_csv(out/'predictions.csv',index=False)
    frame=pd.DataFrame(records); test=frame[frame.role=='test']
    assert len(test)==len(rows)*3
    assert test.groupby(['case_id','method']).size().eq(1).all()
    # Demonstrate that steering was active even where the selected candidate remained unchanged.
    effects={}
    base=test[test.method=='baseline'].set_index('case_id')
    for method in ['seka','pasta']:
        d=test[test.method==method].set_index('case_id').loc[base.index]
        a=np.stack(base.candidate_probabilities); b=np.stack(d.candidate_probabilities)
        effects[method]={'answer_changes':int((d.prediction!=base.prediction).sum()),
            'mean_abs_probability_change':float(np.abs(a-b).mean()),'max_abs_probability_change':float(np.abs(a-b).max())}
        assert effects[method]['max_abs_probability_change']>0
    old=json.loads((ROOT/'results/manifest.json').read_text())
    manifest={'model':'Qwen/Qwen3-4B-Base','model_revision':old['model_revision'],'seka_commit':old['seka_commit'],
        'seka_official_unmodified':subprocess.check_output(['git','-C',str(ROOT/'external/SEKA'),'status','--porcelain'],text=True).strip()=='',
        'versions':{x:importlib.metadata.version(x) for x in ['torch','transformers','numpy','pandas','scipy']},
        'seed':SEED,'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'gpu':torch.cuda.get_device_name(),'dtype':'bfloat16',
        'seconds':time.time()-start,'peak_memory_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
        'peak_memory_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,
        'cases':len(rows),'pairs':len(rows)//2,'clinical_rule_families':len({r['group_id'] for r in rows}),
        'data_sha256':hashlib.sha256((ROOT/'data/clinical/candidates.jsonl').read_bytes()).hexdigest(),
        'scope':'source-grounded clinical rules; simulated patient vignettes; no false guideline counterfactuals',
        'eligibility':protocol['eligibility'],'protocol':protocol,'folds':fold_manifest,'intervention_effects':effects,
        'layers':list(range(26,36)),'kv_heads':list(range(8)),'query_heads':list(range(32)),
        'selection':'controlled single-token conditional probabilities space+A/B/C/D; same options for all methods and pair members',
        'seka_operator':'official positive+negative combination (strength P_pos + 0.25 strength P_neg)/2, min_diff=0, top_pct=.9',
        'pasta_operator':'official unmodified PASTA exclude alpha=exp(-strength), explicit common causal mask'}
    dump(out/'manifest.json',manifest)
    log(f'CLINICAL MODEL RUN COMPLETE seconds={time.time()-start:.1f}')
if __name__=='__main__':
    with torch.inference_mode(): main()

