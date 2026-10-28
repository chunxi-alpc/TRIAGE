#!/usr/bin/env python3
"""Frozen guide-prefix causal pilot, following the current Module 2 protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np

from scripts.run_module2_mechanism import (context_key, jsonable, sha256,
                                          source_fingerprint, write_json)
from src.module2.mechanism_extraction import tokenize_online


CONDITIONS=[
    ('baseline','none',0.),('projected','both',1.),('projected','both',4.),
    ('projected','enhance',4.),('projected','suppress',4.),
    ('raw_mean','both',4.),('random','both',4.),('reversed','both',4.),
    ('direct_bias','both',4.),
]


def compare(reference,actual,*,logit_atol,attention_atol):
    import torch
    if (reference['logits'].shape!=actual['logits'].shape
            or not bool(torch.isfinite(reference['logits']).all())
            or not bool(torch.isfinite(actual['logits']).all())):
        raise ValueError('Full/suffix logits must have the same shape and be finite')
    le=float(torch.max(torch.abs(reference['logits']-actual['logits'])))
    ae=0.;qe=0.
    errors={}
    for name,field in [('attention_mass','per_query_token_mass'),('query_response','scores')]:
        if len(reference[name])!=len(actual[name]):
            raise ValueError(f'{name} diagnostic coverage differs')
        error=0.
        for r,a in zip(reference[name],actual[name]):
            if any(r[k]!=a[k] for k in ('layer','kv_head','rule_id','positions')):
                raise ValueError('Diagnostic target identities differ')
            rv,av=np.asarray(r[field]),np.asarray(a[field])
            if rv.shape!=av.shape or not np.isfinite(rv).all() or not np.isfinite(av).all():
                raise ValueError('Diagnostic arrays must have identical shapes and be finite')
            error=max(error,float(np.abs(rv-av).max()))
        errors[name]=error
    ae,qe=errors['attention_mass'],errors['query_response']
    if le>logit_atol or ae>attention_atol or qe>attention_atol:
        raise RuntimeError(f'Full/suffix-tail comparison failed: logits={le}, attention={ae}, query={qe}')
    return {'logit_max_abs_error':le,'attention_max_abs_error':ae,'query_max_abs_error':qe,
            'logit_atol':logit_atol,'attention_atol':attention_atol}


def plan(method,mode,gain,heads,directions,beta,gold):
    shifts=[];bias=[]
    if method=='baseline':
        return shifts,bias
    for layer,head in heads:
        for rule in ('A','B'):
            if mode=='enhance' and rule!=gold:
                continue
            if mode=='suppress' and rule==gold:
                continue
            sign=1. if rule==gold else -1.
            common={'layer':layer,'kv_head':head,'rule_id':rule}
            if method=='direct_bias':
                bias.append(dict(common,bias=sign*gain*float(beta[layer,head])))
            else:
                shifts.append(dict(common,coefficient=sign*gain,
                                   delta=directions[method][layer,head]))
    return shifts,bias


def run(args):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from src.module3.guideline_prefix import (prepare_guideline_prefix,
        prepare_guideline_tail,run_guideline_tail,run_guideline_suffix)
    start=time.monotonic();source=Path(args.source);out=source/'causal'
    if json.loads((source/'status.json').read_text())['status']!='complete':
        raise ValueError('Representation experiment must complete before causal execution')
    out.mkdir(exist_ok=True)
    (out/'checkpoints').mkdir(exist_ok=True)
    selection=json.loads((source/'selection.json').read_text())
    heads=[tuple(map(int,h)) for h in selection['selected_heads']]
    beta=np.asarray(selection['bias_unit_rms'],float)
    saved=np.load(source/'operators.npz')
    directions={'projected':saved['delta'],'raw_mean':saved['raw_unit'],
                'random':saved['random_direction'],'reversed':-saved['delta']}
    rows=[json.loads(line) for line in Path(args.triplets).read_text().splitlines()]
    canonical=[r for r in rows if r['split']=='confirm' and r['target_rule_id']=='A']
    if len(canonical)!=12:
        raise ValueError('Expected the complete frozen 12 confirmation configurations')
    fingerprint=json.loads((source/'fingerprint.json').read_text())
    if sha256(args.triplets)!=fingerprint['data_sha256']:
        raise ValueError('Representation and causal input differ')
    for name,meta in fingerprint['model_files'].items():
        if sha256(Path(args.model)/name)!=meta['sha256']:
            raise ValueError('Frozen model/tokenizer changed')
    signature={'schema_version':'module2.mechanism.causal.v1',
        'operators_sha256':sha256(source/'operators.npz'),
        'selection_sha256':sha256(source/'selection.json'),
        'representation_fingerprint_sha256':sha256(source/'fingerprint.json'),
        'source':source_fingerprint(['src/module3/guideline_prefix.py',
            'src/module2/mechanism_extraction.py','src/module2/extraction.py',
            'scripts/run_module2_mechanism_causal.py']),
        'conditions':[dict(method=m,mode=mode,gain=g) for m,mode,g in CONDITIONS],
        'heads':heads,'unique_prompts':24,'expected_records':24*len(CONDITIONS),
        'oracle_gains':True,'actual_m1_used':False,'threads':args.threads,
        'tail_validation':'baseline/zero exact; projected4,reverse4,direct4 each physical prompt',
        'full_prefix_logit_atol':1e-4,'tail_logit_atol':1e-5,'tail_attention_atol':1e-6,
        'causal_scope':'ID+action guide keys; all patient/question suffix tokens; full selected head set'}
    if (out/'fingerprint.json').exists():
        if json.loads((out/'fingerprint.json').read_text())!=json.loads(json.dumps(signature,default=jsonable)):
            raise ValueError('Refuse to resume under a different causal fingerprint')
    else:
        write_json(out/'fingerprint.json',signature)
    write_json(out/'conditions.json',signature['conditions'])
    torch.set_num_threads(args.threads)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,
                    dtype=torch.float32,attn_implementation='eager').eval()
    labels=fingerprint['candidate_token_ids']
    targets=[{'layer':l,'kv_head':h,'rule_id':r} for l,h in heads for r in ('A','B')]
    responses=[dict(t,delta=directions['projected'][t['layer'],t['kv_head']]) for t in targets]
    records=[];validations=[];done=0
    expected_keys=set()
    for row in canonical:
        case=context_key(row)
        for variant in ('positive','negative'):
            for m,mode,g in CONDITIONS:
                expected_keys.add((case,variant,m,mode,g))
    for ci,row in enumerate(canonical):
        case=context_key(row)
        prefix=None;original=None
        for vi,variant in enumerate(('positive','negative')):
            cp=out/'checkpoints'/f'prompt_{ci:02d}_{vi}.json'
            if cp.exists():
                previous=json.loads(cp.read_text());records.extend(previous['records']);validations.extend(previous['validations']);done+=1
                continue
            tokenized=tokenize_online(tokenizer,row,variant)
            ids=tokenized['encoded']['input_ids']
            if prefix is None:
                prefix=prepare_guideline_prefix(model,ids,tokenized['prefix_length'],tokenized['rule_positions'])
                original=[(k.clone(),v.clone()) for k,v in prefix.cache.to_legacy_cache()]
            elif prefix.rule_positions!=tokenized['rule_positions']:
                raise ValueError('Opposite patients must share guideline mapping')
            if heads:
                tail=prepare_guideline_tail(model,prefix,ids,min(l for l,h in heads),
                    diagnostic_targets=targets,response_targets=responses)
                baseline=tail.baseline
                def execute(shifts=(),bias=()):
                    return run_guideline_tail(model,tail,shifts,score_bias=bias,
                        diagnostic_targets=targets,response_targets=responses)
            else:
                baseline=run_guideline_suffix(model,prefix,ids)
                def execute(shifts=(),bias=()):
                    return run_guideline_suffix(model,prefix,ids,shifts,score_bias=bias)
            with torch.inference_mode():
                full=model(input_ids=ids,attention_mask=torch.ones_like(ids),
                           use_cache=False,logits_to_keep=1).logits[0,-1].float().cpu()
            full_error=float((full-baseline['logits']).abs().max())
            if not np.isfinite(full_error) or full_error>1e-4:
                raise RuntimeError(f'Full/guide-split baseline error {full_error}')
            zero_shifts=[dict(t,delta=directions['projected'][t['layer'],t['kv_head']],coefficient=0.) for t in targets]
            zero=execute(zero_shifts)
            checks=[dict(case_id=case,variant=variant,condition='zero',
                full_split_error=full_error,**compare(baseline,zero,logit_atol=0.,attention_atol=0.))]
            local=[]
            for method,mode,gain in CONDITIONS:
                shifts,bias=plan(method,mode,gain,heads,directions,beta,row['oracle'][variant]['selected_rule_id'])
                actual=baseline if method=='baseline' else execute(shifts,bias)
                if (method,mode,gain) in {('projected','both',4.),('reversed','both',4.),('direct_bias','both',4.)}:
                    reference=run_guideline_suffix(model,prefix,ids,shifts,score_bias=bias,
                        diagnostic_targets=targets,response_targets=responses)
                    checks.append(dict(case_id=case,variant=variant,condition=method,
                        **compare(reference,actual,logit_atol=1e-5,attention_atol=1e-6)))
                masses={r:[x['target_attention_mass'] for x in actual['attention_mass'] if x['rule_id']==r] for r in ('A','B')}
                logits=actual['logits'][labels].tolist()
                gold=row['oracle'][variant]['selected_rule_id']
                pred='A' if logits[0]>=logits[1] else 'B'
                rec={'case_id':case,'triplet_id':row['id'],'group_id':row['group_id'],
                    'variant':variant,'patient_state':row['patient_facts'][variant][row['decisive_node_index']],
                    'gold_rule':gold,'method':method,'mode':mode,'gain':gain,
                    'mass_A':float(np.mean(masses['A'])) if masses['A'] else None,
                    'mass_B':float(np.mean(masses['B'])) if masses['B'] else None,
                    'logits_A':float(logits[0]),'logits_B':float(logits[1]),
                    'prediction':pred,'correct':pred==gold,
                    'action_permutation':row['controls']['action_permutation'],
                    'rule_order_index':row['controls']['rule_order_index'],
                    'attention_mass':actual['attention_mass'],'query_response':actual['query_response'],
                    'prefix_length':actual['prefix_length'],'suffix_length':actual['suffix_length'],
                    'empty_expert':not bool(heads)}
                local.append(rec)
            for (oldk,oldv),(k,v) in zip(original,prefix.cache.to_legacy_cache()):
                if not torch.equal(oldk,k) or not torch.equal(oldv,v):
                    raise RuntimeError('A patient/condition changed the shared original guideline cache')
            write_json(cp,{'records':local,'validations':checks})
            records.extend(local);validations.extend(checks);done+=1
            write_json(out/'status.json',{'status':'running','prompts_completed':done,
                'expected_prompts':24,'records':len(records),'expected_records':216,
                'elapsed_seconds':time.monotonic()-start})
            print(f'causal {done}/24 | records={len(records)}/216 | elapsed={time.monotonic()-start:.1f}s',flush=True)
    actual_keys=[(r['case_id'],r['variant'],r['method'],r['mode'],r['gain']) for r in records]
    if len(actual_keys)!=len(set(actual_keys)) or set(actual_keys)!=expected_keys:
        raise RuntimeError('Causal coverage differs from the frozen complete grid')
    with (out/'records.jsonl').open('w') as f:
        for r in records:
            f.write(json.dumps(r,ensure_ascii=False,allow_nan=False,default=jsonable)+'\n')
    write_json(out/'validation.json',validations)
    write_json(out/'status.json',{'status':'complete','prompts_completed':done,
        'records':len(records),'expected_records':216,'missing_records':0,
        'selected_heads':heads,'validation_checks':len(validations),
        'elapsed_seconds':time.monotonic()-start})
    print('causal complete',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--source',default='experiments/module2_mechanism_20261028/results')
    p.add_argument('--triplets',default='experiments/module2_mechanism_20261028/inputs/triplets.jsonl')
    p.add_argument('--threads',type=int,default=4)
    run(p.parse_args())


if __name__=='__main__':
    main()
