#!/usr/bin/env python3
"""Real, token-level Module 2 mechanism measurements; no synthetic activations."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.module2.extraction import encode_prompt, validate_qwen3
from src.module2.mechanism_extraction import (ROLES, VARIANTS, capture_tokens,
    prompt_hash, row_metadata, token_pair_second_moment, tokenize_online)
from src.module2.mechanism import fit_operators, evaluate_direction, select_heads
from src.module2.minimal_pairs import audit_triplets


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def write_json(path, obj):
    p = Path(path)
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=jsonable,
                              allow_nan=False)+'\n')
    tmp.replace(p)


def context_key(row):
    c = row['controls']
    return f"{row['pair_cluster_id']}:swap{c['action_permutation']}:order{c['rule_order_index']}"


def contexts(rows):
    result = OrderedDict()
    for i, row in enumerate(rows):
        result.setdefault(context_key(row), []).append(i)
    for indices in result.values():
        if {rows[i]['target_rule_id'] for i in indices} != {'A', 'B'} or len(indices) != 2:
            raise ValueError('Every context needs both opposite target orientations')
    return result


def source_fingerprint(paths):
    return {p: sha256(ROOT/p) for p in paths}


def extract(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    started = time.monotonic()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out/'checkpoints'
    checkpoint.mkdir(exist_ok=True)
    rows = [json.loads(line) for line in Path(args.triplets).read_text().splitlines()]
    data_audit = audit_triplets(rows)
    write_json(out/'data_audit.json', data_audit)
    metadata = [row_metadata(r) for r in rows]
    write_json(out/'metadata.json', metadata)
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                dtype=torch.float32, attn_implementation='eager').eval()
    geometry = validate_qwen3(model)
    labels = [tokenizer.encode(t, add_special_tokens=False) for t in ('A', 'B')]
    if any(len(x) != 1 for x in labels) or labels[0] == labels[1]:
        raise ValueError('A/B must be distinct single tokens')
    candidate_ids = [x[0] for x in labels]
    signature = {
        'model_id':'Qwen/Qwen3-0.6B-Base',
        'model_revision':'da87bfb608c14b7cf20ba1ce41287e8de496c0cd',
        'model_files':{p.name:{'sha256':sha256(p),'bytes':p.stat().st_size}
                       for p in sorted(Path(args.model).glob('*'))
                       if p.suffix in {'.json','.safetensors','.txt'}},
        'data_sha256':sha256(args.triplets), 'geometry':geometry,
        'source':source_fingerprint(['src/module2/mechanism_extraction.py',
            'src/module2/extraction.py','src/module2/minimal_pairs.py',
            'src/module2/mechanism.py','scripts/run_module2_mechanism.py']),
        'torch':torch.__version__, 'transformers':transformers.__version__,
        'numpy':np.__version__, 'threads':args.threads,'dtype':'float32',
        'rank':2,'max_heads':4,'seed':261028,'roles':ROLES,
        'offline_order':'P-Q-G','online_order':'G-P-Q',
        'candidate_labels':['A','B'],'candidate_token_ids':candidate_ids,
    }
    if (out/'fingerprint.json').exists():
        if json.loads((out/'fingerprint.json').read_text()) != json.loads(json.dumps(signature,default=jsonable)):
            raise ValueError('Refuse resuming checkpoints under a different fingerprint')
    else:
        write_json(out/'fingerprint.json',signature)
    L,H,D = geometry['layers'],geometry['kv_heads'],geometry['head_dim']
    shape=(len(rows),3,L,H,D)
    keys=np.zeros(shape,np.float32); offline_q=np.zeros(shape,np.float32)
    offline_mass=np.zeros(shape[:-1],np.float32)
    offline_logits=np.zeros((len(rows),3,2),np.float32)
    omega=np.zeros((L,H,D,D),np.float64)
    audits=[]; forward_count=0; fit_count=0
    grouped=contexts(rows)
    for ci,(case,indices) in enumerate(grouped.items()):
        cp=checkpoint/f'offline_{ci:03d}.npz'
        if cp.exists():
            cached=np.load(cp)
            for name,dest in [('keys',keys),('queries',offline_q),('mass',offline_mass),('logits',offline_logits)]:
                dest[indices]=cached[name]
            if 'moment' in cached.files:
                omega+=cached['moment']; fit_count+=len(indices)
            audits.extend(json.loads(cp.with_suffix('.json').read_text()))
            forward_count+=3
        else:
            canonical=next(rows[i] for i in indices if rows[i]['target_rule_id']=='A')
            encoded={v:encode_prompt(tokenizer,canonical['prompts'][v],canonical['spans'][v],ROLES)
                     for v in VARIANTS}
            reference=encoded['positive']
            for v in VARIANTS:
                e=encoded[v]
                if e['length']!=reference['length'] or e['rule_positions']!=reference['rule_positions']:
                    raise ValueError(f'{case}: three-state token positions/length mismatch')
                for r,positions in e['rule_positions'].items():
                    if not torch.equal(e['encoded']['input_ids'][0,positions],
                                       reference['encoded']['input_ids'][0,positions]):
                        raise ValueError(f'{case}: paired target token IDs mismatch')
            captures={v:capture_tokens(model,encoded[v],candidate_ids) for v in VARIANTS}
            forward_count+=3
            moment=np.zeros_like(omega) if canonical['split']=='fit' else None
            local_audits=[]
            for i in indices:
                row=rows[i]; rule=row['target_rule_id']
                order=VARIANTS if rule=='A' else ('negative','positive','neutral')
                for vi,v in enumerate(order):
                    if row['prompts'][VARIANTS[vi]] != canonical['prompts'][v]:
                        raise ValueError('Sibling target orientations do not share physical prompts')
                    c=captures[v]['rules'][rule]
                    keys[i,vi]=c['token_keys'].mean(1)
                    offline_q[i,vi]=c['effective_query'];offline_mass[i,vi]=c['attention_mass']
                    offline_logits[i,vi]=captures[v]['candidate_logits']
                if moment is not None:
                    moment+=token_pair_second_moment(captures[order[0]]['rules'][rule]['token_keys'],
                                                    captures[order[1]]['rules'][rule]['token_keys'])
                    fit_count+=1
                local_audits.append({'id':row['id'],'case_id':case,'aligned':True,
                    'token_length':reference['length'],'target_positions':reference['rule_positions'][rule],
                    'target_token_ids':reference['encoded']['input_ids'][0,reference['rule_positions'][rule]].tolist(),
                    'prompt_sha256':{v:prompt_hash(row['prompts'][v]) for v in VARIANTS}})
            arrays={'keys':keys[indices],'queries':offline_q[indices],
                    'mass':offline_mass[indices],'logits':offline_logits[indices]}
            if moment is not None:
                omega+=moment;arrays['moment']=moment
            np.savez_compressed(cp,**arrays);write_json(cp.with_suffix('.json'),local_audits)
            audits.extend(local_audits)
        write_json(out/'status.json',{'status':'running','stage':'offline',
            'contexts_completed':ci+1,'contexts_total':len(grouped),'real_forwards':forward_count,
            'elapsed_seconds':time.monotonic()-started})
        print(f'offline {ci+1}/{len(grouped)} | forwards={forward_count} | elapsed={time.monotonic()-started:.1f}s',flush=True)
    fit_mask=np.array([r['split']=='fit' for r in rows])
    if fit_count != int(fit_mask.sum()):
        raise ValueError('Fit moment weighting mismatch')
    omega/=fit_count
    fit=fit_operators(keys[fit_mask,0]-keys[fit_mask,1],omega,rank=2,
                      metadata=[m for m,b in zip(metadata,fit_mask) if b])
    rng=np.random.default_rng(261028)
    random=rng.normal(size=fit['delta'].shape)
    random=random/np.linalg.norm(random,axis=-1,keepdims=True)*np.linalg.norm(fit['delta'],axis=-1,keepdims=True)
    directions={'projected':fit['delta'],'raw_mean':fit['raw_unit'],
                'random':random,'reversed':-fit['delta']}
    operators={k:v for k,v in fit.items() if isinstance(v,np.ndarray)}
    operators['random_direction']=random
    np.savez_compressed(out/'operators.npz',**operators)
    write_json(out/'operator_metadata.json',{k:v for k,v in fit.items() if not isinstance(v,np.ndarray)})
    write_json(out/'offline_audit.json',audits)
    online_q=np.full((len(rows),2,L,H,D),np.nan,np.float32)
    online_mass=np.full((len(rows),2,L,H),np.nan,np.float32)
    online_logits=np.full((len(rows),2,2),np.nan,np.float32)
    rms={n:np.full((len(rows),2,L,H),np.nan,np.float32) for n in directions}
    pq={n:np.full((len(rows),2,L,H,geometry['queries_per_kv_head']),np.nan,np.float32) for n in directions}
    tp={n:np.full((len(rows),2,L,H),np.nan,np.float32) for n in directions}
    online_audits=[]; online_forwards=0
    for split in ('select','confirm'):
        split_contexts=[(ci,case,indices) for ci,(case,indices) in enumerate(grouped.items()) if rows[indices[0]]['split']==split]
        for step,(ci,case,indices) in enumerate(split_contexts):
            cp=checkpoint/f'online_{ci:03d}.npz'
            if cp.exists():
                cached=np.load(cp)
                online_q[indices]=cached['queries'];online_mass[indices]=cached['mass'];online_logits[indices]=cached['logits']
                for n in directions:
                    rms[n][indices]=cached['rms_'+n];pq[n][indices]=cached['pq_'+n];tp[n][indices]=cached['tp_'+n]
                online_audits.extend(json.loads(cp.with_suffix('.json').read_text()))
                online_forwards+=2
            else:
                canonical=next(rows[i] for i in indices if rows[i]['target_rule_id']=='A')
                es={v:tokenize_online(tokenizer,canonical,v) for v in VARIANTS}
                ref=es['positive'];pl=ref['prefix_length']
                for v,e in es.items():
                    if pl!=e['prefix_length'] or not torch.equal(ref['encoded']['input_ids'][:,:pl],e['encoded']['input_ids'][:,:pl]):
                        raise ValueError('G prefix must be identical for all patient states')
                cs={v:capture_tokens(model,es[v],candidate_ids,directions=directions) for v in VARIANTS[:2]}
                online_forwards+=2
                local=[]
                for i in indices:
                    row=rows[i];rule=row['target_rule_id']
                    order=('positive','negative') if rule=='A' else ('negative','positive')
                    key_change=float(np.abs(cs['positive']['rules'][rule]['token_keys']-cs['negative']['rules'][rule]['token_keys']).max())
                    if key_change != 0:
                        raise ValueError('Guide-first keys unexpectedly depend on later patient facts')
                    for vi,v in enumerate(order):
                        c=cs[v]['rules'][rule]
                        online_q[i,vi]=c['effective_query'];online_mass[i,vi]=c['attention_mass'];online_logits[i,vi]=cs[v]['candidate_logits']
                        for n in directions:
                            rms[n][i,vi]=c['unit_rms'][n];pq[n][i,vi]=c['per_query'][n];tp[n][i,vi]=c['token_positive'][n]
                    local.append({'id':row['id'],'case_id':case,'prefix_length':pl,'prefix_identical':True,
                        'online_key_change_max':key_change,'token_length':ref['length'],
                        'rule_positions':ref['rule_positions'],
                        'prompt_sha256':{v:prompt_hash(es[v]['prompt']) for v in VARIANTS}})
                arrays={'queries':online_q[indices],'mass':online_mass[indices],'logits':online_logits[indices]}
                for n in directions:
                    arrays.update({'rms_'+n:rms[n][indices],'pq_'+n:pq[n][indices],'tp_'+n:tp[n][indices]})
                np.savez_compressed(cp,**arrays);write_json(cp.with_suffix('.json'),local);online_audits.extend(local)
            write_json(out/'status.json',{'status':'running','stage':'online_'+split,
                'contexts_completed':step+1,'contexts_total':len(split_contexts),
                'offline_forwards':forward_count,'online_forwards':online_forwards,
                'elapsed_seconds':time.monotonic()-started})
            print(f'online {split} {step+1}/{len(split_contexts)} | forwards={online_forwards} | elapsed={time.monotonic()-started:.1f}s',flush=True)
        mask=np.array([r['split']==split for r in rows])
        evaluation=evaluate_direction(fit['delta'],keys[mask],online_q[mask],
            [m for m,b in zip(metadata,mask) if b],fit_group_ids=fit['fit_group_ids'])
        np.savez_compressed(out/f'{split}_evaluation.npz',**{k:v for k,v in evaluation.items() if isinstance(v,np.ndarray)})
        if split=='select':
            selected=select_heads(fit,evaluation,max_heads=4)
            beta=np.sqrt(np.mean(rms['projected'][mask].astype(np.float64)**2,axis=(0,1)))
            write_json(out/'selection.json',{'selected_heads':selected,'selection_split':'select',
                'confirmation_used':False,'rule':'current main Eq.7; no extra post-hoc thresholds',
                'bias_unit_rms':beta,'rank':2,'max_heads':4})
            print('selection frozen: '+json.dumps(selected),flush=True)
    measurements={'offline_keys':keys,'offline_effective_queries':offline_q,
        'online_effective_queries':online_q,'offline_attention_mass':offline_mass,
        'online_attention_mass':online_mass,'offline_candidate_logits':offline_logits,
        'online_candidate_logits':online_logits,'fit_token_second_moment':omega}
    for n in directions:
        measurements.update({'online_unit_rms_'+n:rms[n], 'online_per_query_'+n:pq[n],
                             'online_token_positive_'+n:tp[n]})
    np.savez_compressed(out/'measurements.npz',**measurements)
    write_json(out/'online_audit.json',online_audits)
    write_json(out/'status.json',{'status':'complete','stage':'representation',
        'offline_forwards':forward_count,'online_forwards':online_forwards,
        'triplets':len(rows),'selected_heads':selected,
        'first_layer_offline_difference_max':float(np.abs(keys[:,0,0]-keys[:,1,0]).max()),
        'elapsed_seconds':time.monotonic()-started})
    print('representation complete',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--triplets',default='experiments/module2_mechanism_20261028/inputs/triplets.jsonl')
    p.add_argument('--output',default='experiments/module2_mechanism_20261028/results')
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    extract(args)


if __name__=='__main__':
    main()
