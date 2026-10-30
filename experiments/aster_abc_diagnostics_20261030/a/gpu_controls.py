"""Actual-model age-direction nuisance controls on an independent output path."""
import argparse
import copy
import json
import time
from pathlib import Path
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from src.aster.manuscript_runtime import PromptBuilder,ManuscriptBackbone,aligned_triplet
from experiments.aster_manuscript_clinical_20261030.data import clinical_validator
from .common import BASE,SEED,read_jsonl,write_json,source_mean_ci,cosine,unit

VARIANTS=('base','paraphrase','action_swap','irrelevant_a','irrelevant_b','position_control')

def variant(row,name):
    out=copy.deepcopy(row)
    if name=='paraphrase':
        out['states']={s:p.replace('Age at current encounter:','Age documented at current visit:') for s,p in out['states'].items()}
    elif name=='action_swap':
        rules=out['guideline']['groups'][0]['rules']
        if len(rules)!=2:raise ValueError('root action swap requires exactly two siblings')
        rules[0]['action'],rules[1]['action']=rules[1]['action'],rules[0]['action']
    elif name in ('irrelevant_a','irrelevant_b'):
        identifier='ALPHA' if name=='irrelevant_a' else 'OMEGA'
        out['states']={s:p+'\nAdministrative record identifier: '+identifier+'.' for s,p in out['states'].items()}
    elif name=='position_control':
        out['states']={s:'Clinical record for this encounter.\n'+p for s,p in out['states'].items()}
    elif name!='base':raise ValueError(name)
    out['variant']=name
    return out

def analyze(output,expert):
    records=[torch.load(p,map_location='cpu',weights_only=False) for p in sorted(Path(output).glob('*.pt'))]
    rows=[];failures=[];counter=0
    for r in records:
        variants=r['variants'];base=variants.get('base')
        if base is None:
            failures.append({'pair_id':r['row']['pair_id'],'reason':'baseline extraction failed'});continue
        for name in expert['selected_heads']:
            h=tuple(map(int,name.split(':')));v=torch.tensor(expert['candidates'][name]['direction'],dtype=torch.float64);mu=unit(expert['candidates'][name]['mean'])
            d0=(base['offline']['positive'][h]-base['offline']['negative'][h]).mean(0).double();base_proj=float(d0@v)
            for control,rec in variants.items():
                dc=(rec['offline']['positive'][h]-rec['offline']['negative'][h]).mean(0).double()
                p0=float(dc@v)
                delta={s:float((rec['offline'][s][h].mean(0)-base['offline'][s][h].mean(0))@v) for s in ['positive','negative','unknown']}
                rows.append({'pair_id':r['row']['pair_id'],'source_id':r['row']['source_id'],'head':name,'variant':control,
                    'direction_cosine':cosine(d0,dc),'base_decisive_projection':base_proj,'control_decisive_projection':p0,'control_mean_projection':float(dc@mu),'spectral_minus_mean_projection':float(dc@(v-mu)),
                    'relative_projection_change':None if abs(base_proj)<=1e-12 else (p0-base_proj)/abs(base_proj),
                    'same_state_projected_change':delta,
                    'target_tokens_base':base['target_tokens'],'target_tokens_control':rec['target_tokens'],
                    'offline_total_tokens':rec['tokens'],'semantics':'age-rule control; not new physician gold'})
            if all(k in variants for k in ['irrelevant_a','irrelevant_b']):
                aa,bb=variants['irrelevant_a'],variants['irrelevant_b']
                for s in ['positive','negative','unknown']:
                    nuisance=(aa['offline'][s][h].mean(0)-bb['offline'][s][h].mean(0)).double()
                    rows.append({'pair_id':r['row']['pair_id'],'source_id':r['row']['source_id'],'head':name,'variant':'irrelevant_a_minus_b','state':s,
                        'nuisance_projection_abs':abs(float(nuisance@v)),'decisive_projection_abs':abs(base_proj),
                        'decisive_minus_nuisance_abs':abs(base_proj)-abs(float(nuisance@v)),
                        'nuisance_to_decisive_ratio':None if abs(base_proj)<=1e-12 else abs(float(nuisance@v))/abs(base_proj)})
        failures.extend(r['failures']);counter+=sum(len(v['offline']) for v in variants.values())
    grouped=defaultdict(list)
    for row in rows:grouped[(row['head'],row['variant'])].append(row)
    metrics={}
    for (head,variant_name),group in grouped.items():
        fields=['direction_cosine','relative_projection_change','spectral_minus_mean_projection'] if variant_name!='irrelevant_a_minus_b' else ['nuisance_projection_abs','decisive_minus_nuisance_abs','nuisance_to_decisive_ratio']
        metrics[head+'/'+variant_name]={field:source_mean_ci(group,field) for field in fields}
    summary={'status':'complete','planned_forward_passes':len(records)*len(VARIANTS)*3,'completed_forward_passes':counter,
        'pairs':len(records),'sources':len({r['row']['source_id'] for r in records}),'technical_failures':failures,'metrics':metrics,
        'scope':'exploratory control study on existing age-rule holdout; no general clinical gold and no efficacy inference',
        'comparison_note':'pre-RoPE key means; paraphrase/position controls can also alter upstream contextual positions; action-swap compares mean over changed target token sets',
        'physician_adjudicated_pairs':0}
    write_json(Path(output)/'summary.json',summary)
    with (Path(output)/'records.jsonl').open('w') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',default=str(BASE));p.add_argument('--output',required=True);p.add_argument('--analyze-only',action='store_true');args=p.parse_args()
    base=Path(args.base);out=Path(args.output);expert=json.loads((base/'analysis/expert.json').read_text())
    if args.analyze_only:
        analyze(out,expert);return
    out.mkdir(parents=True,exist_ok=False);start=time.time();torch.set_num_threads(4);torch.manual_seed(SEED)
    rows=[r for r in read_jsonl(base/'review/accepted_pairs.jsonl') if r['split']=='confirm']
    model_path=base/'models/Qwen3-8B-2602814';tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    builder=PromptBuilder(tok,chat_template_kwargs={'enable_thinking':False})
    model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').to('cuda:0').eval()
    layers=sorted({int(x.split(':')[0]) for x in expert['selected_heads']});backend=ManuscriptBackbone(model,model_id=expert['identity']['model_id'],model_revision=expert['identity']['revision'],layers=layers)
    done=[];failures=[];planned=len(rows)*len(VARIANTS)*3
    for index,row in enumerate(rows):
        clinical_validator(row);result={'row':row,'variants':{},'failures':[]}
        for name in VARIANTS:
            changed=variant(row,name)
            try:
                target,prompts=aligned_triplet(builder,changed)
                snapshots={s:backend.extract(prompt,target,keys=True,queries=False) for s,prompt in prompts.items()}
                result['variants'][name]={'offline':{s:{tuple(map(int,h.split(':'))):snap.keys[tuple(map(int,h.split(':')))] for h in expert['selected_heads']} for s,snap in snapshots.items()},
                    'tokens':{s:int(prompt.input_ids.shape[1]) for s,prompt in prompts.items()},'target_tokens':len(prompts['positive'].positions[target])}
            except Exception as exc:
                failure={'pair_id':row['pair_id'],'variant':name,'reason':type(exc).__name__+': '+str(exc)};result['failures'].append(failure);failures.append(failure)
                if isinstance(exc,torch.cuda.OutOfMemoryError):torch.cuda.empty_cache()
            write_json(out/'status.json',{'status':'running','completed_pairs':done,'current_pair':row['pair_id'],'current_variant':name,'planned_forward_passes':planned,'technical_failures':failures,'elapsed_seconds':time.time()-start})
        torch.save(result,out/f'{index}.pt');done.append(row['pair_id'])
    summary=analyze(out,expert);summary.update(elapsed_seconds=time.time()-start,peak_memory_gb=torch.cuda.max_memory_allocated()/1e9,attention_implementation='eager',model_path=str(model_path))
    write_json(out/'summary.json',summary);write_json(out/'status.json',{'status':'complete','pairs':done,'planned_forward_passes':planned,'technical_failures':failures,'elapsed_seconds':time.time()-start,'peak_memory_gb':summary['peak_memory_gb']})
    print(json.dumps({'status':'complete','pairs':len(done),'technical_failures':len(failures),'elapsed_seconds':summary['elapsed_seconds']}))
if __name__=='__main__':main()
