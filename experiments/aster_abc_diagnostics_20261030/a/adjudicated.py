"""Export actual adjudications and collect non-age keys without the age-only validator."""
import argparse
import copy
import json
import time
from pathlib import Path
import torch
from .common import BASE,SEED,read_jsonl,write_json,source_mean_ci

def export_triplets(candidates,adjudications):
    from experiments.aster_manuscript_clinical_20261030.data import tree_guideline
    by_id={r['candidate_id']:r for r in candidates};rows=[];excluded=[]
    for a in adjudications:
        if a.get('status')!='physician_adjudicated':
            excluded.append({'candidate_id':a.get('candidate_id'),'reason':'not physician adjudicated'});continue
        if not a.get('eligible_for_direction') or a.get('ancestors_reachable') is not True:
            excluded.append({'candidate_id':a['candidate_id'],'reason':'ancestor_not_reachable_or_ineligible','adjudication_retained':a});continue
        labels=a['labels']
        if {labels['a'],labels['b']}!={'true','false'} or labels['missing']!='unknown':
            excluded.append({'candidate_id':a['candidate_id'],'reason':'not an eligible true-false-unknown triplet'});continue
        c=by_id[a['candidate_id']];guide,mapping=tree_guideline(c['source_tree']);positive=next(k for k in ('a','b') if labels[k]=='true');negative='b' if positive=='a' else 'a'
        root_rules=guide['groups'][0]['rules'];yes,no=root_rules[0]['rule_id'],root_rules[1]['rule_id'];texts=a['patient_texts']
        for target in (yes,no):
            states={'positive':texts[positive],'negative':texts[negative],'unknown':texts['missing']}
            if target==no:states['positive'],states['negative']=states['negative'],states['positive']
            rows.append({'kind':'physician_adjudicated_clinical_counterfactual','pair_id':a['candidate_id']+':'+target,'source_id':a['source_id'],
                'split':a['partition'],'guideline':copy.deepcopy(guide),'target_rule_id':target,'states':states,
                'question':'请依据完整指南给出患者的临床决策路径及最终处置。最后输出 PATH: 经过的规则编号；FINAL: 终点处置的L编号。',
                'verification':{'verifier':'independent human clinician adjudication: '+','.join(a['reviewer_ids']),'reference':a['evidence'],
                    'positive_rule_id':target,'negative_rule_id':no if target==yes else yes,'unknown_label':'NONE','ancestors_reachable':True,'independent_of_classifier':True},
                'adjudication':a,'source_guideline_id':c['source_guideline_id'],'source_tree':c['source_tree'],'tree_mapping':mapping,'clinical_feature':c['proposed_measurement']})
    return rows,excluded

def prepare(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);rows,excluded=export_triplets(read_jsonl(args.candidates),read_jsonl(args.adjudications))
    with (out/'pairs.jsonl').open('x') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    write_json(out/'status.json',{'status':'complete','exported_triplets':len(rows),'source_clusters':len({r['source_id'] for r in rows}),'excluded':excluded,'labels_invented':False})

def collect(args):
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from src.aster.manuscript_runtime import PromptBuilder,ManuscriptBackbone,aligned_triplet
    rows=[r for r in read_jsonl(args.pairs) if r['split']==args.split]
    if not rows:raise ValueError('no eligible physician-adjudicated input pairs; no model loaded')
    if any(r.get('kind')!='physician_adjudicated_clinical_counterfactual' or not r.get('adjudication',{}).get('eligible_for_direction') for r in rows):raise ValueError('input must be exported eligible adjudicated rows')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);expert=json.loads(Path(args.expert).read_text());heads=expert['selected_heads'];layers=sorted({int(h.split(':')[0]) for h in heads})
    torch.set_num_threads(4);torch.manual_seed(SEED);start=time.time();tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True);builder=PromptBuilder(tok,chat_template_kwargs={'enable_thinking':False})
    model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').to('cuda:0').eval();backend=ManuscriptBackbone(model,model_id=expert['identity']['model_id'],model_revision=expert['identity']['revision'],layers=layers)
    records=[];failures=[]
    for index,row in enumerate(rows):
        try:
            target,prompts=aligned_triplet(builder,row);snapshots={s:backend.extract(p,target,keys=True,queries=False) for s,p in prompts.items()}
            payload={'row':row,'offline':{s:{tuple(map(int,h.split(':'))):snap.keys[tuple(map(int,h.split(':')))] for h in heads} for s,snap in snapshots.items()}};torch.save(payload,out/f'{index}.pt')
            for name in heads:
                h=tuple(map(int,name.split(':')));v=torch.tensor(expert['candidates'][name]['direction'],dtype=torch.float64);p,n,u=(payload['offline'][s][h] for s in ['positive','negative','unknown'])
                records.append({'pair_id':row['pair_id'],'source_id':row['source_id'],'head':name,'clinical_feature':row['clinical_feature'],'directed_projection':float(((p-n)@v).mean()),'unknown_centroid_projection':float((((p+n)/2-u)@v).mean())})
        except Exception as exc:
            failures.append({'pair_id':row['pair_id'],'source_id':row['source_id'],'reason':type(exc).__name__+': '+str(exc)})
            if isinstance(exc,torch.cuda.OutOfMemoryError):torch.cuda.empty_cache()
    with (out/'records.jsonl').open('x') as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    write_json(out/'summary.json',{'status':'complete','split':args.split,'submitted_pairs':len(rows),'sources':len({r['source_id'] for r in rows}),'completed_pairs':len({r['pair_id'] for r in records}),'technical_failures':failures,
        'per_head':{h:{k:source_mean_ci([r for r in records if r['head']==h],k) for k in ['directed_projection','unknown_centroid_projection']} for h in heads},
        'elapsed_seconds':time.time()-start,'peak_memory_gb':torch.cuda.max_memory_allocated()/1e9,'frozen_age_direction_zero_shot_on_new_feature':True,'parameters_selected':False})

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--candidates',required=True);a.add_argument('--adjudications',required=True);a.add_argument('--output',required=True)
    b=sub.add_parser('collect');b.add_argument('--pairs',required=True);b.add_argument('--split',default='confirm');b.add_argument('--expert',default=str(BASE/'analysis/expert.json'));b.add_argument('--model',default=str(BASE/'models/Qwen3-8B-2602814'));b.add_argument('--output',required=True)
    args=p.parse_args();prepare(args) if args.command=='prepare' else collect(args)
if __name__=='__main__':main()
