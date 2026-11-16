"""CPU-only diagnosis of frozen train tensors and dev outputs; never loads a model."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,collections,csv,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(x) for x in p.read_text().splitlines()]
def metric(task,rows,field):
    groups=collections.defaultdict(list)
    for x in rows:groups[x['case_id'] if task=='N2' else x['family_id']].append(x[field])
    return float(np.mean([all(v) if task in ('N1','N2') else np.mean(v) for v in groups.values()]))
def stats(v):
    return {'n':len(v),'min':float(np.min(v)),'median':float(np.median(v)),'max':float(np.max(v))} if v else {'n':0}
def run(root,out,model):
    out.mkdir(parents=True,exist_ok=True)
    originals=[p for t in ('N1','N2','N3','N5') for p in (root/t).rglob('*') if p.is_file()]
    before={str(p):sha(p) for p in originals}
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    evidence={'new_model_forwards':0,'new_training':False,'test_prediction_content_analyzed':False,'test_files_hashed_only':True,'token_ids':{s:tok.encode(s,add_special_tokens=False) for k in 'ABCDEFGH' for s in (k,' '+k)},'tasks':{}}
    aggregate=[];case_records=[]
    for task in ('N1','N2','N3','N5'):
        rows=lines(root/task/'dev_predictions.jsonl');base={(x['case_id'],json.dumps(x['control'],sort_keys=True)):x for x in rows if x['method']=='baseline'}
        axes=torch.load(root/task/'axes.pt',map_location='cpu',weights_only=True);ae={}
        for name,a in axes.items():
            if 'mean' not in a:continue
            u=a['basis'].double();m=a['mean'].double();d=a['differences'].double();dc=d-d.mean(0);s=a['singular_values'].double()
            ae[name]={'rank':u.shape[1],'mean_norm':float(m.norm()),'projected_mean_norm':float((u@(u.T@m)).norm()),
              'mean_length_retained':float((u@(u.T@m)).norm()/m.norm()),'centered_variance_retained':float(s[:u.shape[1]].square().sum()/dc.square().sum()),
              'difference_rows':len(d),'difference_mean_centroid_mismatch_norm':float((d.mean(0)-m).norm())}
        groups=collections.defaultdict(list)
        for row in rows:groups[(row['method'],row['beta'])].append(row)
        task_rows=[]
        for (method,beta),xs in groups.items():
            augmented=[];rel=[];requested=[];actual=[];qrel=[];centered=[];bias=[];nonzero=0;silent=0;format_invalid=0;output_disagreement=0;corrected=damaged=flips=0;rawflips=rawcorrected=rawdamaged=0
            for x in xs:
                expected=x['expected'];expected=[expected] if isinstance(expected,str) else expected
                raw=x.get('top_unconstrained_token'); normalized=raw.strip() if raw is not None else None
                valid=normalized in x['option_labels'] if normalized is not None else False
                if raw is not None:format_invalid+=not valid;output_disagreement+=normalized!=x['prediction']
                y={**x,'top_token_correct':bool(valid and normalized in expected)};augmented.append(y)
                b=base[(x['case_id'],json.dumps(x['control'],sort_keys=True))]
                flips+=x['prediction']!=b['prediction'];corrected+=x['correct'] and not b['correct'];damaged+=not x['correct'] and b['correct']
                br=b['top_unconstrained_token'].strip();bx=b['expected'];bx=[bx] if isinstance(bx,str) else bx
                bc=br in bx
                if raw is not None:
                    rawflips+=normalized!=br;rawcorrected+=y['top_token_correct'] and not bc;rawdamaged+=not y['top_token_correct'] and bc
                hook=x.get('hook',{});rs=hook.get('residual',[]);bs=hook.get('attention',[])
                nonzero+=any(z['actual_norm']>0 for z in rs) or any(z['actual_score_rms']>0 for z in bs)
                silent+=not rs and not bs
                for z in rs:
                    rel.append(z['relative_norm']);requested.append(z['requested_norm']);actual.append(z['actual_norm']);qrel.append(z['quantization_error']/max(1e-12,z['requested_norm']*len(z['positions'])**.5))
                bias.extend(z['actual_signed_mean'] for z in bs)
                if x.get('candidate_logits'):
                    d=np.array([x['candidate_logits'][k]-b['candidate_logits'][k] for k in x['candidate_logits']]);centered.append(float(abs(d-d.mean()).max()))
                if raw is not None:
                    case_records.append({'task':task,'method':method,'beta':beta,'case_id':x['case_id'],'control':x['control'],'expected':expected,'scored_prediction':x['prediction'],
                     'original_correct':x['correct'],'recorded_top_token':raw,'top_token_normalized':normalized,'top_token_valid':valid,'top_token_correct':y['top_token_correct']})
            rec={'task':task,'method':method,'beta':beta,'n':len(xs),'original_correct':sum(x['correct'] for x in xs),'original_primary':metric(task,xs,'correct'),
             'top_token_correct':sum(x['top_token_correct'] for x in augmented) if xs[0].get('top_unconstrained_token') is not None else None,
             'top_token_primary':metric(task,augmented,'top_token_correct') if xs[0].get('top_unconstrained_token') is not None else None,
             'top_token_invalid':format_invalid,'prediction_disagreements':output_disagreement,'original_prediction_flips':flips,'original_corrected':corrected,'original_damaged':damaged,
             'top_token_prediction_flips':rawflips,'top_token_corrected':rawcorrected,'top_token_damaged':rawdamaged,
             'nonzero_hook_rows':nonzero,'no_hook_rows':silent,'relative_residual_per_edit':stats(rel),'requested_norm_per_edit':stats(requested),'actual_norm_per_edit':stats(actual),
             'rounding_error_relative_per_edit':stats(qrel),'centered_bare_candidate_logit_change':stats(centered),'actual_score_bias_values':sorted(set(bias))}
            task_rows.append(rec);aggregate.append(rec)
        selection=read(root/task/'dev_selection.json');recomputed={}
        for method in selection['selected']:
            possibilities={x['beta']:x['original_primary'] for x in task_rows if x['method']==method}
            best=max(possibilities,key=lambda b:(possibilities[b],-b));assert best==selection['selected'][method]
            recomputed[method]=best
        te={'axes':ae,'selection_reproduced':True,'selected':recomputed,'baseline_raw_top_tokens':dict(collections.Counter(x['top_unconstrained_token'] for x in base.values())),
          'baseline_top_is_bare_candidate':sum(x['unconstrained_is_candidate'] for x in base.values()),'baseline_records':len(base),'grid':task_rows}
        if task=='N2':
            us=axes['source']['basis'].double();separation=float((us@(us.T@axes['source']['mean'].double())).norm());by=collections.defaultdict(dict)
            for x in rows:
                if x['method']=='route' and x['beta']==1:by[x['case_id']][x['control']['source']]=x
            te['policy_separation']={cid:{'projected_C_P_distance':separation,'common_shift_norm':max(0,sum(v[s]['hook']['residual'][0]['requested_norm']**2 for s in ('C','P'))/2-separation**2/4)**.5,
              'C_recorded_top':v['C']['top_unconstrained_token'],'P_recorded_top':v['P']['top_unconstrained_token']} for cid,v in by.items()}
        if task=='N5':
            capture=read(root/task/'train_fit.json')['captures'];p=[x for x in capture if x['state']=='T'];n=[x for x in capture if x['state']=='F'];pairs=list(zip(p,n))
            te['unmatched_fit']={'T':len(p),'F':len(n),'covariance_rows':len(pairs),'F_unused_in_covariance':max(0,len(n)-len(p)),
               'same_case_pairs':sum(a['case_id']==b['case_id'] for a,b in pairs),'same_rule_pairs':sum(a['rule_id']==b['rule_id'] for a,b in pairs),
               'rule_state_counts':{str(k):v for k,v in collections.Counter((x['state'],x['rule_id']) for x in capture).items()}}
        evidence['tasks'][task]=te
    assert before=={str(p):sha(p) for p in originals}
    evidence['all_original_stage_files_unchanged']=True;evidence['original_files_hashed']=len(before)
    (out/'analysis.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    (out/'input_sha256.json').write_text(json.dumps(before,indent=2)+'\n')
    with (out/'per_case_dev_diagnostics.jsonl').open('w') as f:
        for x in case_records:f.write(json.dumps(x,ensure_ascii=False)+'\n')
    fields=['task','method','beta','n','original_correct','original_primary','top_token_correct','top_token_primary','top_token_invalid','prediction_disagreements','original_prediction_flips','original_corrected','original_damaged','top_token_prediction_flips','top_token_corrected','top_token_damaged','nonzero_hook_rows','no_hook_rows']
    with (out/'dev_strength_grid.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(aggregate)
    print(json.dumps({'status':'offline_analysis_complete','new_forwards':0,'original_files_unchanged':len(before),'dev_grid_rows':len(aggregate),'per_case_records':len(case_records)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);p.add_argument('--tokenizer',default='/home/gcx/care_20261016/CARE_Codex_Pilot/models/Qwen3-4B-Base');a=p.parse_args();run(Path(a.run),Path(a.out),a.tokenizer)
