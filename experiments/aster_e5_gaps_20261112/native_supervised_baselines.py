"""Native answer-SFT LoRA and Qwen-adapted LoFiT on frozen E1 endpoints.

Training uses full-vocabulary single-answer-token cross entropy, not binary
margin BCE or any ASTER pair-difference loss. Evaluation preserves the frozen
binary score convention. Existing baselines.Adapter components are reused.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

REPO = Path(__file__).resolve().parents[2]
DATASET = Path('/home/gcx/aster_cr_pair_dataset_20261111/balanced_train.jsonl')
HISTORICAL = Path('/home/gcx/aster_cr_repair_v2_run_20261108')
DEFAULT_OUTPUT = REPO / 'reports/aster_e5_gaps_20261112/native_supervised_e1'
LRS = (0.0001, 0.001, 0.01)
SEEDS = (11, 22, 33)
os.environ.setdefault('ASTER_CR_ROOT', str(HISTORICAL))
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
sys.path.insert(0, str(REPO))


def rows(path):
    if not Path(path).exists():
        return []
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def append(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def job_dir(args, lr=None, seed=None):
    return args.output / ('lr' + str(args.lr if lr is None else lr) + '_seed' + str(args.seed if seed is None else seed))


def prepare(args):
    target = args.output / 'training_endpoints.jsonl'
    info_file = args.output / 'training_manifest.json'
    if info_file.exists() and target.exists():
        existing=read(info_file)
        if existing.get('method')!=args.kind:raise RuntimeError('Output root belongs to another baseline method.')
        return existing
    pairs = rows(args.dataset)
    from collections import Counter
    if len(pairs) != 2540 or Counter(p['kind'] for p in pairs) != Counter(dict(A=635, P=635, C=635, F=635)):
        raise ValueError('Expected frozen E1 Dataset A: 2540 pairs, 635 in each A/P/C/F category.')
    by_prompt = {}
    for pair in pairs:
        for side in ('original', 'modified'):
            endpoint = pair[side]
            text, label = endpoint['prompt'], int(endpoint['label'])
            if label not in (0, 1):
                raise ValueError('Nonbinary endpoint label: ' + pair['pair_id'])
            if text in by_prompt and by_prompt[text]['label'] != label:
                raise ValueError('Conflicting labels for identical prompt: ' + pair['pair_id'])
            if text not in by_prompt:
                by_prompt[text] = dict(example_id='e1_endpoint_' + str(len(by_prompt)), prompt=text, label=label, view_ids=[], source_ids=[], pair_endpoints=[])
            r = by_prompt[text]
            for key, val in [('view_ids', endpoint['view_id']), ('source_ids', pair['source_id'])]:
                if val not in r[key]:
                    r[key].append(val)
            r['pair_endpoints'].append(dict(pair_id=pair['pair_id'], side=side, kind=pair['kind']))
    args.output.mkdir(parents=True, exist_ok=True)
    with target.open('x') as f:
        for record in by_prompt.values():
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    info = dict(audit_reference='docs/ASTER/baseline-audit-20261112.zh.md, branch codex/aster-codex00-01-audit-20261112', dataset=str(args.dataset), pairs=len(pairs), categories=dict(Counter(p['kind'] for p in pairs)), endpoints_before_deduplication=2*len(pairs), training_examples=len(by_prompt), sources=len({p['source_id'] for p in pairs}),
                deduplication='Exact full prompt equality; each unique prompt once per epoch. All pair endpoint memberships retained; conflicting labels fail.',
                objective='Native supervised fine-tuning: full-vocabulary cross entropy of the correct answer token, with prompt tokens masked. No A/P or C/F difference loss.',
                extra_natural_training_nodes=0, model='Qwen3-8B', method=args.kind, target_budget=17, epochs_per_stage=1,
                optimizer='AdamW', weight_decay=0.01, gradient_clip_norm=1.0,
                configuration=('Single layer 18 q_proj, rank 1, alpha 1, dropout 0; local configuration, not a standard multi-layer LoRA baseline.' if args.kind=='lora' else 'Qwen architecture adaptation: all-head per-coordinate scale selection with 1e-3 mean-absolute scale penalty, then discard scales and train one selected-head offset. Not an unchanged official LoFiT model configuration.'),
                stages=['tune'] if args.kind=='lora' else ['select','tune'],
                candidate_lrs=list(LRS), candidate_seed=11, final_seeds=list(SEEDS),
                selection='Maximum development natural-path accuracy; minimum development natural-node BCE breaks ties; ascending learning rate breaks exact remaining ties.',
                test_history='Historical 40-source controlled pool, not a new independent test.', target_tasks_evaluated=[])
    write(info_file, info)
    return info


def train(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from experiments.aster_cr.baselines import Adapter, Engine
    info = prepare(args)
    output = job_dir(args)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'training.json').exists():
        return
    if (output / 'adapter.pt').exists() or (output / 'train_steps.jsonl').exists():
        raise RuntimeError('Partial training output exists; choose a new output root: ' + str(output))
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    started = time.time()
    eng = Engine('qwen', device=args.device)
    adapter = Adapter(eng, args.kind, 17)
    if args.kind=='lora' and (adapter.layer != 18 or adapter.rank != 1):
        raise RuntimeError('Expected fixed single-layer18 / rank1 LoRA configuration.')
    eng.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    eng.model.enable_input_require_grads()
    eng.model.train()
    examples = rows(args.output / 'training_endpoints.jsonl')
    rng = np.random.default_rng(args.seed)
    torch.cuda.reset_peak_memory_stats(eng.device)
    stages=[]
    try:
        for stage in (['select','tune'] if args.kind=='lofit' else ['tune']):
            parameters=adapter.parameters()
            initial=[p.detach().cpu().clone() for p in parameters]
            opt=torch.optim.AdamW(parameters,lr=args.lr,weight_decay=.01)
            losses=[];answer_losses=[];stage_started=time.time()
            for step,idx in enumerate(rng.permutation(len(examples)),1):
                example=examples[int(idx)]
                opt.zero_grad(set_to_none=True)
                ids=eng.ids(example['prompt'])
                hidden=eng.model.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).last_hidden_state[0,-1]
                # The existing frozen model output head yields logits for the
                # entire vocabulary. Only the prompted next answer is supervised.
                logits=eng.model.lm_head(hidden).float().unsqueeze(0)
                target_token=eng.label_ids[0 if example['label']==1 else 1]
                answer_loss=F.cross_entropy(logits,torch.tensor([target_token],device=logits.device))
                loss=answer_loss
                if args.kind=='lofit' and stage=='select':loss=loss+1e-3*adapter.alpha.abs().mean()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite '+stage+' loss at step '+str(step))
                loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.0);opt.step()
                value=float(loss.detach());answer_value=float(answer_loss.detach())
                losses.append(value);answer_losses.append(answer_value)
                append(output/'train_steps.jsonl',dict(stage=stage,step=step,example_id=example['example_id'],loss=value,full_vocabulary_answer_ce=answer_value,elapsed_seconds=time.time()-stage_started))
                if step==1 or step%100==0 or step==len(examples):
                    progress=dict(state='training',method=args.kind,seed=args.seed,lr=args.lr,stage=stage,step=step,total=len(examples),mean_loss=float(np.mean(losses)),mean_answer_ce=float(np.mean(answer_losses)),elapsed_seconds=time.time()-stage_started)
                    write(output/'status.json',progress);print(json.dumps(progress),flush=True)
            changes=[float((p.detach().cpu()-before).norm()) for p,before in zip(parameters,initial)]
            if not any(v>0 for v in changes):raise RuntimeError('No actual parameter update in completed stage '+stage)
            stages.append(dict(stage=stage,steps=len(examples),parameters=sum(p.numel() for p in parameters),mean_loss=float(np.mean(losses)),mean_answer_ce=float(np.mean(answer_losses)),parameter_change_l2=changes,seconds=time.time()-stage_started))
            if args.kind=='lofit' and stage=='select':adapter.select()
        adapter.dump(output/'adapter.pt')
        result=dict(method=args.kind,reported_name='LoRA-single-layer-SFT' if args.kind=='lora' else 'LoFiT-style-Qwen-SFT',seed=args.seed,lr=args.lr,examples=len(examples),epochs_per_stage=1,stages=stages,
                    objective=info['objective'],manifest='../training_manifest.json',configuration=info['configuration'],target_budget=17,strict_parameter_matching=False,
                    layer=adapter.layer if args.kind=='lora' else None,rank=adapter.rank if args.kind=='lora' else None,heads=adapter.heads,
                    deployment_parameters=sum(p.numel() for p in adapter.parameters()),selection_parameters=getattr(adapter,'selection_parameters',0),
                    optimizer='AdamW',weight_decay=.01,gradient_clip_norm=1.0,full_vocabulary_loss=True,prompt_token_loss=False,extra_pair_loss=False,
                    train_seconds=sum(stage['seconds'] for stage in stages),total_seconds=time.time()-started,peak_gib=torch.cuda.max_memory_allocated(eng.device)/2**30)
        write(output/'training.json',result);write(output/'status.json',dict(state='trained',**result))
    finally:adapter.close()


def bootstrap_summary(views, pairs, scores, raw, paths, historical_paths, budget=0.2923349027297149):
    import numpy as np
    sources = sorted({v['source_id'] for v in views})
    index = {v:i for i,v in enumerate(sources)}
    boot = np.random.default_rng(11).multinomial(len(sources), np.full(len(sources), 1/len(sources)), size=2000)
    def aggregate(values, ss):
        values = np.asarray(values, dtype=float)
        if values.ndim == 1: values = values[:, None]
        totals = np.zeros((len(sources), values.shape[1]))
        for i, source in enumerate(ss): totals[index[source]] += values[i]
        return boot @ totals
    def ci(a):
        a=np.asarray(a);a=a[np.isfinite(a)]
        return np.quantile(a,[.025,.975]).tolist() if len(a) else None
    def rate(correct, ss):
        a=np.asarray(correct,dtype=float);b=aggregate(np.column_stack([np.ones(len(a)),a]),ss)
        with np.errstate(divide='ignore',invalid='ignore'): intervals=ci(b[:,1]/b[:,0])
        return dict(n=len(a),correct=int(a.sum()),accuracy=float(a.mean()) if len(a) else None,accuracy_ci=intervals)
    groups={}
    pair_predictions=[]
    for group,kinds in [('A',('A',)),('P',('P',)),('AP',('A','P')),('C',('C',)),('F',('F',)),('CF',('C','F')),('F0',('F0',))]:
        pp=[p for p in pairs if p['kind'] in kinds]
        if not pp: continue
        l=np.array([scores[p['left']] for p in pp]);r=np.array([scores[p['right']] for p in pp]);a=np.array([raw[p['left']] for p in pp]);b=np.array([raw[p['right']] for p in pp])
        yl=np.array([p['left_label'] for p in pp]);yr=np.array([p['right_label'] for p in pp]);d=r-l;d0=b-a;dc=d-d0
        correct=((l>0)==yl)&((r>0)==yr);base=((a>0)==yl)&((b>0)==yr);ss=[p['source_id'] for p in pp]
        bb=aggregate(np.column_stack([np.ones(len(pp)),d0,d0*d0,d,d*d,dc*dc,correct.astype(float),correct.astype(float)-base.astype(float)]),ss)
        with np.errstate(divide='ignore',invalid='ignore'):
            rms_ci=ci(np.sqrt(bb[:,4]/bb[:,0]));correction_ci=ci(np.sqrt(bb[:,5]/bb[:,0]));accuracy_ci=ci(bb[:,6]/bb[:,0]);diff_ci=ci(bb[:,7]/bb[:,0])
            vr_ci=ci(1-(bb[:,4]/bb[:,0]-(bb[:,3]/bb[:,0])**2)/(bb[:,2]/bb[:,0]-(bb[:,1]/bb[:,0])**2))
        m=dict(n=len(pp),sources=len(set(ss)),difference_rms=float(np.sqrt(np.mean(d*d))),difference_rms_ci=rms_ci,correction_rms=float(np.sqrt(np.mean(dc*dc))),correction_rms_ci=correction_ci,
               paired_correct=int(correct.sum()),paired_accuracy=float(correct.mean()),paired_accuracy_ci=accuracy_ci,paired_accuracy_difference_vs_raw=float(np.mean(correct.astype(float)-base.astype(float))),paired_accuracy_difference_vs_raw_ci=diff_ci,
               corrected_pairs_vs_raw=int(np.sum(correct&~base)),new_error_pairs_vs_raw=int(np.sum(~correct&base)))
        if group in ('A','P','AP','F0'):
            m.update(variance_reduction=float(1-np.var(d)/np.var(d0)) if np.var(d0)>0 else None,variance_reduction_ci=vr_ci)
        else:
            signed0=(yr-yl)*d0;signed=(yr-yl)*d
            eb=aggregate(np.column_stack([signed0,signed]),ss)
            with np.errstate(divide='ignore',invalid='ignore'):er_ci=ci(eb[:,1]/eb[:,0])
            m.update(mean_evidence_retention=float(signed.mean()/signed0.mean()) if signed0.mean()!=0 else None,mean_evidence_retention_ci=er_ci,
                     response_direction_degraded=int(np.sum((signed0>0)&(signed<=0))),response_direction_improved=int(np.sum((signed0<=0)&(signed>0))),
                     exceeds_reference_budget_count=int(np.sum(np.abs(dc)>budget+1e-7)),exceeds_reference_budget_fraction=float(np.mean(np.abs(dc)>budget+1e-7)))
        groups[group]=m
        if len(kinds)==1:
            for i,p in enumerate(pp):pair_predictions.append(dict(pair_id=p['pair_id'],source_id=p['source_id'],kind=p['kind'],left=p['left'],right=p['right'],left_label=int(yl[i]),right_label=int(yr[i]),left_score=float(l[i]),right_score=float(r[i]),original_difference=float(d0[i]),corrected_difference=float(d[i]),correction_difference=float(dc[i]),paired_correct=bool(correct[i])))
    natural=[v for v in views if v.get('original') and v.get('variant')=='full']
    ns=np.array([scores[v['view_id']] for v in natural]);ny=np.array([v['label'] for v in natural]);ss=[v['source_id'] for v in natural]
    groups['natural']=dict(**rate((ns>0)==ny,ss),bce=float(np.mean(np.logaddexp(0,ns)-ny*ns)))
    path=rate([r['correct'] for r in paths],[r['source_id'] for r in paths])
    by_case={r['case_id']:r for r in historical_paths}
    diffs=np.array([float(r['correct'])-float(by_case[r['case_id']]['correct']) for r in paths]);ss=[r['source_id'] for r in paths]
    bb=aggregate(np.column_stack([np.ones(len(diffs)),diffs]),ss)
    path.update(difference_vs_raw=float(diffs.mean()),difference_vs_raw_ci=ci(bb[:,1]/bb[:,0]),corrected_vs_raw=sum(r['correct'] and not by_case[r['case_id']]['correct'] for r in paths),new_errors_vs_raw=sum(not r['correct'] and by_case[r['case_id']]['correct'] for r in paths))
    return dict(groups=groups,path=path,uncertainty='2000 jointly sampled source-cluster bootstrap draws; seed 11; percentile 95%; undefined sparse-group draws omitted.'),pair_predictions


def evaluate(args):
    import torch
    from experiments.aster_cr.baselines import Adapter, Engine
    from experiments.aster_cr.common import TASK_KINDS
    from experiments.aster_cr.paths import execute
    from experiments.aster_closed_form.data import canonical_guide, prompt
    output=job_dir(args);split=args.split
    if (output / (split+'_metrics.json')).exists(): return
    training=read(output/'training.json')
    if split=='test':
        selected=read(args.output/'selection.json')
        if selected['lr']!=args.lr or not selected['frozen']: raise RuntimeError('Test requires the development-selected frozen learning rate.')
    torch.set_num_threads(4)
    started=time.time();eng=Engine('qwen',device=args.device);ad=Adapter(eng,args.kind,17,load=output/'adapter.pt');eng.model.eval()
    torch.cuda.reset_peak_memory_stats(eng.device)
    views=[v for v in rows(args.historical/'data/views.jsonl') if v['split']==split and v['kind'] in TASK_KINDS]
    pairs=[p for p in rows(args.historical/'data/pairs.jsonl') if p['split']==split]
    cases=[c for c in rows(args.historical/'data/path_cases.jsonl') if c['split']==split]
    raw={r['view_id']:r['ell'] for p in (args.historical/'qwen/scores').glob(split+'_*.jsonl') for r in rows(p)}
    view_file=output/(split+'_input_predictions.jsonl');path_file=output/(split+'_path_predictions.jsonl');node_file=output/(split+'_path_node_scores.jsonl')
    saved={r['view_id']:r['score'] for r in rows(view_file)}
    cache={v['prompt']:saved[v['view_id']] for v in views if v['view_id'] in saved}
    for r in rows(node_file): cache[r['prompt']]=r['score']
    def score(text):
        if text not in cache: cache[text]=float(ad.score(text))
        return cache[text]
    try:
        for i,v in enumerate(views,1):
            if v['view_id'] not in saved:
                s=score(v['prompt']);saved[v['view_id']]=s
                append(view_file,dict(view_id=v['view_id'],source_id=v['source_id'],label=v['label'],score=s,prediction=int(s>0),raw_score=raw[v['view_id']]))
            if i%100==0: write(output/'status.json',dict(state='evaluating_inputs',split=split,done=i,total=len(views),seconds=time.time()-started))
        done={r['case_id'] for r in rows(path_file)}
        for i,c in enumerate(cases,1):
            if c['case_id'] in done:continue
            tree=canonical_guide(c['guideline']);nodes={n['id']:n for n in tree['nodes']};gid=tree['nodes'][0]['id'];scores={}
            while gid in nodes and gid not in scores:
                text=prompt(c['guideline'],c['definitions'],c['patient'],gid);s=score(text);scores[gid]=s
                append(node_file,dict(case_id=c['case_id'],source_id=c['source_id'],gid=gid,prompt=text,score=s))
                gid=nodes[gid]['yes' if s>0 else 'no']
            append(path_file,dict(case_id=c['case_id'],source_id=c['source_id'],**execute(c,scores)))
            if i%50==0:write(output/'status.json',dict(state='evaluating_paths',split=split,done=i,total=len(cases),seconds=time.time()-started))
        paths=rows(path_file);raw_paths=rows(args.historical/'qwen/path_results'/('full_'+split+'.jsonl'))
        metrics,per_pair=bootstrap_summary(views,pairs,saved,raw,paths,raw_paths)
        pair_target=output/(split+'_pair_predictions.jsonl')
        with pair_target.open('x') as f:
            for r in per_pair:f.write(json.dumps(r)+'\n')
        metrics.update(method=('LoRA_single_layer_SFT_E1' if args.kind=='lora' else 'LoFiT_style_Qwen_SFT_E1'),split=split,seed=args.seed,lr=args.lr,training=training,seconds=time.time()-started,peak_gib=torch.cuda.max_memory_allocated(eng.device)/2**30,engine_counts=eng.counts,
                       historical_evaluation=True,new_independent_test=False,training_objective='full-vocabulary single-answer-token supervised CE',extra_AP_CF_loss=False)
        write(output/(split+'_metrics.json'),metrics);write(output/'status.json',dict(state=split+'_complete',seconds=time.time()-started))
        print(json.dumps(dict(state=split+'_complete',seed=args.seed,lr=args.lr,path=metrics['path'],natural=metrics['groups']['natural'])),flush=True)
    finally:ad.close()


def select(args):
    if (args.output/'selection.json').exists():return read(args.output/'selection.json')
    candidates=[]
    for lr in LRS:
        d=job_dir(args,lr,11);m=read(d/'dev_metrics.json')
        candidates.append(dict(lr=lr,seed=11,path_accuracy=m['path']['accuracy'],natural_bce=m['groups']['natural']['bce'],metrics=str(d/'dev_metrics.json')))
    chosen=sorted(candidates,key=lambda c:(-c['path_accuracy'],c['natural_bce'],c['lr']))[0]
    frozen=dict(lr=chosen['lr'],frozen=True,selection_timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),selection_rule='Maximum dev natural-path accuracy, then minimum dev natural-node BCE, then ascending lr.',candidates=candidates,development_sources=20,test_used_for_selection=False,seeds=list(SEEDS))
    write(args.output/'selection.json',frozen);return frozen


def controller(args):
    prepare(args)
    gpus=[int(s) for s in args.gpus.split(',')]
    if not gpus or any(g not in (0,1,5,6) for g in gpus) or len(set(gpus))!=len(gpus):raise ValueError('Only physical GPUs 0, 1, 5, and 6 may be used.')
    base=[sys.executable,str(Path(__file__).resolve()),'--kind',args.kind,'--output',str(args.output),'--dataset',str(args.dataset),'--historical',str(args.historical)]
    failures=[];failed=threading.Event()
    def run_jobs(jobs):
        q=queue.Queue()
        for job in jobs:q.put(job)
        def worker(gpu):
            while not failed.is_set():
                try:lr,seed,actions=q.get_nowait()
                except queue.Empty:return
                d=job_dir(args,lr,seed);d.mkdir(parents=True,exist_ok=True)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),ASTER_CR_ROOT=str(args.historical))
                for action,split in actions:
                    cmd=base+['--action',action,'--lr',str(lr),'--seed',str(seed),'--split',split,'--device','cuda:0']
                    with (d/(action+'_'+split+'.log')).open('a') as log:
                        proc=subprocess.run(cmd,env=env,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
                    if proc.returncode:
                        error=dict(lr=lr,seed=seed,action=action,split=split,returncode=proc.returncode,log=str(d/(action+'_'+split+'.log')))
                        write(d/'error.json',error);failures.append(error);failed.set();break
                q.task_done()
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:list(pool.map(worker,gpus))
        if failures:raise RuntimeError('A native supervised baseline job failed; see error.json and the saved job log. No automatic retry.')
    write(args.output/'controller_status.json',dict(state='development_candidates_running',gpus=gpus,candidates=list(LRS)))
    # A candidate started on an idle GPU joins this same finite experiment.
    # Process locks below prevent duplicate fitting or concurrent use of a GPU.
    candidate_lrs=list(LRS)
    early=args.output/'early_start.json'
    if early.exists():
        early_lr=read(early)['lr']
        candidate_lrs=[lr for lr in LRS if lr!=early_lr]+[early_lr]
    run_jobs([(lr,11,[('train','dev'),('evaluate','dev')]) for lr in candidate_lrs])
    frozen=select(args);lr=frozen['lr']
    write(args.output/'controller_status.json',dict(state='selected_config_frozen',gpus=gpus,lr=lr))
    run_jobs([(lr,22,[('train','dev'),('evaluate','dev'),('evaluate','test')]),(lr,33,[('train','dev'),('evaluate','dev'),('evaluate','test')]),(lr,11,[('evaluate','test')])])
    finalize(args)


def finalize(args):
    lr=read(args.output/'selection.json')['lr']
    gpus=[int(x) for x in args.gpus.split(',')]
    import numpy as np
    summary=[]
    for seed in SEEDS:
        m=read(job_dir(args,lr,seed)/'test_metrics.json')
        ap,cf,path=m['groups']['AP'],m['groups']['CF'],m['path']
        summary.append(dict(method=('LoRA_single_layer_SFT_E1' if args.kind=='lora' else 'LoFiT_style_Qwen_SFT_E1'),seed=seed,lr=lr,training_examples=m['training']['examples'],
                            AP_n=ap['n'],AP_response_rms=ap['difference_rms'],AP_response_rms_ci95=json.dumps(ap['difference_rms_ci']),
                            AP_variance_reduction=ap['variance_reduction'],AP_variance_reduction_ci95=json.dumps(ap['variance_reduction_ci']),
                            CF_n=cf['n'],CF_correction_rms=cf['correction_rms'],CF_correction_rms_ci95=json.dumps(cf['correction_rms_ci']),
                            CF_paired_correct=cf['paired_correct'],CF_paired_accuracy_ci95=json.dumps(cf['paired_accuracy_ci']),
                            CF_evidence_retention=cf['mean_evidence_retention'],CF_evidence_retention_ci95=json.dumps(cf['mean_evidence_retention_ci']),
                            path_n=path['n'],path_correct=path['correct'],path_accuracy=path['accuracy'],path_accuracy_ci95=json.dumps(path['accuracy_ci']),
                            path_difference_vs_raw=path['difference_vs_raw'],path_difference_vs_raw_ci95=json.dumps(path['difference_vs_raw_ci']),
                            training_seconds=m['training']['train_seconds'],training_peak_gib=m['training']['peak_gib']))
    with (args.output/'section_4_5_supervised_seeds.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    numeric=['AP_response_rms','AP_variance_reduction','CF_correction_rms','CF_evidence_retention','path_accuracy']
    write(args.output/'seed_summary.json',dict(seeds=list(SEEDS),seed_statistics={k:dict(mean=float(np.mean([r[k] for r in summary])),sample_std=float(np.std([r[k] for r in summary],ddof=1))) for k in numeric},
                                             uncertainty_note='Per-seed source-cluster 95% intervals are in the table. Seed standard deviations are not confidence intervals; intervals are not averaged.',records=summary))
    result=dict(state='complete',method=args.kind,lr=lr,seeds=list(SEEDS),gpu_ids=gpus,new_training_jobs=5,test_jobs=3,selection=str(args.output/'selection.json'),training_manifest=str(args.output/'training_manifest.json'),NLI4CT_runs=0,MedPIC_runs=0,ASTER_training_runs=0,
                results=[dict(seed=s,metrics=str(job_dir(args,lr,s)/'test_metrics.json')) for s in SEEDS])
    write(args.output/'completion.json',result);write(args.output/'controller_status.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=['prepare','train','evaluate','select','controller','finalize'],required=True)
    p.add_argument('--kind',choices=['lora','lofit'],required=True)
    p.add_argument('--output',type=Path,default=None)
    p.add_argument('--dataset',type=Path,default=DATASET)
    p.add_argument('--historical',type=Path,default=HISTORICAL)
    p.add_argument('--lr',type=float,choices=LRS,default=.001)
    p.add_argument('--seed',type=int,choices=SEEDS,default=11)
    p.add_argument('--split',choices=['dev','test'],default='dev')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--gpus',default='0,1')
    args=p.parse_args()
    if args.output is None:args.output=DEFAULT_OUTPUT/args.kind
    os.environ['ASTER_CR_ROOT']=str(args.historical)
    try:
        if args.action in ('train','evaluate'):
            import fcntl
            physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES','')
            if physical_gpu not in ('0','1','5','6'):raise ValueError('A single authorized physical GPU is required.')
            work=job_dir(args);work.mkdir(parents=True,exist_ok=True)
            locks=DEFAULT_OUTPUT.parent/'gpu_locks';locks.mkdir(parents=True,exist_ok=True)
            with (work/'execution.lock').open('a') as job_lock, (locks/('gpu_'+physical_gpu+'.lock')).open('a') as gpu_lock:
                fcntl.flock(job_lock,fcntl.LOCK_EX)
                fcntl.flock(gpu_lock,fcntl.LOCK_EX)
                globals()[args.action](args)
        else:globals()[args.action](args)
    except Exception as exc:
        import traceback
        where=args.output if args.action in ('controller','prepare','select','finalize') else job_dir(args)
        write(where/'error.json',dict(action=args.action,seed=args.seed,lr=args.lr,error=str(exc),traceback=traceback.format_exc()))
        if args.action=='controller':write(args.output/'controller_status.json',dict(state='failed',error=str(exc),automatic_retries=0))
        raise


if __name__=='__main__':main()
