"""Run conditional group 3 only after a genuine dev improvement was observed.

All inference is real local-checkpoint CUDA inference. Pass a frozen C-method
configuration chosen in group 1/2; learned directions are freshly extracted
from true training pairs at those exact layers and every KV head.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch

from engine import QwenEngine, read_cases
from key_edit import KeyEditor, learn_paired_directions
from run_attention import summarize


def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--baseline-dev', required=True)
    p.add_argument('--baseline-confirmation', required=True)
    p.add_argument('--direct-dev')
    p.add_argument('--direct-confirmation')
    p.add_argument('--trigger-evidence', required=True,
                   help='Observed real dev result that triggered this conditional experiment')
    args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(Path(args.config).read_text())
    if cfg.get('variant','C')!='C' or cfg.get('with_structure',False):
        raise ValueError('This matched comparison uses C, with no structure text')
    cfg={**cfg,'variant':'C','with_structure':False}
    layers=cfg['layers'];strength=float(cfg['strength']);window=int(cfg.get('query_window',1))
    model=QwenEngine()
    started=time.monotonic()
    save(out/'provenance.json',model.provenance|{
        'argv':sys.argv,'conditional_trigger':args.trigger_evidence,'config':cfg,
        'study_status':'exploratory reuse of existing development and confirmation sets',
        'direction_space':'post-RoPE key vectors',
        'scope':'Key methods affect every causally visible query; direct score bias only affects selected last 1/16 query rows',
        'calibration':'Per selected causal 4-head query group, gamma=z/(z^2+0.01*||mean(Q)||^2/d); no sign gate; no RMS search',
        'query_aligned':'ridge=0.01*mean(diag(A A^T)); A=selected causal Q/sqrt(d)',
        'unknown_training_policy':'Unknown rule states do not define a T-F contrast; every case remains in evaluation',
        'seed':20261019})
    train=read_cases('train');captures={};training_rows=[]
    editor=KeyEditor('capture',0,window)
    model.key_editor=editor
    with (out/'training_predictions.jsonl').open('x') as stream:
        for index,c in enumerate(train):
            editor.set_case(c)
            try:
                row=model.score(c,{**cfg,'method':'capture','strength':0})
                captures[c['case_id']]=editor.capture
            except Exception as exc:
                row={'status':'technical_failure','is_model_prediction':False,
                     'error':type(exc).__name__+': '+str(exc),'traceback':traceback.format_exc()}
            row.update(case_id=c['case_id'],split='train',gold_answer=c['gold_answer'])
            training_rows.append(row)
            stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
            if (index+1)%24==0:
                print(json.dumps({'phase':'training_capture','completed':index+1,'total':len(train)}),flush=True)
    directions,audit=learn_paired_directions(train,captures,layers)
    audit['training_failures']=[r['case_id'] for r in training_rows if r['status']!='completed']
    save(out/'training_directions.json',audit)
    torch.save(directions,out/'directions.pt')
    results={}
    for split in ('dev','confirmation'):
        cases=read_cases(split)
        baseline=read_rows(getattr(args,'baseline_'+split))
        direct_path=getattr(args,'direct_'+split)
        direct=None
        for method in ('direct','query_aligned','learned_mean','random'):
            arm=f'{split}_{method}'
            arm_start=time.monotonic()
            rows=[]
            arm_cfg={**cfg,'name':arm,'method':method}
            if method=='direct' and direct_path:
                rows=read_rows(direct_path)
                if {r['case_id'] for r in rows}!={c['case_id'] for c in cases}:
                    raise ValueError('Reused direct predictions do not cover the full split')
                for r in rows:
                    observed=r.get('config',{})
                    for key in ('layers','strength','variant','query_window'):
                        expected=cfg.get(key,1 if key=='query_window' else 'C' if key=='variant' else None)
                        actual=observed.get(key,1 if key=='query_window' else 'C' if key=='variant' else None)
                        if actual!=expected:
                            raise ValueError(f'Reused direct config mismatch for {key}')
                save(out/(arm+'_reused_source.json'),{'path':str(Path(direct_path).resolve())})
                (out/(arm+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
            else:
                editor=KeyEditor(method,strength,window,directions)
                model.key_editor=editor
                with (out/(arm+'.jsonl')).open('x') as stream:
                    for index,c in enumerate(cases):
                        editor.set_case(c)
                        try:
                            row=model.score(c,arm_cfg)
                        except Exception as exc:
                            row={'status':'technical_failure','is_model_prediction':False,
                                 'error':type(exc).__name__+': '+str(exc),'traceback':traceback.format_exc()}
                        row.update(case_id=c['case_id'],arm=arm,split=split,config=arm_cfg,
                                   source='reference',gold_answer=c['gold_answer'],
                                   reference_state=c['reference_state'],family_id=c['family_id'],
                                   variant=c['variant'])
                        rows.append(row);stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                        if (index+1)%24==0:
                            print(json.dumps({'arm':arm,'completed':index+1,'total':len(cases)}),flush=True)
            if method=='direct':
                direct=rows
            summary=summarize(cases,rows,baseline)
            summary['versus_direct']=summarize(cases,rows,direct)
            summary['elapsed_seconds']=time.monotonic()-arm_start
            groups=[g for r in rows for d in r.get('attention_diagnostics',[])
                    for g in d.get('callback',{}).get('groups',[]) if g.get('status')=='edited']
            if groups:
                summary['key_edit_diagnostics']={
                    'edited_group_records':len(groups),
                    'mean_group_wrong_sign_fraction':sum(g['wrong_sign_fraction'] for g in groups)/len(groups),
                    'mean_group_signed_score_change':sum(g['signed_score_mean'] for g in groups)/len(groups),
                    'mean_group_nonzero_fraction':sum(g['nonzero_fraction'] for g in groups)/len(groups)}
            results[arm]=summary
            save(out/'summary.json',results)
            print(json.dumps({'arm':arm,**summary}),flush=True)
    save(out/'completion.json',{'status':'completed','forward_calls':model.forward_calls,
                               'elapsed_seconds':time.monotonic()-started,
                               'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(),
                               'evaluation_cases':288,'training_cases':len(train)})


if __name__=='__main__':
    main()

