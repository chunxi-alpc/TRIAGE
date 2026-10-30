"""Matched predicted/oracle generation, isolated from official evaluations."""
import argparse,json,time,traceback
from pathlib import Path
import torch
from .audit import read,write,oracle_decisions,strict_score,load_units,paired
from src.aster.manuscript_core import load_expert
from experiments.aster_manuscript_clinical_20261030.classify import model_and_classifier
from experiments.aster_manuscript_clinical_20261030.engine import ClinicalExecutor
from experiments.aster_manuscript_clinical_20261030.native_memory import query_chunked_attention

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--cpu-audit',required=True);p.add_argument('--output',required=True)
    p.add_argument('--limit',type=int,default=43);p.add_argument('--query-chunks',type=int,default=512)
    p.add_argument('--methods',nargs='+',default=['spectral','text_clip'],choices=['spectral','text_clip','mean','reverse'])
    a=p.parse_args();run=Path(a.run);out=Path(a.output)
    gate=json.loads((Path(a.cpu_audit)/'summary.json').read_text())
    if not gate['oracle_direct_map_gate_pass'] or gate['stored_vs_recomputed_plan_mismatches']:raise RuntimeError('CPU oracle/routing gate failed')
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(20261030)
    freeze=json.loads((run/'frozen.json').read_text());expert=load_expert(run/'analysis/expert.json')
    cases=read(run/'cases/confirm.jsonl')[:a.limit];_,classes=load_units(run,'confirm');routes=['predicted','gold_path_diagnostic_only']
    expected=len(cases)*len(a.methods)*len(routes)
    write(out/'manifest.json',{'phase':'confirm','routes':routes,'cases':len(cases),'source_clusters':len({c['source_id'] for c in cases}),
        'methods':a.methods,'source_run':str(run),'query_chunks':a.query_chunks,'frozen_gain':freeze['g_plus'],'max_new_tokens':freeze['max_new_tokens'],
        'official_test_used':False,'predicted_results':'new matched diagnostic runs with same attention implementation as oracle',
        'max_generation_calls':expected+2,'execution':'same chunked eager attention, exact_execution=True, capture=False in every arm',
        'old_predicted_outputs_used_for_causal_contrast':False})
    tok,model,_=model_and_classifier();engine=ClinicalExecutor(model,tok,expert,freeze,exact_execution=True);results=[];start=time.time()
    # Route invariance for the no-intervention control, within the new execution path.
    ref=cases[0];raw=classes[ref['case_id']]['raw_decisions']
    with query_chunked_attention(model,a.query_chunks,balanced=True):
        b1=engine.execute(ref,raw,'zero',capture=False)
        b2=engine.execute(ref,raw,'zero',capture=False,decisions_override=oracle_decisions(ref))
    reference={'same_zero_first_logits':torch.equal(b1['first_logits'],b2['first_logits']),
               'same_zero_tokens':b1['token_ids']==b2['token_ids'],'max_abs_logit_error':float((b1['first_logits']-b2['first_logits']).abs().max())}
    write(out/'reference.json',reference)
    if not all(reference[k] for k in ['same_zero_first_logits','same_zero_tokens']):raise RuntimeError('Zero-edit route invariance failed')
    del b1,b2
    with (out/'records.jsonl').open('x') as f:
        for i,case in enumerate(cases):
            for method in a.methods:
                for route in routes:
                    r={'case_id':case['case_id'],'source_id':case['source_id'],'method':method,'route':route}
                    try:
                        raw=classes[case['case_id']]['raw_decisions']
                        override=oracle_decisions(case) if route=='gold_path_diagnostic_only' else None
                        with query_chunked_attention(model,a.query_chunks,balanced=True):
                            result=engine.execute(case,raw,method,capture=False,decisions_override=override)
                        result.pop('first_logits',None);score=strict_score(case,result['text'])
                        r.update(status='complete',score=score,result=result)
                    except Exception as e:
                        r.update(status='failed',error_type=type(e).__name__,error=str(e));traceback.print_exc();torch.cuda.empty_cache()
                    results.append(r);f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
                    write(out/'status.json',{'status':'running','completed':len(results),'expected':expected,'technical_failures':sum(x['status']=='failed' for x in results),
                                            'elapsed_seconds':time.time()-start})
    paired_rows=[]
    for r in results:
        m=('oracle_' if r['route']=='gold_path_diagnostic_only' else 'predicted_')+r['method']
        paired_rows.append({'case_id':r['case_id'],'source_id':r['source_id'],'method':m,'status':r['status'],
                            'ok':r['status']=='complete' and r['score']['leaf_correct']})
    comparisons=[paired(paired_rows,'oracle_'+m,'predicted_'+m) for m in a.methods]
    summary={'status':'complete','cases':len(cases),'units':len(results),'technical_failures':sum(x['status']=='failed' for x in results),
        'elapsed_seconds':time.time()-start,'peak_memory_gb':torch.cuda.max_memory_allocated()/1e9,'comparisons':comparisons,
        'methods':[{'method':m,'route':route,'denominator':len(cases),
        'correct':sum(r['status']=='complete' and r.get('score',{}).get('leaf_correct',False) for r in results if r['method']==m and r['route']==route),
        'exact_path_correct':sum(r['status']=='complete' and r.get('score',{}).get('exact_path',False) for r in results if r['method']==m and r['route']==route),
        'root_correct':sum(r['status']=='complete' and r.get('score',{}).get('root_correct',False) for r in results if r['method']==m and r['route']==route)} for m in a.methods for route in routes],
        'limitation':'Matched oracle paths reveal source reference and are diagnostic, not deployable. Current confirm has two source clusters.'}
    write(out/'summary.json',summary);write(out/'status.json',summary)
    print(json.dumps({k:summary[k] for k in ['status','cases','units','technical_failures','elapsed_seconds','methods']}))
if __name__=='__main__':main()
