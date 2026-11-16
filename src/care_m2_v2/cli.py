"""Server-only staged CLI. Outputs are new directories or verified journal resumes."""
import argparse,os,json,time,traceback,subprocess,shutil
from pathlib import Path
import torch
from care_m2_inc.io import read_json,read_jsonl,write_json,write_jsonl,sha256_file
from .data import BASE,prepare,verify,digest
from .study import load,train,select,evaluate,summarize,primary,METHODS,followup_decisions
from .engine import Engine
from .smoke import run as smoke


def publish(out,message):
    # Never merge or reset a worktree; only normal pushes to the authorized PR.
    branch='codex/care-module2-n1-n5-only-20261020'
    actual=subprocess.check_output(['git','branch','--show-current'],text=True).strip();assert actual==branch,actual
    subprocess.run(['git','add',str(out)],check=True)
    if subprocess.run(['git','diff','--cached','--quiet']).returncode:
        subprocess.run(['git','commit','-m',message],check=True)
        subprocess.run(['git','push','origin','HEAD:refs/heads/'+branch],check=True)

def save_manifest(engine,out,task,started,status):
    records=read_jsonl(engine.journal) if engine.journal.exists() else []
    result={'task':task,'status':status,'process_exit_code':0,'model':engine.fp,'forward_count':sum(r['payload'].get('forward_count',0) for r in records),
      'elapsed_seconds':time.time()-started,'peak_memory_bytes':torch.cuda.max_memory_allocated(),'outputs':{},'schema':'care.m2.v2.run'}
    for p in Path(out).rglob('*'):
        if p.is_file() and p.name not in ('run_manifest.json',):result['outputs'][str(p.relative_to(out))]=sha256_file(p)
    write_json(Path(out)/'run_manifest.json',result);(Path(out)/'STATUS.md').write_text(status+'\n')
    return result

def run_task(args,task,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);started=time.time()
    shutil.copy(Path(args.data)/'manifest.json',out/'data_manifest.json')
    engine=Engine(args.model,out/'raw');allrows=load(args.data,task)
    try:
        axes=train(engine,[r for r in allrows if r['split']=='train'],out)
        if task=='N2':
            from .knowledge import screen,evaluate_known
            knowledge,eligibility=screen(engine,out)
        selection,dev=select(engine,[r for r in allrows if r['split']=='dev'],axes,out,[r for r in allrows if r['split']!='test'])
        chosen=[p for p in dev if p['beta']==selection['selected'][p['method']]]
        # G2: only freeze N1 test if development shows an advantage over direct score bias.
        go=True
        if task=='N1':
            by={m:primary(task,[p for p in chosen if p['method']==m]) for m in METHODS[task]}
            go=by['route']>by['direct_bias'] and float(axes['R']['mean'].norm())>1e-6 and float(axes['A']['mean'].norm())>1e-6
            write_json(out/'gate.json',{'gate':'G2','go':go,'dev_primary':by,'rule':'route > direct bias on complete four-cell blocks, both trained R/A contrasts nonzero; no extra search'})
        if go:
            preds=evaluate(engine,[r for r in allrows if r['split']=='test'],axes,selection,out,allrows)
            summary=summarize(task,preds,out)
            if task=='N2':
                known=evaluate_known(engine,knowledge,eligibility,selection,axes,out)
                if known:
                    (out/'knowledge_analysis').mkdir();summarize(task,known,out/'knowledge_analysis')
            if task=='N3':followup_decisions(engine,[r for r in allrows if r['split']=='test'],preds,selection,axes,out,allrows)
        else:
            summary=summarize(task,chosen,out,'dev');summary['test_status']='not_run_gate_fail';write_json(out/'summary.json',summary)
        if task in ('N2','N3'):
            metrics=summary['metrics'];go=metrics['route']['primary']>metrics['baseline']['primary'] and metrics['route']['primary']>=metrics['text']['primary'] and selection['selected']['route']>0
            if task=='N2':go=go and metrics['iia_train']['primary']>metrics['baseline']['primary'] and metrics['iia_train']['donor_copy']/max(1,metrics['iia_train']['donor_copy_eligible']) < metrics['iia_train']['correct']/max(1,metrics['iia_train']['total'])
            if task=='N3':go=go and metrics['route']['q_wrong_decision']<=metrics['text']['q_wrong_decision'] and metrics['route']['d_unnecessary_query']<=metrics['text']['d_unnecessary_query']
            write_json(out/'gate.json',{'gate':'G3' if task=='N2' else 'G4','go':go,'rule':'nonzero route > baseline and >= same-information text; N2 cross-content IIA > baseline with receiver correctness exceeding donor-copy rate; N3 neither risk worse than text','metrics':metrics})
        status='gpu_evaluated' if task!='N1' or go else 'not_run_gate_fail_test_dev_negative'
        manifest=save_manifest(engine,out,task,started,status)
        assert manifest['forward_count']<={'N1':1800,'N2':1500,'N3':1200,'N5':800}[task],manifest['forward_count']
        return summary
    except Exception as exc:
        (out/'failure.log').write_text(traceback.format_exc());write_json(out/'failure.json',{'status':'failed','error':str(exc),'forward_count':engine.calls,'exit_code':1})
        raise
    finally:
        del engine;torch.cuda.empty_cache()

def full(args):
    root=Path(args.out);root.mkdir(parents=True,exist_ok=False)
    write_json(root/'frozen_execution.json',{'args':vars(args),'data_manifest_sha256':sha256_file(Path(args.data)/'manifest.json'),
      'phases':['S1','N1','N2','N3','N4 conditional','N5'],'cpu_report':read_json(Path(args.data)/'cpu_report.json')})
    assert read_json(Path(args.data)/'cpu_report.json')['valid']
    e=Engine(args.model,root/'S1/raw');start=time.time()
    rows=[r for t in ('N1','N2','N3','N4','N5') for r in load(args.data,t,'train')]
    smoke(e,rows,root/'S1');save_manifest(e,root/'S1','S1',start,'gpu_smoke_passed');del e;torch.cuda.empty_cache()
    if args.push:publish(root/'S1','results(care-m2): verify v2 local hooks and real score bias')
    summaries={}
    for task in ('N1','N2','N3'):
        summaries[task]=run_task(args,task,root/task)
        if args.push:publish(root/task,'results(care-m2): record '+task+' v2 outputs and gate')
    n4=root/'N4';n4.mkdir()
    n2=read_json(root/'N2/gate.json')['go'];n3=read_json(root/'N3/gate.json')['go']
    if not (n2 and n3):
        write_json(n4/'gate.json',{'status':'not_run_gate_fail','N2_go':n2,'N3_go':n3,'forward_count':0})
        (n4/'STATUS.md').write_text('not_run_gate_fail\nN2/N3 did not both meet the preregistered single-axis gates.\n')
    else:
        from .joint import run_joint
        summaries['N4']=run_joint(args,root,n4)
    if args.push:publish(n4,'results(care-m2): record conditional N4 decision')
    summaries['N5']=run_task(args,'N5',root/'N5')
    if args.push:publish(root/'N5','results(care-m2): evaluate frozen A1 transfer without A1 inference')
    # Holm across executed TEST main comparisons, not dev-only N1.
    ps=sorted((s['primary_comparison']['two_sided_signflip_p'],t) for t,s in summaries.items() if s.get('split')=='test' and s.get('primary_comparison'))
    adjusted={};previous=0.
    for i,(p,t) in enumerate(ps):previous=max(previous,min(1.,p*(len(ps)-i)));adjusted[t]=previous
    write_json(root/'summary.json',{'studies':summaries,'holm_adjusted_p':adjusted,'N4':read_json(n4/'gate.json'),'status':'finished_with_registered_gate_stops',
      'old_experiment_forwards':0,'A1_inference_forwards':0,'limits':'small synthetic mechanisms and historical frozen-A1 exploratory transfer; no clinical validation'})
    report=['# N1–N5 本轮结果','', '旧实验及 A1 推理均未重跑。v1 无效结果完整保留。','']
    for t,s in summaries.items():
        report.append(f"- {t}: {s.get('split')}, "+'; '.join(f"{m} {v['correct']}/{v['total']}，主指标 {v['primary']:.3f}" for m,v in s['metrics'].items() if m in ('baseline','text','direct_bias','route')))
    report+=['- N4: '+read_json(n4/'gate.json')['status'],'','L0：见 S1；L1/L2：分别见各主指标、供体与风险分层；L3：仅固定 A1 历史预测的探索迁移。','不得将 CPU 通过或非零干预写成语义机制成功。']
    (root/'REPORT.zh.md').write_text('\n'.join(report)+'\n')
    if args.push:publish(root,'results(care-m2): summarize N1-N5 and Holm-adjusted comparisons')


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','verify','smoke','fit','select','evaluate','analyze','run','run-task'])
    p.add_argument('--data',default=str(BASE/'data_v2_ready'));p.add_argument('--out',required=True)
    p.add_argument('--model',default='/home/gcx/care_20261016/CARE_Codex_Pilot/models/Qwen3-4B-Base');p.add_argument('--experiment',choices=['N1','N2','N3','N5']);p.add_argument('--push',action='store_true')
    args=p.parse_args()
    if args.command=='prepare':print(prepare(args.out));return
    if args.command=='verify':
        rows=[r for t in ('N1','N2','N3','N4','N5') for r in load(args.data,t)];report=verify(rows);write_json(Path(args.out)/'cpu_report.json',report);assert report['valid'];return
    if args.command=='run':full(args);return
    if args.command=='run-task':run_task(args,args.experiment,args.out);return
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);rows=load(args.data,args.experiment or 'N1')
    if args.command=='analyze':summarize(args.experiment,read_jsonl(out/'test_predictions.jsonl'),out);return
    e=Engine(args.model,out/'raw')
    if args.command=='smoke':smoke(e,rows,out);return
    if args.command=='fit':train(e,[r for r in rows if r['split']=='train'],out);return
    axes=torch.load(out/'axes.pt',weights_only=True)
    if args.command=='select':
        shutil.copy(Path(args.data)/'manifest.json',out/'data_manifest.json');select(e,[r for r in rows if r['split']=='dev'],axes,out,[r for r in rows if r['split']!='test']);return
    evaluate(e,[r for r in rows if r['split']=='test'],axes,read_json(out/'dev_selection.json'),out,rows)
if __name__=='__main__':main()
