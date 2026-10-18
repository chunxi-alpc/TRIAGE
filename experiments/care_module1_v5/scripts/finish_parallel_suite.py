from __future__ import annotations
import datetime,hashlib,json,os,pathlib,shutil,subprocess,time,traceback
ROOT=pathlib.Path('/home/gcx/TRIAGE-v5-pr4');EXP=ROOT/'experiments/care_module1_v5';PY='/home/gcx/n/.venv/bin/python';WORK=EXP/'runs/v5_parallel_generation/private';ENV=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
status={'status':'waiting_for_remaining_parallel_batches','start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpus':[4,5,6,7],'original_outputs_reused':256,'remaining_conditions':236,'definition':'Scheduling/parallelism change requested by user; frozen core, prompts, token budget, batch membership, inputs, labels and evaluation unchanged.'}
STATUS=EXP/'runs/parallel_controller_status.json';LOG=(EXP/'runs/parallel_controller.log').open('a')
def update(**kw):
 status.update(kw);tmp=STATUS.with_suffix('.pending');tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(STATUS)
def run(*args):
 p=subprocess.run([str(x) for x in args],cwd=ROOT,env=ENV,stdout=LOG,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError(f'Command failed: {pathlib.Path(str(args[1])).name if len(args)>1 else args[0]}; exit={p.returncode}')
def merge_replay(work,out,inputs,rules,n):
 merged=work/'merged_trialmatch.jsonl'
 run(PY,EXP/'scripts/parallel_trialmatch.py','merge','--work-dir',work,'--output',merged)
 run(PY,EXP/'scripts/replay_evidence.py','--inputs',inputs,'--rules',rules,'--expected-count',n,'--merged',merged,'--source-status',EXP/'runs/v5_serial_initial/run_status.json','--parallel-manifest',work/'timing_summary_private.json','--output',out,'--device','cuda:6')
 shutil.copy2(merged,out/'private/trialmatch_precomputed.jsonl')
 worker=json.loads((work/'timing_summary_private.json').read_text())
 (out/'parallel_generation_summary.json').write_text(json.dumps(worker,indent=2)+'\n')
try:
 update()
 while True:
  statuses=[json.loads((WORK/f'worker_rank{r}_status_private.json').read_text()) for r in range(4)]
  if any(s['status']=='failed' for s in statuses):raise RuntimeError('A remaining-condition worker failed; see private worker status')
  if all(s['status']=='completed' for s in statuses):break
  time.sleep(60)
 update(status='replaying_regression_evidence')
 old=EXP/'runs/v5_20261018'
 if old.exists():old.rename(EXP/'runs/v5_serial_original_unmodified')
 merge_replay(WORK,old,pathlib.Path('/home/gcx/care_module1_20261018_v4/data/private/inputs.jsonl'),EXP/'data/rules_v5.jsonl',192)
 run(PY,EXP/'scripts/evaluate.py','--labels',EXP/'data/labels_public.jsonl','--pairs',EXP/'data/pairs_public.jsonl','--predictions',old/'predictions_public.jsonl','--qwen-reference',EXP/'data/v4_qwen_direct_public.jsonl','--output',old/'metrics.json','--errors',old/'errors_public.jsonl')
 update(status='parallel_new_synthetic_generation')
 ework=EXP/'exploratory/runs/parallel_generation/private';ework.mkdir(parents=True,exist_ok=True);empty=ework/'empty_prefix.jsonl';empty.write_text('')
 run(PY,EXP/'scripts/parallel_trialmatch.py','prepare','--inputs',EXP/'exploratory/data/inputs_public.jsonl','--rules',EXP/'exploratory/data/private/rules.jsonl','--existing',empty,'--work-dir',ework,'--world-size',4,'--batch-size',8)
 processes=[]
 for rank,device in enumerate([4,5,6,7]):
  log=(ework/f'worker_rank{rank}.log').open('w')
  processes.append(subprocess.Popen([PY,'-u',str(EXP/'scripts/parallel_trialmatch.py'),'worker','--work-dir',str(ework),'--rank',str(rank),'--device',f'cuda:{device}'],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL))
 rc=[p.wait() for p in processes]
 if any(rc):raise RuntimeError(f'New synthetic parallel workers failed: {rc}')
 update(status='replaying_new_synthetic_evidence')
 eout=EXP/'exploratory/runs/frozen_v1'
 merge_replay(ework,eout,EXP/'exploratory/data/inputs_public.jsonl',EXP/'exploratory/data/private/rules.jsonl',24)
 run(PY,EXP/'exploratory/evaluate.py','--predictions',eout/'predictions_public.jsonl','--output',EXP/'exploratory/results')
 run(PY,EXP/'scripts/verify_artifacts.py')
 update(status='completed',end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 run(PY,EXP/'scripts/generate_report.py','--run-status',old/'run_status.json','--metrics',old/'metrics.json','--errors',old/'errors_public.jsonl','--output',EXP/'REPORT.zh.md')
except Exception as exc:
 update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 raise
finally:LOG.close()
