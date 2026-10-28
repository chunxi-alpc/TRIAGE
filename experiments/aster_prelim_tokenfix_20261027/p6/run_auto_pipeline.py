"""Launch generation and scoring on GPUs 0/1, then summarize; no automatic full-run escalation."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-name',required=True);ap.add_argument('--phase',choices=['smoke','baselines','ablation'],required=True);ap.add_argument('--lambda-value',type=float,default=1);a=ap.parse_args()
 assert '/' not in a.run_name
 lock=(B/'private/p6_auto_pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 run=B/'private'/a.run_name;assert not run.exists()
 profile=B/'private/p6_profile_auto100_v1'
 assert (profile/'DONE.json').exists(),'Longest-context profile not completed'
 rows=[json.loads(l) for l in (profile/'outputs.jsonl').read_text().splitlines()]
 lookup={(r['variant'],r['method']):r for r in rows}
 for v in ['native','decoy']:
  assert lookup[v,'Vanilla']['output_ids']==lookup[v,'Matched-Zero']['output_ids']
  assert lookup[v,'ASTER-1-no-green']['nonzero_attention_calls']>0
 status=B/'private'/(a.run_name+'.coordinator.json')
 genlog=(B/'private'/(a.run_name+'.generation.log')).open('x')
 scorelog=(B/'private'/(a.run_name+'.scoring.log')).open('x')
 gen=subprocess.Popen([sys.executable,str(D/'run_auto_pilot.py'),'--phase',a.phase,'--run-dir',str(run),'--lambda-value',str(a.lambda_value)],env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'},stdout=genlog,stderr=subprocess.STDOUT)
 state={'status':'GENERATION_STARTING','phase':a.phase,'run_dir':str(run),'generation_pid':gen.pid,'test_files_read':False};dump(status,state)
 while not (run/'manifest.json').exists() and gen.poll() is None:time.sleep(1)
 if not (run/'manifest.json').exists():state.update(status='FAILED_GENERATION_START',exit_code=gen.wait());dump(status,state);return
 judge=subprocess.Popen([sys.executable,str(D/'score_auto_pilot.py'),'--run-dir',str(run),'--follow'],env={**os.environ,'CUDA_VISIBLE_DEVICES':'1'},stdout=scorelog,stderr=subprocess.STDOUT)
 state.update(status='GENERATION_AND_SCORING_RUNNING',scoring_pid=judge.pid);dump(status,state)
 gc=gen.wait();sc=judge.wait();state.update(generation_exit=gc,scoring_exit=sc)
 if gc or sc:state['status']='FAILED_CHECK_LOGS'
 else:
  summarylog=(B/'private'/(a.run_name+'.summary.log')).open('x')
  code=subprocess.run([sys.executable,str(D/'summarize_auto_pilot.py'),'--run-dir',str(run)],stdout=summarylog,stderr=subprocess.STDOUT).returncode
  state.update(status='COMPLETED' if code==0 else 'SUMMARY_FAILED',summary_exit=code)
 dump(status,state)
if __name__=='__main__':main()
