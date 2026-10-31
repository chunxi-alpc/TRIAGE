"""Small behavior-objective direction search; frozen backbone and independent fit pairs."""
import argparse,json,math,time
from .run import BASE,OUT,read,write
COEFFICIENTS=[[1,0,0],[0,1,0],[.75,.25,0],[.5,.5,0],[.25,.75,0],[1,0,.25],[1,.25,.25]]
def softplus(x):return max(x,0)+math.log1p(math.exp(-abs(x)))
def worker(shard):
 import torch
 from experiments.aster_manuscript_clinical_20261030.classify import model_and_classifier
 from experiments.aster_manuscript_clinical_20261030.engine import ClinicalExecutor
 from experiments.aster_manuscript_clinical_20261030.native_memory import query_chunked_attention
 from src.aster.manuscript_core import load_expert
 from .runtime import execute
 torch.set_num_threads(4);torch.manual_seed(13)
 tok,model,_=model_and_classifier();engine=ClinicalExecutor(model,tok,load_expert(BASE/'analysis/expert.json'),json.loads((BASE/'frozen.json').read_text()))
 heads=json.loads((OUT/'selected_head_control.json').read_text())['heads'];gain=json.loads((OUT/'key_gain_fits.json').read_text())['mean']['rms_gain']
 pairs=[r for r in read(BASE/'review/accepted_pairs.jsonl') if r['split']=='fit'];start=time.time();n=0
 output=OUT/f'behavior_{shard}.jsonl'
 done={(x['pair_id'],x['state'],x['candidate']) for x in read(output)} if output.exists() else set()
 with query_chunked_attention(model,256),(OUT/f'behavior_{shard}.jsonl').open('a') as f:
  for i,row in enumerate(pairs):
   if i%4!=shard:continue
   for state in ['positive','negative']:
    target=row['target_rule_id'];gold=target if state=='positive' else target[:-1]+('N' if target[-1]=='Y' else 'Y')
    wrong=gold[:-1]+('N' if gold[-1]=='Y' else 'Y')
    case={'case_id':row['pair_id']+':'+state,'patient':row['states'][state],'guideline':row['guideline'],'question':row['question'],'gold_rules':[gold]}
    engine.behavior_reference=None
    execute(engine,case,{'kind':'zero','probe_only':True},route=[gold])
    engine.behavior_reference=engine.last_first_log_probs.clone()
    for j,coeffs in enumerate(COEFFICIENTS):
     if (row['pair_id'],state,j) in done:continue
     spec={'kind':'mean','probe_only':True,'gain':gain,'direction_heads':heads,'full_rule_keys':True,'direction_mix':coeffs}
     good=execute(engine,case,spec,route=[gold]);bad=execute(engine,case,spec,route=[wrong])
     loss=softplus(-good['decision_margin'])+softplus(bad['decision_margin'])+.1*(good['format_kl']+bad['format_kl'])
     f.write(json.dumps({'pair_id':row['pair_id'],'source_id':row['source_id'],'state':state,'candidate':j,'coefficients':coeffs,'loss':loss,
      'correct_gate_margin':good['decision_margin'],'wrong_gate_margin':bad['decision_margin'],'format_kl':(good['format_kl']+bad['format_kl'])/2})+'\n');f.flush();n+=1
    engine.behavior_reference=None
 write(OUT/f'behavior_{shard}_status.json',{'status':'complete','records':n,'elapsed':time.time()-start,'peak_gb':torch.cuda.max_memory_allocated()/1e9})
def summarize():
 rows=[]
 for shard in range(4):rows+=read(OUT/f'behavior_{shard}.jsonl')
 summaries=[]
 for j,coef in enumerate(COEFFICIENTS):
  rs=[r for r in rows if r['candidate']==j];summaries.append({'candidate':j,'coefficients':coef,'n':len(rs),'mean_loss':sum(r['loss'] for r in rs)/len(rs)})
 best=min(summaries,key=lambda x:(x['mean_loss'],x['candidate']))
 out={'candidates':summaries,'selected':best,'basis':['normalized mean','normalized low-frequency mean','fixed random direction'],
  'objective':'softplus(-correct-gate margin)+softplus(wrong-gate margin)+0.1*(initial-output-position KLs)',
  'fitting':'finite seven-point direction-mixture search on original fit sources; not a full-vector gradient-training replication',
  'gain':json.loads((OUT/'key_gain_fits.json').read_text())['mean']['rms_gain'],
  'heads':json.loads((OUT/'selected_head_control.json').read_text())['heads']}
 write(OUT/'behavior_summary.json',out);return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int);p.add_argument('--summarize',action='store_true');a=p.parse_args()
 if a.summarize:summarize()
 else:worker(a.shard)
