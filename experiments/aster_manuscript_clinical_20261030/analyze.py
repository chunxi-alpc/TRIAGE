"""CPU sufficient-statistic analysis and expert freeze; no test tuning."""
import argparse,copy,json,random,subprocess
from pathlib import Path
import torch
from src.aster.manuscript_core import PairMoments,FitConfig,ExcludedHead,head_name,select_expert,validation_observation,subspace_overlap,save_expert
from src.aster.calibration import fit_temperature
from src.aster.action_blind import summarize_logits
from .data import write,read

def load(run):
 records=[]
 accepted={r['pair_id'] for r in read(Path(run)/'review/accepted_pairs.jsonl')}
 for shard in range(2):
  root=Path(run)/('collect'+str(shard));status=json.loads((root/'status.json').read_text())
  if status['status']!='complete':raise RuntimeError('collection not complete')
  for i in status['complete']:
   record=torch.load(root/(str(i)+'.pt'),map_location='cpu',weights_only=False)
   if record['row']['pair_id'] in accepted:records.append(record)
 return records
def estimate(rows,head,rank):
 m=PairMoments()
 for r in rows:m.update(r['offline']['positive'][head],r['offline']['negative'][head])
 return m.estimate(FitConfig(rank,4))
def retention(rows,head,basis):
 a=0.;b=0.
 for r in rows:
  d=r['offline']['positive'][head]-r['offline']['negative'][head]
  a+=float((d@basis).square().sum(-1).mean());b+=float(d.square().sum(-1).mean())
 return None if b==0 else a/b
def gold(r,state):return r['row']['target_rule_id'] if state=='positive' else r['row']['verification']['negative_rule_id'] if state=='negative' else 'NONE'
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run);out=run/'analysis';out.mkdir(exist_ok=False)
 torch.set_num_threads(4);torch.manual_seed(20261030);rows=load(run);splits={s:[r for r in rows if r['row']['split']==s and not r['row'].get('control')] for s in ['fit','selection','calibration','development','confirm']}
 fit=splits['fit'];sel=splits['selection'];test=splits['confirm'];heads=list(fit[0]['offline']['positive']);rank_candidates=[1,2,4,8]
 by_rank={};objectives=[]
 for rank in rank_candidates:
  candidates={};excluded={};energies=[]
  for h in heads:
   try:
    c=estimate(fit,h,rank);candidates[h]=c
    e=retention(sel,h,torch.tensor(c['basis'],dtype=torch.float64))
    if e is not None:energies.append(e)
   except ExcludedHead as e:excluded[head_name(h)]=str(e)
  value=sum(energies)/len(energies)-rank/128 if energies else -1e9
  objectives.append({'rank':rank,'mean_selection_all_difference_energy':None if not energies else sum(energies)/len(energies),'objective':value,'candidates':len(candidates),'excluded':excluded})
  by_rank[rank]=candidates
 rank=max(objectives,key=lambda x:(x['objective'],-x['rank']))['rank'];candidates=by_rank[rank]
 observations={h:[] for h in candidates}
 for r in sel:
  for h,c in candidates.items():
   observations[h].append(validation_observation(*(r['offline'][s][h] for s in ['positive','negative','unknown']),
       torch.tensor(c['direction'],dtype=torch.float64),r['queries']['positive'][h],r['queries']['negative'][h],r['cos'],r['sin']))
 provenance={'template_id':fit[0]['template_id'],'fit_sources':sorted({r['row']['source_id'] for r in fit}),
  'selection_sources':sorted({r['row']['source_id'] for r in sel}),'specification_commit':'60027c426a3155bf55c26273f4c3f49fd563ce1f',
  'data_kind':'clinical_counterfactual_vignette','code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
 expert=select_expert(candidates,observations,FitConfig(rank,4),identity=fit[0]['identity'],provenance=provenance)
 save_expert(expert,out/'expert.json')
 # Temperature is fitted only on calibration labels.
 calibration=[{'logits':c['logits'],'label':gold(r,s)} for r in splits['calibration'] for s,c in r['classification'].items() if 'logits' in c]
 temperature=fit_temperature(calibration,split='calibration')
 thresholds=[]
 for prob in [0.,.7,.9]:
  for margin in [0.,.1]:
   accepted=[];total=0
   for r in splits['development']:
    for s,c in r['classification'].items():
     total+=1
     if 'logits' not in c:continue
     d=summarize_logits(c['logits'],temperature=temperature['temperature'],probability_threshold=prob,margin_threshold=margin)
     if not d['abstain']:accepted.append(d['prediction']==gold(r,s))
   acc=sum(accepted)/len(accepted) if accepted else 0.
   thresholds.append({'probability':prob,'margin':margin,'accepted':len(accepted),'total':total,'accepted_accuracy':acc})
 chosen=max(thresholds,key=lambda x:((x['accepted_accuracy']>=.9),x['accepted'] if x['accepted_accuracy']>=.9 else x['accepted_accuracy'],x['accepted'],-x['probability'],-x['margin']))
 # Development-only perturbation scale for direct score bias, same frozen heads and routing.
 bias={}
 for name in expert['selected_heads']:
  h=tuple(map(int,name.split(':')));v=torch.tensor(candidates[h]['direction'],dtype=torch.float64);values=[]
  from src.aster.manuscript_core import rotate
  for r in splits['development']:
   shift=rotate(v[None,:],r['cos'],r['sin'])
   for state in ['positive','negative']: values.extend((r['queries'][state][h]@shift.T/128**.5).flatten().tolist())
  bias[name]=(sum(x*x for x in values)/len(values))**.5 if values else 0.
 freeze={'rank':rank,'top_k':4,'selected_heads':expert['selected_heads'],'temperature':temperature,'thresholds':chosen,
  'threshold_candidates':thresholds,'gain_candidates':[.25,1.,4.],'bias_unit_rms':bias,'random_seed':20261030,
  'max_new_tokens':12,'code_commit':provenance['code_commit'],'scope':'rule-verified clinical counterfactuals; no physician adjudication'}
 write(out/'freeze_pre_gain.json',freeze);write(out/'rank_selection.json',objectives)
 # Full training spectrum, fixed training basis evaluated on heldout per-token differences.
 half_sources=sorted({r['row']['source_id'] for r in fit});left=[r for r in fit if r['row']['source_id'] in half_sources[:len(half_sources)//2]];right=[r for r in fit if r not in left]
 statistics={};rng=torch.Generator().manual_seed(20261030)
 for h,c in candidates.items():
  basis=torch.tensor(c['basis'],dtype=torch.float64);v=torch.tensor(c['direction'],dtype=torch.float64);mean=torch.tensor(c['mean'],dtype=torch.float64)
  vals=torch.tensor(c['eigenvalues']).clamp_min(0);curve=(vals.cumsum(0)/vals.sum()).tolist()
  record={'training_energy_curve':curve,'training_energy_retention':c['spectral_energy_retention'],'mean_norm_retention':c['mean_norm_retention'],
   'heldout_all_difference_energy':retention(test,h,basis),'numerical_rank':c['numerical_rank'],'projected_mean_norm':c['projected_mean_norm']}
  try:
   lb=torch.tensor(estimate(left,h,rank)['basis'],dtype=torch.float64);rb=torch.tensor(estimate(right,h,rank)['basis'],dtype=torch.float64)
   record['guideline_half_overlap']=subspace_overlap(lb,rb)
   record['random_overlap']=[subspace_overlap(lb,torch.linalg.qr(torch.randn(128,rank,generator=rng,dtype=torch.float64)).Q) for _ in range(3)]
  except ExcludedHead as e:record['half_failure']=str(e)
  cosines=[]
  for r in test:
   d=(r['offline']['positive'][h]-r['offline']['negative'][h]).mean(0);den=float(d.norm()*v.norm())
   cosines.append({'pair_id':r['row']['pair_id'],'source_id':r['row']['source_id'],'cosine':None if den<=1e-12 else float(d@v)/den,'mean_norm':float(d.norm())})
  record['unseen_rule_cosines']=cosines
  # Responses per case, GQA query head, target token. Persist raw diagnostic matrices on server.
  response={}
  for r in test:
   shift=rotate(v[None,:],r['cos'],r['sin'])
   response[r['row']['pair_id']]={s:(r['queries'][s][h]@shift.T/128**.5).tolist() for s in ['positive','negative']}
  record['heldout_query_responses']=response
  controls=[]
  for r in rows:
   if r['row'].get('control'):
    controls.append({'pair_id':r['row']['pair_id'],'control':r['row']['control'],'energy':retention([r],h,basis),
     'directed_mean':float(((r['offline']['positive'][h]-r['offline']['negative'][h])@v).mean())})
  record['confound_controls']=controls
  statistics[head_name(h)]=record
 write(out/'statistics.json',statistics)
 # Sample-size sensitivity: guideline-cluster sampling; all rank candidates, three fixed repetitions.
 sensitivity=[]
 sources=sorted({r['row']['source_id'] for r in fit})
 for size in sorted({min(n,len(fit)) for n in [4,8,16,24]}):
  for rep in range(3):
   rr=random.Random(20261030+rep);ss=sources.copy();rr.shuffle(ss);chosen_sources=set(ss[:size//2]);subset=[r for r in fit if r['row']['source_id'] in chosen_sources]
   for rk in rank_candidates:
    train_energy=[];held_energy=[];excluded=0
    for h in heads:
     try:
      c=estimate(subset,h,rk);train_energy.append(c['spectral_energy_retention']);e=retention(sel,h,torch.tensor(c['basis'],dtype=torch.float64))
      if e is not None:held_energy.append(e)
     except ExcludedHead:excluded+=1
    sensitivity.append({'requested_pairs':size,'actual_pairs':len(subset),'repeat':rep,'rank':rk,'excluded_heads':excluded,
      'mean_train_energy':sum(train_energy)/len(train_energy) if train_energy else None,'mean_selection_energy':sum(held_energy)/len(held_energy) if held_energy else None})
 write(out/'sample_sensitivity.json',sensitivity)
 summary={'status':'complete','submitted_and_accepted':{s:len(v) for s,v in splits.items()},'rank':rank,'valid_candidates':len(candidates),
  'selected_heads':expert['selected_heads'],'mean_training_retention':sum(c['spectral_energy_retention'] for c in candidates.values())/len(candidates),
  'mean_holdout_retention':sum(r['heldout_all_difference_energy'] for r in statistics.values() if r['heldout_all_difference_energy'] is not None)/sum(r['heldout_all_difference_energy'] is not None for r in statistics.values()),
  'calibration_records':len(calibration),'rule_verified_clinical_pairs':len(rows),'physician_adjudicated_pairs':0}
 write(out/'summary.json',summary)
 # Standard static scientific plot.
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 curves=torch.tensor([s['training_energy_curve'] for s in statistics.values()])
 fig,ax=plt.subplots(figsize=(6,4));ax.plot(range(1,129),curves.mean(0).numpy(),label='Training spectrum, mean KV head')
 ax.axvline(rank,color='gray',ls='--',label=f'Chosen rank {rank}');ax.set(xlabel='Rank',ylabel='Cumulative difference energy',ylim=(0,1.02));ax.legend();fig.tight_layout();fig.savefig(out/'spectrum.png',dpi=180);plt.close(fig)
 print(json.dumps(summary))
if __name__=='__main__':main()

