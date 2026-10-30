"""Merge development shards and summarize complete cluster-paired controls."""
import argparse,json,random,subprocess
from pathlib import Path
from collections import defaultdict
from .prepare import read,write
from .controls import METHODS

def freeze(run):
 out=run/'gain';out.mkdir(exist_ok=False);records=[]
 sources=[]
 for shard in [0,1]:
  base=run/('gain'+str(shard));s=json.loads((base/'status.json').read_text());assert s['status']=='complete';sources.append(json.loads((base/'frozen.json').read_text()));records.append(s['results'])
 result=[]
 for i,g in enumerate([.25,1.,4.]):
  n=sum(r[i]['denominator'] for r in records)
  result.append({'gain':g,'denominator':n,'first_token_accuracy':sum(r[i]['first_token_accuracy']*r[i]['denominator'] for r in records)/n,
   'rule_accuracy':sum(r[i]['rule_accuracy']*r[i]['denominator'] for r in records)/n})
 chosen=max(result,key=lambda r:(r['first_token_accuracy'],-r['gain']))['gain']
 config=sources[0];config.update(g_plus=chosen,g_minus=chosen,gain_selection_results=result,scope='synthetic mechanism only; exploratory heldout; no clinical gold')
 write(out/'frozen.json',config);write(out/'status.json',{'status':'complete','gain':chosen,'results':result})

def interval(values):
 rng=random.Random(20261030);keys=sorted(values);samples=[]
 for _ in range(1000):
  chosen=[rng.choice(keys) for _ in keys];nums=[x for k in chosen for x in values[k]];samples.append(sum(nums)/len(nums))
 samples.sort();return [samples[24],samples[974]]

def summarize(run):
 out=run/'summary';out.mkdir(exist_ok=False);rows=[]
 for shard in [0,1]:
  root=run/('evaluate'+str(shard));s=json.loads((root/'status.json').read_text());assert s['status']=='complete';rows+=read(root/'results.jsonl')
 main=[r for r in rows if r['control'] is None];index={(r['pair_id'],r['state'],r['route'],r['method']):r for r in rows}
 comparisons=[]
 for control in [None,'nondetermining_fact','paraphrase','action_mapping']:
  for route in ['gold','predicted']:
   for method in METHODS:
    group=[r for r in rows if r['control']==control and r['route']==route and r['method']==method]
    if not group:continue
    delta=defaultdict(list);accuracy=defaultdict(list);corrected=regressed=0;covered=0;attention=[]
    for r in group:
     b=index[r['pair_id'],r['state'],route,'baseline'];d=int(r['correct'])-int(b['correct']);delta[r['source_id']].append(d);accuracy[r['source_id']].append(int(r['correct']))
     corrected+=d>0;regressed+=d<0;res=r.get('result',{});covered+=any(res.get('plan',{}).get('gains',{}).values())
     if r['gold'] in ['R1','R2']:
      sibling='R2' if r['gold']=='R1' else 'R1'
      byhead=defaultdict(list)
      for item in res.get('attention',[]):byhead[item['layer'],item['kv_head']].append(item)
      for key,items in byhead.items():
       for pos,item in enumerate(items):
        c=item['mass'].get(r['gold'],[0]*4);s=item['mass'].get(sibling,[0]*4)
        attention.append({'stage':'first_query' if pos==0 else 'generation','correct':sum(c)/len(c),'sibling':sum(s)/len(s),
          'ratio':None if sum(s)==0 else sum(c)/sum(s)})
    att={}
    for stage in ['first_query','generation']:
     vals=[x for x in attention if x['stage']==stage];ratios=[x['ratio'] for x in vals if x['ratio'] is not None]
     if vals:att[stage]={'correct_mass':sum(x['correct'] for x in vals)/len(vals),'sibling_mass':sum(x['sibling'] for x in vals)/len(vals),'mean_mass_ratio':sum(ratios)/len(ratios) if ratios else None}
    comparisons.append({'control':control,'route':route,'method':method,'denominator':len(group),'guidelines':len(accuracy),
     'correct':sum(r['correct'] for r in group),'accuracy':sum(r['correct'] for r in group)/len(group),'accuracy_cluster_ci':interval(accuracy),
     'delta_accuracy':sum(x for v in delta.values() for x in v)/len(group),'delta_cluster_ci':interval(delta),
     'corrected':corrected,'regressed':regressed,'technical_failures':sum(r['status']=='failed' for r in group),
     'invalid_outputs':sum(r.get('invalid_output',False) for r in group),'scoring_failures':0,'routing_edit_coverage':covered/len(group),
     'generated_tokens':sum(len(r.get('result',{}).get('token_ids',[])) for r in group),'attention':att})
 write(out/'comparisons.json',comparisons)
 stats=json.loads((run/'analysis/statistics.json').read_text());selected=json.loads((run/'gain/frozen.json').read_text())['selected_heads']
 structure={}
 for name in selected:
  s=stats[name];cos=[c['cosine'] for c in s['unseen_rule_cosines'] if c['cosine'] is not None];responses=[v for d in s['heldout_query_responses'].values() for vv in d.values() for v in vv for v in v]
  structure[name]={'train_energy':s['training_energy_retention'],'heldout_energy':s['heldout_all_difference_energy'],'mean_norm_retention':s['mean_norm_retention'],
   'guideline_half_overlap':s.get('guideline_half_overlap'),'random_overlap_mean':sum(s.get('random_overlap',[0]))/len(s.get('random_overlap',[0])),
   'unseen_rule_cosine_mean':sum(cos)/len(cos) if cos else None,'unseen_rule_positive_fraction':sum(c>0 for c in cos)/len(cos) if cos else None,
   'heldout_query_positive_fraction':sum(v>0 for v in responses)/len(responses) if responses else None}
 write(out/'selected_structure.json',structure)
 status={'status':'complete','matrix_units':len(rows),'expected_units':1440,'technical_failures':sum(r['status']=='failed' for r in rows),
  'invalid_outputs':sum(r.get('invalid_output',False) for r in rows),'clinical_end_to_end_run':False,
  'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'mechanism_only':True}
 assert len(rows)==1440
 write(out/'status.json',status);print(json.dumps(status))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','summarize']);p.add_argument('--run',required=True);a=p.parse_args()
 (freeze if a.stage=='freeze' else summarize)(Path(a.run))

