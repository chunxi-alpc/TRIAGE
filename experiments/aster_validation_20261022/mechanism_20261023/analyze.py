"""Paired branch/action statistics; no invented gold or causal conclusion."""
import argparse,collections,json,math
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from report import render

def bootstrap(values,diseases,reps=2000):
 if not values:return None
 groups=collections.defaultdict(list)
 for x,d in zip(values,diseases):groups[d].append(x)
 if len(groups)<2:return None
 sums=np.array([sum(v) for v in groups.values()]);ns=np.array([len(v) for v in groups.values()]);rng=np.random.default_rng(20261023);ix=rng.integers(len(groups),size=(reps,len(groups)));return np.quantile(sums[ix].sum(1)/ns[ix].sum(1),[.025,.975]).tolist()
def paired(a,b):
 assert {r['id'] for r in a}=={r['id'] for r in b}
 bb={r['id']:r for r in b};d=[int(bb[r['id']]['correct'])-int(r['correct']) for r in a];win=d.count(1);lose=d.count(-1)
 return {'n':len(d),'delta':sum(d)/len(d) if d else None,'disease_bootstrap_ci':bootstrap(d,[r['disease'] for r in a]),'wins':win,'losses':lose,'p':binomtest(win,win+lose,.5).pvalue if win+lose else 1}
def cell(rows):
 n=len(rows);return {'n':n,'accuracy':sum(r['correct'] for r in rows)/n if n else None,'accuracy_disease_ci':bootstrap([r['correct'] for r in rows],[r['disease'] for r in rows]),'parse_failure_rate':sum(r['parse_failure'] for r in rows)/n if n else None,**{key:sum(r.get(key,False) for r in rows)/n if n else None for key in ['original_action_selected','sibling_action_selected','branch_permitted_action']}}
def harmonic(xs):return len(xs)/sum(1/x for x in xs) if xs and min(xs)>0 else 0.
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 manifest=json.loads(a.input.with_suffix('.manifest.json').read_text());rows=list(map(json.loads,a.input.read_text().splitlines()));cases={r['id']:r for r in map(json.loads,a.dataset.read_text().splitlines())};assert manifest['complete'] and len(rows)==manifest['rows']
 keys=[(r['id'],r['variant'],r['method']) for r in rows];assert len(keys)==len(set(keys));lookup={k:r for k,r in zip(keys,rows)};ids={r['id'] for r in rows};K={r['id'] for r in rows if r['stratum']=='K'}
 buckets=collections.defaultdict(list)
 for r in rows:
  for st in ['all',r['stratum']]:buckets[r['variant'],r['method'],st].append(r)
 report={'n_cases':len(ids),'K':len(K),'U':len(ids-K),'cells':{'/'.join(k):cell(v) for k,v in buckets.items()},'comparisons':{},'decomposition':{},'action_induced_branch_shift':{},'harmonic':{},'notes':['K/U fixed from original closed-book; not parameter-memory proof.','Identical condition paths form one branch; some branches permit multiple actions.','Branch and final action are separate calls; paired associations do not prove a single internal reasoning trajectory.','Single-source-path exact action correctness and branch-permitted action correctness are separate.','Memory is operational original-action selection only.','CS has strict verified coverage, not imputed cases.']}
 methods=sorted({r['method'] for r in rows if r['kind']=='action'})
 for v in ['A','B','C','D','CS']:
  for method in methods:
   for reference in ['existing_baseline','existing_prune','gold_branch_prune','existing_signed','visible_prune']:
    aa=buckets.get((v,reference,'all'),[]);bb=buckets.get((v,method,'all'),[])
    if aa and bb and reference!=method:report['comparisons'][v+'/'+reference+'->'+method]=paired(aa,bb)
  for mode in ['hidden','visible','hide_inapp']:
   for unique in [False,True]:
    selected=[i for i in ids if (i,v,'branch_'+mode) in lookup and (i,v,'existing_baseline') in lookup and (not unique or cases[i]['single_action_gold_branch'])];correct=[i for i in selected if lookup[i,v,'branch_'+mode]['correct']];bad=[i for i in correct if not lookup[i,v,'existing_baseline']['correct']]
    report['decomposition'][v+'/'+mode+('/unique_action_branch' if unique else '/all')]={'n':len(selected),'branch_correct':len(correct),'branch_correct_action_wrong':len(bad),'joint_rate':len(bad)/len(selected) if selected else None,'action_error_given_branch_correct':len(bad)/len(correct) if correct else None}
 # Harmful D flip: correct in A/hidden -> sibling in D/visible; reverse direction separately.
 for reference in ['branch_hidden','branch_visible']:
  for typ in ['all','judgment','objective_candidate','ambiguous','treatment_judgment']:
   for st in ['all','K','U']:
    eligible=[i for i in ids if (i,'A',reference) in lookup and (i,'D','branch_visible') in lookup and (typ=='all' or cases[i]['type']==typ) and (st=='all' or (i in K)==(st=='K'))];valid=[i for i in eligible if lookup[i,'A',reference]['prediction'] is not None and lookup[i,'D','branch_visible']['prediction'] is not None];harm=[];reverse=[];flips=[]
    for i in valid:
     b=lookup[i,'A',reference]['prediction'];d=lookup[i,'D','branch_visible']['prediction'];gold=chr(65+cases[i]['app_branch']);sib=chr(65+cases[i]['sib_branch']);harm.append(int(b==gold and d==sib));reverse.append(int(b==sib and d==gold));flips.append(int(b!=d))
    delta=[x-y for x,y in zip(harm,reverse)];h=sum(harm);rv=sum(reverse)
    report['action_induced_branch_shift']['/'.join([reference,typ,st])]={'n':len(eligible),'valid_pairs':len(valid),'invalid_pairs':len(eligible)-len(valid),'flip_rate':sum(flips)/len(valid) if valid else None,'harmful':h,'reverse':rv,'net_harmful_rate':sum(delta)/len(delta) if delta else None,'net_disease_ci':bootstrap(delta,[cases[i]['disease'] for i in valid]),'p':binomtest(h,h+rv,.5).pvalue if h+rv else 1}
 for method in methods:
  for variants,name in [(['B','C','D','CS'],'primary_BCDCS'),(['B','C','D'],'secondary_BCD')]:
   common=sorted(i for i in K if all((i,v,method) in lookup for v in variants));value=None;ci=None
   if common:
    matrix=np.array([[lookup[i,v,method]['correct'] for v in variants] for i in common],dtype=float);value=harmonic(matrix.mean(0).tolist());groups=collections.defaultdict(list)
    for j,i in enumerate(common):groups[cases[i]['disease']].append(j)
    if len(groups)>1:
     cs=list(groups.values());rng=np.random.default_rng(20261023);boots=[]
     for _ in range(2000):
      selected=np.concatenate([cs[j] for j in rng.integers(len(cs),size=len(cs))]);boots.append(harmonic(matrix[selected].mean(0).tolist()))
     ci=np.quantile(boots,[.025,.975]).tolist()
   report['harmonic'][method+'/'+name]={'n_common_K':len(common),'value':value,'disease_bootstrap_ci':ci,'variants':variants,'descriptive_only':len(common)<3 or ci is None,'note':'NA if no common cases; CS two cases in one disease are descriptive only, not a general method ranking'}
 report['A_no_decrease']={}
 for method in methods:
  comparison=report['comparisons'].get('A/existing_baseline->'+method)
  if comparison:
   ci=comparison['disease_bootstrap_ci'];report['A_no_decrease'][method]={'paired_delta':comparison['delta'],'ci':ci,'lower_bound_nonnegative':bool(ci and ci[0]>=0),'note':'A nonsignificant difference alone does not establish no degradation'}
 report['harmonic_comparisons']={}
 for method in methods:
  for reference in ['existing_baseline','visible_prune']:
   if method==reference:continue
   for variants,name in [(['B','C','D','CS'],'primary_BCDCS'),(['B','C','D'],'secondary_BCD')]:
    common=sorted(i for i in K if all((i,v,m) in lookup for v in variants for m in [method,reference]));delta=None;ci=None
    if common:
     aa=np.array([[lookup[i,v,reference]['correct'] for v in variants] for i in common],float);bb=np.array([[lookup[i,v,method]['correct'] for v in variants] for i in common],float);delta=harmonic(bb.mean(0).tolist())-harmonic(aa.mean(0).tolist());clusters=collections.defaultdict(list)
     for j,i in enumerate(common):clusters[cases[i]['disease']].append(j)
     if len(clusters)>1:
      cs=list(clusters.values());rng=np.random.default_rng(20261023);boots=[]
      for _ in range(2000):
       ix=np.concatenate([cs[j] for j in rng.integers(len(cs),size=len(cs))]);boots.append(harmonic(bb[ix].mean(0).tolist())-harmonic(aa[ix].mean(0).tolist()))
      ci=np.quantile(boots,[.025,.975]).tolist()
    report['harmonic_comparisons'][reference+'->'+method+'/'+name]={'n_common_K':len(common),'paired_delta':delta,'disease_bootstrap_ci':ci}
 # Hidden versus original visible actions isolates visibility without action swapping.
 report['original_action_visibility']={}
 for st in ['all','K','U']:
  aa=buckets.get(('A','branch_hidden',st),[]);bb=buckets.get(('A','branch_visible',st),[])
  if aa and bb:report['original_action_visibility'][st]=paired(aa,bb)
 # Multiple-comparison correction includes branch-shift and action comparisons.
 tests=list(report['comparisons'].values())+list(report['action_induced_branch_shift'].values())+list(report['original_action_visibility'].values());tests.sort(key=lambda r:r['p']);prev=0
 for j,t in enumerate(tests):prev=max(prev,min(1,t['p']*(len(tests)-j)));t['p_holm']=prev
 gate=report['action_induced_branch_shift']['branch_visible/all/K'];ci=gate['net_disease_ci'];required=bool(ci and ci[0]>0 and gate['p_holm']<.05)
 report['attention_gate']={'triggered':required,'criterion':'K: D visible harmful-minus-reverse branch flips vs A visible has disease-cluster CI lower>0 and Holm p<0.05','next':'development-only layer/head selection required before test blocking' if required else 'do not run attention blocking; systematic branch bias not established under prespecified gate'}
 a.out.mkdir(parents=True,exist_ok=True);(a.out/'summary.json').write_text(json.dumps(report,indent=2));(a.out/'scores.jsonl').write_text(''.join(json.dumps({k:v for k,v in r.items() if k!='completion'})+'\n' for r in rows));(a.out/'REPORT.zh.md').write_text(render(report));print('mechanism analysis complete',len(ids))
if __name__=='__main__':main()
