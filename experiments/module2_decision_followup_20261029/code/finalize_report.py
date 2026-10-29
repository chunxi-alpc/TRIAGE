from pathlib import Path
import json, csv, hashlib, collections, statistics, random, zipfile
import numpy as np
root=Path('/home/gcx/TRIAGE-worktrees')
run=root/'applicability-decision-20261029-run02'
code=root/'applicability-decision-20261029-code'
out=run/'final_review'
out.mkdir(exist_ok=True)
def load(name): return json.loads((run/name).read_text())
def lines(name): return [json.loads(x) for x in (run/name).read_text().splitlines() if x]
def dump(name,v): (out/name).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvout(name,data):
 with (out/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
frozen=load('select/frozen.json')
hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in frozen['hashes'].items()}
assert all(hashes.values()),'Frozen artifact changed'
expected={'audit/predictions.jsonl':288,'localize/patches.jsonl':14688,'train/optimization.jsonl':480,'select/predictions.jsonl':76128,'confirm/predictions.jsonl':1152,'confirm/ablations.jsonl':3456}
coverage={}
for file,n in expected.items():
 data=lines(file);assert len(data)==n,(file,len(data),n)
 if 'predictions' in file or 'ablations' in file:
  keys=[(x['id'],x.get('method','baseline'),x.get('head_set'),x.get('mode'),x.get('tau')) for x in data]
 elif 'patches' in file:
  keys=[(x['id'],x['role'],x['component'],x['layer'],x['head']) for x in data]
 else: keys=[(x['head_set'],x['method'],x['step']) for x in data]
 assert len(set(keys))==len(keys),file
 coverage[file]={'rows':len(data),'unique_cells':len(set(keys)),'expected':n}
dump('completion_validation.json',{'coverage':coverage,'frozen_hashes_match':hashes,'pipeline':load('pipeline_exit.json'),'new_gpu_runs_during_review':0})
summary=load('confirm/summary.json')
metrics=[]
for row in summary:
 gs=row['per_guideline'];metrics.append({k:row[k] for k in ['method','head_set','mode','tau','n','guidelines','accuracy','paired_accuracy','margin']}|{'corrected':sum(x['corrected'] for x in gs),'regressed':sum(x['regressed'] for x in gs),'improved_guidelines':sum(x['accuracy']>.5 for x in gs)})
csvout('heldout_metrics.csv',metrics)
csvout('heldout_ablation_metrics.csv',[{k:x[k] for k in ['method','head_set','mode','tau','accuracy','paired_accuracy','margin']} for x in load('confirm/ablation_summary.json')])
csvout('heldout_by_guideline.csv',[{'method':r['method'],**g} for r in summary for g in r['per_guideline']])
patch=collections.defaultdict(list)
for x in lines('localize/patches.jsonl'): patch[(x['component'],x['role'],x['layer'],x['head'])].append(x['margin_change'])
csvout('localization_summary.csv',[dict(zip(['component','role','layer','head'],k),n=len(v),mean_margin_change=statistics.mean(v),positive_fraction=sum(y>0 for y in v)/len(v)) for k,v in patch.items()])
audit=lines('audit/predictions.jsonl');scoring=[]
for split in ['fit','select']:
 q=[x for x in audit if x['split']==split];pairs=collections.defaultdict(list);correct=[]
 for x in q:
  z=x['secondary_scores'];prediction='A' if z[' A']['score']>z[' B']['score'] else 'B'
  ok=prediction==x['gold'];correct.append(ok);pairs[x['pair']].append(ok)
 scoring.append({'split':split,'n':len(q),'bare_accuracy':sum(x['correct'] for x in q)/len(q),'bare_predicted_B':sum(x['pred']=='B' for x in q),'space_prefix_accuracy':statistics.mean(correct),'space_prefix_paired':statistics.mean(all(v) for v in pairs.values()),'max_cached_full_error':max(x['cached_vs_full_error'] for x in q),'min_tokens':min(x['length'] for x in q),'max_tokens':max(x['length'] for x in q)})
csvout('scoring_audit.csv',scoring)
durations=[{'stage':p.parent.name,**json.loads(p.read_text())} for p in sorted(run.glob('*/status.json'))]
dump('runtime.json',{'stages':durations,'total_stage_seconds':sum(x.get('elapsed_seconds',0) for x in durations)})
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
rng=np.random.default_rng(261029)
plotrows=[]
for r in summary:
 vals=np.array([x['accuracy']-.5 for x in r['per_guideline']])*100
 boots=rng.choice(vals,size=(10000,len(vals)),replace=True).mean(axis=1)
 lo,hi=np.quantile(boots,[.025,.975])
 plotrows.append({'method':r['method'],'gain_pp':float(vals.mean()),'lower_95':float(lo),'upper_95':float(hi)})
csvout('figure_data.csv',plotrows)
fig,ax=plt.subplots(figsize=(9,5.6))
yy=np.arange(len(plotrows));means=np.array([x['gain_pp'] for x in plotrows])
ax.barh(yy,means,color=['#248377' if x['method']=='optimized_subspace' else '#7b8ea3' for x in plotrows])
ax.errorbar(means,yy,xerr=np.array([[x['gain_pp']-x['lower_95'] for x in plotrows],[x['upper_95']-x['gain_pp'] for x in plotrows]]),fmt='none',ecolor='#303642',capsize=3)
ax.set_yticks(yy,[x['method'] for x in plotrows]);ax.invert_yaxis();ax.axvline(0,color='black',lw=.7)
ax.set_xlabel('Accuracy change versus no edit (percentage points)')
ax.set_title('Held-out rule-choice results: 12 guidelines, 96 prompts')
ax.text(0,-.19,'Descriptive 95% paired guideline bootstrap intervals. Primary tests use 97.5% intervals.\nAll configurations selected before confirmation; oracle applicability routing.',transform=ax.transAxes,fontsize=8)
ax.grid(axis='x',alpha=.15);fig.tight_layout()
fig.savefig(out/'heldout_accuracy.png',dpi=180,bbox_inches='tight')
fig.savefig(out/'heldout_accuracy.pdf',bbox_inches='tight')
print(json.dumps({'coverage':coverage,'review':'completed','output':str(out)},ensure_ascii=False))
