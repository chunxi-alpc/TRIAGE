"""Paired tree-cluster CIs; descriptive engineering results, no E2 pass claim."""
import collections,json
from pathlib import Path
import numpy as np
P=Path('/home/gcx/aster_e0_e8_20261024/private');OUT=Path(__file__).parent/'results'
def paired_ci(rows,a,b):
 trees=sorted({r['tree'] for r in rows});sums=np.array([sum(int(r['arms'][a]['correct'])-int(r['arms'][b]['correct']) for r in rows if r['tree']==t) for t in trees]);counts=np.array([sum(r['tree']==t for r in rows) for t in trees]);rng=np.random.default_rng(0);ix=rng.integers(0,len(trees),(10000,len(trees)));boot=sums[ix].sum(1)/counts[ix].sum(1)
 return {'difference':float(sums.sum()/counts.sum()),'tree_cluster_95CI':np.quantile(boot,[.025,.975]).tolist(),'trees':len(trees),'resamples':10000,'seed':0}
def main():
 reports={}
 for name in ['llama_e1_v2','qwen_e1']:
  folder=P/name
  if not (folder/'summary.json').exists():continue
  s=json.loads((folder/'summary.json').read_text());rows=list(map(json.loads,(folder/'scores.jsonl').read_text().splitlines()));result={'run':s,'by_variant':{}}
  for v in ['NC','AC','SW']:
   rr=[r for r in rows if r['variant']==v];result['by_variant'][v]={'n':len(rr),'accuracy':{arm:sum(r['arms'][arm]['correct'] for r in rr)/len(rr) for arm in rr[0]['arms']},'paired_hard_vs_crop':paired_ci(rr,'group9_hard','group3'),'paired_strict_vs_crop':paired_ci(rr,'strict_diagnostic','group3')}
  reports[name]=result;(OUT/(name+'.json')).write_text(json.dumps(result,indent=2));(OUT/(name+'_scores.jsonl')).write_text((folder/'scores.jsonl').read_text())
 if not reports:return
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,len(reports),figsize=(6*len(reports),4),squeeze=False)
 for ax,(name,result) in zip(axs[0],reports.items()):
  arms=['group3','group4','group9_hard','strict_diagnostic'];x=np.arange(3)
  for j,arm in enumerate(arms):ax.bar(x+(j-1.5)*.2,[result['by_variant'][v]['accuracy'][arm]*100 for v in ['NC','AC','SW']],width=.19,label=arm)
  ax.set_xticks(x,['NC','AC','SW']);ax.set_ylim(0,100);ax.set_ylabel('Accuracy (%)');ax.set_title(name+' / development pilot');ax.legend(fontsize=7)
 fig.tight_layout();fig.savefig(OUT/'E1_pilot.png',dpi=160);plt.close(fig)
 (OUT/'summary.json').write_text(json.dumps(reports,indent=2))
if __name__=='__main__':main()
