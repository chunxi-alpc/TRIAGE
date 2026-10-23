"""Static research curves from saved numeric summaries only."""
import json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(sys.argv[1]);d=json.loads(p.read_text())
if d['phase']=='diagnostic':
 fig,axes=plt.subplots(1,3,figsize=(13,3.6))
 for method in ('app','signed','negative'):
  rs=sorted([r for r in d['curve'] if r['variant'].startswith('B_') and r['method']==method],key=lambda r:r['gain'])
  axes[0].plot([r['gain'] for r in rs],[r['accuracy'] for r in rs],'-o',label=method)
  axes[1].plot([r['gain'] for r in rs],[r['margin_prior'] for r in rs],'-o',label=method)
 rs=sorted([r for r in d['curve'] if r['variant'].startswith('C_') and r['method']=='negative'],key=lambda r:r['gain'])
 axes[2].plot([r['gain'] for r in rs],[r['accuracy'] for r in rs],'-o',label='C wrong-sibling accuracy')
 axes[2].plot([r['gain'] for r in rs],[r['target_attention_mass'][1] for r in rs],'-s',label='C sibling attention mass')
 for ax,title in zip(axes,('B decision accuracy (development)','B gold minus original logit','C negative control (development)')):
  ax.set_title(title);ax.set_xlabel('gain');ax.legend(fontsize=8);ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(p.parent/'gain_curves.svg');plt.close(fig)
else:
 fig,axes=plt.subplots(1,2,figsize=(10,4))
 for method in ('annotate','prune','prune_cad','app','unsigned','signed','annotate_signed'):
  if method=='unsigned' and not any('/unsigned/' in k for k in d['noise']):continue
  for ax,variant,stratum,metric in ((axes[0],'B_applicable_rule_changed','K','accuracy'),(axes[1],'C_sibling_rule_changed','all','sibling_misuse_rate')):
   vals=[]
   for rate in (0,.1,.2,.3):
    v=[d['noise']['/'.join(map(str,('heldout',seed,rate,variant,method,stratum)))][metric] for seed in (11,22,33)]
    vals.append(sum(v)/len(v) if all(x is not None for x in v) else float('nan'))
   ax.plot([0,.1,.2,.3],vals,'-o',label=method)
 for ax,title in zip(axes,('Heldout B / K accuracy','Heldout C sibling misuse')):
  ax.set_title(title);ax.set_xlabel('corrupted applicability fraction');ax.legend(fontsize=7);ax.grid(alpha=.2)
 fig.suptitle('Mean over 3 shared corruption seeds; points are not independent replicates',fontsize=9);fig.tight_layout();fig.savefig(p.parent/'noise_curves.svg');plt.close(fig)
