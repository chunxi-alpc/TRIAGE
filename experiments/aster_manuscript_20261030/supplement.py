"""Registered same-rank random references and source-cluster energy intervals."""
import argparse,json,random
from collections import defaultdict
from pathlib import Path
import torch
from .analyze import load,retention
from .prepare import read,write
from .report import interval
from src.aster.manuscript_core import parse_head

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run);torch.set_num_threads(4)
 rows=[r for r in load(run) if r['row']['split']=='heldout_exploratory' and not r['row'].get('control')]
 expert=json.loads((run/'analysis/expert.json').read_text());rng=torch.Generator().manual_seed(20261030)
 sources=sorted({r['row']['source_id'] for r in rows});numerators=[];denominators=[];refs={}
 for name,c in expert['candidates'].items():
  h=parse_head(name);basis=torch.tensor(c['basis'],dtype=torch.float64);rank=basis.shape[1]
  random_basis=[torch.linalg.qr(torch.randn(128,rank,generator=rng,dtype=torch.float64)).Q for _ in range(3)]
  refs[name]=[retention(rows,h,b) for b in random_basis]
  aa=[];bb=[]
  for source in sources:
   subset=[r for r in rows if r['row']['source_id']==source];n=d=0.
   for r in subset:
    diff=r['offline']['positive'][h]-r['offline']['negative'][h]
    n+=float((diff@basis).square().sum(-1).mean());d+=float(diff.square().sum(-1).mean())
   aa.append(n);bb.append(d)
  numerators.append(aa);denominators.append(bb)
 aa=torch.tensor(numerators,dtype=torch.float64);bb=torch.tensor(denominators,dtype=torch.float64)
 rng2=random.Random(20261030);rep=[]
 for _ in range(1000):
  index=[rng2.randrange(len(sources)) for _ in sources]
  den=bb[:,index].sum(1);valid=den>0
  rep.append(float((aa[:,index].sum(1)[valid]/den[valid]).mean()))
 rep.sort();reference=[v for vals in refs.values() for v in vals if v is not None]
 write(run/'summary/random_energy_reference.json',{'seed':20261030,'repeats':3,'rank':expert['config']['rank'],'per_head':refs,
  'mean_random_heldout_energy':sum(reference)/len(reference),'heads':len(refs)})
 write(run/'summary/energy_cluster_interval.json',{'guideline_clusters':len(sources),'repeats':1000,'mean_energy':float((aa.sum(1)/bb.sum(1)).mean()),'ci95':[rep[24],rep[974]],
  'scope':'fixed training basis; source-cluster bootstrap of exploratory heldout only'})

def attention_intervals(run):
 rows=read(run/'evaluate0/results.jsonl')+read(run/'evaluate1/results.jsonl')
 rows=[x for x in rows if x['control'] is None and x['route']=='predicted' and x['gold']!='NONE']
 def ratio(x):
  byhead={}
  for item in x['result']['attention']:
   key=item['layer'],item['kv_head']
   if key not in byhead:byhead[key]=item
  good=x['gold'];other='R2' if good=='R1' else 'R1'
  vals=[sum(item['mass'][good])/sum(item['mass'][other]) for item in byhead.values() if sum(item['mass'][other])>0]
  return sum(vals)/len(vals) if vals else None
 index={(x['pair_id'],x['state'],x['method']):x for x in rows}
 comparisons=[]
 for method,reference in [(m,'baseline') for m in ['spectral','mean','random','reverse','score_bias','enhance_only','suppress_only']]+[('spectral','mean'),('spectral','score_bias')]:
  grouped=defaultdict(list)
  for x in rows:
   if x['method']==method:grouped[x['source_id']].append(ratio(x)-ratio(index[x['pair_id'],x['state'],reference]))
  flat=[v for vals in grouped.values() for v in vals]
  comparisons.append({'method':method,'reference':reference,'ratio_change':sum(flat)/len(flat),'cluster_ci95':interval(grouped),'guidelines':len(grouped)})
 write(run/'summary/attention_cluster_intervals.json',comparisons)

if __name__=='__main__':
 main()
 import sys
 attention_intervals(Path(sys.argv[sys.argv.index('--run')+1]))

