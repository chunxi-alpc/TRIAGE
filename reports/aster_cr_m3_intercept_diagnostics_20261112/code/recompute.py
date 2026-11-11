"""Recompute tables from saved predictions only; no model or original data access."""
import run as r
from collections import defaultdict
import numpy as np

def additional(scope,nodes,pairs,questions,qps):
 boot=r.ev.SourceBootstrap([n['source_id'] for n in nodes]);names=list(nodes[0]['scores']);datasets=defaultdict(lambda:defaultdict(dict));nb={n['view_id']:n for n in nodes};qby={(q['method'],q['item_id'],q['variant']):q for q in questions}
 for p in qps:
  for f in ['paired_correct','relation_correct']:datasets[p['group']+'_'+f][p['method']][p['pair_id']]=(p['source_id'],p[f])
 for p in pairs:
  if p['group'] not in ['native_preserving','native_changing']:continue
  a=nb[p['left']];b=nb[p['right']];qa=qby[(p['method'],a['item_id'],a['variant'])];qb=qby[(p['method'],b['item_id'],b['variant'])]
  types=['single' if len(q['gold'])==1 else 'multiple' for q in [qa,qb]];kind=types[0] if types[0]==types[1] else 'mixed_single_multiple';none='both_gold_none' if qa['gold_none'] and qb['gold_none'] else 'one_gold_none' if qa['gold_none'] or qb['gold_none'] else 'no_gold_none'
  for s in [kind,none]:
   for f in ['paired_correct','relation_correct']:datasets[p['group']+'_'+f+'/'+s][p['method']][p['pair_id']]=(p['source_id'],p[f])
 for level,methods in datasets.items():
  vals={}
  for m,rr in methods.items():
   x=list(rr.values());vals[m]=boot.mean([a[1] for a in x],[a[0] for a in x]);r.ALL_BOOT.append(dict(scope=scope,comparison=m,metric=level,**boot.summary(vals[m]),sources=len(boot.sources),draws=2000))
   next(t for t in r.ALL_METRICS if t['scope']==scope and t['method']==m).update({level:r.ev.finite(vals[m][0]),level+'_n':len(x),level+'_correct':sum(a[1] for a in x)})
  for a,b in r.pairs_to_compare(names):
   if a not in methods or b not in methods:continue
   ar=methods[a];br=methods[b];keys=sorted(ar.keys()&br.keys());sg=[ar[k][0] for k in keys];wr=[ar[k][1] and not br[k][1] for k in keys];rw=[br[k][1] and not ar[k][1] for k in keys];comp=a+' minus '+b
   r.ALL_BOOT.append(dict(scope=scope,comparison=comp,metric=level,**boot.summary(vals[a]-vals[b]),sources=len(boot.sources),draws=2000))
   r.ALL_TRANS.append(dict(scope=scope,comparison=comp,level=level,valid_denominator=len(keys),wrong_to_right=sum(wr),right_to_wrong=sum(rw),kept_right=sum(ar[k][1] and br[k][1] for k in keys),kept_wrong=sum(not ar[k][1] and not br[k][1] for k in keys)))
   for tag,xx in [('wrong_to_right',wr),('right_to_wrong',rw)]:r.ALL_BOOT.append(dict(scope=scope,comparison=comp,metric=level+'/'+tag,**boot.summary(boot.sums(xx,sg)),sources=len(boot.sources),draws=2000))

def main():
 for d in sorted((r.OUT/'predictions').iterdir()):
  if not (d/'nodes.jsonl.gz').exists():continue
  data=[r.load('predictions/'+d.name+'/'+n+'.jsonl') for n in ['nodes','pairs','paths','questions']]
  r.summarize(d.name,*data)
  if d.name.startswith('medpic'):additional(d.name,data[0],data[1],data[3],r.load('predictions/'+d.name+'/question_pairs.jsonl'))
 for name,data in [('metrics',r.ALL_METRICS),('source_bootstrap',r.ALL_BOOT),('transitions',r.ALL_TRANS)]:r.ev.write_csv(r.OUT/(name+'.csv'),data)
 r.write(r.OUT/'recomputed.json',dict(from_saved_predictions_only=True,metric_rows=len(r.ALL_METRICS),bootstrap_rows=len(r.ALL_BOOT),transitions=len(r.ALL_TRANS)))
if __name__=='__main__':main()
