"""Independent CPU replay of shared S, fusion, keys and reported metrics."""
import argparse,json,math,gzip
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
 data=[json.loads(l) for l in (a.root/'dev.jsonl').read_text().splitlines()]
 expected={(c['id'],c['variant']):c for c in data};assert len(expected)==800
 summary=json.loads((a.report/'summary.json').read_text());result={}
 for model in ('qwen','llama'):
  with gzip.open(a.report/f'{model}_predictions.jsonl.gz','rt') as f:rows=[json.loads(l) for l in f]
  assert len(rows)==800 and {(r['id'],r['variant']) for r in rows}==set(expected)
  for r in rows:
   assert r['gold_option']==expected[r['id'],r['variant']]['gold_option']
   assert r['seed']==0 and r['split']=='dev'
   s=[sum(p for p,g in zip(r['p'],r['rule_to_action']) if g==j) for j in range(len(r['S']))]
   assert all(abs(a-b)<1e-9 for a,b in zip(s,r['S']))
   os=[s[j] if j>=0 else 0 for j in r['option_to_action']]
   assert all(abs(a-b)<1e-9 for a,b in zip(os,r['option_S']))
   logits=r['scores']['original'];ex=[math.exp(x-max(logits)) for x in logits];base=[x/sum(ex) for x in ex]
   direct=[x/sum(os) for x in os] if sum(os)>0 else base
   assert r['answers']['direct_map']==max(range(len(direct)),key=lambda i:direct[i])
   for w in (.25,.5,.75):
    mix=[(1-w)*b+w*s for b,s in zip(base,direct)]
    assert all(abs(x-y)<1e-6 for x,y in zip(mix,r['scores'][f'fusion_{w}']))
   for m,scores in r['scores'].items():assert r['answers'][m]==max(range(len(scores)),key=lambda i:scores[i])
  for m,metrics in summary[model]['metrics'].items():
   for group,v in metrics.items():
    rr=rows if group=='all' else [r for r in rows if r['variant']==group]
    correct=sum(r['answers'][m]==r['gold_option'] for r in rr)
    harm=sum(r['answers']['original']==r['gold_option'] and r['answers'][m]!=r['gold_option'] for r in rr)
    assert v['n']==len(rr) and v['correct']==correct and v['harmed']==harm
  result[model]={'verified_rows':len(rows),'all_methods':len(summary[model]['metrics']),'passed':True}
 out=a.report/'verification.json';assert not out.exists();out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
