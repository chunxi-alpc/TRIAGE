"""Prepare all available aligned pairs; preserve the prior pilot in development."""
import sys,json,hashlib,random,argparse
from pathlib import Path
from collections import Counter
sys.path.append(str(Path(__file__).resolve().parents[1]/'aster_medguide_20261022'))
from prepare_dataset import build_example,norm

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 import pyarrow.parquet as pq
 src=Path(a.source);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 paths=json.loads((src/'all_paths.json').read_text());groups=[];counts=Counter()
 for i,row in enumerate(pq.read_table(src/'train.parquet').to_pylist()):
  try:
   group=build_example(row,paths,i,20261022)
   for r in group:
    r['replacement_elsewhere']=any(norm(q[-1])==norm(r['replacement_action']) for q in paths[r['disease']])
    r['condition_text']=r['path'][-2]
    # Exploratory text classification, not expert condition labels.
    import re
    r['condition_type']='judgment_candidate' if re.search(r'\b(?:eligib\w*|suitab\w*|toler\w*|fit|unfit|candidate\w*|resectab\w*|operab\w*|adequate)\b',r['condition_text'],re.I) else 'other_unreviewed'
   groups.append(group);counts['eligible']+=1
  except (ValueError,KeyError,TypeError) as e:counts[str(e)]+=1
 random.Random(20261022).shuffle(groups)
 splits={'all':groups,'dev':groups[:40],'test':groups[40:]}
 manifest={'seed':20261022,'counts':dict(counts),'source_sha256':{n:hashlib.sha256((src/n).read_bytes()).hexdigest() for n in ('train.parquet','all_paths.json')},'splits':{}}
 for name,gs in splits.items():
  f=out/(name+'.jsonl');f.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for g in gs for r in g))
  manifest['splits'][name]={'n':len(gs),'ids':[g[0]['id'] for g in gs],'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'disease_counts':dict(Counter(g[0]['disease'] for g in gs)),'judgment_candidates':sum(g[0]['condition_type']=='judgment_candidate' for g in gs)}
 (out/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2))
 print(json.dumps({k:v['n'] for k,v in manifest['splits'].items()}))
if __name__=='__main__':main()
