"""Reproduce public P1 metrics with Python stdlib; no private data or GPU."""
import argparse,gzip,json
from analysis import apply,metric,bootstrap
ap=argparse.ArgumentParser();ap.add_argument('inputs');a=ap.parse_args()
with gzip.open(a.inputs,'rt') as f:rows=[json.loads(x) for x in f]
for model in sorted({r['model'] for r in rows}):
 rr=apply([r for r in rows if r['model']==model],{'kind':'direct'})
 print(json.dumps({'model':model,**metric(rr),'cluster_inference':bootstrap(rr)}))
