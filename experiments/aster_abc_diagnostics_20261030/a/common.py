"""Read-only helpers for source-clustered diagnostic analyses."""
from pathlib import Path
import json
import math
import random
from collections import defaultdict
import torch

BASE = Path('/home/gcx/aster_manuscript_20261030/clinical01')
SEED = 20261030

def read_jsonl(path):
    return [json.loads(s) for s in Path(path).open() if s.strip()]

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n')

def load_records(base=BASE):
    accepted = {r['pair_id'] for r in read_jsonl(Path(base)/'review/accepted_pairs.jsonl')}
    records = []
    for shard in (0, 1):
        root = Path(base)/f'collect{shard}'
        status = json.loads((root/'status.json').read_text())
        for index in status['complete']:
            r = torch.load(root/f'{index}.pt', map_location='cpu', weights_only=False)
            if r['row']['pair_id'] in accepted:
                records.append(r)
    return records

def mean_difference(record, head):
    return (record['offline']['positive'][head]-record['offline']['negative'][head]).double().mean(0)

def unit(v):
    v = torch.as_tensor(v, dtype=torch.float64)
    n = float(v.norm())
    return None if n <= 1e-12 else v/n

def cosine(a, b):
    ua, ub = unit(a), unit(b)
    return None if ua is None or ub is None else float(ua@ub)

def source_mean_ci(rows, field, repeats=2000, seed=SEED):
    """Equal-weight source means; all repeated/paired observations stay together."""
    groups = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is not None and math.isfinite(float(value)):
            groups[row['source_id']].append(float(value))
    values = [sum(v)/len(v) for _, v in sorted(groups.items())]
    if not values:
        return {'estimate':None, 'ci95':None, 'sources':0, 'observations':0}
    estimate = sum(values)/len(values)
    if len(values) < 2:
        return {'estimate':estimate, 'ci95':None, 'sources':len(values), 'observations':sum(map(len,groups.values())), 'inference':'insufficient independent sources'}
    rng = random.Random(seed)
    boot = sorted(sum(rng.choice(values) for _ in values)/len(values) for _ in range(repeats))
    return {'estimate':estimate, 'ci95':[boot[int(.025*repeats)],boot[min(repeats-1,int(.975*repeats))]],
            'sources':len(values), 'observations':sum(map(len,groups.values())),
            'inference':'exploratory; two-source intervals do not establish population generalization' if len(values)==2 else 'source-cluster percentile bootstrap'}

def holm(pvalues):
    order=sorted(pvalues,key=pvalues.get); output={}; prior=0.
    for i,key in enumerate(order):
        prior=max(prior,min(1.,pvalues[key]*(len(order)-i)));output[key]=prior
    return output
