"""Verify complete anonymized outputs and exact CPU reproduction."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reproduced',type=Path,required=True);a=ap.parse_args()
    r=json.loads((HERE/'results.json').read_text());other=json.loads((a.reproduced/'results.json').read_text())
    assert r==other,'CPU reproduction differs'
    rows=[json.loads(x) for x in (HERE/'numeric/predictions.jsonl').read_text().splitlines()]
    assert len(rows)==800 and len({x['id'] for x in rows})==200
    assert all(x['id'].startswith('case_') and x['split']=='dev' and x['seed']==0 for x in rows)
    assert not any(k in x for x in rows for k in ('patient','rules','options','condition_key','assessments'))
    assert sum(v['status']['forward_calls'] for v in r['resources'].values())==8800
    assert all(v['gpu_tests']['passed'] for v in r['resources'].values())
    checked=0
    for kind,v in r['methods'].items():
        assert len(v['all_candidates'])==27
        for c in v['all_candidates']:
            p=c['predictions'];assert len(p)==800
            assert c['metrics']['correct']==sum(x==y['gold_option'] for x,y in zip(p,rows))
            assert sum(x['n'] for x in c['metrics']['by_variant'].values())==800
            checked+=1
    max_residual=0.
    from sparse_engine import configurations
    for row in rows:
        for name,c in configurations().items():
            delta=np.asarray(row['scores'][name])-row['scores']['zero_'+c['geometry']]
            shift=.25*np.tanh(delta/2.)
            assert np.isfinite(shift).all() and np.max(np.abs(shift))<=.25
            max_residual=max(max_residual,float(np.max(np.abs(shift))))
    result=dict(passed=True,rows=800,patients=200,gpu_forward_calls=8800,all_fusion_candidates_checked=checked,
        exact_cpu_reproduction=True,max_abs_bounded_residual=max_residual,
        artifacts_sha256={str(p.relative_to(HERE)):digest(p) for p in [HERE/'results.json',HERE/'numeric/predictions.jsonl',HERE/'predictions.csv']})
    (HERE/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
