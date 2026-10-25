"""Reuse exact-payload condition scores; run only missing sparse GPU forwards."""
import argparse, hashlib, json, os, subprocess, time
from pathlib import Path
import numpy as np
import torch
from sparse_engine import SparseEngine, configurations, GEOMETRIES
from sparse_attention import sparse_attention
from improve_branch import condition_key
from mass_operator import action_scores

HERE=Path(__file__).resolve().parent
DEFAULT_OLD=Path('/home/gcx/aster_dynamic_review_20261025')
DEFAULT_OUT=Path('/home/gcx/aster_sparse_dynamic_qwen_seed0_20261025')
MODEL=Path('/home/gcx/TRIAGE/models/modelscope/Qwen3-8B')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,obj):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
def sign(case_id):return 1 if hashlib.sha256(('seed0:'+case_id).encode()).digest()[0]%2 else -1

def inputs(old):
    data=old/'dev.jsonl';cache=old/'qwen/predictions.jsonl'
    manifest=json.loads((old/'qwen/manifest.json').read_text())
    assert sha(data)==manifest['data_sha256']
    assert sha(MODEL/'config.json')==manifest['model_config_sha256']
    rows=[json.loads(x) for x in data.read_text().splitlines()]
    saved=[json.loads(x) for x in cache.read_text().splitlines()]
    by={(r['id'],r['variant']):r for r in saved}
    assert len(rows)==len(saved)==len(by)==800 and len({r['id'] for r in rows})==200
    for c in rows:
        r=by[c['id'],c['variant']]
        assert c['split']=='dev' and c['seed']==0
        assert all(c[k]==r[k] for k in ('id','tree','seed','variant','split','gold_option'))
        assert [x['rid'] for x in c['rules']]==list(range(len(c['rules'])))
        safe={k:c[k] for k in ('patient','rules','options')}
        assert condition_key(safe)==r['condition_key']
        actions,s=action_scores(safe,r['p'])
        assert np.allclose(s,r['S'],rtol=0,atol=1e-12)
        assert [actions.index(o) if o in actions else -1 for o in safe['options']]==r['option_to_action']
        assert np.allclose([s[actions.index(o)] if o in actions else 0. for o in safe['options']],r['option_S'],rtol=0,atol=1e-12)
    return sorted(rows,key=lambda c:(c['id'],c['variant'])),by

def gpu_test():
    torch.manual_seed(0)
    q=torch.randn(1,4,257,16,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(1,2,257,16,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
    args=(q,k,v,None,.25,[i%3 for i in range(257)],[.2,.7,.1],[0,2],253)
    zero=sparse_attention(*args,strength=0.)
    ref=torch.nn.functional.scaled_dot_product_attention(q,k.repeat_interleave(2,1),v.repeat_interleave(2,1),is_causal=True)
    edited,d=sparse_attention(*args,return_details=True)
    err=float((zero-ref).abs().max());assert err<.04
    assert torch.equal(zero[:,:,:253],edited[:,:,:253])
    assert torch.equal(zero[:,[1,3]],edited[:,[1,3]])
    assert torch.isfinite(edited).all() and float(d['bias'].abs().max())<=2
    assert float(d['after'][~d['allowed']].abs().sum())==0
    return dict(passed=True,bf16_zero_vs_sdpa_max_abs=err,seed=0)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--old',type=Path,default=DEFAULT_OLD)
    ap.add_argument('--out',type=Path,default=DEFAULT_OUT);ap.add_argument('--shard',type=int,choices=[0,1,6],required=True)
    ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(a.shard)
    rows,by=inputs(a.old);ids=sorted({c['id'] for c in rows});selected=set(ids[[0,1,6].index(a.shard)::3])
    rows=[r for r in rows if r['id'] in selected]
    if a.limit:rows=rows[:a.limit]
    out=a.out/(('pilot' if a.limit else 'shard')+str(a.shard));out.mkdir(parents=True,exist_ok=False)
    start=time.time()
    atomic(out/'manifest.json',dict(seed=0,shard=a.shard,rows=len(rows),cases=len({r['id'] for r in rows}),
        data_sha256=sha(a.old/'dev.jsonl'),cache_sha256=sha(a.old/'qwen/predictions.jsonl'),
        model=str(MODEL),model_config_sha256=sha(MODEL/'config.json'),torch=torch.__version__,
        transformers=__import__('transformers').__version__,gpu=torch.cuda.get_device_name(),
        physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],configs=configurations(),geometries=GEOMETRIES,
        code_sha256={p.name:sha(p) for p in HERE.glob('*.py')},base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        condition_forward_calls=0,cached_original_repeated=False))
    e=SparseEngine(str(MODEL));atomic(out/'gpu_tests.json',gpu_test())
    with (out/'predictions.jsonl').open('x') as f:
        for i,c in enumerate(rows):
            old=by[c['id'],c['variant']];safe={k:c[k] for k in ('patient','rules','options')}
            scores={};seconds={};answers={};case_start=time.time()
            for name,kw in configurations().items():
                torch.cuda.synchronize();t=time.time()
                pred,values=e.answer(safe,mode='sparse',p=old['p'],case_sign=sign(c['id']),**kw)
                torch.cuda.synchronize();seconds[name]=time.time()-t
                assert np.isfinite(values).all();scores[name]=values;answers[name]=pred
            numeric={k:old[k] for k in ('id','tree','seed','variant','split','gold_option','p','S','option_S','rule_to_action','option_to_action','covered','represented_mass')}
            numeric.update(scores=scores,answers=answers,seconds=seconds,case_sign=sign(c['id']),tokens=len(e.render(safe)[0]),total_seconds=time.time()-case_start)
            f.write(json.dumps(numeric)+'\n');f.flush()
            atomic(out/'status.json',dict(completed=i+1,total=len(rows),complete=i+1==len(rows),elapsed_seconds=time.time()-start,
                peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
                forward_calls=(i+1)*len(configurations())))
    atomic(out/'DONE.json',dict(complete=True,elapsed_seconds=time.time()-start))

if __name__=='__main__':main()
