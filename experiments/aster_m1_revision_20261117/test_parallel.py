"""CPU checks for disjoint four-GPU work and exact serial equivalence."""
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from experiments.aster_cr_retrain_paired_v2_20261111.common import read,write,rows
from . import parallel
from .test_revision import pair
from .revision import ExactPatcher,Config
from dataclasses import asdict

class ParallelTests(unittest.TestCase):
    def test_four_shards_cover_every_item_once(self):
        items=list(range(1255))
        chunks=[parallel.shard(items,i) for i in range(4)]
        self.assertEqual(sorted(x for c in chunks for x in c),items)
        self.assertLessEqual(max(map(len,chunks))-min(map(len,chunks)),1)

    def test_extraction_preserves_float32_and_existing_cache(self):
        with tempfile.TemporaryDirectory() as td:
            private=Path(td)/"private";out=Path(td)/"out"
            ids=["u"+str(i) for i in range(9)]
            write(out/"models/qwen/gradient_plan.json",dict(required_ids=ids))
            request=dict(session="test",stage="extract_fit",kind="extract",scope="fit",items=ids)
            inputs={uid:dict(prompt=uid) for uid in ids}
            class Engine:
                def prepare(self,prompt):return prompt
                def run(self,context,**kwargs):
                    return dict(ell=1.,heads=torch.tensor([[1.234567]],dtype=torch.float32),
                                grads=torch.tensor([[0.0000001]],dtype=torch.float32))
            with patch.object(parallel,"PRIVATE",private),patch.object(parallel,"OUT",out),patch("torch.cuda.max_memory_allocated",return_value=0):
                for worker in range(4):parallel.run_shard("qwen",request,Engine(),inputs,worker)
                # Resume all four shards without overwriting scores or arrays.
                for worker in range(4):
                    result=parallel.run_shard("qwen",request,Engine(),inputs,worker)
                    self.assertEqual(result["newly_computed"],0)
            scores=rows(private/"qwen/cache/scores_fit.jsonl")
            self.assertEqual(len(scores),9)
            self.assertEqual({r["uid"] for r in scores},set(ids))
            array=np.load(scores[0]["feature"])
            self.assertEqual(array["heads"].dtype,np.float32)
            self.assertGreater(float(array["grads"][0,0]),0)

    def test_parallel_pairs_match_serial_exact_measurement(self):
        pairs=[pair("F","s"+str(i),"p"+str(i)) for i in range(7)]
        inputs={}
        values={}
        for p in pairs:
            for side,value in [("left",0.),("right",1.)]:
                uid=p[side]
                inputs[uid]=dict(prompt=str(value))
                values[uid]=dict(heads=np.array([[value]],np.float32),ell=value)
        class Store:
            def get(self,uid):return values[uid]
            def ell(self,uid):return values[uid]["ell"]
        class Engine:
            def prepare(self,prompt):return float(prompt)
            def run(self,context,replacements=None,**kwargs):
                return dict(ell=float(next(iter(replacements.values()))[0]) if replacements else context)
        cfg=Config()
        expected=[ExactPatcher(Store(),inputs,Engine(),cfg).measure(p,[0]) for p in pairs]
        with tempfile.TemporaryDirectory() as td:
            private=Path(td)/"private";out=Path(td)/"out"
            request=dict(session="test",stage="exact_patching",kind="patch",items=pairs,
                         candidates={"F":{"heads":[0]}},config=asdict(cfg))
            with patch.object(parallel,"PRIVATE",private),patch.object(parallel,"OUT",out),patch("experiments.aster_m1_revision_20261117.runtime.FeatureStore",return_value=Store()),patch("torch.cuda.max_memory_allocated",return_value=0):
                for worker in range(4):parallel.run_shard("qwen",request,Engine(),inputs,worker)
            actual=[read(private/"qwen/paired"/(p["pair_id"]+".json")) for p in pairs]
            self.assertEqual(actual,expected)

if __name__=="__main__":unittest.main()
