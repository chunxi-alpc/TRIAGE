"""Minimal checks for GPU assignment and disjoint path-mask resume."""
import json,tempfile,unittest
from pathlib import Path
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
from . import pipeline
from .common import rows,write_rows

class FakeGenerator:
    calls=[]
    def __init__(self):
        self.torch=SimpleNamespace(cuda=SimpleNamespace(max_memory_reserved=lambda:0))
    def generate(self,instruction,payload,max_new=600):
        self.calls.append(payload["current_node"])
        return dict(keep_action_ids=[],keep_definition_indices=[]),"{}",1,1

class ParallelPipelineTests(unittest.TestCase):
    def test_path_masks_partition_resume_and_merge(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            entries=[dict(key=str(i),gid=str(i),condition="condition",guideline={},definitions=[]) for i in range(7)]
            write_rows(root/"data/path_semantic_masks.jsonl",[dict(key="0",status="complete",value={})])
            FakeGenerator.calls=[]
            with patch("experiments.aster_closed_form.paths.enumerate_contexts",return_value=entries), \
                 patch("experiments.aster_closed_form.construct.Generator",FakeGenerator), \
                 patch("experiments.aster_closed_form.data.canonical_guide",return_value={"leaves":{}}):
                pipeline.mask_paths(root,0,2)
                pipeline.mask_paths(root,1,2)
                self.assertEqual(sorted(FakeGenerator.calls),[str(i) for i in range(1,7)])
                pipeline.mask_paths(root,0,2)
                self.assertEqual(len(FakeGenerator.calls),6)
            first={r["key"] for r in rows(root/"data/path_semantic_masks_0.jsonl")}
            second={r["key"] for r in rows(root/"data/path_semantic_masks_1.jsonl")}
            self.assertFalse(first & second)
            pipeline.merge_path_masks(root,2)
            self.assertEqual({r["key"] for r in rows(root/"data/path_semantic_masks.jsonl")},{str(i) for i in range(7)})
            self.assertEqual(json.loads((root/"status/path_semantic_extraction_0.json").read_text())["state"],"complete")
    def test_dispatcher_uses_both_gpus_for_bulk_stages(self):
        calls=[]
        def fake_spawn(root,stage,gpu,**kwargs):
            calls.append((stage,gpu,kwargs));return SimpleNamespace(pid=100+len(calls))
        with tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
            for name in ("join","fit_grid","select_and_ablations","select_amplification","report","merge_path_masks"):
                stack.enter_context(patch.object(pipeline,name))
            stack.enter_context(patch.object(pipeline,"spawn",side_effect=fake_spawn))
            stack.enter_context(patch("experiments.aster_closed_form.construct.merge_construction_shards"))
            stack.enter_context(patch("experiments.aster_closed_form.data.build"))
            pipeline.all_stages(Path(folder))
        for stage in ("construct","path-masks","collect","grid","amp-dev","final","paths"):
            actual=[(gpu,kw) for st,gpu,kw in calls if st==stage]
            self.assertEqual(actual,[(0,dict(shard=0,shards=2)),(1,dict(shard=1,shards=2))])
        self.assertTrue(all(gpu in (0,1) for _,gpu,_ in calls))

if __name__=="__main__":unittest.main()
