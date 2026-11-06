"""CPU mocks for construction shard ownership and resumable merging."""
import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from . import construct as module
from .common import rows,write_rows,append

def group(i,cohort="main",source=None):
    gid="g"+str(i)
    question="Has observation reached 1 for "+gid+"?"
    return dict(group_id=gid,source_id=source or "s"+str(i%2),split="fit",cohort=cohort,
        gid="G0",ancestors=[],definitions=[],shared_record="Fixed facts.",facts=["Observation 0.","Observation 1."],
        guideline={"groups":[dict(group_id="G0",rules=[
            dict(rule_id="R0Y",condition=question,action="L1: action one"),
            dict(rule_id="R0N",condition=question,action="L2: action two")])]},
        source_rule_evidence=[],definition_links=[],existing_patient="")

def value_for(question):
    return dict(facts=[{"observation":0},{"observation":1}],
        predicate=dict(op=">=",field="observation",value=1),ancestor_predicates=[],
        decisive_sentences=["Observation 0.","Observation 1."],protected_action_ids=[],
        condition_change=dict(valid=True,base_state=0,
            edits=[dict(old=question,new=question.replace("reached 1","reached 0"))],
            predicate=dict(op=">=",field="observation",value=0)),
        clinical_background=dict(valid=False),treatment_preference=dict(valid=False))

class FakeGenerator:
    instances=[]
    def __init__(self):
        self.calls=[]
        self.torch=SimpleNamespace(cuda=SimpleNamespace(max_memory_reserved=lambda:0))
        self.instances.append(self)
    def generate(self,instruction,payload,max_new=2600):
        self.calls.append((instruction,copy.deepcopy(payload)))
        if instruction==module.MASK_INSTRUCTION:
            value=dict(keep_action_ids=[],keep_definition_indices=[])
        elif instruction.startswith(module.DEF_INSTRUCTION):
            value=dict(definition_valid=False,reason="No coherent definition pair.")
        else:
            value=value_for(payload["current_condition"])
        return value,json.dumps(value),30,40

class TestParallelConstruction(unittest.TestCase):
    def setUp(self):
        FakeGenerator.instances=[]
    def prepare(self,root,groups):
        write_rows(root/"data/groups_main.jsonl",[g for g in groups if g["cohort"]=="main"])
        write_rows(root/"data/groups_definition.jsonl",[g for g in groups if g["cohort"]=="definition"])

    def test_two_shards_cover_every_group_without_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            groups=[group(i) for i in range(6)]+[group(6,"definition")]
            self.prepare(root,groups)
            expected=module.ordered_groups(root)
            with patch.object(module,"Generator",FakeGenerator):
                module.construct(root,shard=0,shards=2)
                module.construct(root,shard=1,shards=2)
            a={r["group_id"] for r in rows(root/"data/constructions_0.jsonl")}
            b={r["group_id"] for r in rows(root/"data/constructions_1.jsonl")}
            self.assertFalse(a&b)
            self.assertEqual(a|b,{g["group_id"] for g in expected})
            self.assertEqual(a,{g["group_id"] for g in expected[0::2]})
            self.assertEqual(b,{g["group_id"] for g in expected[1::2]})
            self.assertFalse((root/"data/constructions.jsonl").exists())
            self.assertFalse((root/"data/semantic_masks.jsonl").exists())
            self.assertFalse((root/"logs/construction_raw.jsonl").exists())
            for shard in range(2):
                status=json.loads((root/f"status/construction_{shard}.json").read_text())
                self.assertEqual(status["state"],"complete")
                self.assertEqual(status["done"],len(expected[shard::2]))
                self.assertEqual(status["total"],len(expected[shard::2]))
            summary=module.merge_construction_shards(root,shards=2)
            self.assertEqual(summary["constructions"],7)
            self.assertEqual(summary["semantic_masks"],18)
            records=rows(root/"data/constructions.jsonl")
            self.assertEqual(len(records),len({r["group_id"] for r in records}))
            again=module.merge_construction_shards(root,shards=2)
            self.assertEqual(again,summary)

    def test_resume_retains_master_and_completes_condition_mask(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);groups=[group(i,source="s0") for i in range(3)]
            self.prepare(root,groups)
            original=dict(group_id="g0",status="complete",value=value_for(module.condition(groups[0])),
                          preserved_marker="original successful record")
            write_rows(root/"data/constructions.jsonl",[original])
            master_masks=[dict(key="g0:"+name,group_id="g0",status="complete",value={})
                          for name in ("base","swap")]
            write_rows(root/"data/semantic_masks.jsonl",master_masks)
            original_bytes=(root/"data/constructions.jsonl").read_bytes()
            masks_bytes=(root/"data/semantic_masks.jsonl").read_bytes()
            with patch.object(module,"Generator",FakeGenerator):
                module.construct(root,shard=0,shards=2)
            self.assertEqual((root/"data/constructions.jsonl").read_bytes(),original_bytes)
            self.assertEqual((root/"data/semantic_masks.jsonl").read_bytes(),masks_bytes)
            generated=rows(root/"data/constructions_0.jsonl")
            self.assertEqual([r["group_id"] for r in generated],["g2"])
            masks=rows(root/"data/semantic_masks_0.jsonl")
            self.assertIn("g0:condition",{r["key"] for r in masks})
            self.assertNotIn("g0:base",{r["key"] for r in masks})
            with patch.object(module,"Generator",side_effect=AssertionError("No model should load")):
                resumed=module.construct(root,shard=0,shards=2)
            self.assertEqual(resumed["done"],2)
            self.assertEqual(resumed["calls"],0)
            status=json.loads((root/"status/construction_0.json").read_text())
            self.assertEqual((status["state"],status["done"],status["total"]),("complete",2,2))
            append(root/"data/constructions_0.jsonl",dict(group_id="g0",status="failed",error="old failure"))
            append(root/"data/semantic_masks_0.jsonl",dict(key="g0:base",status="failed"))
            module.merge_construction_shards(root,shards=2)
            merged={r["group_id"]:r for r in rows(root/"data/constructions.jsonl")}
            self.assertEqual(merged["g0"],original)
            merged_masks=rows(root/"data/semantic_masks.jsonl")
            self.assertEqual(len(merged_masks),len({r["key"] for r in merged_masks}))
            self.assertEqual(next(r for r in merged_masks if r["key"]=="g0:base"),master_masks[0])

    def test_serial_paths_and_empty_work(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.prepare(root,[group(0)])
            with patch.object(module,"Generator",FakeGenerator):
                module.construct(root)
            self.assertTrue((root/"data/constructions.jsonl").exists())
            self.assertTrue((root/"status/construction.json").exists())
            self.assertFalse((root/"data/constructions_0.jsonl").exists())
            with patch.object(module,"Generator",side_effect=AssertionError("No model should load")):
                module.construct(root)
                empty=module.construct(root,limit=0,shard=1,shards=2)
            self.assertEqual((empty["done"],empty["total"]),(0,0))

if __name__=="__main__":
    unittest.main()
