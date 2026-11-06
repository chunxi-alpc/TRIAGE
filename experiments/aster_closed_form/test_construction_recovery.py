"""CPU mocks for current-revision construction recovery and response preservation."""
import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from . import construct as module
from .common import rows,write_rows
from .test_parallel_construct import group,value_for

class RecoveryGenerator:
    instances=[]
    fail_first_parse=False
    def __init__(self):
        self.calls=[]
        self.main_calls=0
        self.last_response=None
        self.torch=SimpleNamespace(cuda=SimpleNamespace(max_memory_reserved=lambda:0))
        self.instances.append(self)
    def generate(self,instruction,payload,max_new=2600):
        self.calls.append((instruction,copy.deepcopy(payload)))
        self.last_response=None
        if instruction==module.MASK_INSTRUCTION:
            value=dict(keep_action_ids=[],keep_definition_indices=[])
        elif instruction==module.CONDITION_INSTRUCTION:
            value=dict(condition_change=value_for(payload["current_condition"])["condition_change"],
                       facts=[{"unrequested_replacement":True}])
        else:
            self.main_calls+=1
            if self.fail_first_parse and self.main_calls==1:
                self.last_response=dict(text='{"facts": [BROKEN_OUTPUT',input_tokens=51,output_tokens=17)
                raise json.JSONDecodeError("Expecting value",self.last_response["text"],11)
            value=value_for(payload["current_condition"])
        raw=json.dumps(value)
        self.last_response=dict(text=raw,input_tokens=30,output_tokens=40)
        return value,raw,30,40

class ParseFailureGenerator(RecoveryGenerator):
    fail_first_parse=True

class TestConstructionRecovery(unittest.TestCase):
    def setUp(self):
        RecoveryGenerator.instances=[]
        ParseFailureGenerator.instances=[]
    def prepare(self,root):
        g=group(0)
        write_rows(root/"data/groups_main.jsonl",[g])
        write_rows(root/"data/groups_definition.jsonl",[])
        return g
    def master_masks(self,root,names=("base","swap")):
        write_rows(root/"data/semantic_masks.jsonl",[
            dict(key="g0:"+name,group_id="g0",status="complete",value={"old_mask":name})
            for name in names])

    def test_old_failed_is_reconstructed_and_upgraded(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);g=self.prepare(root)
            old=dict(group_id="g0",status="failed",error="Old JSON failure")
            write_rows(root/"data/constructions.jsonl",[old])
            self.master_masks(root)
            with patch.object(module,"Generator",RecoveryGenerator):
                module.construct(root)
            records=rows(root/"data/constructions.jsonl")
            self.assertEqual(records[0],old)
            latest=records[-1]
            self.assertEqual(latest["status"],"complete")
            self.assertEqual(latest["construction_revision"],module.REVISION)
            self.assertEqual(latest["value"]["facts"],value_for(module.condition(g))["facts"])
            generator=RecoveryGenerator.instances[-1]
            self.assertEqual(generator.main_calls,1)
            payload=generator.calls[0][1]
            self.assertEqual([p["state"] for p in payload["patients"]],[0,1])
            self.assertEqual([p["label"] for p in payload["patients"]],[0,1])
            masks={r["key"]:r for r in rows(root/"data/semantic_masks.jsonl")}
            self.assertTrue(all(masks["g0:"+name]["construction_revision"]==module.REVISION
                                for name in ("base","swap","condition")))

    def test_existing_main_success_with_unavailable_condition_is_kept(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);g=self.prepare(root)
            value=value_for(module.condition(g))
            value["condition_change"]=dict(valid=False,reason="Previously unavailable")
            value["preserved_field"]={"sentinel":[1,2,3]}
            old=dict(group_id="g0",source_id=g["source_id"],split="fit",cohort="main",
                     status="complete",value=copy.deepcopy(value),construction_revision=2,
                     validation=dict(available=True),preserved_record="original")
            write_rows(root/"data/constructions.jsonl",[old])
            self.master_masks(root)
            with patch.object(module,"Generator",side_effect=AssertionError("Existing main success must remain")):
                result=module.construct(root)
            self.assertEqual(result["calls"],0)
            self.assertEqual(rows(root/"data/constructions.jsonl"),[old])
            self.assertEqual([r["key"] for r in rows(root/"data/semantic_masks.jsonl")],
                             ["g0:base","g0:swap"])

    def test_merge_prefers_current_revision_and_success_within_revision(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.prepare(root)
            write_rows(root/"data/constructions.jsonl",[
                dict(group_id="g0",status="complete",value={"marker":"old"})])
            current=dict(group_id="g0",status="complete",construction_revision=module.REVISION,
                         value={"marker":"repaired"})
            write_rows(root/"data/constructions_0.jsonl",[current])
            write_rows(root/"data/constructions_1.jsonl",[
                dict(group_id="g0",status="failed",construction_revision=module.REVISION,error="stale")])
            write_rows(root/"data/semantic_masks.jsonl",[
                dict(key="g0:base",status="complete",value={"marker":"old"})])
            current_mask=dict(key="g0:base",status="complete",construction_revision=module.REVISION,
                              value={"marker":"new"})
            write_rows(root/"data/semantic_masks_0.jsonl",[current_mask])
            module.merge_construction_shards(root,shards=2)
            self.assertEqual(rows(root/"data/constructions.jsonl"),[current])
            self.assertEqual(rows(root/"data/semantic_masks.jsonl"),[current_mask])

    def test_parse_failure_raw_is_logged_and_supplied_to_retry(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.prepare(root)
            with patch.object(module,"Generator",ParseFailureGenerator):
                module.construct(root)
            generator=ParseFailureGenerator.instances[-1]
            main=[c for c in generator.calls if c[0]!=module.MASK_INSTRUCTION and
                  c[0]!=module.CONDITION_INSTRUCTION]
            self.assertEqual(len(main),2)
            self.assertEqual(main[1][1]["previous_response"],'{"facts": [BROKEN_OUTPUT')
            self.assertIn("Expecting value",main[1][1]["validation_error"])
            raw=rows(root/"logs/construction_raw.jsonl")
            failed=next(r for r in raw if "parse_error" in r)
            self.assertEqual(failed["text"],'{"facts": [BROKEN_OUTPUT')
            self.assertEqual((failed["input_tokens"],failed["output_tokens"]),(51,17))
            self.assertEqual(failed["construction_revision"],module.REVISION)
            self.assertEqual(rows(root/"data/constructions.jsonl")[-1]["status"],"complete")

if __name__=="__main__":
    unittest.main()
