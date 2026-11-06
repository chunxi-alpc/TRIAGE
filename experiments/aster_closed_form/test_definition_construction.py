"""Necessary CPU checks for definition schema, retry, and original ancestor truth."""
import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from . import construct as module
from .common import rows,write_rows
from .test_parallel_construct import group,value_for

def definition_group():
    g=group(0,"definition")
    g["definitions"]=[{"quote":"The observation criterion is a value of at least 1."}]
    g["definition_links"]=[{"target_term":"observation"}]
    return g

def definition_value(g):
    v=value_for(module.condition(g))
    v.update(definition_valid=True,short_condition="Does the observation meet its criterion?",
             shared_record="Other patient facts are unchanged.",
             fact_sentences=["Observation 0.","Observation 1."],reason="Only the observation differs.")
    v["condition_change"].update(edits=[{
        "old":g["definitions"][0]["quote"],
        "new":"The observation criterion is a value of at least 0."}])
    return v

class DefinitionGenerator:
    instances=[]
    first_mode="missing"
    def __init__(self):
        self.calls=[];self.definition_calls=0;self.last_response=None
        self.torch=SimpleNamespace(cuda=SimpleNamespace(max_memory_reserved=lambda:0))
        self.instances.append(self)
    def generate(self,instruction,payload,max_new=2600):
        self.calls.append((instruction,copy.deepcopy(payload),max_new))
        self.last_response=None
        if instruction==module.MASK_INSTRUCTION:
            value=dict(keep_action_ids=[],keep_definition_indices=[0])
        else:
            if not instruction.startswith(module.DEF_INSTRUCTION):
                raise AssertionError("Definition must use its independent instruction")
            self.definition_calls+=1
            if self.definition_calls==1 and self.first_mode=="truncated":
                self.last_response=dict(text='{"definition_valid":true,"facts":[',input_tokens=100,
                                        output_tokens=max_new)
                raise json.JSONDecodeError("Expecting value",self.last_response["text"],32)
            value=definition_value(definition_group())
            if self.definition_calls==1 and self.first_mode=="missing":
                for key in ("definition_valid","short_condition","shared_record","fact_sentences"):
                    value.pop(key)
        text=json.dumps(value)
        self.last_response=dict(text=text,input_tokens=100,output_tokens=80)
        return value,text,100,80

class TruncatedDefinitionGenerator(DefinitionGenerator):
    first_mode="truncated"

class TestDefinitionConstruction(unittest.TestCase):
    def setUp(self):
        DefinitionGenerator.instances=[]
        TruncatedDefinitionGenerator.instances=[]
    def prepare(self,root):
        g=definition_group()
        write_rows(root/"data/groups_main.jsonl",[])
        write_rows(root/"data/groups_definition.jsonl",[g])
        return g

    def test_missing_flag_or_required_field_is_not_unavailable(self):
        g=definition_group();v=definition_value(g)
        missing=copy.deepcopy(v);missing.pop("definition_valid")
        with self.assertRaisesRegex(ValueError,"explicit boolean"):
            module.validate(g,missing)
        missing=copy.deepcopy(v);missing["definition_valid"]="false"
        with self.assertRaisesRegex(ValueError,"explicit boolean"):
            module.validate(g,missing)
        missing=copy.deepcopy(v);missing.pop("short_condition")
        with self.assertRaisesRegex(ValueError,"missing required fields"):
            module.validate(g,missing)
        self.assertFalse(module.validate(g,{"definition_valid":False,"reason":"No coherent example"})["available"])
        self.assertTrue(module.validate(g,copy.deepcopy(v))["available"])

    def test_previous_missing_schema_reconstructs_and_retries(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);g=self.prepare(root)
            old=dict(group_id=g["group_id"],status="complete",construction_revision=2,
                     value=value_for(module.condition(g)),validation={"available":False})
            write_rows(root/"data/constructions.jsonl",[old])
            with patch.object(module,"Generator",DefinitionGenerator):
                module.construct(root)
            gen=DefinitionGenerator.instances[-1]
            self.assertEqual(gen.definition_calls,2)
            calls=[c for c in gen.calls if c[0]!=module.MASK_INSTRUCTION]
            self.assertEqual([c[2] for c in calls],[4096,4096])
            self.assertIn("previous_response",calls[1][1])
            self.assertIn("definition_valid",calls[1][1]["validation_error"])
            self.assertTrue(all(c[2]==1200 for c in gen.calls if c[0]==module.MASK_INSTRUCTION))
            latest=rows(root/"data/constructions.jsonl")[-1]
            self.assertEqual(latest["construction_revision"],3)
            self.assertTrue(latest["validation"]["available"])
            self.assertTrue(latest["value"]["definition_valid"])
            self.assertEqual(len(rows(root/"data/semantic_masks.jsonl")),3)

    def test_truncated_definition_retry_has_more_output_capacity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.prepare(root)
            with patch.object(module,"Generator",TruncatedDefinitionGenerator):
                module.construct(root)
            gen=TruncatedDefinitionGenerator.instances[-1]
            self.assertEqual([c[2] for c in gen.calls if c[0]!=module.MASK_INSTRUCTION],[4096,6144])
            logs=rows(root/"logs/construction_raw.jsonl")
            self.assertEqual(logs[0]["output_tokens"],4096)
            self.assertIn("parse_error",logs[0])
            self.assertTrue(rows(root/"data/constructions.jsonl")[-1]["validation"]["available"])

    def test_negative_ancestor_is_original_question_truth(self):
        g=group(0)
        g["ancestors"]=[dict(gid="G9",condition="Is the marker present?",label=0)]
        v=value_for(module.condition(g))
        for facts in v["facts"]:facts["marker_present"]=False
        v["ancestor_predicates"]=[dict(gid="G9",predicate=dict(op="==",field="marker_present",value=True))]
        self.assertTrue(module.validate(g,copy.deepcopy(v))["available"])
        v["ancestor_predicates"][0]["predicate"]["value"]=False
        with self.assertRaisesRegex(ValueError,"expected 0"):
            module.validate(g,v)

if __name__=="__main__":
    unittest.main()
