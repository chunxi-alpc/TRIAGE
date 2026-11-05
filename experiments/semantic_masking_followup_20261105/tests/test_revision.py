"""Focused regression checks for the five requested pilot checks."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import CONFIG, DATA, ROOT, blocks_for, read_rows, render
from predicates import evaluate, patient_for, compare, between, all_of, any_of, negate
from revision import views_for, normalize_annotation
from score import reusable, rebind_score
from analyze_revision import masking_metrics, fact_metrics, fit_logistic

def score(p):
    import math
    return dict(p_yes=p,margin=math.log(p/(1-p)),prediction=int(p>=0.5))

class RepairedConditions(unittest.TestCase):
    def setUp(self):
        self.specs={s["node_id"]:s for s in read_rows(DATA/"construction_specs.jsonl")}

    def test_age_range_keeps_both_bounds_and_inclusive_endpoints(self):
        spec=self.specs["2c44dbfd26d455b456beca2920543d6d:17"]
        for age,expected in [(39,False),(40,True),(49,True),(50,True),(51,False)]:
            self.assertEqual(evaluate(spec["predicate"],{"measurement":age}),expected)

    def test_compound_tmb_does_not_drop_other_disjuncts(self):
        spec=self.specs["2c44dbfd26d455b456beca2920543d6d:74"]
        facts=dict(spec["fixed_facts"],measurement=9)
        self.assertFalse(evaluate(spec["predicate"],facts))
        self.assertTrue(evaluate(spec["predicate"],dict(facts,msi_high=True)))
        self.assertTrue(evaluate(spec["predicate"],dict(facts,dmmr=True)))
        self.assertTrue(evaluate(spec["predicate"],dict(facts,measurement=11)))

    def test_boolean_operators_and_units_do_not_become_labels(self):
        expression=all_of(between("value",40,50),any_of(compare("x","==",True),negate(compare("y","==",True))))
        self.assertTrue(evaluate(expression,dict(value=45,x=False,y=False)))
        self.assertFalse(evaluate(expression,dict(value=55,x=True,y=False)))
        self.assertFalse(evaluate(expression,dict(value=45,x=False,y=True)))

    def test_fixed_facts_background_and_ancestors(self):
        self.assertEqual(len(self.specs),50)
        for spec in self.specs.values():
            if not spec["eligible"]:
                continue
            cases={}
            for level in ("below","above","same_side"):
                neutral,facts=patient_for(spec,level,"neutral")
                cue,other=patient_for(spec,level,"cue")
                self.assertEqual(facts,other)
                self.assertEqual(neutral.rsplit("\n",1)[0],cue.rsplit("\n",1)[0])
                for expression in spec["ancestor_constraints"]:
                    self.assertTrue(evaluate(expression,facts))
                cases[level]=facts
            changed={k for k in cases["below"] if cases["below"][k]!=cases["above"][k]}
            self.assertEqual(changed,{"measurement"})
            self.assertNotEqual(evaluate(spec["predicate"],cases["below"]),evaluate(spec["predicate"],cases["above"]))
            self.assertEqual(evaluate(spec["predicate"],cases["above"]),evaluate(spec["predicate"],cases["same_side"]))

    def test_ancestor_forbidding_crossing_is_reported(self):
        for nid in ("d8d6f57b62b5893ca0efe91c59e3470c:26","d9906cd6be177a867400393324954892:21"):
            self.assertFalse(self.specs[nid]["eligible"])
            self.assertIn("ancestor",self.specs[nid]["reason"].lower())

class SourceEdits(unittest.TestCase):
    def test_retained_definition_can_still_support_original_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"source.md"
            text="Assess high risk.\n\nHigh risk means >10.\n\nGive drug A."
            path.write_text(text)
            node=dict(node_id="test",source_id="test",source_file=str(path),node_mapping=["Assess high risk."],
                blocks=blocks_for(text),designated_management=[dict(block_id=2,quote="Give drug A.")],
                designated_delete_block_ids=[2],management_location_status="located")
            value=dict(source_semantic_dependency="yes",
                missing_definitions=[dict(block_id=1,quote="High risk means >10.")],
                necessary_restores=[],conservative_cuts=[dict(block_id=2,quote="Give drug A.")])
            result=normalize_annotation(node,value)
            self.assertIs(result["annotation"]["source_semantic_dependency"],True)
            self.assertFalse(result["annotation"]["definition_removed"])
            self.assertTrue(result["validation"]["dependency_evidence_valid"])

    def test_conservative_cut_keeps_embedded_definition(self):
        from conservative import preserve_notes
        text="Use treatment A (high risk means >10)."
        blocks=blocks_for(text)
        cuts,retained=preserve_notes([dict(block_id=0,quote=text)],{0:blocks[0]})
        revised=render(blocks,cuts=cuts)
        self.assertIn("(high risk means >10)",revised)
        self.assertNotIn("Use treatment A",revised)
        self.assertEqual(len(retained),1)

    def test_real_mapped_action_is_present_and_deleted(self):
        nodes=read_rows(DATA/"nodes.jsonl")
        node=next(n for n in nodes if n["node_id"]=="afd5d03b38791ef112c4c90e963feed0:15")
        annotation=normalize_annotation(node,{"source_semantic_dependency":False})
        views,changes=views_for(node,annotation)
        self.assertTrue(changes["direct_changed"])
        for span in node["designated_management"]:
            self.assertIn(span["quote"],views["full"])
        self.assertEqual(views["direct"],render(node["blocks"],deleted=node["designated_delete_block_ids"]))

    def test_original_position_restore_and_independent_cut(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"source.md"
            text="Before.\n\nHigh means >10. Give drug A.\n\nAfter.\n\nCopyright 2026 publisher. All rights reserved."
            path.write_text(text)
            node=dict(node_id="test",source_id="test",source_file=str(path),
                blocks=blocks_for(text),designated_management=[dict(block_id=1,quote="Give drug A.")],
                designated_delete_block_ids=[1],management_location_status="located")
            value=dict(rule_evidence=[dict(block_id=1,quote="High means >10.")],
                source_semantic_dependency=True,source_condition_action_mixed=True,dataset_definition_completed=False,
                necessary_restores=[dict(block_id=1,quote="High means >10.")],
                conservative_cuts=[dict(block_id=1,quote="Give drug A.")])
            views,changes=views_for(node,normalize_annotation(node,value))
            self.assertIn("Before.\n\nHigh means >10.\n\nAfter.",views["necessary_restore"])
            self.assertNotIn("drug A",views["conservative"])
            self.assertIn("High means >10.",views["conservative"])
            self.assertTrue(changes["matched_irrelevant_available"])
            self.assertNotIn("High means >10.",views["irrelevant_restore"])

    def test_noop_deletion_not_in_effect_denominator(self):
        common=dict(population="original",role="statistical",source_semantic_dependency=False,
            dataset_definition_completed=False,gold=1,scores=dict(full=score(.8),direct=score(.2)))
        rows=[dict(common,changes=dict(direct_changed=changed,conservative_changed=False,
                    restoration_available=False,matched_irrelevant_available=False)) for changed in (True,False)]
        result=masking_metrics(rows)["all"]
        self.assertEqual(result["full_vs_direct"]["n"],1)
        self.assertEqual(result["unchanged_input_count"],1)

class CacheAndMetrics(unittest.TestCase):
    def test_cache_rejects_changed_input_and_model(self):
        cached=dict(status="complete",score_id="unchanged_id",rendered_prompt="complete input",model_settings={"model":"qwen"})
        self.assertTrue(reusable(cached,"complete input",{"model":"qwen"}))
        self.assertFalse(reusable(cached,"modified complete input",{"model":"qwen"}))
        self.assertFalse(reusable(cached,"complete input",{"model":"other"}))

    def test_label_only_correction_reuses_score_and_rebinds_label(self):
        cached=dict(status="complete",score_id="old",prediction=1,gold=0,correct=False)
        row=dict(score_id="new",example_id="e",node_id="n",source_id="s",population="constructed",
            view="full",gold=1,split="development",ancestors=[],context="",condition="Age >=40?",patient="Age 45.")
        bound=rebind_score(cached,row,"test")
        self.assertEqual(bound["gold"],1)
        self.assertTrue(bound["correct"])
        self.assertEqual(cached["gold"],0)

    def test_pb_is_one_neutral_original_pair_per_node(self):
        rows=[]
        for background in ("neutral","cue"):
            for level,gold,p in (("below",0,.1),("above",1,.8),("same_side",1,.7)):
                # The cue variant is wrong, but cannot inflate the primary P_b.
                rows.append(dict(node_id="n",population="constructed",background=background,
                    wording="original",wording_changed=False,metric_level=level,gold=gold,
                    construction={},scores=dict(full=score(p if background=="neutral" else .2))))
        result=fact_metrics(rows)["full"]
        self.assertEqual(result["P_b"]["denominator"],1)
        self.assertEqual(result["P_b"]["rate"],0)
        self.assertEqual(result["P_b_strata"]["cue/original"]["rate"],1)

    def test_calibration_has_only_one_input_feature_and_grouped_dev(self):
        rows=[]
        for source in range(4):
            for gold in (0,1):
                rows.append(dict(gold=gold,source_id=str(source),
                    scores=dict(full=score(.7 if gold else .2),direct=score(.6 if gold else .1))))
        model,info=fit_logistic(rows,["full"])
        self.assertEqual(info["feature_order"],["full"])
        self.assertEqual(len(info["coefficients"][0]),1)
        for candidate in info["cv_candidates"]:
            for fold in candidate["folds"]:
                self.assertFalse(set(fold["training_sources"]) & set(fold["validation_sources"]))

if __name__=="__main__":
    unittest.main()
