import tempfile
import unittest
from pathlib import Path
from common import blocks_for
from revision import normalize_annotation, views_for
from dependency_rules import row_dependency
from analyze_revision import masking_metrics

class FollowupRegression(unittest.TestCase):
    def run_case(self, retained=False, wrong_focus=False):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        text = "Assess high risk.\n\nHigh risk means age >60. Give drug A.\n\nAfter."
        if retained:
            text += "\n\nHigh risk means age >60."
        text += "\n\nCopyright 2026 publisher. All rights reserved."
        path = Path(directory.name)/"source.md"
        path.write_text(text)
        condition = "Is the tumor RET fusion-positive?" if wrong_focus else "Is the patient high risk?"
        node = dict(node_id="test", source_id="test", condition=condition, source_file=str(path),
            node_mapping=["Assess high risk."], blocks=blocks_for(text),
            designated_delete_block_ids=[1], designated_management=[dict(block_id=1, quote="Give drug A.")],
            management_location_status="located")
        value = dict(semantic_review_version=3, source_semantic_dependency="yes",
            source_condition_action_mixed="no", dataset_definition_completed="no",
            necessary_restores=[], conservative_cuts=[dict(block_id=1,quote="Give drug A.")],
            definition_links=[dict(focus_quote="high risk",target_term="high risk",
                relation="needed_to_interpret_target",kind="definition",
                definition_span=dict(block_id=1,quote="High risk means age >60."),
                equivalent_spans=[])])
        return node, normalize_annotation(node, value)

    def test_deleted_definition_is_restored_without_model_restore_field(self):
        node, record = self.run_case()
        views, change = views_for(node, record)
        self.assertTrue(change["restoration_available"])
        self.assertIn("Assess high risk.\n\nHigh risk means age >60.\n\nAfter.", views["necessary_restore"])
        self.assertNotIn("Give drug A.", views["necessary_restore"])
        self.assertTrue(change["matched_irrelevant_available"])

    def test_other_node_definition_does_not_count_as_dependency(self):
        node, record = self.run_case(wrong_focus=True)
        self.assertIsNone(record["annotation"]["source_semantic_dependency"])
        self.assertFalse(record["annotation"]["necessary_restores"])

    def test_retained_literal_copy_does_not_count_as_definition_loss(self):
        node, record = self.run_case(retained=True)
        self.assertTrue(record["annotation"]["source_semantic_dependency"])
        self.assertTrue(record["annotation"]["removed_definition_spans"])
        self.assertFalse(record["annotation"]["definition_lost"])
        self.assertFalse(record["annotation"]["necessary_restores"])

    def test_second_normalization_preserves_derived_restore(self):
        node, record = self.run_case()
        again = normalize_annotation(node, record["annotation"])
        self.assertEqual(record["annotation"]["necessary_restores"], again["annotation"]["necessary_restores"])
        self.assertEqual(record["annotation"]["definition_links"], again["annotation"]["definition_links"])

    def test_q_and_masking_require_same_traceable_resolved_evidence(self):
        common = dict(population="original", role="statistical", gold=1,
            source_semantic_dependency=True, dataset_definition_completed=True,
            source_traceable=True, dependency_evidence_valid=True, unresolved_references=[],
            scores={}, changes=dict(direct_changed=False,conservative_changed=False,
                                   restoration_available=False,matched_irrelevant_available=False))
        rows = [common,dict(common,source_traceable=False),
                dict(common,unresolved_references=["missing external definition"])]
        result = masking_metrics(rows)
        self.assertEqual(result["dependent"]["n"],1)
        self.assertEqual(result["unknown_dependency"]["n"],2)
        self.assertEqual(result["definition_already_completed"]["n"],1)
