#!/usr/bin/env python3
"""CPU-only tests for official label mapping, valid input boundary and scoring."""
import unittest
from prepare import LABEL_MAP, QUESTION, records_to_inputs_and_labels
from evaluate import class_metrics, decode_a1, cluster_bootstrap, hash_id

def rec(i, typ, label, evidence="[]"):
    return {
        "annotation_id": i, "patient_id": "patient-" + str(i % 2),
        "trial_id": "trial-1", "note": "Patient is an adult with prior therapy.",
        "criterion_type": typ, "criterion_text": "Age at least 18 years.",
        "expert_eligibility": label, "expert_sentences": evidence,
        "training": False, "gpt4_eligibility": "leak_control",
        "gpt4_explanation": "leak_control",
    }

class DatasetTests(unittest.TestCase):
    def test_predeclared_labels(self):
        self.assertEqual(len(LABEL_MAP), 8)
        self.assertEqual(LABEL_MAP[("inclusion", "included")], "T")
        self.assertEqual(LABEL_MAP[("exclusion", "excluded")], "T")
        self.assertEqual(LABEL_MAP[("exclusion", "not excluded")], "F")
        self.assertEqual(LABEL_MAP[("exclusion", "not enough information")], "U")
        self.assertIsNone(LABEL_MAP[("inclusion", "not applicable")])

    def test_no_gold_in_input_and_full_population(self):
        samples=[rec(0,"inclusion","included"),
                 rec(1,"exclusion","not excluded","[0]"),
                 rec(2,"inclusion","not applicable")]
        inputs,labels=records_to_inputs_and_labels(samples)
        self.assertEqual(len(inputs),len(labels))
        self.assertEqual([r["mapped_gold"] for r in labels],["T","F",None])
        self.assertEqual([r["evidence_present"] for r in labels],[False,True,False])
        self.assertEqual(inputs[0]["question"],QUESTION)
        for row in inputs:
            self.assertEqual(set(row),{"case_id","question","patient","rules"})
            self.assertEqual(len(row["rules"]),1)
            self.assertEqual(set(row["rules"][0]),{"rule_id","text","action"})
            self.assertNotIn("leak_control",str(row))
            self.assertNotIn("expert_eligibility",str(row))

    def test_duplicate_and_wrong_labels_fail(self):
        with self.assertRaises(ValueError):
            records_to_inputs_and_labels([rec(1,"inclusion","included"),
                                          rec(1,"exclusion","excluded")])
        with self.assertRaises(ValueError):
            records_to_inputs_and_labels([rec(1,"inclusion","not excluded")])

    def test_a1_parser_and_failure_denominator(self):
        a={"status":"ok","rules":[{"rule_id":"R_CRITERION","state":"U"}]}
        self.assertEqual(decode_a1(a),"U")
        self.assertEqual(decode_a1(None),"ERROR")
        self.assertEqual(decode_a1({"status":"parse_failure","rules":[]}),"ERROR")
        rows=[{"mapped_gold":"T","prediction":"T"},
              {"mapped_gold":"F","prediction":"ERROR"},
              {"mapped_gold":"U","prediction":"F"}]
        score=class_metrics(rows)
        self.assertEqual((score["correct"],score["total"]),(1,3))
        self.assertEqual(score["confusion"]["F"]["ERROR"],1)

    def test_patient_bootstrap(self):
        rows=[{"patient_id":"p1","mapped_gold":"T","prediction":"T"},
              {"patient_id":"p1","mapped_gold":"F","prediction":"F"},
              {"patient_id":"p2","mapped_gold":"U","prediction":"ERROR"}]
        b=cluster_bootstrap(rows,"patient_id",replicates=40)
        self.assertEqual(b["clusters"],2)
        self.assertTrue(all(0<=v<=1 for v in b["ci95"]))
        self.assertEqual(hash_id("patient","p1"),hash_id("patient","p1"))

if __name__=="__main__":
    unittest.main(verbosity=2)

