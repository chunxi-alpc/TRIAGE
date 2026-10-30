import unittest
from .statistics import paired_summary,holm
class SourceStatisticsTests(unittest.TestCase):
    def test_source_weighting_and_nested_cases(self):
        rows=[dict(case_id=str(i),source_id="a",baseline=0,treatment=1) for i in range(9)]
        rows+=[dict(case_id="b",source_id="b",baseline=1,treatment=0)]
        s=paired_summary(rows,repeats=200)
        self.assertEqual(s["source_equal_paired_difference"],0)
        self.assertAlmostEqual(s["case_weighted_paired_difference"],.8)
        self.assertFalse(s["pass_inference"])
    def test_missing_and_failed_stay_in_denominator(self):
        s=paired_summary([dict(case_id="1",source_id="a",baseline=1,treatment=None),dict(case_id="2",source_id="b",baseline=0,treatment=1,treatment_failed=True)],repeats=100)
        self.assertEqual(s["planned_cases"],2)
        self.assertEqual(s["treatment_coverage"],0)
        self.assertEqual(s["corrections"],0)
        self.assertEqual(s["degradations"],1)
    def test_duplicate_cases_rejected(self):
        r=dict(case_id="x",source_id="a",baseline=0,treatment=1)
        with self.assertRaises(ValueError): paired_summary([r,r])
    def test_seed_reproducibility(self):
        rows=[dict(case_id=str(i),source_id=str(i),baseline=0,treatment=i/4) for i in range(5)]
        self.assertEqual(paired_summary(rows,repeats=300),paired_summary(rows,repeats=300))
    def test_exact_sign_flip_uses_clusters(self):
        rows=[dict(case_id=str(i),source_id=str(i),baseline=0,treatment=1) for i in range(4)]
        s=paired_summary(rows,repeats=100)
        self.assertEqual(s["paired_p"],.125)
    def test_holm_unsorted_and_ties(self):
        self.assertEqual(holm([.04,.01,.03]),[.06,.03,.06])
        self.assertEqual(holm([.01,.01]),[.02,.02])
    def test_empty_no_inference(self):
        self.assertEqual(paired_summary([])["status"],"no_data")
    def test_invalid_score(self):
        with self.assertRaises(ValueError): paired_summary([dict(case_id="1",source_id="a",baseline=0,treatment=float("nan"))])
if __name__=="__main__": unittest.main()
