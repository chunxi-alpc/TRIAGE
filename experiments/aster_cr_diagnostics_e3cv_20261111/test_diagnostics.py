"""Focused tests for score decomposition, path branching, native decoding and OOF aggregation."""
import unittest
import numpy as np
from .score_variant import score_variant
from experiments.aster_cr_retrain_paired_v2_20261111 import evaluation as ev

class DiagnosticsTests(unittest.TestCase):
    def test_intercepts_preserve_pair_differences(self):
        raw=np.array([-2.,.3,1.]);c=np.array([.1,-.2,.4]);b=.7;b0=-.3
        full=raw+c+b
        np.testing.assert_allclose(score_variant(raw,full,b,b0,"pair_only"),raw+c)
        for kind in ("raw","independent_intercept","original_intercept_only"):
            np.testing.assert_allclose(np.diff(score_variant(raw,full,b,b0,kind)),np.diff(raw))
        np.testing.assert_allclose(np.diff(full),np.diff(score_variant(raw,full,b,b0,"pair_only")))

    def test_changed_path_replays_new_branch(self):
        nodes={"G0":{"yes":"G1","no":"G2"},"G1":{"yes":"L1","no":"L2"},"G2":{"yes":"L3","no":"L4"}}
        case={"gold_path":[{"gid":"G0","label":0},{"gid":"G2","label":1}],"gold_leaf":"L3"}
        first=ev._execute_prepared_path(case,{"G0":.1,"G1":-.2,"G2":.1},nodes,"G0")
        changed=ev._execute_prepared_path(case,{"G0":-.1,"G1":-.2,"G2":.1},nodes,"G0")
        self.assertEqual(first["leaf"],"L2");self.assertFalse(first["correct"])
        self.assertEqual(changed["leaf"],"L3");self.assertTrue(changed["correct"])
        self.assertEqual(changed["path"][1]["gid"],"G2")

    def test_medpic_legal_set_with_intercept(self):
        item={"options":{"A":"first","B":"second","C":"None of the above"}}
        raw={"A":.2,"B":.1,"C":.4}
        self.assertEqual(ev.decode_medpic(item,raw),["C"])
        shifted={k:v+.5 for k,v in raw.items()}
        self.assertEqual(ev.decode_medpic(item,shifted),["A","B"])

    def test_unknown_labels_remain_null(self):
        views=[{"view_id":"a","label":None},{"view_id":"b","label":None}]
        pair=dict(pair_id="p",source_id="s",left="a",right="b",kind="C",left_label=None,right_label=None)
        records=ev.observations(views,[pair],{"raw":{"a":-.2,"b":.2}},{"raw":{"a":0,"b":1}})
        self.assertIsNone(records[0]["paired_correct"]);self.assertIsNone(records[0]["direction_degraded"])
        mm,cc,_=ev.pair_statistics(records,ev.SourceBootstrap(["s"]))
        self.assertEqual(cc["cf_paired_labeled_count"],0)
        self.assertTrue(np.isnan(mm["cf_paired_accuracy"][0]))

    def test_pooled_vr_recomputes_global_variance(self):
        raw=np.array([-1.,1.,-10.,10.]);adj=np.array([0.,0.,-9.,9.])
        records=[dict(method="m",group="AP",source_id=str(i),raw_difference=float(x),
            difference=float(y),paired_correct=True,relation_correct=True,direction_degraded=False,
            over_budget=False,left_label=0,right_label=0) for i,(x,y) in enumerate(zip(raw,adj))]
        metrics,_,_=ev.pair_statistics(records,ev.SourceBootstrap([str(i) for i in range(4)]))
        expected=1-np.var(adj)/np.var(raw)
        self.assertAlmostEqual(metrics["ap_vr"][0],expected)
        self.assertNotAlmostEqual(metrics["ap_vr"][0],(1+(1-.81))/2)

if __name__=="__main__":unittest.main()
