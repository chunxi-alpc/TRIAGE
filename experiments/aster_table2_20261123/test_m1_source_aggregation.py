"""Regression for source-level exact-patch aggregation."""
import tempfile
import unittest
from pathlib import Path
from experiments.aster_cr_retrain_paired_v2_20261111 import m1_plan

class SourceAggregation(unittest.TestCase):
    def test_independent_source_values_reach_the_selection_test(self):
        sources=["source"+str(i) for i in range(10)]
        kinds={}
        for kind in ("A","P","C","F"):
            folds=[]
            for fold in range(5):
                held=sources[2*fold:2*fold+2]
                reps=[]
                for source in held:
                    for role in ("observed","null"):
                        reps.append(dict(pair_id=kind+source+role,source_id=source,left="L",right="R",
                           left_label=0,right_label=0 if kind in ("A","P") else 1,
                           m1_role=role,m1_weight=1.0))
                folds.append(dict(fold=fold+1,candidates=[0],attribution=[1.0],robust_z=[3.0],
                    discovery_sources=[s for s in sources if s not in held],validation_sources=held,
                    representatives=reps,representative_diagnostics={}))
            kinds[kind]=dict(folds=folds)
        plan=dict(method_version=m1_plan.METHOD_VERSION,group="E1",inner_source_folds=5,kinds=kinds,
            training_sources=sources,fit_pair_counts={k:10 for k in kinds},plan_seconds=0.)
        class Store:
            def ell(self,uid):return 0.
        def exact(pair,heads):
            effect=1. if pair["m1_role"]=="observed" else 0.
            return [(h,effect,-effect) for h in heads]
        with tempfile.TemporaryDirectory(prefix="aster-m1-source-test-") as td:
            heads,result=m1_plan.finish_plan(plan,Store(),exact,Path(td))
        self.assertEqual(heads,[0])
        for kind in kinds:
            stat=result["selection_metrics"][kind]["0"]
            self.assertEqual(stat["validation_source_count"],10)
            self.assertEqual(stat["effect_mean"],1.)
            self.assertTrue(stat["selected"])

if __name__=="__main__":unittest.main()
