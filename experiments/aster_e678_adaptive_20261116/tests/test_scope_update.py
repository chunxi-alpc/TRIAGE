import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from experiments.aster_e678_adaptive_20261116 import adaptive_m1 as m
from experiments.aster_e678_adaptive_20261116.evaluate import evaluate
from experiments.aster_cr_retrain_paired_v2_20261111 import evaluation as legacy
from experiments.aster_cr_retrain_paired_v2_20261111.common import lines
from test_adaptive import Store,pair

class ScopeUpdateTests(unittest.TestCase):
    def test_no_interval_calculation_or_interval_gate(self):
        d=np.array([1.]*19+[-18.])[:,None]
        d=np.repeat(d,2,axis=1)
        with patch.object(np,"quantile",side_effect=AssertionError("CI computation removed")):
            records,summary=m.confirm(d,[str(i) for i in range(20)],
                [dict(kind="A",head=0),dict(kind="C",head=0)],1,1)
        self.assertEqual(summary["heads"],[0])
        self.assertTrue(all(r["confirmed"] and r["excess_mean"]>0 for r in records))
        self.assertTrue(all("ci_low" not in r and "ci_high" not in r for r in records))

    def test_evaluation_has_no_fixed_model_or_interval_draws(self):
        pp=[pair("s"+str(i),kind=k) for i,k in enumerate("APCF")]
        data={};views=[]
        for i,p in enumerate(pp):
            for j,side in enumerate(("left","right")):
                uid=p[side]
                ell=float((-1 if j==0 else 1)*(i+1))
                data[uid]=dict(heads=np.array([[ell]]),ell=ell)
                views.append(dict(view_id=uid,source_id=p["source_id"],split="dev",
                    kind="original" if j==0 else p["kind"],original=j==0,label=j))
        obj=dict(heads=np.array([0]),alpha=np.array([-.1]),offset=0.,
                 metadata=dict(delta=.2923349),directory="test_adaptive")
        before=legacy.SourceBootstrap
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for name,rr in (("dev_views.jsonl",views),("dev_evaluation_pairs.jsonl",pp),
                            ("dev_path_views.jsonl",[]),("dev_cases.jsonl",[])):
                lines(root/"data"/name,rr)
            with patch.object(legacy,"model_parameters",return_value=obj) as load, patch.object(
                np,"quantile",side_effect=AssertionError("Evaluation CI removed")):
                result=evaluate(root,"dev",Store(data))
                load.assert_called_once_with(root,"E1")
            self.assertEqual(set(result["models"]),{"raw","E1"})
            self.assertNotIn("bootstrap",result)
            self.assertTrue(all("low" not in r and "high" not in r for r in result["point_estimates"]))
            self.assertFalse((root/"source_bootstrap.csv").exists())
            self.assertTrue((root/"point_estimates.csv").exists())
        self.assertIs(legacy.SourceBootstrap,before)

if __name__=="__main__":unittest.main()
