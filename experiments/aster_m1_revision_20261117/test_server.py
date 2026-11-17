"""Server CPU integration tests; synthetic data are never experimental results."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from experiments.aster_cr_retrain_paired_v2_20261111.common import write,read,lines
from .revision import fit_selected, Config, ExactPatcher, paired_effect
from .test_revision import pair
from .evaluate import independent_intercept, prediction_models, PointSourceAggregation, public_names
from .run import freeze_parameters
from .runtime import FeatureStore
from experiments.aster_cr_retrain_paired_v2_20261111 import evaluation as legacy

class Store:
    def __init__(self):
        self.values={}
    def ell(self,uid):return self.values[uid]["ell"]
    def get(self,uid):return self.values[uid]
    def heads(self,uid,heads):return self.values[uid]["heads"][heads].reshape(-1)

def training_fixture():
    rng=np.random.default_rng(17)
    ps=[];natural=[];store=Store()
    for i in range(16):
        for k in ("A","P","C","F"):
            p=pair(k,"s"+str(i),k+str(i));ps.append(p)
            left=rng.normal(size=(3,2)).astype(np.float32)
            right=left+rng.normal(size=(3,2)).astype(np.float32)
            for uid,a,label in ((p["left"],left,p["left_label"]),(p["right"],right,p["right_label"])):
                store.values[uid]=dict(heads=a,ell=float(a.sum()*.2+label-0.5))
            natural.append(dict(view_id=p["left"],source_id=p["source_id"],split="fit",label=i%2,variant="full"))
    return ps,natural,store

class ServerIntegration(unittest.TestCase):
    def test_real_zero_rank_fallback_and_four_predictions(self):
        ps,natural,store=training_fixture()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);target=root/"models/E1/full"
            result=fit_selected(ps,store,[],natural,target,
                    dict(heads=[],status="intercept_only_fallback"),"synthetic")
            self.assertEqual(result["status"],"intercept_only_fallback")
            self.assertTrue(result["M3"]["success"])
            self.assertEqual(result["M3"]["max_CF_correction"],0)
            write(root/"independent_intercept.json",independent_intercept(natural,store))
            model=prediction_models(root)
            scores=legacy.score_views(store,natural,model)
            self.assertEqual(set(scores),{"raw","adaptive_full","adaptive_no_intercept","independent_intercept"})
            for uid in scores["raw"]:
                self.assertEqual(scores["raw"][uid],scores["adaptive_no_intercept"][uid])
                self.assertAlmostEqual(scores["adaptive_full"][uid],scores["independent_intercept"][uid])

    def test_real_m2_m3_retains_budget(self):
        ps,natural,store=training_fixture()
        with tempfile.TemporaryDirectory() as td:
            result=fit_selected(ps,store,[],natural,Path(td),dict(heads=[0,1,2],status="features_selected"),"synthetic")
            self.assertTrue(result["M3"]["success"])
            self.assertLessEqual(result["M2"]["N_rank"],16)
            self.assertLessEqual(result["M3"]["constraint_max_violation"],1e-7)
            self.assertEqual(result["M3"]["AP_pairs"],32)
            self.assertEqual(result["M3"]["CF_pairs"],32)

    def test_no_intercept_preserves_centering(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);target=root/"models/E1/full";target.mkdir(parents=True)
            np.savez(target/"m2.npz",heads=[0],mu=[3.,4.],scale=[2.,5.],N=np.eye(2))
            np.savez(target/"m3.npz",w=[2.,10.],b=7.)
            write(target/"model.json",{})
            write(root/"independent_intercept.json",dict(b=-2.))
            models=prediction_models(root)
            self.assertEqual(models["adaptive_no_intercept"]["offset"],-11.)
            self.assertEqual(models["adaptive_full"]["offset"],-4.)
            store=Store();store.values["x"]=dict(heads=np.array([[3.,4.]]),ell=1.)
            legacy._LINEAR_SCORE_CACHE.clear()
            scores=legacy.score_views(store,[dict(view_id="x")],models)
            self.assertEqual(scores["adaptive_no_intercept"]["x"],1.)
            self.assertEqual(scores["adaptive_full"]["x"],8.)
            self.assertEqual(scores["independent_intercept"]["x"],-1.)

    def test_independent_intercept_uses_fit_only(self):
        ps,natural,store=training_fixture()
        with self.assertRaises(ValueError):
            independent_intercept([dict(v,split="test") for v in natural],store)

    def test_point_aggregation_never_resamples(self):
        with patch("numpy.random.default_rng",side_effect=AssertionError("No resampling")):
            agg=PointSourceAggregation(["a","a","b"])
            self.assertEqual(agg.mean([1,0,1],["a","a","b"]).tolist(),[2/3])
            self.assertEqual(set(agg.summary([.5])),{"estimate"})

    def test_freeze_detects_parameter_change(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);target=root/"models/E1/full";target.mkdir(parents=True)
            write(root/"independent_intercept.json",dict(b=0))
            np.savez(target/"m3.npz",w=[1.])
            before=freeze_parameters(root)
            np.savez(target/"m3.npz",w=[2.])
            self.assertNotEqual(before,freeze_parameters(root))

    def test_global_head_number_preserved_for_non_qwen_geometry(self):
        p=pair("F");store=Store()
        store.values[p["left"]]=dict(heads=np.zeros((6,2)),ell=0.)
        store.values[p["right"]]=dict(heads=np.ones((6,2)),ell=1.)
        class Engine:
            def prepare(self,prompt):return float(prompt)
            def run(self,context,replacements=None,capture=False):
                if replacements:
                    self.asserted_head=next(iter(replacements))
                    assert self.asserted_head==5
                    return dict(ell=float(replacements[5][0]))
                return dict(ell=context)
        engine=Engine()
        row=ExactPatcher(store,{p["left"]:dict(prompt="0"),p["right"]:dict(prompt="1")},engine).measure(p,[5])[0]
        self.assertEqual(row["head"],5)
        self.assertGreater(row["magnitude"],0)

    def test_named_raw_prediction_and_no_interval_fields(self):
        value=public_names(dict(models=dict(raw=dict(estimate=.5,low=.1,high=.9)),
                               comparison="adaptive_full minus raw",draws=0))
        self.assertIn("frozen_backbone",value["models"])
        self.assertEqual(value["models"]["frozen_backbone"],dict(estimate=.5))
        self.assertEqual(value["comparison"],"adaptive_full minus frozen_backbone")
        self.assertNotIn("draws",value)

    def test_missing_gradient_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);np.savez(root/"a.npz",heads=np.zeros((2,2)))
            lines(root/"cache/scores_fit.jsonl",[dict(uid="a",ell=0.,feature=str(root/"a.npz"),has_grads=True)])
            store=FeatureStore(root)
            with self.assertRaises(ValueError):store.validate(["a"],{"a"},2,2)

    def test_solver_failure_blocks_complete(self):
        from experiments.aster_cr_retrain_paired_v2_20261111 import training
        ps,natural,store=training_fixture()
        with tempfile.TemporaryDirectory() as td,patch.object(training,"refit_m3",return_value=dict(success=False)):
            with self.assertRaises(RuntimeError):
                fit_selected(ps,store,[],natural,Path(td),dict(heads=[],status="intercept_only_fallback"),"synthetic")
            self.assertFalse((Path(td)/"model.json").exists())


    def test_queue_dependencies_and_authorized_cards(self):
        from .queue import manifest
        data=manifest();jobs={j["id"]:j for j in data["jobs"]}
        for key in ("llama31","gemma2","gemma3","medgemma","qwen_base","gemma2_base"):
            self.assertEqual(jobs["m1_revision_"+key]["dependencies"],["m1_revision_qwen"])
        for job in data["jobs"]:
            if job["gpu_count"]:self.assertEqual(job["allowed_gpus"],[0,1,5,6])

    def test_failed_numerical_patch_is_preserved(self):
        p=pair("F");store=Store()
        store.values[p["left"]]=dict(heads=np.zeros((1,1)),ell=0.)
        store.values[p["right"]]=dict(heads=np.ones((1,1)),ell=1.)
        class Engine:
            def prepare(self,prompt):return float(prompt)
            def run(self,context,replacements=None,capture=False):
                return dict(ell=context+.5 if replacements else context)
        saved=[]
        with self.assertRaises(ValueError):
            ExactPatcher(store,{p["left"]:dict(prompt="0"),p["right"]:dict(prompt="1")},Engine()).measure(p,[0],on_record=saved.append)
        self.assertEqual(len(saved),1)
        self.assertEqual(saved[0]["self_left"],.5)

    def test_evaluation_writes_four_point_predictions(self):
        from . import evaluate as module
        ps,natural,store=training_fixture()
        p=ps[2]
        views=[dict(view_id=p[k],source_id=p["source_id"],split="dev",label=p[k+"_label"],
                    kind="original",variant="full",original=True) for k in ("left","right")]
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/"run";source=Path(td)/"source";hist=Path(td)/"hist"
            target=root/"models/E1/full";target.mkdir(parents=True)
            np.savez(target/"m2.npz",heads=[],mu=[],scale=[],N=np.empty((0,0)))
            np.savez(target/"m3.npz",w=[],b=.2)
            write(target/"model.json",dict(M3=dict(delta=.2923349)))
            write(root/"independent_intercept.json",dict(b=-.1))
            write(root/"fit_frozen.json",dict(frozen=True))
            lines(source/"data/historical_views.jsonl",views)
            lines(source/"data/historical_pairs.jsonl",[dict(p,split="dev")])
            with patch.object(module,"SOURCE",source),patch.object(module,"HIST",hist),patch("numpy.random.default_rng",side_effect=AssertionError("No bootstrap")):
                result=module.evaluate(root,"dev",store)
            self.assertEqual(set(result["models"]),{"frozen_backbone","adaptive_full","adaptive_no_intercept","independent_intercept"})
            self.assertIn("point_estimates",result)
            self.assertNotIn("bootstrap",result)
            self.assertFalse((root/"source_bootstrap.csv").exists())
            self.assertTrue((root/"point_estimates.csv").exists())
            self.assertEqual(result["models"]["adaptive_full"]["cf_max_budget_excess"],0.)
            prediction=read_json_lines(root/"evaluation/dev/predictions.jsonl")[0]
            self.assertIn("frozen_backbone",prediction["scores"])

def read_json_lines(path):
    from experiments.aster_cr_retrain_paired_v2_20261111.common import rows
    return rows(path)

if __name__=="__main__":unittest.main()

