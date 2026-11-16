import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from experiments.aster_e678_adaptive_20261116 import adaptive_m1 as m
from experiments.aster_e678_adaptive_20261116 import staged
from experiments.aster_e678_adaptive_20261116.prepare import OUT,KEYS
from experiments.aster_e678_adaptive_20261116.run import require_queue,SplitStore,extract
from experiments.aster_cr_retrain_paired_v2_20261111 import training,m1_plan,patch_queue
from test_adaptive import pair,Store

class IntegrationTests(unittest.TestCase):
    def test_prepared_real_source_isolation(self):
        audit=json.loads((OUT/"data_audit.json").read_text())
        self.assertEqual(audit["source_counts"],{"fit":80,"dev":20,"test":40})
        self.assertEqual(audit["dev_category_sources"],dict.fromkeys("APCF",20))
        self.assertFalse(audit["protocol_conflict"])
        for key in KEYS:
            root=OUT/"models"/key
            read=lambda n:[json.loads(s) for s in (root/"data"/n).read_text().splitlines()]
            fit=read("E1_pairs.jsonl");dev=read("dev_pairs.jsonl")
            test=[p for p in read("historical_pairs.jsonl") if p["split"]=="test"]
            sets=[set(p["source_id"] for p in rr) for rr in (fit,dev,test)]
            self.assertTrue(all(not sets[a]&sets[b] for a,b in ((0,1),(0,2),(1,2))))
            self.assertEqual(len(fit),2540)
            stages=json.loads((root/"stage_ids.json").read_text())
            self.assertTrue(m.pair_ids(dev)<=set(stages["dev"]))
            self.assertFalse(set(stages["fit"])&set(stages["test"]))
            extra=read("extra_inputs.jsonl")
            self.assertEqual(len(extra),43)
            self.assertTrue(all(r["split"]=="dev" and r["source_id"] in sets[1] for r in extra))
            self.assertTrue(all(r["labels"]==["是","否"] and "?"*4 not in r["prompt"] for r in extra))

    def test_same_stats_in_both_entry_points(self):
        n,h,dim=20,8,20
        fit=[pair("f"+str(i),kind=k,split="fit") for i in range(n) for k in "APCF"]
        dev=[pair("d"+str(i),kind=k) for i in range(n) for k in "APCF"]
        data={}
        for i in range(n):
            for k in "APCF":
                p=pair("f"+str(i),kind=k,split="fit")
                v=np.tile(np.eye(n)[i],(h,1))
                data[p["left"]]=dict(heads=np.zeros((h,dim)),grads=v,ell=0)
                data[p["right"]]=dict(heads=v,ell=1)
        class Patcher:
            def effects(self,target,donor,heads,sham=False):
                return np.array([1. if sham else (2. if (target["kind"]=="A" and head<4)
                    or (target["kind"]=="C" and 4<=head<7) else 0.) for head in heads])
        store=Store(data)
        with tempfile.TemporaryDirectory() as td:
            base=Path(td)
            heads,a=m.select(fit,dev,store,None,Patcher(),base/"sync",h,4)
            plan=m1_plan.make_plan("E1",fit,store,base/"staged",mode="adaptive",
                adaptive=dict(dev_pairs=dev,total_heads=h,heads_per_layer=4))
            hb,b=m1_plan.finish_plan(plan,None,Patcher(),base/"staged")
            self.assertEqual(heads,list(range(7)))
            self.assertEqual(heads,hb)
            self.assertEqual(a,b)
            with self.assertRaises(ValueError):
                patch_queue.build_queue([base/"staged/m1_plan.json"],out=base/"old_queue")
            self.assertFalse((base/"old_queue").exists())

    def test_training_dispatch_small_rank_and_zero_rank_m3(self):
        pairs=[pair("f"+str(i),kind=k,split="fit") for i in range(20) for k in "APCF"]
        rng=np.random.default_rng(8)
        data={uid:dict(heads=rng.normal(size=(9,1)),ell=float(rng.normal()))
              for uid in m.pair_ids(pairs)}
        natural=[dict(view_id=p["left"],source_id=p["source_id"],split="fit",variant="full",label=i%2)
                 for i,p in enumerate(pairs)]
        with tempfile.TemporaryDirectory() as td, patch(
            "experiments.aster_e678_adaptive_20261116.pipeline.select",
            return_value=(list(range(7)),{"status":"complete"})):
            result=training.fit_group("E1",pairs,Store(data),[],natural,td,m1_mode="adaptive",
                patcher=None,adaptive=dict(dev_pairs=[pair("d")],dev_features=None,total_heads=9,heads_per_layer=3))
            self.assertEqual(result["heads"],list(range(7)))
            self.assertEqual(result["M2"]["dimension"],7)
            self.assertLessEqual(result["M2"]["S_rank"],7)
            self.assertEqual(result["M3"]["rank"],0)
            self.assertTrue(result["M3"]["success"])

    def test_test_labels_cannot_change_selection(self):
        # Unavailable test UIDs are rejected; labels in dev pairs are not read by confirm.
        d=np.ones((20,8))
        hypotheses=[dict(kind=k,head=h) for k in "AC" for h in range(4)]
        a=m.confirm(d,[str(i) for i in range(20)],hypotheses,4,2)
        b=m.confirm(d,[str(i) for i in range(20)],[dict(h,test_label=99) for h in hypotheses],4,2)
        self.assertEqual(a[1]["heads"],b[1]["heads"])
        np.testing.assert_allclose([r["p_confirm"] for r in a[0]],[r["p_confirm"] for r in b[0]])

    def test_history_files_untouched(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td)
            history=base/"reports/aster_e678_20261112/models/qwen"
            history.mkdir(parents=True)
            sentinel=history/"fixed_result.json"
            sentinel.write_text('{"heads":[1,2,3],"note":"original"}')
            before=(sentinel.read_bytes(),sentinel.stat().st_mtime_ns)
            output=base/"reports/aster_e678_adaptive_20261116"
            with patch("experiments.aster_e678_adaptive_20261116.pipeline.select",
                       return_value=([],{"status":"selection_incomplete"})):
                training.fit_group("E1",[pair("f",split="fit")],Store({}),[],[],output,
                    m1_mode="adaptive",patcher=None,adaptive=dict(dev_pairs=[pair("d")],
                    dev_features=None,total_heads=4,heads_per_layer=2))
            self.assertEqual(before,(sentinel.read_bytes(),sentinel.stat().st_mtime_ns))
            self.assertEqual(list(history.iterdir()),[sentinel])
            self.assertTrue((output/"model.json").exists())

    def test_fit_store_does_not_read_test_and_test_extraction_needs_freeze(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            (root/"cache").mkdir()
            # Deliberately invalid test JSON would fail if fitting opened it.
            (root/"cache/scores_test.jsonl").write_text("invalid test data")
            self.assertEqual(SplitStore(root).records,{})
            with self.assertRaises(ValueError):extract(root,"test",None,None)

    def test_legacy_fixed_mode_preserved(self):
        pairs=[pair("f",kind=k,split="fit") for k in "APCF"]
        data={}
        for p in pairs:
            data[p["left"]]=dict(heads=np.zeros((80,1)),grads=np.ones((80,1)),ell=0)
            data[p["right"]]=dict(heads=np.arange(80).reshape(80,1),ell=1)
        store=Store(data)
        patcher=lambda p,heads:[(h,float(h),float(-h)) for h in heads]
        with tempfile.TemporaryDirectory() as td:
            a,_=training._m1("E1",pairs,store,patcher,Path(td)/"direct")
            plan=m1_plan.make_plan("E1",pairs,store,Path(td)/"planned")
            b,_=m1_plan.finish_plan(plan,store,patcher,Path(td)/"planned")
            self.assertEqual(len(a),51)
            self.assertEqual(a,b)

    def test_standalone_gpu_launch_denied(self):
        with patch.dict("os.environ",{"CUDA_VISIBLE_DEVICES":"0"}):
            with self.assertRaises(RuntimeError):require_queue()
        with patch.dict("os.environ",{"CUDA_VISIBLE_DEVICES":"4"}):
            with self.assertRaises(RuntimeError):require_queue()

if __name__=="__main__":unittest.main()
