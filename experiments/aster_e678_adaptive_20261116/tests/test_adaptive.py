import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.stats import binomtest
from experiments.aster_e678_adaptive_20261116 import adaptive_m1 as m
from experiments.aster_e678_adaptive_20261116.patcher import ShamDonorPatcher
from experiments.aster_e678_adaptive_20261116.pipeline import fit_group
from experiments.aster_cr_retrain_paired_v2_20261111 import training

class Store:
    def __init__(self, data): self.data=data; self.access=[]
    def get(self, uid): self.access.append(uid); return self.data[uid]
    def ell(self, uid): return self.get(uid)["ell"]
    def heads(self, uid, selected=None):
        x=self.get(uid)["heads"]
        return x if selected is None else x[selected].reshape(-1)

def pair(source,kind="A",split="dev",suffix=""):
    return dict(source_id=source,kind=kind,split=split,pair_id=source+kind+suffix,
                left=source+kind+suffix+"l",right=source+kind+suffix+"r",
                left_label=0,right_label=1)

class AdaptiveTests(unittest.TestCase):
    def hypotheses(self,n):
        return [dict(kind=k,head=h) for k in "APCF" for h in range(n)]
    def confirm(self,d,n):
        return m.confirm(d,[str(s) for s in range(len(d))],self.hypotheses(n),n,4)

    def test_exactly_seven_no_fixed_k(self):
        h=90
        d=-np.ones((20,4*h))
        d[:,:4]=2
        d[:,2*h+4:2*h+7]=2
        rr,ss=self.confirm(d,h)
        self.assertEqual(ss["heads"],list(range(7)))
        self.assertEqual(ss["status"],"complete")
        self.assertAlmostEqual(ss["N_eff"]["A"],4)

    def test_no_effects_no_padding(self):
        rr,ss=self.confirm(np.zeros((20,360)),90)
        self.assertEqual(ss["heads"],[])
        self.assertEqual(ss["status"],"selection_incomplete")
        self.assertTrue(all(r["p_confirm"]==1 for r in rr))

    def test_final_can_exceed_51_and_candidates_exceed_64(self):
        g=np.repeat(np.eye(80)[:,None,:],100,axis=1)
        observed,p=m.screen_arrays(g,g)
        q=m.adjust(np.tile(p,4))
        self.assertTrue(np.all(q<=.20))
        self.assertEqual(len(p),100)
        rr,ss=self.confirm(np.ones((20,400)),100)
        self.assertEqual(ss["selected_count"],100)
        self.assertEqual(ss["shared_count"],100)

    def test_permutation_matches_literal_definition(self):
        rng=np.random.default_rng(4)
        g=rng.normal(size=(7,3,2));d=rng.normal(size=g.shape)
        cfg=m.Config(permutations=23)
        obs,p=m.screen_arrays(g,d,cfg)
        rng=np.random.default_rng(cfg.seed)
        null=np.array([np.abs((g*d[rng.permutation(7)]).sum(-1)).mean(0) for _ in range(23)])
        np.testing.assert_allclose(obs,np.abs((g*d).sum(-1)).mean(0))
        np.testing.assert_allclose(p,(1+(null>=obs).sum(0))/24)

    def test_adjustment_known_values_and_ties(self):
        p=[.01,.04,.03,.002]
        expected=np.array([.02,.04,.04,.008])
        np.testing.assert_allclose(m.adjust(p),expected)
        np.testing.assert_allclose(m.adjust(p,"BY"),expected*(1+.5+1/3+.25))
        np.testing.assert_allclose(m.adjust([1,1,0]),[1,1,0])
        self.assertEqual(len(m.adjust([])),0)

    def test_sign_test_ties_missing_and_exact_reference(self):
        d=np.array([1]*15+[-1]*3+[0,np.nan])
        self.assertAlmostEqual(float(m.sign_p(d)),binomtest(15,18,.5,alternative="greater").pvalue)

    def test_source_count_no_relaxation(self):
        rr,ss=self.confirm(np.ones((15,8)),2)
        self.assertEqual(ss["heads"],[])
        self.assertTrue(all(r["evidence_status"]=="insufficient_statistical_evidence" for r in rr))

    def test_source_bootstrap_and_representatives(self):
        pp=[pair("s1",suffix="b"),pair("s1",suffix="a"),pair("s2")]
        reps=m.representatives(pp,"dev")["A"]
        self.assertEqual(len(reps),2)
        self.assertEqual(reps[0]["pair_id"],"s1Aa")
        w=m.bootstrap_weights(["s1","s2"],2000,4)
        self.assertTrue(np.all(w.sum(1)==2))
        with self.assertRaises(ValueError):m.bootstrap_weights(["s1","s1"],20,4)

    def test_donors_are_distinct_nested_and_reproducible(self):
        reps=m.representatives([pair("s"+str(i)) for i in range(20)],"dev")
        plans=m.donor_plan(reps)
        self.assertEqual(plans,m.donor_plan(reps))
        for (_,s),dd in plans.items():
            self.assertEqual(len(dd),10)
            self.assertNotIn(s,[p["source_id"] for p in dd])
            self.assertEqual(len(set(p["source_id"] for p in dd)),10)

    def test_bidirectional_protocol_and_cache_identity(self):
        a,b=pair("a"),pair("b")
        data={}
        for p in [a,b]:
            for idx,side in enumerate(["left","right"]):
                data[p[side]]=dict(heads=np.array([[1+idx+(10 if p==b else 0)]]),ell=10+idx)
        class Engine:
            def __init__(self):self.calls=[]
            def prepare(self,prompt):return prompt
            def run(self,context,replacements,capture):
                self.calls.append((context,replacements[0].copy()))
                return {"ell":float(replacements[0][0])}
        eng=Engine()
        with tempfile.TemporaryDirectory() as td:
            p=ShamDonorPatcher(td,Store(data),{x:{"prompt":x} for x in data},eng,[a,b],{"model":"q"})
            true=p.effects(a,a,[0])
            sham=p.effects(a,b,[0],True)
            self.assertEqual([(c,float(v[0])) for c,v in eng.calls],
                             [(a["left"],2),(a["right"],1),(a["left"],12),(a["right"],11)])
            np.testing.assert_allclose(true,[9])
            np.testing.assert_allclose(sham,[1])
            p.effects(a,b,[0],True)
            self.assertEqual(len(eng.calls),4)
            with self.assertRaises(ValueError):p.effects(a,a,[0],True)
            with self.assertRaises(ValueError):p.effects(dict(a,split="test"),b,[0],True)
            p.close()
            with self.assertRaises(ValueError):
                ShamDonorPatcher(td,Store(data),{},eng,[a,b],{"model":"other"})

    def test_checkpoint_head_geometry(self):
        records=[dict(kind=k,head=h,confirmed=True) for k in ("A","C") for h in (0,15,16,31,32)]
        a=m.combine_sets(records,64,16)
        b=m.combine_sets(records,64,8)
        self.assertEqual(a["layer_distribution"],{"0":2,"1":2,"2":1,"3":0})
        self.assertEqual(b["layer_distribution"]["4"],1)
        with self.assertRaises(ValueError):m.combine_sets(records,32,8)

    def test_scoped_store_rejects_test_before_read(self):
        store=Store({})
        scoped=m.ScopedStore(store,["fit"])
        for call in (scoped.get,scoped.ell,scoped.heads):
            with self.assertRaises(ValueError):call("test")
        self.assertEqual(store.access,[])

    def test_screen_and_confirmation_split_restrictions(self):
        with self.assertRaises(ValueError):m.screen_candidates([pair("t",split="test")],Store({}))
        with self.assertRaises(ValueError):m.measure([pair("f",split="fit")],[],None)

    def test_incomplete_stops_m2_m3(self):
        pp=[pair("fit",split="fit")]
        with tempfile.TemporaryDirectory() as td, patch(
            "experiments.aster_e678_adaptive_20261116.pipeline.select",
            return_value=([],{"status":"selection_incomplete"})), patch.object(training,"_m2") as m2, patch.object(training,"refit_m3") as m3:
            r=fit_group("E1",pp,Store({}),[],[],td,dev_pairs=[pair("dev")],
                         dev_features=Store({}),patcher=None,total_heads=8,heads_per_layer=2)
            self.assertFalse(r["complete"]);m2.assert_not_called();m3.assert_not_called()

    def test_m2_m3_reject_dev_test(self):
        for split in ["dev","test"]:
            with self.assertRaises(ValueError):
                fit_group("E1",[pair("f",split="fit")],Store({}),
                          [dict(view_id="x",source_id="d",split=split)],[],Path("/unused"),
                          dev_pairs=[pair("d")],dev_features=Store({}),patcher=None,
                          total_heads=8,heads_per_layer=2)

    def test_m2_small_dimension_no_added_heads(self):
        rng=np.random.default_rng(7)
        pairs=[pair("s"+str(i),kind=k,split="fit") for i in range(20) for k in "APCF"]
        data={uid:dict(heads=rng.normal(size=(3,2)),ell=0.) for uid in m.pair_ids(pairs)}
        with tempfile.TemporaryDirectory() as td:
            support,summary=training._m2("E1",[0,1],pairs,Store(data),[],Path(td))
            self.assertEqual(summary["dimension"],4)
            self.assertLessEqual(summary["S_rank"],4)
            np.testing.assert_array_equal(support["heads"],[0,1])

    def test_missing_donors_not_silently_reduced(self):
        screen=[dict(kind=k,head=0,q_screen=.001) for k in "AC"]
        effects=[dict(kind=k,head=0,source_id=str(s),true_effect=2,sham_effects=[0]*3)
                 for k in "AC" for s in range(20)]
        with tempfile.TemporaryDirectory() as td:
            main=m.analyze(screen,effects,[str(s) for s in range(20)],1,1,td)
            self.assertEqual(main["heads"],[])
            self.assertEqual(json.loads((Path(td)/"donors_3.json").read_text())["heads"],[0])

if __name__=="__main__": unittest.main()
