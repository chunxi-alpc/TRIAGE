"""CPU-only regression tests; synthetic examples are not model performance data."""
import unittest
import numpy as np
from .revision import (Config, balanced_mean, confirm, mass_select, paired_effect,
                       partition_sources, screen, symmetric_attribution, validate_pairs,
                       ExactPatcher, select_and_fit)


def pair(kind="A", source="s0", identifier="p0"):
    return dict(pair_id=identifier, source_id=source, kind=kind, split="fit",
                left=identifier+"L", right=identifier+"R", left_label=0,
                right_label=int(kind in ("C", "F")))


def patch(p, head=0, effect=1.):
    return dict(pair_id=p["pair_id"], source_id=p["source_id"], kind=p["kind"], head=head,
                left_label=p["left_label"], right_label=p["right_label"],
                base_left=0., base_right=2., paired_forward=effect,
                paired_reverse=2.-effect, self_left=0., self_right=2.)


def fixture():
    ps=[pair(k, "s"+str(i), k+str(i)) for k in ("A", "P", "C", "F") for i in range(2)]
    candidates={k:dict(heads=[0]) for k in ("A", "P", "C", "F")}
    return ps, candidates


class Regression(unittest.TestCase):
    def test_constant_gradient_old_screen_false_negative(self):
        # Equivalent to the old source-permutation statistic, for every permutation.
        g=np.ones((4,1,1)); d=np.arange(1.,5.).reshape(4,1,1)
        cross=np.einsum('shd,thd->sth',g,d)
        observed=np.abs(cross[np.arange(4),np.arange(4)]).mean(0)
        from itertools import permutations
        null=[np.abs(cross[np.arange(4),p]).mean(0) for p in permutations(range(4))]
        self.assertTrue(np.all(np.asarray(null)==observed))
        p_old=(1+sum(np.all(v>=observed) for v in null))/(len(null)+1)
        self.assertEqual(p_old,1.)
        left=dict(heads=np.zeros((1,1)),grads=np.ones((1,1)))
        right=dict(heads=np.ones((1,1))*2,grads=np.ones((1,1)))
        self.assertEqual(symmetric_attribution(left,right)[0],2.)
        self.assertEqual(mass_select(symmetric_attribution(left,right),.95),[0])

    def test_random_replacement_not_valid_gate(self):
        p=pair(); r=patch(p,effect=.1)
        new=paired_effect(r)["magnitude"]
        old=.1-10. # Larger unrelated replacement would reject an effective head.
        self.assertLess(old,0); self.assertGreater(new,0)

    def test_symmetric_attribution_uses_right(self):
        l=dict(heads=np.zeros((1,1)),grads=np.zeros((1,1)))
        r=dict(heads=np.ones((1,1)),grads=np.ones((1,1))*4)
        self.assertEqual(symmetric_attribution(l,r)[0],2)

    def test_missing_right_gradient_rejected(self):
        with self.assertRaises(ValueError):
            symmetric_attribution(dict(heads=[[0]],grads=[[1]]),dict(heads=[[1]]))

    def test_source_equal_weight(self):
        values=[[10]]*10+[[0]]
        self.assertEqual(balanced_mean(values,["a"]*10+["b"])[0],5)

    def test_mass_ties_not_fixed_k(self):
        self.assertEqual(mass_select([4,2,2,0],.70),[0,1,2])

    def test_mass_zero_has_no_backfill(self):
        self.assertEqual(mass_select([0,0],.90),[])

    def test_mass_all(self):
        self.assertEqual(mass_select([1,2,3],1),[0,1,2])

    def test_mass_overflow_resistant(self):
        self.assertEqual(mass_select([1e308,1e308],.5),[0,1])

    def test_mass_invalid(self):
        for scores in ([-1,2],[float("nan")],[float("inf")]):
            with self.assertRaises(ValueError): mass_select(scores,.9)

    def test_identity_failure_stops(self):
        row=patch(pair());row["self_left"]=.5
        with self.assertRaises(ValueError): paired_effect(row)

    def test_nonfinite_patch_stops(self):
        row=patch(pair());row["paired_forward"]=float("nan")
        with self.assertRaises(ValueError): paired_effect(row)

    def test_no_effect_is_zero(self):
        self.assertEqual(paired_effect(patch(pair(),effect=0))["magnitude"],0)

    def test_negative_evidence_direction_preserved(self):
        e=paired_effect(patch(pair("F"),effect=-1))
        self.assertLess(e["label_aligned"],0);self.assertGreater(e["magnitude"],0)

    def test_common_mode_has_zero_pair_effect(self):
        row=patch(pair("F"))
        row.update(paired_forward=5.,paired_reverse=7.)
        effect=paired_effect(row)
        self.assertEqual(effect["u_L"],5.)
        self.assertEqual(effect["u_R"],5.)
        self.assertEqual(effect["common_mode"],5.)
        self.assertEqual(effect["c"],0.)
        self.assertEqual(effect["magnitude"],0.)
        self.assertEqual(effect["label_aligned"],0.)

    def test_common_mode_never_selects_head(self):
        ps,candidates=fixture()
        records=[dict(patch(p),paired_forward=5.,paired_reverse=7.) for p in ps]
        result=confirm(ps,candidates,records,1,Config(min_sources=2))
        self.assertEqual(result["heads"],[])
        self.assertEqual(result["status"],"intercept_only_fallback")

    def test_common_mode_does_not_change_pair_ranking(self):
        row=patch(pair("C"),effect=-2.)
        shifted=dict(row,paired_forward=row["paired_forward"]+10.,
                     paired_reverse=row["paired_reverse"]+10.)
        a,b=paired_effect(row),paired_effect(shifted)
        self.assertEqual(a["magnitude"],b["magnitude"])
        self.assertEqual(a["label_aligned"],b["label_aligned"])
        self.assertLess(b["label_aligned"],0.)

    def test_pair_effect_subtracts_both_numerical_errors(self):
        row=patch(pair("C"))
        row.update(self_left=5e-5,self_right=2.-4e-5)
        effect=paired_effect(row)
        floor=.5*(5e-5+4e-5+1e-4+1.2e-4)
        self.assertAlmostEqual(effect["numerical_floor"],floor)
        self.assertAlmostEqual(effect["magnitude"],abs(effect["c"])-floor)

    def test_endpoint_reversal_invariance(self):
        p=pair("F");r=patch(p)
        reversed_row=dict(r,base_left=r["base_right"],base_right=r["base_left"],
                          paired_forward=r["paired_reverse"],paired_reverse=r["paired_forward"],
                          self_left=r["self_right"],self_right=r["self_left"],left_label=1,right_label=0)
        a,b=paired_effect(r),paired_effect(reversed_row)
        self.assertAlmostEqual(a["magnitude"],b["magnitude"])
        self.assertAlmostEqual(a["label_aligned"],b["label_aligned"])

    def test_dev_test_rejected(self):
        for split in ("dev","test"):
            with self.assertRaises(ValueError): validate_pairs([dict(pair(),split=split)])

    def test_label_transition_rejected(self):
        with self.assertRaises(ValueError): validate_pairs([dict(pair("F"),right_label=0)])

    def test_duplicate_pair_rejected(self):
        with self.assertRaises(ValueError): validate_pairs([pair(),pair()])

    def test_cross_source_endpoint_rejected(self):
        with self.assertRaises(ValueError):
            validate_pairs([pair(),dict(pair(source="s1",identifier="p1"),left="p0L")])

    def test_partition_disjoint_reproducible(self):
        ps=[pair(k,"s"+str(i),k+str(i)) for i in range(8) for k in ("A","P","C","F")]
        a,b=partition_sources(ps,Config(min_sources=2))
        self.assertFalse(set(a)&set(b));self.assertEqual(len(a)+len(b),8)
        self.assertEqual((a,b),partition_sources(list(reversed(ps)),Config(min_sources=2)))

    def test_insufficient_coverage_rejected(self):
        with self.assertRaises(ValueError): partition_sources([pair()],Config(min_sources=2))

    def test_single_evidence_group_can_proceed(self):
        ps,c=fixture(); rows=[patch(p,effect=int(p["kind"]=="F")) for p in ps]
        result=confirm(ps,c,rows,1,Config(min_sources=2))
        self.assertEqual(result["heads"],[0]);self.assertEqual(result["nuisance_response_heads"],[])
        self.assertEqual(result["status"],"features_selected")

    def test_empty_union_is_explicit_fallback(self):
        ps,c=fixture(); rows=[patch(p,effect=0) for p in ps]
        self.assertEqual(confirm(ps,c,rows,1,Config(min_sources=2))["status"],"intercept_only_fallback")

    def test_missing_result_not_negative_evidence(self):
        ps,c=fixture()
        with self.assertRaises(ValueError): confirm(ps,c,[patch(p) for p in ps[:-1]],1,Config(min_sources=2))

    def test_duplicate_result_rejected(self):
        ps,c=fixture();rows=[patch(p) for p in ps]
        with self.assertRaises(ValueError): confirm(ps,c,rows+[rows[0]],1,Config(min_sources=2))

    def test_wrong_metadata_rejected(self):
        ps,c=fixture();rows=[patch(p) for p in ps];rows[0]["source_id"]="other"
        with self.assertRaises(ValueError): confirm(ps,c,rows,1,Config(min_sources=2))

    def test_config_rejects_invalid_thresholds(self):
        with self.assertRaises(ValueError):Config(retained_mass=1.1)
        with self.assertRaises(ValueError):Config(identity_atol=-1)

    def test_engine_adapter_with_linear_model(self):
        p=pair("F")
        class Store:
            def get(self,uid):return dict(heads=np.array([[0.] if uid.endswith('L') else [2.]]))
            def ell(self,uid):return 0. if uid.endswith('L') else 2.
        class Engine:
            def prepare(self,prompt):return float(prompt)
            def run(self,context,replacements=None,capture=False):
                return dict(ell=float(replacements[0][0]) if replacements else context)
        inputs={p['left']:dict(prompt='0'),p['right']:dict(prompt='2')}
        row=ExactPatcher(Store(),inputs,Engine()).measure(p,[0])[0]
        self.assertEqual(row['paired_forward'],2)
        self.assertGreater(paired_effect(row)['label_aligned'],0)

    def test_changed_baseline_rejected(self):
        p=pair("F")
        class Store:
            def get(self,uid):return dict(heads=np.ones((1,1)))
            def ell(self,uid):return 5.
        class Engine:
            def prepare(self,prompt):return prompt
            def run(self,*args,**kwargs):return dict(ell=0.)
        inputs={p['left']:dict(prompt='x'),p['right']:dict(prompt='x')}
        with self.assertRaises(ValueError):ExactPatcher(Store(),inputs,Engine()).measure(p,[0])

    def test_fit_requires_identity_before_any_model_calls(self):
        with self.assertRaises(ValueError):
            select_and_fit([pair()],None,[],[],{},None,"never_created",{})


    def test_complete_entry_with_mock_training(self):
        # Wiring test only: M2/M3 are mocks, never reported as trained models.
        import sys, types, tempfile
        from pathlib import Path
        from unittest.mock import patch as mock_patch
        ps=[pair(k,"s"+str(i),k+str(i)) for i in range(8) for k in ("A","P","C","F")]
        class Store:
            def get(self,uid):
                return dict(heads=np.array([[0. if uid.endswith('L') else 2.]]),grads=np.ones((1,1)))
            def ell(self,uid):return 0. if uid.endswith('L') else 2.
            def heads(self,uid,selected):return self.get(uid)['heads'][selected].reshape(-1)
        class Engine:
            def prepare(self,prompt):return float(prompt)
            def run(self,context,replacements=None,capture=False):
                return dict(ell=float(replacements[0][0]) if replacements else context)
        inputs={p[k]:dict(prompt='0' if k=='left' else '2') for p in ps for k in ('left','right')}
        natural=[dict(view_id=p['left'],source_id=p['source_id'],split='fit',label=i%2) for i,p in enumerate(ps)]
        module=types.ModuleType('experiments.aster_cr_retrain_paired_v2_20261111')
        module.training=types.SimpleNamespace(_m2=lambda *args: ({},dict(rank=16)),
                                             refit_m3=lambda *args:dict(rank=1,success=True))
        identity={k:'0'*64 for k in ('checkpoint_sha256','tokenizer_sha256','inputs_sha256','scoring_sha256')}
        with tempfile.TemporaryDirectory() as td, mock_patch.dict(sys.modules,{module.__name__:module}):
            out=Path(td)/'new'
            result=select_and_fit(ps,Store(),[],natural,inputs,Engine(),out,identity,Config(min_sources=2))
            self.assertEqual(result['heads'],[0]); self.assertFalse(result['performance_evaluated'])
            self.assertTrue((out/'protocol.json').is_file())
            with self.assertRaises(FileExistsError):
                select_and_fit(ps,Store(),[],natural,inputs,Engine(),out,identity,Config(min_sources=2))

    def test_resource_preflight_never_truncates(self):
        import tempfile
        from pathlib import Path
        ps=[pair(k,"s"+str(i),k+str(i)) for i in range(8) for k in ("A","P","C","F")]
        class Store:
            def get(self,uid):return dict(heads=[[0 if uid.endswith('L') else 2]],grads=[[1]])
        natural=[dict(view_id='u',source_id='s0',split='fit',label=i) for i in (0,1)]
        identity={k:'0'*64 for k in ('checkpoint_sha256','tokenizer_sha256','inputs_sha256','scoring_sha256')}
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'new'
            with self.assertRaises(RuntimeError):
                select_and_fit(ps,Store(),[],natural,{},None,out,identity,Config(min_sources=2,max_patch_forwards=1))
            self.assertFalse(out.exists())


if __name__=="__main__":unittest.main()
