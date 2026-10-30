import copy
import unittest
import torch
from src.aster.manuscript_core import PairMoments,FitConfig
from .common import cosine,source_mean_ci,holm
from .gpu_controls import variant
from .candidates import validate_annotation

class NumericalTests(unittest.TestCase):
    def test_pair_equal_weight_and_mean_cosine_identity(self):
        m=PairMoments();m.update(torch.tensor([[2.,1.]]),torch.zeros(1,2));m.update(torch.tensor([[0.,2.]]*5),torch.zeros(5,2))
        c=m.estimate(FitConfig(1,1));self.assertTrue(torch.allclose(torch.tensor(c['mean']),torch.tensor([1.,1.5])))
        self.assertAlmostEqual(cosine(c['direction'],c['mean']),c['mean_norm_retention'],places=12)
    def test_cluster_bootstrap_does_not_count_repeated_methods(self):
        rows=[{'source_id':'a','x':0.},{'source_id':'b','x':1.}]
        a=source_mean_ci(rows,'x');b=source_mean_ci(rows+rows,'x')
        self.assertEqual(a['estimate'],b['estimate']);self.assertEqual(a['ci95'],b['ci95']);self.assertEqual(b['sources'],2)
    def test_holm(self):self.assertEqual(holm({'a':.01,'b':.04}),{'a':.02,'b':.04})
    def test_variant_does_not_mutate_input_or_swap_conditions(self):
        row={'states':{'positive':'Age at current encounter: 18 years.','negative':'Age at current encounter: 17 years.','unknown':'Age at current encounter: not available.'},'guideline':{'groups':[{'rules':[{'condition':'yes','action':'A'},{'condition':'no','action':'B'}]}]}}
        original=copy.deepcopy(row);changed=variant(row,'action_swap')
        self.assertEqual(row,original);self.assertEqual(changed['guideline']['groups'][0]['rules'][0]['condition'],'yes');self.assertEqual(changed['guideline']['groups'][0]['rules'][0]['action'],'B')
    def test_missing_adjudication_never_produces_labels(self):
        with self.assertRaises(ValueError):validate_annotation({'candidate_id':'x'},[])
    def test_machine_review_cannot_be_physician_gold(self):
        candidate={'candidate_id':'x'};reviews=[{'candidate_id':'x','reviewer_id':x,'role':'independent_reviewer','reviewer_type':'model'} for x in ['a','b']]
        with self.assertRaises(ValueError):validate_annotation(candidate,reviews)

    def annotation_fixture(self, labels=None, reachable=True):
        candidate={'candidate_id':'fixture:x','source_id':'fixture:source','partition':'confirm'}
        reviews=[{'candidate_id':'fixture:x','reviewer_id':reviewer,'role':'independent_reviewer','reviewer_type':'human_clinician',
                  'labels':labels or {'a':'true','b':'false','missing':'unknown'},'ancestors_reachable':reachable,
                  'corrected_patient_texts':{'a':'synthetic fixture true','b':'synthetic fixture false','missing':'synthetic fixture unknown'},'evidence':'synthetic test fixture only'}
                 for reviewer in ['fixture_reviewer_1','fixture_reviewer_2']]
        return candidate,reviews
    def test_unknown_cannot_replace_negative_state(self):
        candidate,reviews=self.annotation_fixture({'a':'true','b':'unknown','missing':'unknown'})
        with self.assertRaises(ValueError):validate_annotation(candidate,reviews)
    def test_missing_state_must_be_unknown(self):
        candidate,reviews=self.annotation_fixture({'a':'true','b':'false','missing':'true'})
        with self.assertRaises(ValueError):validate_annotation(candidate,reviews)
    def test_unreachable_adjudication_is_retained_but_ineligible(self):
        candidate,reviews=self.annotation_fixture(reachable=False)
        result=validate_annotation(candidate,reviews)
        self.assertEqual(result['status'],'physician_adjudicated');self.assertFalse(result['eligible_for_direction']);self.assertEqual(result['exclusion_reason'],'ancestor_not_reachable')

    def test_export_preserves_target_truth_and_source_dependency(self):
        from .adjudicated import export_triplets
        c={'candidate_id':'fixture:x','source_id':'fixture:source','partition':'confirm','source_guideline_id':0,'proposed_measurement':'synthetic C',
           'source_tree':{'index':0,'description':'Synthetic predicate C','yes':{'index':1,'description':'Synthetic action A'},'no':{'index':2,'description':'Synthetic action B'}}}
        a={'candidate_id':'fixture:x','source_id':'fixture:source','partition':'confirm','status':'physician_adjudicated','eligible_for_direction':True,'ancestors_reachable':True,
           'labels':{'a':'true','b':'false','missing':'unknown'},'patient_texts':{'a':'C true: synthetic fixture','b':'C false: synthetic fixture','missing':'C absent: synthetic fixture'},'evidence':'synthetic fixture only','reviewer_ids':['fixture1','fixture2']}
        rows,excluded=export_triplets([c],[a])
        self.assertEqual(len(rows),2);self.assertEqual(excluded,[]);self.assertEqual(rows[0]['source_id'],rows[1]['source_id'])
        self.assertEqual(rows[0]['states']['positive'],a['patient_texts']['a']);self.assertEqual(rows[1]['states']['positive'],a['patient_texts']['b'])

if __name__=='__main__':unittest.main()
