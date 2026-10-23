"""Synthetic tests and public-only invariants; never need patient data in CI."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from step0_private_structure_audit import retained, structural, transition, first_step_mask_bound, select_cad, sha
from step0_clinical_review import make_packets
from test_step0_extended_audit import fixture
from step0_reanalysis import ROOT


def complete_fixture():
    c,rows=fixture()
    c.update(patient='Synthetic patient',original_answer='A',source_variants={'A':{'original_sibling_action_label':'B'}})
    for b in c['branches']:b['condition']=c['variant_rules']['A'][b['rule_indices'][0]]['condition']
    return c,rows


class PrivateAuditTests(unittest.TestCase):
    def test_other_to_other_is_changed_only_if_label_changes(self):
        self.assertEqual(transition('C','C','A','B'),'unchanged')
        self.assertEqual(transition('C','D','A','B'),'wrong_to_wrong_changed')
        self.assertEqual(transition('A','C','A','B'),'gold_to_other')
        self.assertEqual(transition('A','B','A','B'),'gold_to_sibling')

    def test_literal_absence_is_not_semantic_label(self):
        c,rows=complete_fixture();x=structural([c],rows,['synthetic'])
        self.assertEqual(x['cohorts']['A24'][0]['literal_class'],2)
        self.assertIsNone(x['semantic_categories_3_4']['counts'])
        self.assertTrue(x['cohorts']['A24'][0]['original_in_D'])
        self.assertFalse(x['cohorts']['A24'][0]['original_in_A'])

    def test_partition_and_condition_link_fail_closed(self):
        c,rows=complete_fixture();c['branches'][1]['rule_indices']=[-1,1]
        with self.assertRaisesRegex(ValueError,'partition'):structural([c],rows,['synthetic'])
        c,rows=complete_fixture();c['branches'][1]['condition']='different'
        with self.assertRaisesRegex(ValueError,'condition linkage'):structural([c],rows,['synthetic'])

    def test_exact_and_normalized_matches_are_distinct(self):
        c,_=complete_fixture();c['options']['A']='  ACTION ONE. '
        r=retained(c,'A',[0]);self.assertEqual(r['allowed_option_labels'],['A']);self.assertEqual(r['exact_option_labels'],[])

    def test_review_packet_blinding(self):
        c,rows=complete_fixture();s=structural([c],rows,['synthetic'])
        with tempfile.TemporaryDirectory() as d:
            result=make_packets([c],s,Path(d));self.assertEqual(result['items_per_reviewer'],4)
            packet=(Path(d)/'reviewer1.jsonl').read_text()
            for key in ['stratum','prediction','gold_answer','variant','synthetic','outcome','control']:
                self.assertNotIn('"'+key+'"',packet)
            with self.assertRaises(ValueError):make_packets([c],s,Path(d))

    def test_legacy_missing_assigned_wrong_and_duplicate_rejection(self):
        r=dict(id='case',variant='B_original',method='cad')
        self.assertEqual(select_cad([r],['case'])['case'],r)
        with self.assertRaises(ValueError):select_cad([r,r],['case'])
        with self.assertRaises(ValueError):select_cad([dict(r,assigned_wrong=True)],['case'])

    def test_mask_bound_is_sufficient_not_necessary(self):
        self.assertTrue(first_step_mask_bound({'A':-0.1,'B':-4},'B',0.1))
        self.assertFalse(first_step_mask_bound({'A':-10,'B':-11},'B',0.1))
        # The latter can still be masked if an unrecorded vocabulary token has logp=-0.1.
        self.assertTrue(first_step_mask_bound({'A':-10,'B':-11,'unrecorded':-0.1},'B',0.1))

    def test_frozen_private_structure_public_invariants(self):
        p=ROOT/'step0_review/private_structure_summary.json'
        if not p.exists():self.skipTest('snapshot not yet written')
        x=json.loads(p.read_text());self.assertEqual(x['provenance']['n'],200)
        self.assertEqual(x['provenance']['script_sha256'],sha(Path(__file__).with_name('step0_private_structure_audit.py')))
        self.assertEqual(x['provenance']['frozen_public_sha256'],sha(ROOT/'step0_review/extended_public.json'))
        self.assertEqual(len(x['cohorts']['A24']),24)
        self.assertEqual(sum(r['literal_class']==2 for r in x['cohorts']['A24']),5)
        self.assertEqual(sum(r['literal_class']==1 for r in x['cohorts']['D6']),6)
        for g,n in [('all',200),('K',76),('U',124)]:self.assertEqual(sum(x['direction_summary'][g].values()),n)
        self.assertEqual(x['sources']['n_dataset_trees'],23)
        self.assertIsNone(x['sources']['independent_clinical_sources'])
        for v,n in [('B',34),('D',22)]:
            rr=x['cohorts'][v+'_oracle_errors'];self.assertEqual(len(rr),n);self.assertFalse(any(r['permitted'] for r in rr))
        self.assertEqual(len(x['cad']['cases']),18)
        self.assertEqual(sum(r['first_step_gold_certainly_masked_from_option_lower_bound'] for r in x['cad']['cases']),16)
        for r in x['cad']['cases']:
            self.assertFalse(r['per_step_masks_available'])

if __name__=='__main__':unittest.main()
