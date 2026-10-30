import unittest
from src.aster.manuscript_core import plan_gains
from src.aster.action_blind import summarize_logits
from experiments.aster_manuscript_clinical_20261030.engine import parse_prediction
from .audit import strict_parse,strict_score,oracle_decisions,paired

class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.guide={'groups':[{'group_id':'G0','mutually_exclusive':True,'rules':[
            {'rule_id':'R0Y','condition':'age>=18','action':'继续 G1'},
            {'rule_id':'R0N','condition':'age<18','action':'L2：minor'}]},
            {'group_id':'G1','mutually_exclusive':True,'parent_group_id':'G0','parent_rule_id':'R0Y',
             'ancestor_conditions':['age>=18'],'rules':[{'rule_id':'R1Y','condition':'fever','action':'L3：yes'},
             {'rule_id':'R1N','condition':'no fever','action':'L4：no'}]}]}
    def test_unreachable_child_cannot_edit(self):
        p=plan_gains(self.guide,{'G0':{'prediction':'R0N','abstain':False},'G1':{'prediction':'R1Y','abstain':False}},g_plus=1,g_minus=1)
        self.assertFalse(p['groups']['G1']['reachable']);self.assertEqual(p['gains']['R1Y'],0);self.assertEqual(p['gains']['R1N'],0)
    def test_NONE_abstain_failure_separate(self):
        for decision,reason in [(None,'classification_failure'),({'prediction':'NONE','abstain':False},'none'),({'prediction':'R0Y','abstain':True},'abstain')]:
            p=plan_gains(self.guide,{'G0':decision,'G1':None},g_plus=1,g_minus=1)
            self.assertEqual(p['groups']['G0']['reason'],reason);self.assertTrue(all(v==0 for v in p['gains'].values()))
    def test_zero_threshold_exact_tie_still_abstains(self):
        self.assertTrue(summarize_logits({'R0Y':0,'NONE':0},temperature=1,probability_threshold=0,margin_threshold=0)['abstain'])
    def test_duplicate_path_reproduces_legacy_bug(self):
        text='PATH: R0Y,R0Y,R1Y\nFINAL: L3'
        self.assertEqual(parse_prediction(text)[1],['R0Y','R1Y'])
        self.assertTrue(strict_parse(text)[2]);self.assertEqual(strict_parse(text)[1],['R0Y','R0Y','R1Y'])
    def test_missing_output_is_not_NONE(self):
        self.assertIsNone(strict_parse('')[0]);self.assertTrue(strict_parse('')[2])
        self.assertEqual(strict_parse('PATH: \nFINAL: NONE')[0],'NONE')
    def test_oracle_path_direct_map(self):
        from experiments.aster_manuscript_clinical_20261030.engine import direct_text
        c={'guideline':self.guide,'gold_rules':['R0Y','R1N'],'gold_leaf':'L4'}
        p=plan_gains(self.guide,oracle_decisions(c),g_plus=.25,g_minus=.25)
        score=strict_score(c,direct_text(c,p));self.assertTrue(score['leaf_correct']);self.assertTrue(score['exact_path'])
    def test_cluster_bootstrap_failures_not_omitted(self):
        rows=[{'case_id':str(i),'method':m,'source_id':str(i//2),'ok':ok,'status':status}
              for i in range(4) for m,ok,status in [('spectral',i!=0,'failed' if i==0 else 'complete'),('baseline',i==0,'complete')]]
        s=paired(rows,'spectral','baseline');self.assertEqual(s['paired_cases'],4);self.assertEqual(s['source_clusters'],2)
        self.assertEqual(s['corrected'],3);self.assertEqual(s['regressed'],1);self.assertEqual(s['paired_success_coverage'],.75)

if __name__=='__main__':unittest.main()
