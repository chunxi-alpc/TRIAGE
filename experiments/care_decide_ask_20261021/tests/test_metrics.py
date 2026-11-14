import unittest
from care_decide_ask.metrics import evaluate

class MetricsTests(unittest.TestCase):
    def test_wrong_decision_and_overask_denominators(self):
        gold={'a':{'family_id':'f1','requires_ask':True,'gold_action':None,'acceptable_questions':['eligible']},
              'b':{'family_id':'f1','requires_ask':False,'gold_action':'APPROVE','acceptable_questions':[]},
              'c':{'family_id':'f2','requires_ask':False,'gold_action':'DEFER','acceptable_questions':[]}}
        out=[{'case_id':'a','status':'ASK','ask_attribute':'eligible'},
             {'case_id':'b','status':'ASK','ask_attribute':'archive_ok'},
             {'case_id':'c','status':'DECIDE','action':'DEFER'}]
        report=evaluate(out,gold,bootstrap_repetitions=20)
        self.assertEqual(report['wrong_decision_rate_all_cases'],0)
        self.assertEqual(report['unnecessary_ask_rate_among_decidable'],.5)
        self.assertEqual(report['question_hit_rate_among_need_to_ask'],1)
        self.assertEqual(report['task_behavior_accuracy'],2/3)
        self.assertEqual(report['n_families'],2)

    def test_missing_sample_not_dropped(self):
        with self.assertRaisesRegex(ValueError,'missing evaluated'):
            evaluate([{'case_id':'a','status':'REVIEW'}],{
                'a':{'family_id':'f','requires_ask':False,'gold_action':'A'},
                'b':{'family_id':'f','requires_ask':False,'gold_action':'B'}})
