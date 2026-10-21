import unittest
from care_decide_ask import decide_or_ask,derive_truth_states
from care_decide_ask.calibration import calibrate_joint,make_prediction_sets,case_nonconformity
from care_decide_ask.a1_adapter import frozen_a1_hard_sets,require_a1_probabilities


def policy():
    yes={'and':[{'atom':'eligible'},{'atom':'clearance'}]}
    return {'atoms':['eligible','clearance','exception','archive_ok'],
            'rules':[{'rule_id':'R_EXCEPTION','priority':0,'when':{'atom':'exception'},'action':'ESCALATE'},
                     {'rule_id':'R_APPROVE','priority':1,'when':yes,'action':'APPROVE'},
                     {'rule_id':'R_DEFER','priority':2,'when':{'not':yes},'action':'DEFER'}],
            'default_action':'REVIEW'}


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.p=policy()
        self.full={r['rule_id']:['T','F','U'] for r in self.p['rules']}

    def test_known_true_decide(self):
        c={'eligible':'T','clearance':'T','exception':'F','archive_ok':'U'}
        r=decide_or_ask(self.p,c,self.full)
        self.assertEqual((r['status'],r['action']),('DECIDE','APPROVE'))
        self.assertEqual(derive_truth_states(self.p,c)['R_APPROVE'],'T')

    def test_critical_unknown_asks_specific_attribute(self):
        c={'eligible':'U','clearance':'T','exception':'F','archive_ok':'U'}
        r=decide_or_ask(self.p,c,self.full)
        self.assertEqual((r['status'],r['ask_attribute']),('ASK','eligible'))
        self.assertEqual(r['possible_actions'],['APPROVE','DEFER'])

    def test_unknown_irrelevant_does_not_force_question(self):
        c={'eligible':'T','clearance':'T','exception':'F','archive_ok':'U'}
        r=decide_or_ask(self.p,c,self.full)
        self.assertEqual(r['status'],'DECIDE')

    def test_unknown_but_known_false_other_condition_decide(self):
        c={'eligible':'U','clearance':'F','exception':'F','archive_ok':'U'}
        r=decide_or_ask(self.p,c,self.full)
        self.assertEqual((r['status'],r['action']),('DECIDE','DEFER'))

    def test_exception_override_preserved(self):
        c={'eligible':'U','clearance':'U','exception':'T','archive_ok':'U'}
        r=decide_or_ask(self.p,c,self.full)
        self.assertEqual((r['status'],r['action']),('DECIDE','ESCALATE'))

    def test_inconsistent_labels_fail_closed(self):
        c={'eligible':'T','clearance':'T','exception':'F','archive_ok':'F'}
        invalid=dict(self.full,R_APPROVE=['F'])
        r=decide_or_ask(self.p,c,invalid)
        self.assertEqual(r['status'],'REVIEW')
        self.assertEqual(r['world_count'],0)

    def test_invalid_priority_rejected(self):
        p=policy();p['rules'][1]['priority']=0
        with self.assertRaises(ValueError):decide_or_ask(p,{},self.full)

    def test_unanswerable_attributes_trigger_review_instead_of_fake_question(self):
        p=policy();p['askable_atoms']=['clearance','archive_ok']
        row={'eligible':'U','clearance':'T','exception':'F','archive_ok':'T'}
        result=decide_or_ask(p,row,self.full)
        self.assertEqual(result['status'],'REVIEW')
        self.assertEqual(result['reason'],'multiple_actions_but_no_answerable_single_attribute')

    def test_restrict_questions_to_answerable_attributes(self):
        p=policy();p['askable_atoms']=['eligible','archive_ok']
        row={'eligible':'U','clearance':'T','exception':'F','archive_ok':'U'}
        result=decide_or_ask(p,row,self.full)
        self.assertEqual((result['status'],result['ask_attribute']),('ASK','eligible'))

    def test_no_empty_constraint_silent_answer(self):
        p=policy();p['constraints']=[{'and':[{'atom':'exception'}, {'not':{'atom':'exception'}}]}]
        self.assertEqual(decide_or_ask(p,{},self.full)['status'],'REVIEW')


class CalibrationTests(unittest.TestCase):
    def test_case_level_max_nonconformity(self):
        row={'truth':{'a':'T','b':'U'},
             'probs':{'a':{'T':.9,'F':.05,'U':.05},'b':{'T':.3,'F':.3,'U':.4}}}
        self.assertAlmostEqual(case_nonconformity(row),.6)
        report=calibrate_joint([row]*19,.1)
        self.assertEqual(report['quantile_index_one_based'],18)
        self.assertAlmostEqual(report['threshold'],.6)
        groups=make_prediction_sets(row['probs'],report)
        self.assertIn('T',groups['a']);self.assertIn('U',groups['b'])
        self.assertFalse(report['conditional_decision_error_bound'])

    def test_n_small_infinite_quantile(self):
        r={'truth':{'r':'T'},'probs':{'r':{'T':.9,'F':.05,'U':.05}}}
        q=calibrate_joint([r]*3,.05)
        self.assertTrue(q['infinite_threshold'])
        self.assertEqual(make_prediction_sets({'r':r['probs']['r']},q)['r'],['T','F','U'])

    def test_missing_scores_fall_back_to_full_sets(self):
        r={'truth':{'r':'F'},'probs':{'r':None}}
        self.assertEqual(case_nonconformity(r),0)
        q=calibrate_joint([r]*30,.1)
        self.assertEqual(make_prediction_sets({'r':None},q)['r'],['T','F','U'])

    def test_hard_A1_cannot_fake_guarantee(self):
        pred={'arm':'A1','rules':[{'rule_id':'r1','state':'T'}, {'rule_id':'r2','state':'U'}]}
        groups=frozen_a1_hard_sets(pred,['r1','r2','r3'])
        self.assertEqual(groups['r1'],['T'])
        self.assertEqual(groups['r2'],['T','F','U'])
        self.assertEqual(groups['r3'],['T','F','U'])
        with self.assertRaisesRegex(ValueError,'FROZEN_A1_NO_PROBABILITIES'):
            require_a1_probabilities(pred,['r1','r2'])

    def test_invalid_score_rejected(self):
        r={'truth':{'r':'T'},'probs':{'r':{'T':.7,'F':.5,'U':.2}}}
        with self.assertRaises(ValueError):case_nonconformity(r)


if __name__=='__main__':unittest.main()

class JointFamilyTests(unittest.TestCase):
    def test_family_calibration_uses_family_max(self):
        base={'truth':{'r':'T'},'probs':{'r':{'T':.95,'F':.025,'U':.025}}}
        rows=[]
        for k in range(30):
            for j in range(2):
                row=dict(base, family_id=f'family_{k:02d}')
                rows.append(row)
        out=calibrate_joint(rows,.1,unit='family')
        self.assertEqual(out['n_units'],30)
        self.assertEqual(out['n_cases'],60)
        self.assertEqual(out['quantile_index_one_based'],28)
        with self.assertRaises(ValueError):calibrate_joint([base]*20,.1,unit='family')

    def test_frozen_case_technical_failure_falls_back(self):
        pred={'arm':'A1','status':'parse_failure',
              'rules':[{'rule_id':'r','state':'T'}]}
        self.assertEqual(frozen_a1_hard_sets(pred,['r']),{'r':['T','F','U']})
