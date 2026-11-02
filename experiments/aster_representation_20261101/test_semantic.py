import tempfile
import argparse
import signal
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.aster_representation_20261101 import semantic as s


def score_fixture(matrix=None, outcome=None):
    matrix=matrix or [[5,1],[1,5]]
    return {'status':'complete','judge_model':'Qwen/Qwen3-8B','expected_draws':5,
            'canonical_text_private':'selected path', 'path_draws':{'results':[matrix]*5},
            'outcome_draws':{'results':outcome or [5]*5}}


class SemanticTests(unittest.TestCase):
    def test_perfect_alignment_and_outcome(self):
        m=s.compute_metrics(score_fixture())
        self.assertEqual(m['DTW_PCC'],0.)
        self.assertEqual(m['Acc'],1.)
        self.assertEqual(m['LNDS_PCS'],1.)
        self.assertLess(m['OT_PCC'],1e-4)

    def test_lnds_includes_exact_threshold_and_preserves_order(self):
        score=score_fixture()
        score['path_draws']['results']=[[[5,1],[1,5]]]*3+[[[4,1],[1,4]]]*2
        m=s.compute_metrics(score)
        self.assertAlmostEqual(m['DTW_PCC'],.1)
        self.assertEqual(m['LNDS_PCS'],1.)
        reverse=s.compute_metrics(score_fixture([[1,5],[5,1]]))
        self.assertEqual(reverse['LNDS_PCS'],.5)

    def test_five_draw_mean_and_reuse_requirements(self):
        m=s.compute_metrics(score_fixture(outcome=[1,2,3,4,5]))
        self.assertAlmostEqual(m['Acc'],.5)
        good=score_fixture()
        self.assertTrue(s.reusable(good,'selected path'))
        self.assertFalse(s.reusable(good,'different path'))
        self.assertFalse(s.reusable(dict(good,status='failed'),'selected path'))

    def test_canonical_uses_predicted_rules_and_terminal(self):
        case={'case_id':'test:1','guideline_id':1,'path_id':1,'filename':'a','language':'en',
              'guideline':{'groups':[{'rules':[
                  {'rule_id':'R0Y','condition':'condition is true'},
                  {'rule_id':'R0N','condition':'condition is false'}]}]},
              'tree_mapping':{'1':{'node':1,'leaf':True,'description':'chosen outcome'},
                              '2':{'node':2,'leaf':True,'description':'other outcome'}},
              'gold_rules':['R0N'],'gold_leaf':'L2'}
        rendered=s.canonical(case,'PATH: R0Y\nFINAL: L1')
        self.assertIn('condition is true',rendered)
        self.assertIn('chosen outcome',rendered)
        self.assertNotIn('other outcome',rendered)
        self.assertNotIn('condition is false',rendered)

    def test_finished_successes_and_failures_are_not_repeated(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(s,'OUT',Path(directory)):
            case={'case_id':'test:1'}
            p=s.OUT/'units'/'test_1'/'no_js.json'
            s.write(p,{'status':'failed'})
            self.assertTrue(s.unit_finished(case,'no_js'))
            self.assertFalse(s.unit_finished(case,'no_js',True))
            s.write(p,{'status':'complete'})
            self.assertTrue(s.unit_finished(case,'no_js',True))

    def test_manuscript_columns_require_full_coverage_and_preserve_external_rows(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(s,'OUT',Path(directory)):
            path=Path(directory)/'manuscript.md'
            rows='\n'.join('| '+name+' | — | — | — |' for name in s.PAPER_LABELS)
            external='| Qwen3-8B | — | 47.88 | 60.94 |'
            table='| 模型／方法 | a | b | c |\n|---|---:|---:|---:|\n'+external+'\n'+rows
            original='before\n<!-- ASTER_FINAL_MAIN_START -->\n\n'+table+'\n\n<!-- ASTER_FINAL_MAIN_END -->\n'
            original+='between\n<!-- ASTER_FINAL_PATH_START -->\n\n'+table+'\n\n<!-- ASTER_FINAL_PATH_END -->\nafter\n'
            path.write_text(original)
            metrics={'DTW_PCC':.1,'Acc':.8,'OT_PCC':.2,'LNDS_PCS':.7}
            summary={'methods':{k:{'status':'incomplete','registered':2,'recorded':1,
                                  'all_registered':metrics} for k in s.METHODS},
                     'strict':{k:{'recorded':1,'registered':2,'PathAcc':None} for k in s.METHODS}}
            self.assertEqual(s.update_manuscript(summary,path,expected_cases=2)['status'],'unchanged')
            self.assertEqual(path.read_text(),original)
            summary['strict']['no_js']={'recorded':2,'registered':2,'PathAcc':1.}
            s.update_manuscript(summary,path,expected_cases=2)
            text=path.read_text()
            self.assertIn('| **处置盲内部干预（本文）** | **100.00** | — | — |',text)
            self.assertIn('| **处置盲内部干预（本文）** | — | — | — |',text)
            summary['methods']['no_js'].update(status='complete',recorded=2)
            result=s.update_manuscript(summary,path,expected_cases=2)
            text=path.read_text()
            self.assertEqual(result['updated_methods'],['no_js'])
            self.assertIn('| **处置盲内部干预（本文）** | **100.00** | **10.00** | **80.00** |',text)
            self.assertIn('| **处置盲内部干预（本文）** | **10.00** | **20.00** | **70.00** |',text)
            self.assertIn('| 普通低秩干预 | — | — | — |',text)
            self.assertEqual(text.count(external),2)
            self.assertIn('START -->\n\n| 模型',text)
            self.assertIn(' |\n\n<!-- ASTER_FINAL_PATH_END',text)
            self.assertTrue(text.startswith('before\n'))
            self.assertTrue(text.endswith('after\n'))

    def test_claims_prevent_concurrent_case_scoring(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(s,'OUT',Path(directory)):
            case={'case_id':'test:1'}
            first=s.claim(case,0)
            self.assertIsNotNone(first)
            with patch.object(s,'pid_live',return_value=True):
                self.assertIsNone(s.claim(case,1))
            first.unlink()
            self.assertIsNotNone(s.claim(case,1))


    def primary_args(self,**overrides):
        values=dict(worker=0,wait_for_generation=False,pause_generation=True,
                    primary_only=True,pilot=False,retry_failed=False,memory_fraction=.4)
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_primary_worker_exits_without_waiting_for_controls(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(s,'OUT',Path(directory)), \
             patch.object(s,'manifest',return_value={}), \
             patch.object(s.official,'all_cases',return_value=[{'case_id':'case:1'}]), \
             patch.object(s,'unit_finished',return_value=True), \
             patch.object(s,'generation_running',side_effect=AssertionError('unnecessary wait')), \
             patch.object(s,'aggregate_primary') as aggregate, \
             patch.object(s,'pid_live',return_value=True), \
             patch.object(s.time,'sleep',side_effect=AssertionError('idle GPU waited for other claim')):
            s.write(s.OUT/'claims'/'other.json',{'pid':999})
            s.worker(self.primary_args())
            aggregate.assert_called_once()

    def test_pause_skips_an_already_stopped_worker(self):
        paused={}
        with patch.object(s,'read_json',return_value={'workers':{'0':101,'1':102}}), \
             patch.object(s.official,'live',return_value=True), \
             patch.object(s,'process_state',side_effect=['T','S']),patch.object(s.os,'kill') as kill:
            s.pause_generation_workers(paused)
            self.assertEqual(paused,{1:102})
            kill.assert_called_once_with(102,signal.SIGSTOP)
            s.resume_generation_workers(paused)
            self.assertEqual(paused,{})
            self.assertEqual(kill.call_args.args,(102,signal.SIGCONT))

    def test_launch_resumes_each_gpu_as_its_scorer_exits(self):
        class Process:
            def __init__(self,pid,polls):
                self.pid=pid
                self.polls=iter(polls)
            def poll(self):
                return next(self.polls,0)
        with tempfile.TemporaryDirectory() as directory,patch.object(s,'OUT',Path(directory)), \
             patch.object(s,'manifest',return_value={}), \
             patch.object(s,'read_json',return_value={'status':'complete'}), \
             patch.object(s,'pause_generation_workers',side_effect=lambda paused:paused.update({0:101,1:102})), \
             patch.object(s.official,'live',return_value=True), \
             patch.object(s.os,'kill') as kill,patch.object(s.signal,'signal'), \
             patch.object(s.time,'sleep'),patch.object(s,'aggregate_primary'), \
             patch.object(s.subprocess,'Popen',side_effect=[Process(201,[0]),Process(202,[None,0])]):
            s.launch(self.primary_args())
            self.assertEqual([call.args for call in kill.call_args_list],
                             [(101,signal.SIGCONT),(102,signal.SIGCONT)])

    def test_launch_failure_restores_paused_controls(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(s,'OUT',Path(directory)), \
             patch.object(s,'manifest',return_value={}), \
             patch.object(s,'read_json',return_value={'status':'complete'}), \
             patch.object(s,'pause_generation_workers',side_effect=lambda paused:paused.update({0:101,1:102})), \
             patch.object(s.official,'live',return_value=True), \
             patch.object(s.os,'kill') as kill,patch.object(s.signal,'signal'), \
             patch.object(s.subprocess,'Popen',side_effect=RuntimeError('startup failed')):
            with self.assertRaisesRegex(RuntimeError,'startup failed'):
                s.launch(self.primary_args())
            self.assertEqual([call.args for call in kill.call_args_list],
                             [(101,signal.SIGCONT),(102,signal.SIGCONT)])

    def test_primary_table_only_updates_requested_metrics(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(s,'OUT',Path(directory)):
            path=Path(directory)/'paper.md'
            original=('<!-- ASTER_FINAL_MAIN_START -->\n\n'
                '| 方法 | PathAcc↑ | DTW-PCC↓ | Acc↑ |\n'
                '|---|---:|---:|---:|\n'
                '| **处置盲内部干预（本文）** | **89.21** | — | — |\n\n'
                '<!-- ASTER_FINAL_MAIN_END -->\n')
            path.write_text(original)
            summary={'status':'complete','registered':2,'all_registered':{'DTW_PCC':.1,'Acc':.8}}
            s.update_primary_manuscript(summary,path,expected_cases=2)
            self.assertIn('| **处置盲内部干预（本文）** | **89.21** | **10.00** | **80.00** |',path.read_text())


if __name__=='__main__':
    unittest.main()
