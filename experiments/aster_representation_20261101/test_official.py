import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import official


class OfficialRunnerTests(unittest.TestCase):
    def test_resume_retains_success_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'records.jsonl'
            path.write_text('\n'.join(json.dumps(r) for r in [
                {'case_id':'a','kind':'full','status':'complete'},
                {'case_id':'a','kind':'context','status':'failed'}])+'\n')
            self.assertEqual(set(official.saved_units(path)), {('a','full'),('a','context')})

    def test_live_claim_is_not_duplicated_and_finished_tree_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(official,'OUT',Path(tmp)), patch.object(official,'live',return_value=True):
            jobs=[{'guideline_id':1},{'guideline_id':2},{'guideline_id':3}]
            official.write(Path(tmp)/'done/1.json',{'status':'complete'})
            self.assertEqual(official.claim_next(jobs,0)['guideline_id'],2)
            self.assertEqual(official.claim_next(jobs,1)['guideline_id'],3)
            self.assertIsNone(official.claim_next(jobs,0))

    def test_primary_path_metric_and_failure_denominator(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(official,'OUT',Path(tmp)):
            root=Path(tmp)
            cases=[{'case_id':'a','source_id':'s1','language':'en'},
                   {'case_id':'b','source_id':'s2','language':'zh'}]
            official.write(root/'manifest.json',{'expected_units':14})
            (root/'results').mkdir()
            records=[]
            for kind in official.METHODS:
                for case in cases:
                    success=not(kind=='context' and case['case_id']=='a')
                    records.append({**case,'kind':kind,'status':'complete' if success else 'failed',
                        'score':{'exact_path':case['case_id']=='a' and success,
                                 'leaf_correct':success,'invalid':False}})
            (root/'results/1.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            with patch.object(official,'all_cases',return_value=cases):
                result=official.aggregate()
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['methods']['full']['exact_path'],1)
            self.assertEqual(result['methods']['full']['leaf_correct'],2)
            self.assertEqual(result['methods']['context']['registered'],2)
            self.assertEqual(result['methods']['context']['technical_failures'],1)
            self.assertEqual(result['primary_vs_controls']['context']['corrected'],1)
            self.assertEqual(result['primary_vs_controls']['context']['paired_delta'],.5)

    def test_primary_is_claimed_first_and_controls_cannot_share_an_active_tree(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(official,'OUT',Path(tmp)), patch.object(official,'live',return_value=True):
            root=Path(tmp)
            manifest={'primary_method':'no_js','prioritize_primary':True,
                      'jobs':[{'guideline_id':1},{'guideline_id':2}]}
            a=official.claim_work(manifest,0)
            b=official.claim_work(manifest,1)
            self.assertEqual(a['methods'],['no_js'])
            self.assertEqual(b['phase'],'primary')
            self.assertNotEqual(a['guideline_id'],b['guideline_id'])
            self.assertIsNone(official.claim_work(manifest,0))
            official.write(root/'done/primary/1.json',{'status':'complete'})
            c=official.claim_work(manifest,0)
            self.assertEqual(c['guideline_id'],1)
            self.assertEqual(c['phase'],'controls')
            self.assertNotIn('no_js',c['methods'])


if __name__=='__main__':unittest.main()
