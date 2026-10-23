import unittest,tempfile,json,os
from pathlib import Path
from unittest.mock import patch
from delegate_pending import wait_if_delegated
class DelegationTest(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.r=Path(self.t.name);self.out=self.r/'out.jsonl';self.plan=self.r/'plan.json';self.task={'output':str(self.out),'input':'/input','model':'/model','owner':'worker0','rows':10,'input_sha256':'digest','kind':'main','owner_status':str(self.r/'status')};self.plan.write_text(json.dumps({'tasks':[self.task]}));self.args=['--output',str(self.out),'--input','/input','--model','/model']
 def tearDown(self):self.t.cleanup()
 def test_owner_only_starts_missing_output(self):
  with patch.dict(os.environ,{'TRIAGE_JOB_OWNER':'worker0'}):self.assertIsNone(wait_if_delegated(self.plan,self.args))
 def test_complete_is_reused(self):
  self.out.with_suffix('.manifest.json').write_text(json.dumps({'complete':True,'rows':10,'input_sha256':'digest'}))
  with patch.dict(os.environ,{'TRIAGE_JOB_OWNER':'consumer'}),self.assertRaises(SystemExit):wait_if_delegated(self.plan,self.args)
 def test_failed_owner_does_not_trigger_duplicate(self):
  (self.r/'status').write_text('failed test')
  with patch.dict(os.environ,{'TRIAGE_JOB_OWNER':'consumer'}),self.assertRaises(RuntimeError):wait_if_delegated(self.plan,self.args)
 def test_partial_main_is_preserved(self):
  self.out.write_text('partial')
  with patch.dict(os.environ,{'TRIAGE_JOB_OWNER':'worker0'}),self.assertRaises(RuntimeError):wait_if_delegated(self.plan,self.args)
 def test_changed_input_rejected(self):
  with self.assertRaises(AssertionError):wait_if_delegated(self.plan,self.args[:-1]+['/another_model'])
 def test_unclaimed_task_unaffected(self):self.assertIsNone(wait_if_delegated(self.plan,['--output','/unclaimed']))
if __name__=='__main__':unittest.main()
