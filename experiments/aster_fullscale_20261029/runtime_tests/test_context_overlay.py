"""A replay must preserve raw records and only replace original context failures."""
import unittest,tempfile,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from context_replay import materialize
class ContextOverlayTests(unittest.TestCase):
 def test_overlay_retains_source_and_rejects_replacing_success(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);base=root/'serial/confirm/worker_0';base.mkdir(parents=True);replay=root/'context_replay/confirm/worker_job';replay.mkdir(parents=True)
   original=[{'case_id':'long','method':'baseline','technical_failure':True,'failure_reason':'context_overflow'},{'case_id':'short','method':'baseline','technical_failure':False}]
   path=base/'mega_confirm.jsonl';raw=''.join(json.dumps(r)+'\n' for r in original);path.write_text(raw)
   recovered={'case_id':'long','method':'baseline','technical_failure':False,'supersedes_original_context_failure':True}
   dest=replay/'mega_confirm.jsonl';dest.write_text(json.dumps(recovered)+'\n');materialize(root,'confirm')
   output=[json.loads(line) for line in (root/'analysis/serial/confirm/worker_0/mega_confirm.jsonl').read_text().splitlines()]
   self.assertEqual(path.read_text(),raw);self.assertEqual(len(output),2);self.assertFalse(output[0]['technical_failure']);self.assertEqual(output[1],original[1])
   dest.write_text(json.dumps(original[1])+'\n')
   with self.assertRaises(ValueError):materialize(root,'confirm')
if __name__=='__main__':unittest.main()
