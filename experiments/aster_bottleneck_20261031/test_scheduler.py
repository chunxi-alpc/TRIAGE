import concurrent.futures,json,tempfile,threading,unittest
from pathlib import Path
from unittest.mock import patch
from . import scheduler
from . import advanced
class SchedulerTests(unittest.TestCase):
 def test_free_gpu_runs_next_job_while_other_gpu_is_busy(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);release=threading.Event();began=threading.Event();locations={}
   class FakeProcess:
    def __init__(self,args,env,**kw):
     self.stage=args[args.index('--worker')+1];self.shard=args[args.index('--shard')+1];self.pid=12345
     locations[self.stage]=int(env['CUDA_VISIBLE_DEVICES'])
    def wait(self):
     if self.stage=='slow':began.set();release.wait(5)
     (root/f'{self.stage}_{self.shard}_status.json').write_text('{"status":"complete"}')
     return 0
   with patch.object(scheduler,'OUT',root),patch.object(scheduler,'M1',root),patch.object(scheduler.subprocess,'Popen',FakeProcess):
    pool=scheduler.GPUPool(gpus=[0,6])
    slow=pool.submit('advanced','slow',0,10);fast=pool.submit('advanced','fast',0,10)
    following=pool.submit('advanced','following',0,10)
    self.assertIs(pool.submit('advanced','slow',0,10),slow)
    pool.start()
    try:
     self.assertTrue(began.wait(3))
     following.result(timeout=3)
     self.assertFalse(slow.done())
     self.assertEqual(locations['fast'],locations['following'])
     self.assertNotEqual(locations['slow'],locations['following'])
    finally:release.set();pool.close()
    self.assertEqual(pool.phase,'complete')
 def test_error_parts_are_aggregated_without_duplicate_cases(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);row={'case_id':'a','arm':'bias','status':'complete'}
   (root/'errors_dev_part000_0.jsonl').write_text(json.dumps(row)+'\n')
   (root/'errors_dev_part001_0.jsonl').write_text(json.dumps({**row,'case_id':'b'})+'\n')
   (root/'errors_dev_0.jsonl').write_text(json.dumps(row)+'\n')
   with patch.object(advanced,'OUT',root):self.assertEqual(len(advanced.records('errors_dev')),2)
if __name__=='__main__':unittest.main()
