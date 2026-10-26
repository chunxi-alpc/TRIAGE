"""Server-only synthetic queue tests: no model, real GPU query, or child process.

Set ASTER_DRIVER_DIR to the directory containing both coordinator scripts.
"""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(os.environ.get('ASTER_DRIVER_DIR', Path(__file__).resolve().parents[1]))


def module(filename):
    spec = importlib.util.spec_from_file_location('test_'+filename, ROOT/(filename+'.py'))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


STAGE = module('run_stage_transition')
QUEUE = module('run_baseline_queue')


class QueueDrivers(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='aster_queue_fixture_')
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self,name,value):
        path=self.root/name
        path.write_text(json.dumps(value)+'\n')
        return path

    def stage_plan(self):
        return dict(wait_for_pids=[],required_run_manifests=[],verification_steps=[],cwd=str(self.root),
                    clinical_log=str(self.root/'clinical.log'),clinical_command=['synthetic-clinical-command'])

    def test_existing_state_atomically_prevents_both_drivers_starting(self):
        for driver,plan,name in [(STAGE,self.stage_plan(),'transition'),
                                 (QUEUE,dict(gpu=0,wait_for_pids=[],jobs=[]),'queue')]:
            with self.subTest(driver=name):
                path=self.write(name+'.json',plan)
                state=self.write(name+'_state.json',{'status':'running','pid':12345})
                old=state.read_bytes()
                with patch.object(driver.sys,'argv',[name,str(path)]), \
                     patch.object(driver.subprocess,'run') as run, \
                     patch.object(driver.subprocess,'Popen') as popen, \
                     patch.object(driver.os,'pidfd_open') as pidfd, \
                     patch.object(Path,'exists',return_value=False):
                    # Even a stale/misleading exists() check cannot bypass x mode.
                    with self.assertRaises(FileExistsError):
                        driver.main()
                    run.assert_not_called()
                    popen.assert_not_called()
                    pidfd.assert_not_called()
                self.assertEqual(state.read_bytes(),old)

    def test_queue_opens_all_pidfds_and_retains_failed_job_before_next_job(self):
        path=self.write('queue.json',dict(gpu=0,wait_for_pids=[111,222],jobs=[
            dict(name='first',command=['synthetic-job','--gpu','0'],log=str(self.root/'first.log')),
            dict(name='second',command=['synthetic-job','--gpu','0'],log=str(self.root/'second.log'))]))
        state_path=self.root/'queue_state.json'
        opened=[]
        def open_pid(pid):
            opened.append(pid)
            return 1000+pid
        testcase=self
        class Poll:
            def register(self,*values):
                pass
            def poll(self):
                testcase.assertEqual(opened,[111,222])
                return [(1,1)]
        launched=[]
        def run(*args,**kwargs):
            index=len(launched)
            state=json.loads(state_path.read_text())
            self.assertEqual(state['jobs'][index]['status'],'running')
            if index:
                self.assertEqual(state['jobs'][0]['status'],'failed')
            launched.append(index)
            return SimpleNamespace(returncode=1 if index==0 else 0)
        with patch.object(QUEUE.sys,'argv',['queue',str(path)]), \
             patch.object(QUEUE,'LOCK_ROOT',self.root/'synthetic_locks'), \
             patch.object(QUEUE.os,'pidfd_open',side_effect=open_pid), \
             patch.object(QUEUE.os,'close') as close, \
             patch.object(QUEUE.select,'poll',side_effect=Poll), \
             patch.object(QUEUE.fcntl,'flock'), \
             patch.object(QUEUE.subprocess,'check_output',return_value='0\n'), \
             patch.object(QUEUE.subprocess,'run',side_effect=run):
            self.assertEqual(QUEUE.main(),1)
        state=json.loads(state_path.read_text())
        self.assertEqual(state['status'],'failed')
        self.assertEqual([job['status'] for job in state['jobs']],['failed','finished_pending_independent_verification'])
        results=json.loads((self.root/'queue_results.json').read_text())
        self.assertEqual([row['exit_code'] for row in results],[1,0])
        self.assertEqual(close.call_count,2)

    def test_failed_verifier_records_failure_and_never_launches_clinical(self):
        plan=self.stage_plan()
        prerequisite=self.write('prerequisite.json',dict(process_state='exited',exit_code=0))
        plan['required_run_manifests']=[str(prerequisite)]
        plan['verification_steps']=[dict(name='synthetic-audit',report=str(self.root/'audit.json'),
                                         command=['synthetic-verifier'],log=str(self.root/'audit.log'))]
        path=self.write('transition.json',plan)
        state_path=self.root/'transition_state.json'
        def run(*args,**kwargs):
            self.assertEqual(json.loads(state_path.read_text())['steps'][0]['status'],'running')
            return SimpleNamespace(returncode=1)
        with patch.object(STAGE.sys,'argv',['transition',str(path)]), \
             patch.object(STAGE.subprocess,'run',side_effect=run), \
             patch.object(STAGE.subprocess,'Popen') as popen:
            with self.assertRaisesRegex(RuntimeError,'Verification failed'):
                STAGE.main()
            popen.assert_not_called()
        state=json.loads(state_path.read_text())
        self.assertEqual(state['status'],'failed')
        self.assertEqual(state['steps'][0]['status'],'failed')
        self.assertEqual(state['steps'][0]['exit_code'],1)


if __name__=='__main__':
    unittest.main()
