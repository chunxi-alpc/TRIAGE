"""Finite background generation -> evaluation -> CPU export -> branch push."""
import fcntl
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time


def main():
    path = Path(sys.argv[1]).resolve(); plan = json.loads(path.read_text())
    root = Path(plan['repository']); p = root / 'experiments/aster_seka_execution'
    r = p / 'runtime'; state_path = r / 'clinical_v3_pipeline_state.json'
    if plan['gpu'] not in [0, 1, 6]:
        raise ValueError('GPU outside user authorization')
    state = {'status': 'waiting_for_predecessor', 'pid': os.getpid(), 'gpu': plan['gpu'],
             'started_at': time.time(), 'completed_jobs': []}
    with state_path.open('x') as handle: json.dump(state, handle)
    def save():
        temporary = state_path.with_suffix('.tmp'); temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(state_path)
    try:
        handles = []
        for pid in plan['wait_for_pids']:
            try: handles.append(os.pidfd_open(pid))
            except ProcessLookupError: pass
        for handle in handles:
            event = select.poll(); event.register(handle, select.POLLIN); event.poll(); os.close(handle)
        env = dict(os.environ, PYTHONHASHSEED='0', PYTHONPATH=str(p))
        for job in plan['jobs']:
            state.update(status='waiting_for_gpu', active_job=job['name']); save()
            with Path(f"/home/gcx/.aster_gpu_locks/gpu{plan['gpu']}.lock").open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
            for attempt in range(30):
                used = int(subprocess.check_output(['nvidia-smi', '-i', str(plan['gpu']),
                           '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
                if used <= 1024: break
                time.sleep(2)
            else: raise RuntimeError('GPU not released after prior process')
            state.update(status='running', active_job=job['name']); save()
            with Path(job['log']).open('x') as log:
                subprocess.run(job['command'], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            state['completed_jobs'].append(job['name']); save()
        state.update(status='exporting', active_job='CPU verification and export'); save()
        with (r / 'clinical_v3_export.log').open('x') as log:
            subprocess.run([plan['python'], str(p / 'export_clinical_v3_results.py')], cwd=root,
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        state.update(status='publishing', active_job='result commit and push'); save()
        branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=root, text=True).strip()
        if branch != 'codex/aster-seka-execution-20261025':
            raise RuntimeError('Refusing result publication on unexpected branch')
        with (r / 'clinical_v3_publication.log').open('x') as log:
            def git(*args):
                subprocess.run(['git', *args], cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
            git('add', 'experiments/aster_seka_execution/clinical_v3_results_verified',
                'experiments/aster_seka_execution/CLINICAL_V3_RESULTS.zh.md',
                'experiments/aster_seka_execution/task_ledger.json')
            git('diff', '--cached', '--check')
            git('commit', '-m', 'Publish verified clinical v3 strength and probability fusion results')
            git('-c', 'core.sshCommand=ssh -p 443 -o Hostname=ssh.github.com -o ConnectTimeout=15 -o BatchMode=yes',
                'push', 'origin', branch)
        state.update(status='complete_verified', commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                     completed_at=time.time()); save()
    except BaseException as error:
        state.update(status='blocked' if state['status'] == 'publishing' else 'failed',
                     error_type=type(error).__name__, error=str(error), stopped_at=time.time()); save()
        raise


if __name__ == '__main__':
    main()
