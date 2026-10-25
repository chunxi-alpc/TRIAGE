"""Wait for the coordinator GPU lock, then run each independent baseline job.

Waiting uses the existing OS file lock, without polling model progress.
An unsuccessful baseline retains its result and does not cancel the next one.
Each actual model process still uses run_official_job's own exclusive lock.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time


LOCK_ROOT = Path('/home/gcx/.aster_gpu_locks')


def main():
    path = Path(sys.argv[1]).resolve()
    queue = json.loads(path.read_text())
    gpu = int(queue['gpu'])
    if gpu not in (0, 1, 6):
        raise ValueError('GPU outside the user-authorized set')
    lock_path = LOCK_ROOT / f'gpu{gpu}.lock'
    state_path = path.with_name(path.stem + '_state.json')
    results_path = path.with_name(path.stem + '_results.json')
    state = {'status': 'waiting_for_predecessors', 'pid': os.getpid(), 'gpu': gpu,
             'plan_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
             'started_at': time.time(), 'wait_handles': [],
             'jobs': [{'name': job['name'], 'status': 'planned'} for job in queue['jobs']]}
    # Exclusive creation is the one-time claim for this queue plan. Existing
    # state is never overwritten by a restarted or duplicate driver process.
    with state_path.open('x') as handle:
        json.dump(state, handle, indent=2)
        handle.write('\n')

    def save():
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(state_path)

    results = []
    handles = []
    active_job = None
    try:
        # Open every handle before waiting, so another predecessor cannot exit
        # and have its PID reused during a long wait on the first process.
        for pid in queue.get('wait_for_pids', []):
            try:
                descriptor = os.pidfd_open(int(pid))
            except ProcessLookupError:
                state['wait_handles'].append({'pid': pid, 'state': 'already_exited'})
                continue
            handles.append((pid, descriptor))
            state['wait_handles'].append({'pid': pid, 'state': 'live_handle_opened'})
        save()
        for pid, descriptor in handles:
            event = select.poll()
            event.register(descriptor, select.POLLIN)
            event.poll()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        for index, job in enumerate(queue['jobs']):
            active_job = state['jobs'][index]
            active_job['status'] = 'waiting_for_gpu'
            state['status'] = 'waiting_for_gpu'
            save()
            command = job['command']
            if command[command.index('--gpu') + 1] != str(gpu):
                raise ValueError('Job GPU differs from the coordinator queue')
            with lock_path.open('a') as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
            # The driver may release the allocation shortly after process exit.
            # This resource handoff check never inspects model progress.
            deadline = time.monotonic() + 60
            while True:
                used = int(subprocess.check_output([
                    'nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used',
                    '--format=csv,noheader,nounits'], text=True).strip())
                if used <= 1024:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(f'GPU {gpu} still uses {used} MiB after the prior process exited')
                time.sleep(2)
            started = time.time()
            active_job.update(status='running', started_at=started)
            state['status'] = 'running'
            save()
            with Path(job['log']).open('x') as log:
                run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, PYTHONHASHSEED='0'))
            result = {'name': job['name'], 'exit_code': run.returncode,
                      'started_at': started, 'completed_at': time.time()}
            results.append(result)
            active_job.update(result, status='finished_pending_independent_verification' if run.returncode == 0 else 'failed')
            results_path.write_text(json.dumps(results, indent=2) + '\n')
            save()
            active_job = None
        code = int(any(row['exit_code'] for row in results))
        state.update(status='failed' if code else 'finished_pending_independent_verification',
                     exit_code=code, finished_at=time.time())
        save()
        return code
    except BaseException as error:
        description = type(error).__name__ + ': ' + str(error)
        if active_job is not None:
            active_job.update(status='failed', error=description)
        state.update(status='failed', error=description, finished_at=time.time())
        save()
        raise
    finally:
        for _, descriptor in handles:
            os.close(descriptor)


if __name__ == '__main__':
    raise SystemExit(main())
