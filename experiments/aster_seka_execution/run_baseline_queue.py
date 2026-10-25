"""Wait for the coordinator GPU lock, then run each independent baseline job.

Waiting uses the existing OS file lock, without polling model progress.
An unsuccessful baseline retains its result and does not cancel the next one.
Each actual model process still uses run_official_job's own exclusive lock.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    path = Path(sys.argv[1])
    queue = json.loads(path.read_text())
    gpu = int(queue['gpu'])
    if gpu not in (0, 1, 6):
        raise ValueError('GPU outside the user-authorized set')
    lock_path = Path('/home/gcx/.aster_gpu_locks') / f'gpu{gpu}.lock'
    results = []
    for job in queue['jobs']:
        command = job['command']
        if command[command.index('--gpu') + 1] != str(gpu):
            raise ValueError('Job GPU differs from the coordinator queue')
        with lock_path.open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
        # Let the previous process release its CUDA allocation after unlocking.
        time.sleep(1)
        started = time.time()
        with Path(job['log']).open('x') as log:
            run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, PYTHONHASHSEED='0'))
        results.append({'name': job['name'], 'exit_code': run.returncode,
                        'started_at': started, 'completed_at': time.time()})
        path.with_name(path.stem + '_results.json').write_text(json.dumps(results, indent=2) + '\n')
    return int(any(row['exit_code'] for row in results))


if __name__ == '__main__':
    raise SystemExit(main())
