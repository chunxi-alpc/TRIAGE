"""Unified scheduler: one coordinated process per allowed GPU, then packaging."""
import fcntl
import subprocess
import time
import sys
from common import *

GPUS = [0, 1, 5, 6]

def gpu_busy(gpu):
    r = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True, check=True)
    return bool(r.stdout.strip())

def main():
    lock = (OUT / '.queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = OUT / 'queue_manifest.json'
    status_path = OUT / 'queue_status.json'
    status = read(status_path, {'jobs': {}})
    status.update(state='running', pid=os.getpid(), allowed_gpus=GPUS, started=time.time())
    for item in status['jobs'].values():
        if item.get('state') == 'running':
            raise RuntimeError('Existing running state requires process reconciliation before restart')
    live = {}
    write(status_path, status)
    while True:
        manifest = read(manifest_path)
        for gpu, (process, job, log) in list(live.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            status['jobs'][job['id']].update(state='complete' if code == 0 else 'failed',
                                          exit_code=code, finished=time.time())
            del live[gpu]
        for job in manifest['jobs']:
            if status['jobs'].get(job['id'], {}).get('state') in ['running', 'complete', 'failed']:
                continue
            if any(status['jobs'].get(dep, {}).get('state') != 'complete' for dep in job.get('dependencies', [])):
                continue
            if job.get('ready_file') and not Path(job['ready_file']).exists():
                continue
            candidates = [job['preferred_gpu']] if 'preferred_gpu' in job else GPUS
            free = next((gpu for gpu in candidates if gpu not in live and not gpu_busy(gpu)), None)
            if free is None:
                continue
            logpath = OUT / 'logs' / (job['id'] + '.log')
            logpath.parent.mkdir(exist_ok=True)
            log = logpath.open('a')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(free), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                       MKL_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1',
                       ASTER_CR_ROOT=str(HIST))
            process = subprocess.Popen([sys.executable, str(CODE / job['script']), *job.get('args', [])],
                                       cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
            live[free] = process, job, log
            status['jobs'][job['id']] = dict(state='running', pid=process.pid, gpu=free, started=time.time(),
                                           log=str(logpath), units=job['units'])
        status['updated'] = time.time()
        status['active'] = len(live)
        write(status_path, status)
        terminal = all(status['jobs'].get(j['id'], {}).get('state') in ['complete', 'failed'] for j in manifest['jobs'])
        if manifest.get('closed') and terminal and not live:
            break
        time.sleep(20)
    failed = [key for key, value in status['jobs'].items() if value['state'] == 'failed']
    status.update(state='failed' if failed else 'complete', finished=time.time(), failed_jobs=failed)
    write(status_path, status)
    if (CODE / 'finalize.py').exists():
        with (OUT / 'finalize.log').open('a') as f:
            code = subprocess.call([sys.executable, str(CODE / 'finalize.py')], cwd=PROJECT,
                                   env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='4'), stdout=f, stderr=subprocess.STDOUT)
        status['finalize_exit_code'] = code
        write(status_path, status)

if __name__ == '__main__':
    main()
