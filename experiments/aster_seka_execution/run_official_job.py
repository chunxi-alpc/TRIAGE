"""Run an upstream module on one coordinator-owned GPU; research-server only."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, choices=[0, 1, 6], required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--module', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('module_args', nargs=argparse.REMAINDER)
    a = parser.parse_args()
    if a.module_args[:1] == ['--']:
        a.module_args = a.module_args[1:]
    a.output.mkdir(parents=True, exist_ok=True)
    path = a.output / 'run_manifest.json'
    if path.exists():
        raise RuntimeError('Existing run manifest: choose a new output directory; never overwrite a run')
    lockroot = Path('/home/gcx/.aster_gpu_locks')
    lockroot.mkdir(exist_ok=True)
    lock = (lockroot / f'gpu{a.gpu}.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    snapshot = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-gpu=uuid,name,memory.used,memory.total', '--format=csv,noheader,nounits'], text=True).strip()
    used = int(snapshot.split(',')[2].strip())
    if used > 1024:
        raise RuntimeError(f'GPU {a.gpu} already uses {used} MiB; no job started')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.chdir(a.source)
    sys.path.insert(0, str(a.source))
    import torch
    import transformers
    import datasets
    import tokenizers
    import numpy as np
    import random
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    state = {
        'process_state': 'running', 'verification_status': 'pending',
        'started_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'pid': os.getpid(), 'gpu_physical': a.gpu, 'gpu_snapshot_before': snapshot,
        'command': sys.argv, 'module': a.module, 'module_args': a.module_args,
        'source_commit': subprocess.check_output(['git', '-C', str(a.source), 'rev-parse', 'HEAD'], text=True).strip(),
        'source_dirty': subprocess.check_output(['git', '-C', str(a.source), 'status', '--porcelain'], text=True).strip(),
        'python': sys.version, 'torch': torch.__version__, 'transformers': transformers.__version__,
        'datasets': datasets.__version__, 'tokenizers': tokenizers.__version__,
        'requested_seed': a.seed,
        'seed_caveat': 'Upstream modules may set their own seed; official synthetic builder sets Python random.seed(42). No sample shuffle is used there.',
        'model_path': str(a.model), 'model_files_sha256': {},
        'source_files_sha256': {}, 'input_files_sha256': {},
        'forward_calls': 0, 'first_forward_parameter_dtype': None,
    }
    save(path, state)
    started = time.monotonic()
    hook = None
    code = 0
    try:
        for p in sorted(a.model.iterdir()):
            if p.is_file() and (p.suffix in ['.safetensors', '.json', '.txt'] or p.name == 'tokenizer.model'):
                state['model_files_sha256'][p.name] = sha(p)
        for base in ['src', 'pastalib', 'benchmarks']:
            for p in sorted((a.source / base).rglob('*.py')):
                state['source_files_sha256'][str(p.relative_to(a.source))] = sha(p)
        for item in a.module_args:
            p = Path(item)
            if p.is_file():
                state['input_files_sha256'][str(p.resolve())] = sha(p)
            elif p.is_dir() and '--data_path' in a.module_args:
                for name in ['counterfact.jsonl', 'biasbios.jsonl']:
                    if (p/name).is_file():
                        state['input_files_sha256'][str((p/name).resolve())] = sha(p/name)
        save(path, state)
        def count(module, inputs):
            if module.__class__.__name__ == 'Qwen3ForCausalLM':
                state['forward_calls'] += 1
                if state['first_forward_parameter_dtype'] is None:
                    state['first_forward_parameter_dtype'] = str(next(module.parameters()).dtype)
        hook = torch.nn.modules.module.register_module_forward_pre_hook(count)
        torch.cuda.reset_peak_memory_stats()
        sys.argv = [a.module] + a.module_args
        runpy.run_module(a.module, run_name='__main__')
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code in [None, 0]:
            pass
        else:
            code = exc.code if isinstance(exc, SystemExit) and isinstance(exc.code, int) else 1
            state['error'] = type(exc).__name__ + ': ' + str(exc)
            traceback.print_exc()
    finally:
        if hook:
            hook.remove()
        state.update(process_state='exited', exit_code=code,
                     finished_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                     elapsed_seconds=time.monotonic()-started,
                     peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                     peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        save(path, state)
    return code


if __name__ == '__main__':
    sys.exit(main())
