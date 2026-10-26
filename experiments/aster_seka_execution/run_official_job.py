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


def input_fingerprints(args, source, output, model, state):
    """Hash declared input paths and projection paths referenced by expert JSON."""
    output_flags = {'output', 'output_dir', 'save', 'save_dir', 'save_path',
                    'results_dir', 'result_dir', 'log', 'log_dir', 'log_file'}
    excluded = [output]
    values = []
    flag = None
    for item in args:
        if item.startswith('--'):
            key, equal, value = item[2:].partition('=')
            flag = key.replace('-', '_')
            if equal:
                values.append((flag, value))
        else:
            values.append((flag, item))
    for flag, value in values:
        if flag in output_flags:
            excluded.append((source / value).resolve())
    visited = set()
    def include(path, origin, parse_expert_references=False):
        path = path.resolve()
        if path == model or any(path == p or p in path.parents for p in excluded):
            return
        if path in visited or not path.exists():
            return
        visited.add(path)
        if path.is_dir():
            for child in sorted(path.rglob('*')):
                if child.is_file():
                    include(child, origin, parse_expert_references=parse_expert_references)
            return
        state['input_files_sha256'][str(path)] = sha(path)
        state['input_file_origins'][str(path)] = origin
        # Dataset files can be JSONL even when named .json. Only explicitly
        # designated expert configuration JSON is parsed for indirect inputs.
        if not parse_expert_references or path.suffix.lower() != '.json':
            return
        value = json.loads(path.read_text())
        def references(item):
            if isinstance(item, dict):
                for child in item.values():
                    references(child)
            elif isinstance(item, list):
                for child in item:
                    references(child)
            elif isinstance(item, str) and Path(item).suffix.lower() in {
                    '.pt', '.pth', '.bin', '.safetensors', '.json', '.jsonl', '.npy', '.npz'}:
                # The official adaptive loader resolves expert paths from cwd.
                reference = (source / item).resolve()
                if reference.exists():
                    include(reference, 'expert JSON reference in '+str(path), parse_expert_references=reference.suffix.lower()=='.json')
                else:
                    state['unresolved_input_references'].append(
                        {'json': str(path), 'value': item, 'resolved_from_source': str(reference)})
        references(value)
    for flag, value in values:
        if flag in output_flags or flag in {'model', 'model_path', 'model_name_or_path'}:
            continue
        path = (source / value).resolve()
        if path.is_dir() and flag in {'adapter_dir','reader_dir','baseline_dir','adapter_root','seka_root','official_root'}:
            # Source directories can be this experiment root, which also owns
            # runtime outputs, models and datasets. Fingerprint code/templates
            # without treating those unrelated trees as method inputs.
            for child in sorted(path.rglob('*')):
                relative = child.relative_to(path)
                if any(part in {'runtime','.git','__pycache__','.venv','venv'} for part in relative.parts):
                    continue
                if child.is_symlink():
                    resolved = child.resolve()
                    if not resolved.is_relative_to(path):
                        raise ValueError('Source/template symlink escapes declared source directory: '+str(child))
                    if any(part in {'runtime','.git','__pycache__','.venv','venv'} for part in resolved.relative_to(path).parts):
                        raise ValueError('Source/template symlink targets an excluded directory: '+str(child))
                if child.is_file() and child.suffix in {'.py','.txt','.jinja','.jinja2'}:
                    include(child, '--'+flag+' source/template')
        elif flag == 'dataset_root':
            # Development methods read vignette inputs and original guidelines,
            # never train_val reference paths/answers or the test split.
            # A missing/incomplete snapshot must not fall through to recursive
            # hashing of the entire dataset, including withheld references.
            if not (path/'train_val'/'vignette_cases.jsonl').is_file() or not (path/'train_val'/'guideline_md').is_dir():
                raise FileNotFoundError('MEGA dataset root requires train_val/vignette_cases.jsonl and train_val/guideline_md: '+str(path))
            include(path/'train_val'/'vignette_cases.jsonl','--dataset-root train_val method inputs')
            include(path/'train_val'/'guideline_md','--dataset-root train_val guidelines')
        elif path.is_dir() and flag in {'data','data_path','data_dir'}:
            if (path/'counterfact.jsonl').is_file():
                include(path/'counterfact.jsonl','--'+flag+' (CounterFact input only)')
            elif (path/'biasbios.json').is_file():
                include(path/'biasbios.json','--'+flag+' (BiasBios input only)')
            elif (path/'biasbios.jsonl').is_file():
                include(path/'biasbios.jsonl','--'+flag+' (BiasBios input only)')
            else:
                # Hash the declared dataset directory without interpreting its
                # contents as configuration or recursively following references.
                include(path,'--'+flag,parse_expert_references=False)
        elif path.exists():
            include(path, '--'+str(flag),parse_expert_references=flag in {
                'adaptive_expert_path','expert_path','expert_config','experts_config'})


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
    a.source, a.model, a.output = (p.expanduser().resolve() for p in (a.source, a.model, a.output))
    a.output.mkdir(parents=True, exist_ok=True)
    path = a.output / 'run_manifest.json'
    if path.exists():
        raise RuntimeError('Existing run manifest: choose a new output directory; never overwrite a run')
    started = time.monotonic()
    state = {
        'process_state': 'running', 'stage': 'preflight', 'verification_status': 'pending',
        'started_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'pid': os.getpid(), 'gpu_physical': a.gpu,
        'command': list(sys.argv), 'module': a.module, 'module_args': a.module_args,
        'python': sys.version, 'runner_sha256': sha(Path(__file__).resolve()),
        'requested_seed': a.seed,
        'python_hash_seed': os.environ.get('PYTHONHASHSEED'),
        'python_hash_seed_scope': 'Recorded startup environment only; an unset value means Python randomizes string hashing. Setting it after process start cannot control set iteration in this process.',
        'seed_domains': {'python_numpy_torch_requested': a.seed,
                         'python_hash_seed_startup': os.environ.get('PYTHONHASHSEED'),
                         'data_order': 'Upstream subset/order; separate from generation and projection seeds. CounterFact paraphrase uses set deduplication.',
                         'upstream_synthetic_builder_python_seed': 42},
        'seed_caveat': 'Upstream modules may set their own seed; official synthetic builder sets Python random.seed(42). No sample shuffle is used there.',
        'model_path': str(a.model), 'model_files_sha256': {},
        'source_files_sha256': {}, 'input_files_sha256': {},
        'input_file_origins': {}, 'unresolved_input_references': [],
        'forward_attempts': 0, 'forward_completed': 0,
        'completed_calls_with_input_shape': 0, 'completed_calls_without_input_shape': 0,
        'completed_input_batch_rows': 0, 'completed_input_positions': 0,
        'forward_count_scope': 'Qwen3ForCausalLM calls only; attempts include failures; completed means forward returned. Batch rows and input positions include repeated generation steps and padding, not unique examples or generated tokens.',
        'forward_input_trace': str(a.output/'forward_inputs.jsonl'),
        'forward_input_trace_rows': 0, 'forward_input_hashing_seconds': 0.0,
        'forward_input_hashing_scope': 'Post-forward wall time for device-to-CPU copy, SHA256 and JSONL flush; includes synchronization wait and is included in total elapsed time. No token IDs or text are stored.',
        'first_forward_parameter_dtype': None,
        'elapsed_scope': 'Process wall time from preflight, including hashing/imports/loading; separate GPU lock wall time is not GPU-active time.',
    }
    # Exclusive creation prevents concurrent invocations from replacing a run.
    with path.open('x') as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    torch = None
    lock = None
    lock_started = None
    hooks = []
    trace_handle = None
    cuda_ready = False
    code = 0
    try:
        lockroot = Path('/home/gcx/.aster_gpu_locks')
        lockroot.mkdir(exist_ok=True)
        lock = (lockroot / f'gpu{a.gpu}.lock').open('a+')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_started = time.monotonic()
        snapshot = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-gpu=uuid,name,memory.used,memory.total', '--format=csv,noheader,nounits'], text=True).strip()
        state['gpu_snapshot_before'] = snapshot
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
        state['stage'] = 'environment_import'
        save(path, state)
        import torch
        import transformers
        import datasets
        import tokenizers
        import numpy as np
        import random
        from transformers import Qwen3ForCausalLM
        random.seed(a.seed)
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
        state.update(torch=torch.__version__, transformers=transformers.__version__,
                     datasets=datasets.__version__, tokenizers=tokenizers.__version__,
                     source_commit=subprocess.check_output(['git', '-C', str(a.source), 'rev-parse', 'HEAD'], text=True).strip(),
                     source_dirty=subprocess.check_output(['git', '-C', str(a.source), 'status', '--porcelain'], text=True).strip(),
                     stage='fingerprints')
        save(path, state)
        for p in sorted(a.model.iterdir()):
            if p.is_file() and (p.suffix in ['.safetensors', '.bin', '.json', '.txt'] or p.name == 'tokenizer.model'):
                state['model_files_sha256'][p.name] = sha(p)
        for base in ['src', 'pastalib', 'benchmarks']:
            for p in sorted((a.source / base).rglob('*.py')):
                state['source_files_sha256'][str(p.relative_to(a.source))] = sha(p)
        input_fingerprints(a.module_args, a.source, a.output, a.model, state)
        save(path, state)
        trace_path = a.output/'forward_inputs.jsonl'
        trace_handle = trace_path.open('x')
        def tensor_fingerprint(tensor):
            if tensor is None:
                return None
            if not isinstance(tensor,torch.Tensor):
                return {'available':False,'reason':'input is not a torch.Tensor'}
            value = tensor.detach().contiguous().cpu()
            header = {'dtype':str(value.dtype),'shape':list(value.shape)}
            digest = hashlib.sha256()
            digest.update(json.dumps(header,sort_keys=True,separators=(',',':')).encode('utf-8'))
            digest.update(b'\x00')
            digest.update(value.view(torch.uint8).numpy().tobytes(order='C'))
            return {**header,'sha256':digest.hexdigest(),
                    'encoding':'canonical dtype/shape JSON + NUL + contiguous raw bytes'}
        def count_attempt(module, inputs):
            if isinstance(module, Qwen3ForCausalLM):
                state['forward_attempts'] += 1
                if state['first_forward_parameter_dtype'] is None:
                    state['first_forward_parameter_dtype'] = str(next(module.parameters()).dtype)
                    state['observed_model_config_name_or_path'] = str(getattr(module.config, '_name_or_path', ''))
        def count_completed(module, inputs, kwargs, result):
            if not isinstance(module, Qwen3ForCausalLM):
                return
            state['forward_completed'] += 1
            tensor = kwargs.get('input_ids')
            if tensor is None and inputs:
                tensor = inputs[0]
            if tensor is None:
                tensor = kwargs.get('inputs_embeds')
            if isinstance(tensor, torch.Tensor) and tensor.ndim >= 2:
                state['completed_calls_with_input_shape'] += 1
                state['completed_input_batch_rows'] += int(tensor.shape[0])
                state['completed_input_positions'] += int(tensor.shape[0] * tensor.shape[1])
            else:
                state['completed_calls_without_input_shape'] += 1
            hash_started = time.monotonic()
            try:
                input_ids = kwargs.get('input_ids')
                if input_ids is None and inputs:
                    input_ids = inputs[0]
                attention_mask = kwargs.get('attention_mask')
                # Qwen3 forward's second positional parameter is attention_mask.
                if attention_mask is None and len(inputs) > 1:
                    attention_mask = inputs[1]
                trace = {'completed_call_index':state['forward_completed'],
                         'attempt_index':state['forward_attempts'],
                         'input_ids':tensor_fingerprint(input_ids),
                         'attention_mask':tensor_fingerprint(attention_mask)}
                trace_handle.write(json.dumps(trace,sort_keys=True)+'\n')
                trace_handle.flush()
                state['forward_input_trace_rows'] += 1
            finally:
                state['forward_input_hashing_seconds'] += time.monotonic()-hash_started
        # PyTorch 2.8: global pre-hook has no kwargs; post-hook supports them.
        # Hooks return None and do not alter inputs, outputs, or model forwards.
        hooks.append(torch.nn.modules.module.register_module_forward_pre_hook(count_attempt))
        hooks.append(torch.nn.modules.module.register_module_forward_hook(count_completed, with_kwargs=True))
        torch.cuda.reset_peak_memory_stats()
        cuda_ready = True
        state['stage'] = 'upstream_module'
        save(path, state)
        sys.argv = [a.module] + a.module_args
        runpy.run_module(a.module, run_name='__main__')
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code in [None, 0]:
            pass
        else:
            code = (exc.code % 256) if isinstance(exc, SystemExit) and isinstance(exc.code, int) else 1
            if code == 0:
                code = 1
            state['error'] = type(exc).__name__ + ': ' + str(exc)
            state['failed_stage'] = state['stage']
            traceback.print_exc()
    finally:
        for hook in hooks:
            hook.remove()
        if trace_handle is not None:
            trace_handle.close()
            state['forward_input_trace_sha256'] = sha(a.output/'forward_inputs.jsonl')
        if cuda_ready:
            try:
                torch.cuda.synchronize()
                state.update(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                             peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            except BaseException as exc:
                state['cuda_finalization_error'] = type(exc).__name__ + ': ' + str(exc)
                code = code or 1
        else:
            state.update(peak_allocated_gib=None, peak_reserved_gib=None)
        state.update(process_state='exited', exit_code=code,
                     finished_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                     elapsed_seconds=time.monotonic()-started,
                     gpu_lock_wall_seconds=time.monotonic()-lock_started if lock_started is not None else 0,
                     forward_attempts_without_completed=state['forward_attempts']-state['forward_completed'])
        save(path, state)
        if lock is not None:
            lock.close()
    return code


if __name__ == '__main__':
    sys.exit(main())
