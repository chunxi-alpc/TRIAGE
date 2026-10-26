"""Read-only CPU verification of official SEKA traditional projection artifacts.

Run on the research server. This verifies artifacts, not benchmark efficacy or
an exact paper reproduction. No original manifest or projection is rewritten.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback


PINNED_COMMIT = 'aab9cfed80dc012fb9e83f8bdefb8675c33f40e7'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def option(arguments, name, default=None):
    for i, value in enumerate(arguments):
        if value == name and i+1 < len(arguments):
            return arguments[i+1]
        if value.startswith(name+'='):
            return value.split('=',1)[1]
    return default


def layer_selection(spec, count):
    if spec == 'all':
        return list(range(count))
    if spec.startswith('last'):
        return list(range(count-int(spec[4:]),count))
    if spec.startswith('first'):
        return list(range(int(spec[5:])))
    return [int(x) for x in spec.split(',')]


def verify_manifest(args, report):
    manifest = json.loads(args.run_manifest.read_text())
    report['run_manifest_sha256'] = sha(args.run_manifest)
    report['run_manifest_path'] = str(args.run_manifest)
    require(manifest.get('process_state') == 'exited' and manifest.get('exit_code') == 0,
            'Builder must have exited successfully; incomplete or failed process is not a verified projection')
    require(manifest.get('module') == 'src.custom_builders.synthetic_qa_builder', 'Unexpected builder module')
    require(manifest.get('source_commit') == PINNED_COMMIT, 'Unexpected official source commit')
    require(Path(manifest['model_path']).resolve() == args.model, 'Checkpoint path differs from run manifest')
    arguments = manifest['module_args']
    upstream_model = option(arguments,'--model')
    require(upstream_model is not None and (args.source/upstream_model).resolve() == args.model,
            'Upstream --model does not match manifest checkpoint')
    observed = manifest.get('observed_model_config_name_or_path')
    if observed:
        require((args.source/observed).resolve() == args.model, 'Observed model config path differs')
    recorded = manifest.get('model_files_sha256',{})
    require('config.json' in recorded, 'Checkpoint config fingerprint missing')
    weight_names = [name for name in recorded if name.endswith(('.safetensors','.bin'))]
    require(weight_names, 'Checkpoint weight fingerprints missing')
    actual = {}
    for name, expected in recorded.items():
        actual[name] = sha(args.model/name)
        require(actual[name] == expected, 'Checkpoint fingerprint mismatch: '+name)
    index_path = args.model/'model.safetensors.index.json'
    if index_path.exists():
        shards = set(json.loads(index_path.read_text())['weight_map'].values())
        require(shards <= set(recorded), 'Model index contains weight shards absent from manifest')
    report['checkpoint_files_sha256'] = actual
    report['checkpoint_weight_shards_checked'] = weight_names
    current_commit = subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()
    require(current_commit == manifest['source_commit'], 'Official checkout HEAD changed since builder run')
    source_hashes = manifest.get('source_files_sha256',{})
    needed = {'src/model/projection_builder_base.py','src/custom_builders/synthetic_qa_builder.py','src/utils.py'}
    require(needed <= set(source_hashes), 'Core builder source fingerprints missing')
    for name, expected in source_hashes.items():
        require(sha(args.source/name) == expected, 'Builder source changed since run: '+name)
    report['source_commit'] = current_commit
    report['source_files_checked'] = len(source_hashes)
    report['source_dirty_at_run'] = manifest.get('source_dirty')
    inputs = manifest.get('input_files_sha256',{})
    data_arg = option(arguments,'--data')
    require(data_arg is not None, 'Synthetic calibration data argument missing')
    data_path = (args.source/data_arg).resolve()
    require(str(data_path) in inputs, 'Synthetic calibration data fingerprint missing')
    for name, expected in inputs.items():
        require(sha(Path(name)) == expected, 'Calibration/input fingerprint mismatch: '+name)
    report['input_files_sha256'] = inputs
    attempts = manifest.get('forward_attempts')
    completed = manifest.get('forward_completed')
    require(isinstance(attempts,int) and isinstance(completed,int), 'Distinct attempted/completed forward counters missing')
    require(attempts == completed and completed > 0, 'Builder has failed/incomplete or absent forward calls')
    max_samples = int(option(arguments,'--max_samples','1000'))
    with data_path.open() as handle:
        available_rows = sum(1 for _ in handle)
    effective_rows = min(max_samples,available_rows)
    require(completed <= effective_rows*6 and completed % 3 == 0, 'Synthetic builder forward count inconsistent with available rows and triplets')
    if args.expected_forwards is not None:
        require(completed == args.expected_forwards, 'Unexpected completed forward count')
    report['builder_cost'] = {key:manifest.get(key) for key in [
        'forward_attempts','forward_completed','elapsed_seconds','gpu_lock_wall_seconds',
        'peak_allocated_gib','peak_reserved_gib','first_forward_parameter_dtype',
        'completed_input_batch_rows','completed_input_positions','gpu_physical']}
    report['max_calibration_rows_requested'] = max_samples
    report['calibration_file_rows'] = available_rows
    report['effective_calibration_rows'] = effective_rows
    report['effective_row_count_basis'] = 'Official synthetic builder iterates min(requested max_samples, file line count); verified exit code 0'
    report['maximum_forwards_for_effective_rows'] = effective_rows*6
    report['completed_triplets'] = completed//3
    report['calibration_row_count_caveat'] = 'Up to two triplets per row; a skipped span can remove a triplet. Forward count does not identify which rows survived.'
    output_arg = option(arguments,'--output_dir')
    require(output_arg is not None, 'Builder output directory argument missing')
    output_dir = (args.source/output_arg).resolve()
    require(args.positive.parent == output_dir and args.negative.parent == output_dir,
            'Projection paths differ from builder output directory')
    config = json.loads((args.model/'config.json').read_text())
    if 'text_config' in config:
        config = config['text_config']
    n_layers = config['num_hidden_layers']
    n_kv = config['num_key_value_heads']
    head_dim = config.get('head_dim',config['hidden_size']//config['num_attention_heads'])
    layers = layer_selection(option(arguments,'--layers','all'),n_layers)
    require(layers and len(set(layers)) == len(layers) and all(0 <= x < n_layers for x in layers),
            'Invalid or duplicate selected layer ids')
    report['expected_layers'] = layers
    report['expected_shape'] = [len(layers),n_kv,head_dim,head_dim]
    report['builder_parameters'] = {name:option(arguments,'--'+name) for name in
                                     ['layers','top_pct','min_diff','max_samples','feature']}
    return layers, report['expected_shape']


def verify_tensor(path, expected_layers, expected_shape, tolerance, torch):
    loaded = torch.load(path,map_location='cpu',weights_only=True)
    require(isinstance(loaded,dict) and 'layers' in loaded and 'proj' in loaded,
            'Expected official dictionary with layers and proj: '+str(path))
    layers = loaded['layers']
    if isinstance(layers,torch.Tensor):
        layers = layers.tolist()
    require(list(layers) == expected_layers,'Projection layer metadata mismatch: '+str(path))
    tensor = loaded['proj']
    require(isinstance(tensor,torch.Tensor) and tensor.is_floating_point(),'Projection must be a floating tensor')
    require(list(tensor.shape) == expected_shape,'Projection shape/KV head mismatch: '+str(path))
    require(bool(torch.isfinite(tensor).all()),'Projection contains non-finite values: '+str(path))
    # Recompute errors on CPU in float64; the artifact itself remains untouched.
    numeric = tensor.double()
    rows = []
    for index,layer in enumerate(expected_layers):
        for head in range(expected_shape[1]):
            projection = numeric[index,head]
            nonzero = bool(torch.count_nonzero(projection))
            symmetry = float((projection-projection.T).abs().max())
            residual = projection@projection-projection
            idempotency = float(residual.abs().max())
            norm = float(torch.linalg.vector_norm(projection))
            relative = float(torch.linalg.vector_norm(residual))/norm if norm > 0 else 0.
            trace = float(torch.trace(projection))
            rows.append(dict(layer=layer,kv_head=head,nonzero=nonzero,
                             symmetry_max_abs=symmetry,idempotency_max_abs=idempotency,
                             idempotency_relative_frobenius=relative,trace=trace,
                             approximate_rank=round(trace)))
            require(symmetry <= tolerance and idempotency <= tolerance,
                    f'Projection algebra failed in {path.name}, layer={layer}, kv_head={head}: symmetry={symmetry}, idempotency={idempotency}, tolerance={tolerance}')
    return dict(path=str(path),sha256=sha(path),layers=expected_layers,shape=list(tensor.shape),
                dtype=str(tensor.dtype),finite=True,total_heads=len(rows),
                nonzero_heads=sum(r['nonzero'] for r in rows),zero_heads=sum(not r['nonzero'] for r in rows),
                max_symmetry_error=max(r['symmetry_max_abs'] for r in rows),
                max_idempotency_error=max(r['idempotency_max_abs'] for r in rows),heads=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--positive',type=Path,required=True)
    parser.add_argument('--negative',type=Path,required=True)
    parser.add_argument('--run-manifest',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-forwards',type=int)
    parser.add_argument('--algebra-atol',type=float,default=1e-4)
    args = parser.parse_args()
    for name in ['positive','negative','run_manifest','model','source','output']:
        setattr(args,name,getattr(args,name).expanduser().resolve())
    require(not args.output.exists(),'Audit output already exists; choose a new path')
    require(args.algebra_atol > 0,'Tolerance must be positive')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    started = time.monotonic()
    report = dict(status='running',scope='Official traditional projection artifact and run provenance verification only',
                  standard_benchmark_reproduction_complete=False,new_gpu_forwards=0,
                  verified_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  algebra_absolute_tolerance=args.algebra_atol,
                  zero_heads_permitted=True,
                  negative_projection_semantics='Un[:, kn:] @ Un[:, kn:].T; independent negative-covariance complement, not I minus the positive projector',
                  checkpoint_claim='Matches the recorded run checkpoint; this does not establish equivalence to the paper Qwen3-8B-Base checkpoint')
    code = 0
    try:
        import torch
        torch.set_num_threads(4)
        report['torch'] = torch.__version__
        layers, shape = verify_manifest(args,report)
        report['positive'] = verify_tensor(args.positive,layers,shape,args.algebra_atol,torch)
        report['negative'] = verify_tensor(args.negative,layers,shape,args.algebra_atol,torch)
        # Skipped heads are zero in both tensors. An active negative complement
        # may itself be zero if kn equals head_dim; therefore masks need not match.
        report['positive_has_nonzero_signal'] = report['positive']['nonzero_heads'] > 0
        report['negative_has_nonzero_signal'] = report['negative']['nonzero_heads'] > 0
        report['status'] = 'complete_verified'
    except BaseException as error:
        code = 1
        report['status'] = 'failed'
        report['error'] = type(error).__name__+': '+str(error)
        traceback.print_exc()
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        report['checker_sha256'] = sha(Path(__file__).resolve())
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as handle:
            json.dump(report,handle,indent=2)
            handle.write('\n')
    print(json.dumps({key:report[key] for key in ['status','scope','standard_benchmark_reproduction_complete','new_gpu_forwards']}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
