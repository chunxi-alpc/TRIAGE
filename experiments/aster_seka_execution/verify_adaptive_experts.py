"""Read-only CPU audit of four official AdaSEKA expert SVD artifacts.

Run only on the research server. No model is loaded or GPU forward executed.
An absolute-path expert config is emitted only after all artifact checks pass.
Neither the original projections nor manifests/logs are modified.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import traceback


PINNED_COMMIT = 'aab9cfed80dc012fb9e83f8bdefb8675c33f40e7'
MODULES = {
    'synthetic': 'src.custom_builders.adaptive.synthetic_qa_builder_adaptive',
    'counterfact': 'src.custom_builders.adaptive.counterfact_qa_builder',
    'biasbios': 'src.custom_builders.adaptive.biasbios_qa_builder',
    'hotpotqa': 'src.custom_builders.adaptive.hotpot_qa_builder',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def option(arguments, name, default=None):
    for index, value in enumerate(arguments):
        if value == name and index+1 < len(arguments):
            return arguments[index+1]
        if value.startswith(name+'='):
            return value.split('=', 1)[1]
    return default


def mapping(values):
    result = {}
    for value in values:
        name, separator, path = value.partition('=')
        require(separator and name not in result, 'Use unique NAME=PATH arguments')
        require(name in MODULES, 'Unknown expert source: '+name)
        result[name] = Path(path).expanduser().resolve()
    return result


def layers_from_spec(spec, count):
    if spec == 'all':
        layers = list(range(count))
    elif spec.startswith('last'):
        layers = list(range(count-int(spec[4:]), count))
    elif spec.startswith('first'):
        layers = list(range(int(spec[5:])))
    else:
        layers = [int(value) for value in spec.split(',')]
    require(layers and len(layers) == len(set(layers)) and all(0 <= layer < count for layer in layers),
            'Invalid selected layer ids')
    return layers


def verify_storage(build_path, validation_path, wrapper_path, manifest, source, data_path, upstream_arguments):
    build = json.loads(build_path.read_text())
    validation = json.loads(validation_path.read_text())
    require(build.get('mode') == 'build' and build.get('status') == 'complete_pending_projection_audit',
            'Storage build must have completed; validation alone is not a projection build')
    require(validation.get('mode') == 'validate' and validation.get('status') == 'complete_verified'
            and validation.get('equivalence_verified') is True, 'Storage equivalence validation missing or failed')
    require(build.get('validation_report_sha256') == sha(validation_path), 'Storage validation report hash differs')
    require(build.get('upstream_args') == upstream_arguments, 'Storage report arguments differ from actual run manifest')
    require(build.get('upstream_module') == MODULES['hotpotqa'], 'Storage source module differs')
    for field in ['wrapper_sha256', 'upstream_builder_sha256', 'upstream_checker_sha256', 'model_path',
                  'model_config_sha256', 'calibration_data_sha256', 'torch', 'transformers', 'upstream_module']:
        require(build.get(field) == validation.get(field), 'Storage validation/build provenance mismatch: '+field)
    require(sha(wrapper_path) == build['wrapper_sha256'], 'Storage wrapper changed since validation/build')
    require(sha(source/'src/model/projection_builder_base_adapsvd.py') == build['upstream_builder_sha256'],
            'Storage upstream builder hash differs')
    require(Path(build['model_path']).resolve() == Path(manifest['model_path']).resolve(), 'Storage model path differs')
    require(build['model_config_sha256'] == manifest['model_files_sha256']['config.json'], 'Storage model config differs')
    require(build['calibration_data_sha256'] == sha(data_path), 'Storage calibration data differs')
    for field in ['torch', 'transformers']:
        require(build[field] == manifest[field], 'Storage/runner environment differs: '+field)
    detail = validation['validation_case']
    require(detail['original_included'] == detail['optimized_included']
            and detail['selected_attention_sums_exact_equal'] is True, 'Storage selection equivalence failed')
    for result in detail['comparisons'].values():
        for tensor in [result['logits'], *result['keys']]:
            require(tensor['exact_equal'] is True and tensor['max_abs_error'] == 0
                    and tensor['original_sha256'] == tensor['optimized_sha256'],
                    'Storage key/logit equivalence failed')
    trace_path = Path(build['selection_trace_path']).expanduser().resolve()
    require(sha(trace_path) == build['selection_trace_sha256'], 'Storage selection trace hash differs')
    kept = discarded = 0
    with trace_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            require(row['selected_layers'] == validation['selected_layers']
                    and row['attention_threshold'] == validation['attention_threshold'],
                    'Storage build selection configuration differs')
            kept += int(row['included'] is True)
            discarded += int(row['included'] is False)
    return dict(build_report_sha256=sha(build_path), validation_report_sha256=sha(validation_path),
                wrapper_sha256=build['wrapper_sha256'], selection_trace_sha256=sha(trace_path),
                selection_trace_rows=kept+discarded, kept_triplets=kept, discarded_triplets=discarded,
                validation_forward_calls=validation['verification_forward_calls'],
                validation_elapsed_seconds=validation['elapsed_seconds'],
                equivalence_scope='First eligible calibration triplet: exact included, layer sums, extracted keys and full logits',
                adaptation_label=build['adaptation_label'])


def audit_expert(name, projection, manifest_path, log_path, args, torch, checked_hashes):
    def check(path, expected, label):
        path = Path(path).resolve()
        if path not in checked_hashes:
            checked_hashes[path] = sha(path)
        require(checked_hashes[path] == expected, label+': '+str(path))

    manifest = json.loads(manifest_path.read_text())
    require(manifest.get('process_state') == 'exited' and manifest.get('exit_code') == 0,
            name+': builder is incomplete or failed')
    require(manifest.get('source_commit') == PINNED_COMMIT, name+': official source commit differs')
    require(Path(manifest['model_path']).resolve() == args.model, name+': checkpoint path differs')
    arguments = manifest['module_args']
    wrapped = manifest['module'] == 'adaptive_storage'
    if wrapped:
        require(option(arguments, '--mode') == 'build', name+': validation is not a build')
        actual_module = option(arguments, '--upstream-module')
        require('--' in arguments, 'Storage wrapper must separate original upstream arguments')
        arguments = arguments[arguments.index('--')+1:]
    else:
        actual_module = manifest['module']
    require(actual_module == MODULES[name], name+': expert source module does not match its source label')
    require((args.source/option(arguments, '--model')).resolve() == args.model, name+': upstream model differs')
    require(projection.parent == (args.source/option(arguments, '--output_dir')).resolve(), name+': output path differs')
    fingerprints = manifest.get('model_files_sha256', {})
    require('config.json' in fingerprints and any(path.endswith(('.safetensors', '.bin')) for path in fingerprints),
            name+': checkpoint fingerprints missing')
    for path, expected in fingerprints.items():
        check(args.model/path, expected, name+': checkpoint changed')
    index_path = args.model/'model.safetensors.index.json'
    if index_path.exists():
        shards = set(json.loads(index_path.read_text())['weight_map'].values())
        require(shards <= set(fingerprints), name+': checkpoint shards absent from manifest')
    source_hashes = manifest.get('source_files_sha256', {})
    required_sources = {actual_module.replace('.', '/')+'.py', 'src/model/projection_builder_base_adapsvd.py', 'src/utils.py'}
    require(required_sources <= set(source_hashes), name+': core builder fingerprints missing')
    for path, expected in source_hashes.items():
        check(args.source/path, expected, name+': source changed')
    data_path = (args.source/option(arguments, '--data')).resolve()
    if data_path.is_dir():
        require(name == 'counterfact', 'Only CounterFact uses a dataset directory in this audit')
        data_path = data_path/'counterfact.jsonl'
    inputs = manifest.get('input_files_sha256', {})
    require(str(data_path) in inputs, name+': actual calibration data fingerprint missing')
    for path, expected in inputs.items():
        check(path, expected, name+': input changed')
    attempts, completed = manifest.get('forward_attempts'), manifest.get('forward_completed')
    require(isinstance(attempts, int) and attempts == completed and completed > 0 and completed % 2 == 0,
            name+': incomplete or unexpected pair-of-forward count')
    require(manifest.get('first_forward_parameter_dtype') == 'torch.float32', name+': original calibration FP32 changed')
    log = log_path.read_text(errors='replace')
    summaries = re.findall(r'Finished key extraction\. Kept (\d+) samples, discarded (\d+)\.', log)
    require(len(summaries) == 1, name+': expected exactly one completed key-extraction summary in this job log')
    kept, discarded = map(int, summaries[0])
    require(kept+discarded == completed//2, name+': log selection totals differ from actual forward calls')
    failures = re.findall(r'SVD failed for layer (\d+), head (\d+):', log)
    config = json.loads((args.model/'config.json').read_text())
    config = config.get('text_config', config)
    layers = layers_from_spec(option(arguments, '--layers', 'all'), config['num_hidden_layers'])
    dimension = config.get('head_dim', config['hidden_size']//config['num_attention_heads'])
    kv_heads = config['num_key_value_heads']
    saved = torch.load(projection, map_location='cpu', weights_only=True)
    require(isinstance(saved, dict) and {'layers', 'U_matrices', 'singular_values'} <= set(saved),
            name+': expected official SVD dictionary')
    actual_layers = saved['layers'].tolist() if isinstance(saved['layers'], torch.Tensor) else list(saved['layers'])
    require(actual_layers == layers, name+': saved layer ids differ from calibration command')
    u, s = saved['U_matrices'], saved['singular_values']
    require(isinstance(u, torch.Tensor) and isinstance(s, torch.Tensor), name+': SVD values are not tensors')
    require(u.dtype == s.dtype == torch.float32, name+': official saved SVD dtype differs')
    require(list(u.shape) == [len(layers), kv_heads, dimension, dimension]
            and list(s.shape) == [len(layers), kv_heads, dimension], name+': SVD shape differs from model KV heads')
    head_rows = []
    identity = torch.eye(dimension, dtype=torch.float64)
    for li, layer in enumerate(layers):
        for head in range(kv_heads):
            uh, sh = u[li, head].double(), s[li, head].double()
            invalid = bool(torch.isnan(uh).all() and torch.isnan(sh).all())
            row = dict(layer=layer, kv_head=head, official_nan_skip=invalid)
            if not invalid:
                require(bool(torch.isfinite(uh).all() and torch.isfinite(sh).all()),
                        f'{name}: partial NaN/Inf at layer {layer}, head {head}')
                orthogonality = float((uh.T@uh-identity).abs().max())
                require(bool((sh >= 0).all() and (sh[:-1] >= sh[1:]).all()),
                        f'{name}: singular values not nonnegative/descending at {layer}/{head}')
                row.update(orthogonality_max_abs=orthogonality, max_singular=float(sh[0]), min_singular=float(sh[-1]),
                           orthogonality_warning=orthogonality > args.algebra_atol,
                           column_norm_min=float(torch.linalg.vector_norm(uh,dim=0).min()),
                           column_norm_max=float(torch.linalg.vector_norm(uh,dim=0).max()),
                           identity_and_unit_singular_values=bool(torch.equal(uh, identity) and torch.equal(sh, torch.ones_like(sh))))
            head_rows.append(row)
    result = dict(path=str(projection), sha256=sha(projection), manifest_path=str(manifest_path),
                  manifest_sha256=sha(manifest_path), log_sha256=sha(log_path), source_module=actual_module,
                  source_commit=manifest['source_commit'], source_dirty_at_run=manifest.get('source_dirty'),
                  source_files_checked=len(source_hashes), checkpoint_files_sha256=fingerprints,
                  calibration_path=str(data_path), calibration_sha256=inputs[str(data_path)],
                  requested_max_samples=int(option(arguments, '--max_samples')),
                  example_subset=option(arguments, '--example_subset'),
                  parameter_values={key:option(arguments, '--'+key) for key in ['layers', 'top_pct', 'min_diff', 'feature']},
                  kept_triplets=kept, discarded_triplets=discarded,
                  span_valid_triplets=completed//2, total_source_rows_processed='Not inferred from forward count; missing spans can skip forwards',
                  svd_fallback_heads=[{'layer':layers[int(li)], 'kv_head':int(head)} for li, head in failures],
                  all_svd_computations_succeeded=not failures,
                  layers=layers, u_shape=list(u.shape), s_shape=list(s.shape), dtype=str(u.dtype), heads=head_rows,
                  valid_heads=sum(not row['official_nan_skip'] for row in head_rows),
                  official_nan_skipped_heads=sum(row['official_nan_skip'] for row in head_rows),
                  orthogonality_warning_heads=sum(row.get('orthogonality_warning',False) for row in head_rows),
                  maximum_orthogonality_error=max((row['orthogonality_max_abs'] for row in head_rows if not row['official_nan_skip']),default=None),
                  costs={key:manifest.get(key) for key in ['forward_attempts','forward_completed','elapsed_seconds',
                    'gpu_lock_wall_seconds','peak_allocated_gib','peak_reserved_gib','forward_input_hashing_seconds','gpu_physical']})
    if name == 'synthetic':
        with data_path.open() as handle:
            result['available_source_rows'] = sum(1 for _ in handle)
        result['effective_source_row_cap'] = min(result['available_source_rows'], result['requested_max_samples'])
        require(completed <= 2*result['effective_source_row_cap'], 'Synthetic adaptive builder only takes context_1 per row')
    if wrapped:
        require(name == 'hotpotqa' and args.storage_build_report and args.storage_validation_report and args.storage_wrapper,
                'Storage adapter audit requires actual build/validation reports and wrapper file')
        result['storage_adaptation'] = verify_storage(args.storage_build_report, args.storage_validation_report,
                                                      args.storage_wrapper, manifest, args.source, data_path, arguments)
        for field, expected in [('kept_triplets', kept), ('discarded_triplets', discarded)]:
            require(result['storage_adaptation'][field] == expected, 'Storage trace and original run log differ: '+field)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert', action='append', default=[], help='NAME=actual SVD path (all four sources required)')
    parser.add_argument('--manifest', action='append', default=[])
    parser.add_argument('--log', action='append', default=[])
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--storage-build-report', type=Path)
    parser.add_argument('--storage-validation-report', type=Path)
    parser.add_argument('--storage-wrapper', type=Path)
    parser.add_argument('--algebra-atol', type=float, default=1e-4,
                        help='Reporting threshold for orthogonality warnings only, never a rejection threshold')
    args = parser.parse_args()
    for field in ['model','source','output_dir','storage_build_report','storage_validation_report','storage_wrapper']:
        if getattr(args, field) is not None:
            setattr(args, field, getattr(args, field).expanduser().resolve())
    require(not args.output_dir.exists(), 'Use a new output directory; preserve previous audits')
    args.output_dir.mkdir(parents=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    started = time.monotonic()
    report = dict(status='running', scope='Four official-source SVD artifact structure/provenance and measured numerical diagnostics; not exact orthogonality or performance reproduction',
                  verified_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), new_gpu_forwards=0,
                  standard_benchmark_reproduction_complete=False, clinical_sota_claim=False,
                  paper_checkpoint_equivalence_claim=False, exact_orthogonality_claim=False,
                  orthogonality_warning_threshold=args.algebra_atol,
                  orthogonality_threshold_is_acceptance_gate=False,
                  config_ready=False, experts={}, warnings=[])
    code = 0
    try:
        require(args.algebra_atol > 0, 'Tolerance must be positive')
        experts, manifests, logs = (mapping(getattr(args, field)) for field in ['expert','manifest','log'])
        require(set(experts) == set(manifests) == set(logs) == set(MODULES), 'Supply all four actual sources, manifests and logs')
        current_commit = subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'], text=True).strip()
        require(current_commit == PINNED_COMMIT, 'Official checkout HEAD differs')
        import torch
        torch.set_num_threads(4)
        report['torch'] = torch.__version__
        checked_hashes = {}
        for name in MODULES:
            report['experts'][name] = audit_expert(name, experts[name], manifests[name], logs[name], args, torch, checked_hashes)
        reference = report['experts']['synthetic']
        for name, result in report['experts'].items():
            require(result['layers'] == reference['layers'] and result['u_shape'] == reference['u_shape'],
                    name+': incompatible expert layer/head layout')
            require(result['checkpoint_files_sha256'] == reference['checkpoint_files_sha256'], name+': checkpoint fingerprints differ')
            require(result['parameter_values']['feature'] == reference['parameter_values']['feature'],
                    name+': expert feature transformations differ; one inference feature function cannot match this bank')
            if not result['all_svd_computations_succeeded']:
                report['warnings'].append(name+': upstream SVD fallback occurred; file validity does not mean successful SVD for every head')
            if result['orthogonality_warning_heads']:
                report['warnings'].append(name+': measured float32 U orthogonality error exceeds the diagnostic threshold in '
                                          +str(result['orthogonality_warning_heads'])+' heads; artifacts are unchanged and numerical effects require real inference controls')
        calibration_hashes = [result['calibration_sha256'] for result in report['experts'].values()]
        require(len(set(calibration_hashes)) == 4, 'Four source labels must not reuse the same calibration file')
        routing = []
        for index, head in enumerate(reference['heads']):
            invalid = [name for name in MODULES if report['experts'][name]['heads'][index]['official_nan_skip']]
            routing.append(dict(layer=head['layer'], kv_head=head['kv_head'], invalid_experts=invalid,
                                remaining_valid_experts=4-len(invalid),
                                official_uniform_fallback_due_to_nan=bool(invalid),
                                dynamic_projection_zero_when_all_invalid=len(invalid)==4))
        report['routing_heads'] = routing
        report['routing_nan_uniform_fallback_heads'] = sum(row['official_uniform_fallback_due_to_nan'] for row in routing)
        report['routing_all_valid_heads'] = sum(not row['invalid_experts'] for row in routing)
        report['routing_semantics'] = 'Pinned upstream uses signed maximum-absolute coefficient normalization, not softmax. Any invalid expert produces -inf, then nonfinite normalization, then uniform coefficients; invalid U is skipped when summing projections.'
        report['evaluation_constraint'] = 'Use batch_size=1: official routing reads only batch element 0, last token; include its additional model forward in cost.'
        report['paper_sampling_difference'] = 'Released builders take initial rows; synthetic Git source has 100 context_1 pairs. This does not assert paper seed42 random-200 calibration reproduction.'
        config_path = args.output_dir/'expert_config.json'
        with config_path.open('x') as handle:
            json.dump({name:str(experts[name]) for name in MODULES}, handle, indent=2)
            handle.write('\n')
        report.update(status='complete_verified', config_ready=True,
                      expert_config_path=str(config_path), expert_config_sha256=sha(config_path))
    except BaseException as error:
        code = 1
        report.update(status='failed', error=type(error).__name__+': '+str(error))
        traceback.print_exc()
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        report['checker_sha256'] = sha(Path(__file__).resolve())
        with (args.output_dir/'audit.json').open('x') as handle:
            json.dump(report, handle, indent=2)
            handle.write('\n')
    print(json.dumps({key:report[key] for key in ['status','config_ready','new_gpu_forwards','standard_benchmark_reproduction_complete']}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
