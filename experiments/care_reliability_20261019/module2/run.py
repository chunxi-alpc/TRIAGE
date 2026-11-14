#!/usr/bin/env python3
"""Frozen real-model direction/calibration diagnostic. No GPU starts on import.

Run one fold at a time. The parent orchestration decides when and on which
authorized GPU to execute. --preflight uses only CPU and never reports model
performance. Missing model/dependencies cause an explicit blocked record.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from geometry import random_controls
from compatibility import load_frozen_manifest, config_audit, environment_audit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DIRECTIONS = ('b4', 'mean', 'random_projector', 'random_vector', 'shuffled_mean')
METHODS = tuple(f'{name}_{mode}' for name in DIRECTIONS for mode in ('raw', 'calibrated')) + ('direct_bias',)


class RunBlocked(RuntimeError):
    pass


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def installed_versions(names):
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def source_fingerprints(args):
    sources = list(HERE.glob('*.py')) + list(HERE.glob('*.json')) + list(HERE.glob('*.md'))
    sources += [args.legacy_signed / 'diagnostic.py', args.legacy_signed / 'support.py']
    return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): digest(p)
            for p in sources if p.exists()}


def recursive_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from recursive_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from recursive_strings(child)


def audit_rows(data, mode):
    """Text/schema checks only; true clinical labels get an independent oracle below."""
    pairs = {}
    for row in data:
        required = ('case_id', 'pair_id', 'group_id', 'prompt', 'spans', 'target_action',
                    'applicable', 'gold_answer', 'option_map')
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f'Missing fields {missing} in {row.get("case_id")}')
        if type(row['applicable']) is not bool:
            raise ValueError('The T/F module-two set requires JSON Boolean applicability')
        if set(row['option_map']) != set('ABCD') or row['gold_answer'] not in ('A', 'B', 'C', 'D'):
            raise ValueError('Exactly four A/B/C/D options are required')
        target_answer = row.get('target_answer', row.get('inapplicable_action'))
        if target_answer not in ('A', 'B', 'C', 'D'):
            raise ValueError('Missing target_answer / legacy inapplicable_action')
        row['target_answer'] = target_answer
        if row['applicable'] != (row['gold_answer'] == target_answer):
            raise ValueError('Designated action and independent answer disagree on T/F')
        start, end = row['spans']['action']
        if not 0 <= start < end <= len(row['prompt']):
            raise ValueError('Invalid action character span')
        if row['prompt'][start:end] != row['target_action']:
            raise ValueError('Action span does not recover the exact fixed action text')
        option_start = row['prompt'].find('\nOptions:')
        if option_start < 0 or end >= option_start:
            raise ValueError('Target action must precede answer options')
        if mode == 'sealed_mechanism':
            if row.get('split') != 'confirmation':
                raise ValueError('Sealed mode accepts confirmation cases only')
            if row.get('dataset_id') != 'sealed_mechanism_20261019':
                raise ValueError('Wrong sealed dataset ID')
            if row.get('outer_fold') not in (0, 1, 2):
                raise ValueError('Sealed cases need a frozen outer_fold asset assignment')
        pairs.setdefault(row['pair_id'], []).append(row)
    if len({row['case_id'] for row in data}) != len(data):
        raise ValueError('Duplicate case IDs')
    for pid, members in pairs.items():
        if len(members) != 2 or {r['applicable'] for r in members} != {True, False}:
            raise ValueError(f'Incomplete T/F pair: {pid}')
        if len({r['group_id'] for r in members}) != 1:
            raise ValueError('Pair crosses a family boundary')
        if len({r['target_action'] for r in members}) != 1:
            raise ValueError('Target action changed within a minimal pair')
        if members[0]['option_map'] != members[1]['option_map']:
            raise ValueError('Answer option mapping changed within a pair')
        if members[0]['target_answer'] != members[1]['target_answer']:
            raise ValueError('Target answer changed within a pair')
        if mode == 'sealed_mechanism' and members[0]['outer_fold'] != members[1]['outer_fold']:
            raise ValueError('Pair uses different frozen operator folds')
    return dict(cases=len(data), pairs=len(pairs), families=len({r['group_id'] for r in data}),
                patient_data='simulated', mode=mode)


def preflight(args, protocol):
    required = [args.legacy_signed / 'diagnostic.py', args.legacy_signed / 'support.py',
                args.artifacts / 'data/clinical/candidates.jsonl',
                args.artifacts / 'results/manifest.json']
    folder = args.legacy_signed / f'results/fold{args.fold}'
    required += [folder / name for name in ('split.json', 'selection.json', 'training_audit.json', 'operators.pt')]
    if args.mode == 'sealed_mechanism':
        if args.data is None or args.data_manifest is None:
            raise RunBlocked('Sealed mode requires --data and --data-manifest')
        required += [args.data, args.data_manifest]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RunBlocked('Missing required frozen assets: ' + ', '.join(missing))
    original = load_frozen_manifest(args.artifacts / 'results/manifest.json', protocol)
    split, selection, old_audit = [read(folder / name) for name in ('split.json', 'selection.json', 'training_audit.json')]
    frozen_assets = read(HERE / 'frozen_learned_assets.json')
    if digest(folder / 'operators.pt') != frozen_assets['folds'][str(args.fold)]['operator_sha256']:
        raise AssertionError('Archived learned operator bytes do not match the new frozen asset manifest')
    fixed = protocol['frozen_folds'][str(args.fold)]
    if old_audit['location'] != [fixed['layer'], fixed['kv_head']] or old_audit['rank'] != fixed['rank']:
        raise AssertionError('Archived head/rank differ from the frozen protocol')
    if selection['common_tau'] != fixed['score_rms']:
        raise AssertionError('Archived strength differs from the frozen protocol')
    if [len(split[x]) for x in ('train', 'dev', 'test')] != [6, 2, 4]:
        raise AssertionError('Unexpected archived family split')
    if set(split['train']) & (set(split['dev']) | set(split['test'])) or set(split['dev']) & set(split['test']):
        raise AssertionError('Family overlap')
    if digest(args.artifacts / 'data/clinical/candidates.jsonl') != protocol['legacy_data_sha256']:
        raise AssertionError('The old clinical data bytes differ from the frozen baseline')
    legacy_data = rows(args.artifacts / 'data/clinical/candidates.jsonl')
    legacy_audit = audit_rows(legacy_data, 'legacy384_regression')
    if (legacy_audit['cases'], legacy_audit['pairs'], legacy_audit['families']) != (384, 192, 12):
        raise AssertionError('Old training data changed')
    if args.mode == 'legacy384_regression':
        data = [row for row in legacy_data if row['group_id'] in split['test']]
        data_digest = digest(args.artifacts / 'data/clinical/candidates.jsonl')
    else:
        sealed = rows(args.data)
        audit_rows(sealed, 'sealed_mechanism')
        data_digest = digest(args.data)
        manifest = read(args.data_manifest)
        if manifest.get('status') != 'frozen_before_model_execution' or manifest.get('is_clinical') is not False:
            raise RunBlocked('The mechanism manifest must be frozen and explicitly nonclinical')
        if data_digest != protocol['sealed_data_sha256']:
            raise AssertionError('New mechanism data changed after this protocol was frozen')
        if data_digest not in set(recursive_strings(manifest)):
            raise RunBlocked('Sealed data SHA256 is not recorded in the supplied manifest')
        mapping = {}
        for row in sealed:
            mapping.setdefault(row['group_id'], row['outer_fold'])
            if mapping[row['group_id']] != row['outer_fold']:
                raise AssertionError('A new family uses more than one frozen fold')
        data = [row for row in sealed if row['outer_fold'] == args.fold]
        if set(row['group_id'] for row in sealed) & set(row['group_id'] for row in legacy_data):
            raise AssertionError('New mechanism family IDs overlap the old clinical families')
    if not data:
        raise RunBlocked('This fold has no test cases; do not create an empty inference result')
    audit = audit_rows(data, args.mode)
    audit.update(fold=args.fold, data_sha256=data_digest, fixed_configuration=fixed,
                 primary_cases=(sum(r.get('variant') == 'canonical' for r in data)
                                if args.mode == 'sealed_mechanism' else len(data)),
                 old_training_cases=len([r for r in legacy_data if r['group_id'] in split['train']]),
                 test_outcomes_used_for_selection=False,
                 dependencies_available={name: importlib.util.find_spec(name) is not None
                                         for name in ('torch', 'transformers', 'numpy')},
                 model_manifest_sha256=protocol['model_manifest_sha256'],
                 environment_compatibility=environment_audit(installed_versions(original['versions']), original, protocol),
                 model_directory_exists=bool(args.model and args.model.is_dir()),
                 status='cpu_preflight_only', no_model_inference_performed=True)
    return audit, legacy_data, data, split


def setup_model(args, protocol):
    if importlib.util.find_spec('torch') is None or importlib.util.find_spec('transformers') is None:
        raise RunBlocked('PyTorch/Transformers unavailable; real model inference has not run')
    original = load_frozen_manifest(args.artifacts / 'results/manifest.json', protocol)
    compatibility = dict(status='checking_before_real_inference',
                         original_manifest_sha256=protocol['model_manifest_sha256'],
                         environment=environment_audit(installed_versions(original['versions']), original, protocol))
    audit_path = args.output / 'model_compatibility.json'
    save(audit_path, compatibility)
    if not compatibility['environment']['compatible']:
        raise RunBlocked('Unvalidated Torch/Transformers version; see model_compatibility.json')
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RunBlocked('A BF16-capable authorized CUDA environment is required')
    if args.model is None or not args.model.is_dir():
        raise RunBlocked('Explicit fixed local --model directory is required; no download fallback')
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RunBlocked('Set CUDA_VISIBLE_DEVICES to the GPU authorized by the run coordinator')
    if not args.device.startswith('cuda'):
        raise RunBlocked('This real-model runner does not silently substitute a CPU/random model')
    compatibility['runtime'] = dict(
        torch_runtime_version=str(torch.__version__), cuda_build_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(args.device),
        original_gpu=original.get('gpu'), gpu_differs_from_original=torch.cuda.get_device_name(args.device) != original.get('gpu'),
        original_cuda_build_and_driver_versions_recorded=False,
        cuda_driver_version_recorded=False,
        cuda_driver_note='Neither original manifest nor this stdlib audit records the driver version; no driver equality is claimed',
    )
    resolved = AutoConfig.from_pretrained(args.model, local_files_only=True)
    compatibility['resolved_config'] = config_audit(resolved.to_dict(), original, protocol)
    save(audit_path, compatibility)
    if not compatibility['resolved_config']['compatible']:
        raise RunBlocked('Local resolved model config differs from the frozen geometry; see model_compatibility.json')
    weights = {path.name: digest(path) for path in args.model.glob('*.safetensors')}
    expected = {key: value['sha256'] for key, value in original['weights'].items()}
    if not weights or weights != expected:
        raise AssertionError('The model weight shards do not match the original fixed model')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=resolved, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation=protocol['attention']).to(args.device).eval()
    config = model.config
    compatibility['loaded_config'] = config_audit(config.to_dict(), original, protocol)
    compatibility['loaded_attention'] = getattr(config, '_attn_implementation', None)
    compatibility['loaded_precision'] = str(model.dtype)
    compatible = (compatibility['loaded_config']['compatible']
                  and compatibility['loaded_attention'] == protocol['attention']
                  and compatibility['loaded_precision'] == 'torch.' + protocol['precision'])
    compatibility['status'] = 'passed_configuration_and_environment_gates' if compatible else 'incompatible_loaded_model'
    save(audit_path, compatibility)
    if not compatible:
        raise RunBlocked('Loaded model config, attention or precision violates the frozen geometry; see model_compatibility.json')
    return model, tokenizer, weights


def fit_operators(args, model, tokenizer, legacy, support, training_rows, protocol):
    import torch
    fixed = protocol['frozen_folds'][str(args.fold)]
    location = (fixed['layer'], fixed['kv_head'])
    fresh = {}
    for row in sorted(training_rows, key=lambda x: x['case_id']):
        captured, indices, _, _ = legacy.collect(model, tokenizer, row, [location])
        fresh[row['case_id']] = {location: captured[location[0]]['k'][indices, location[1]].float().cpu().numpy()}
    _, signal, background, controls = support.training_matrices(training_rows, fresh, location)
    if background is None:
        raise AssertionError('Frozen training background controls are missing')
    basis = support.contrastive_basis(signal, background, fixed['rank'])
    projector = (basis @ basis.T).astype(np.float32)
    mean = signal.mean(0).astype(np.float32)
    old_folder = args.legacy_signed / f'results/fold{args.fold}'
    reference = torch.load(old_folder / 'operators.pt', map_location='cpu', weights_only=True)
    p_error = float(np.max(np.abs(projector - reference['B4_projector'].cpu().numpy())))
    mean_error = float(np.max(np.abs(mean - reference['d_app'].cpu().numpy())))
    if p_error > 1e-6 or mean_error > 1e-6:
        raise AssertionError(f'Fresh training geometry differs from archive: P={p_error}, mean={mean_error}')
    old_audit = read(old_folder / 'training_audit.json')
    if sorted(fresh) != sorted(old_audit['train_case_ids']):
        raise AssertionError('Training IDs changed')
    seed = protocol['direction_seed'] + args.fold
    controls_random = random_controls(128, fixed['rank'], signal, seed)
    swaps = controls_random.pop('pair_label_swaps')
    operators = dict(b4=projector, mean=mean, **controls_random)
    operators = {key: np.asarray(value, dtype=np.float32) for key, value in operators.items()}
    np.savez(args.output / 'operators.npz', **operators)
    save(args.output / 'training_audit.json', dict(
        train_case_ids=sorted(fresh), training_pair_ids=sorted({r['pair_id'] for r in training_rows}),
        pair_label_swaps=swaps.tolist(), direction_seed=seed, training_control_count=len(controls),
        original_projector_max_error=p_error, original_mean_max_error=mean_error,
        location=list(location), rank=fixed['rank'], controls=controls,
        shuffled_control_is_first_moment=True, no_test_labels_in_direction_fit=True,
        source_training_data_sha256=digest(args.artifacts / 'data/clinical/candidates.jsonl')))
    return operators


def predict_record(row, method, probabilities, logits, disturbance, mode, fold, tau):
    gold = 'ABCD'.index(row['gold_answer'])
    prediction = 'ABCD'[int(np.argmax(probabilities))]
    return dict(
        case_id=row['case_id'], pair_id=row['pair_id'], group_id=row['group_id'],
        applicable=row['applicable'], gold_answer=row['gold_answer'], target_answer=row['target_answer'],
        prediction=prediction, correct=int(prediction == row['gold_answer']),
        misuse=int(not row['applicable'] and prediction == row['target_answer']),
        candidate_probabilities=probabilities, candidate_logits=logits,
        gold_logit_margin=logits[gold] - max(v for i, v in enumerate(logits) if i != gold),
        gold_probability_margin=probabilities[gold] - max(v for i, v in enumerate(probabilities) if i != gold),
        method=method, disturbance=disturbance, mode=mode, fold=fold, tau=tau,
        dataset_id=row.get('dataset_id', 'legacy_clinical_384_20261016'),
        variant=row.get('variant'), variant_group_id=row.get('variant_group_id'),
        prompt_tokens=row.get('prompt_tokens'),
        oracle_applicability=True, oracle_action_span=True,
        eligible=row.get('eligible'), no_conflict=row.get('no_conflict'),
    )


def execute(args, protocol, legacy_data, test_data, split):
    from backend import load_python, InterventionGeometry
    model, tokenizer, weights = setup_model(args, protocol)
    import torch
    legacy = load_python(args.legacy_signed / 'diagnostic.py', 'care_old_signed_diagnostic')
    support = load_python(args.legacy_signed / 'support.py', 'care_old_signed_support')
    for row in legacy_data:
        if support.classify(row['kind'], row['value']) != row['gold_label']:
            raise AssertionError('Independent old numerical oracle disagrees with stored gold')
    fixed = protocol['frozen_folds'][str(args.fold)]
    location = (fixed['layer'], fixed['kv_head'])
    training = [row for row in legacy_data if row['group_id'] in split['train']]
    with torch.inference_mode():
        operators = fit_operators(args, model, tokenizer, legacy, support, training, protocol)
        screen_path = args.legacy_signed / 'inputs/prior_screen.csv'
        if args.mode == 'legacy384_regression' and screen_path.exists():
            screen = {row['case_id']: row for row in csv.DictReader(screen_path.open())}
            for row in test_data:
                prior = screen[row['case_id']]
                row['eligible'] = prior['eligible'] == 'True'
                row['no_conflict'] = prior['stable'] == 'True' and prior['prior_prediction'] == prior['gold_answer']
        record_count = 0
        no_op_count = 0
        checks = []
        torch.cuda.reset_peak_memory_stats()
        with (args.output / 'predictions.jsonl').open('w', buffering=1) as stream:
            for index, row in enumerate(sorted(test_data, key=lambda r: r['case_id'])):
                encoded, _, _ = legacy.tokenize_span(tokenizer, row['prompt'], row['spans']['action'])
                row['prompt_tokens'] = int(encoded['input_ids'].shape[1])
                if row['prompt_tokens'] > protocol['maximum_prompt_tokens']:
                    raise AssertionError('Prompt exceeds the frozen token limit; no automatic truncation')
                captured, indices, base_prob, base_logits = legacy.collect(model, tokenizer, row, [location])
                geometry = InterventionGeometry(model, captured, indices, location, protocol)
                sign = 1 if row['applicable'] else -1
                def append(method, prob, logits, disturbance, tau):
                    nonlocal record_count
                    record = predict_record(row, method, prob, logits, disturbance, args.mode, args.fold, tau)
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                    record_count += 1
                append('baseline', base_prob, base_logits, {}, 0.0)
                for method in METHODS:
                    # A zero score bias changes the mask head dimension in the
                    # legacy hook. Check the actual SDPA path for EVERY input,
                    # including every prompt length, before counting effects.
                    if index == 0 or method == 'direct_bias':
                        if method == 'direct_bias':
                            context = legacy.bias_scores(model, location, indices, 0.0)
                        else:
                            name, cal = method.rsplit('_', 1)
                            unchanged, zero_audit = geometry.key_intervention(name, cal, operators, sign, 0.0)
                            assert torch.equal(unchanged, geometry.key) and zero_audit['actual_rms'] == 0
                            context = legacy.edit_keys(model, location, indices, unchanged, geometry.key)
                        zero_prob, zero_logits = legacy.score(model, tokenizer, row, context)
                        check = dict(method=method, case_id=row['case_id'], tau=0.0,
                                     prompt_tokens=row['prompt_tokens'],
                                     candidate_logits=zero_logits, candidate_probabilities=zero_prob,
                                     identical_candidate_logits=zero_logits == base_logits,
                                     identical_candidate_probabilities=zero_prob == base_prob,
                                     mask_head_expansion_checked=method == 'direct_bias')
                        checks.append(check)
                        if not check['identical_candidate_logits'] or not check['identical_candidate_probabilities']:
                            save(args.output / 'zero_checks.json', dict(
                                status='failed_real_model_zero_check', policy=protocol['zero_checks'],
                                expected_direct_bias_cases=len(test_data), checks=checks))
                            raise AssertionError(f'Zero intervention changed model outputs: {method}')
                    tau = fixed['score_rms']
                    if method == 'direct_bias':
                        disturbance = geometry.direct_bias_audit(sign, tau)
                        context = legacy.bias_scores(model, location, indices, sign*tau)
                    else:
                        name, cal = method.rsplit('_', 1)
                        changed, disturbance = geometry.key_intervention(name, cal, operators, sign, tau)
                        context = legacy.edit_keys(model, location, indices, changed, geometry.key)
                    if disturbance['calibration_status'] == 'no_eligible_direction':
                        # This is a recorded policy no-op, not a matched nonzero budget.
                        prob, logits = base_prob, base_logits
                        no_op_count += 1
                    else:
                        prob, logits = legacy.score(model, tokenizer, row, context)
                    append(method, prob, logits, disturbance, tau)
                if (index + 1) % 16 == 0:
                    print(json.dumps(dict(event='progress', completed_cases=index+1,
                                          total_cases=len(test_data), fold=args.fold, mode=args.mode)), flush=True)
        expected_zero_checks = len(test_data) + len(METHODS) - 1
        if record_count != len(test_data) * (len(METHODS) + 1) or len(checks) != expected_zero_checks:
            raise AssertionError('Incomplete inference journal or zero checks')
        save(args.output / 'zero_checks.json', dict(
            status='passed_real_model_checks', policy=protocol['zero_checks'],
            expected_direct_bias_cases=len(test_data), checks=checks))
        return dict(status='completed_real_inference', records=record_count, cases=len(test_data),
                    interventions=len(METHODS), training_cases=len(training), no_eligible_no_op_records=no_op_count,
                    actual_model_forwards=len(training)+len(test_data)*(len(METHODS)+1)+len(checks)-no_op_count,
                    zero_check_forwards=len(checks),
                    model_weights=weights, zero_checks_passed=True,
                    model_compatibility=read(args.output / 'model_compatibility.json'),
                    peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                    peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
                    dependency_versions={name: importlib.metadata.version(name)
                                         for name in ('torch', 'transformers', 'numpy')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['legacy384_regression', 'sealed_mechanism'], required=True)
    parser.add_argument('--fold', type=int, choices=[0, 1, 2], required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--data', type=Path)
    parser.add_argument('--data-manifest', type=Path)
    parser.add_argument('--artifacts', type=Path, default=ROOT/'experiments/care_qwen3_4b_base_20261016')
    parser.add_argument('--legacy-signed', type=Path, default=ROOT/'experiments/care_module2_signed_20261019')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = read(HERE / 'protocol.json')
    save(args.output / 'protocol_snapshot.json', protocol)
    start = time.time()
    state = dict(status='preparing', mode=args.mode, fold=args.fold,
                 source_fingerprints=source_fingerprints(args), model_revision=protocol['model_revision'],
                 physical_gpu_visibility=os.environ.get('CUDA_VISIBLE_DEVICES'),
                 test_outcomes_used_for_selection=False, real_model_inference_completed=False)
    save(args.output / 'run_manifest.json', state)
    try:
        audit, legacy_data, test_data, split = preflight(args, protocol)
        save(args.output / 'data_audit.json', audit)
        save(args.output / 'split.json', split)
        state.update(data_sha256=audit['data_sha256'], test_case_ids=sorted(r['case_id'] for r in test_data),
                     test_families=sorted({r['group_id'] for r in test_data}), configuration=audit['fixed_configuration'])
        if args.preflight:
            state.update(status='cpu_preflight_completed', no_model_inference_performed=True,
                         dependencies_available=audit['dependencies_available'],
                         model_directory_exists=audit['model_directory_exists'])
        else:
            state.update(execute(args, protocol, legacy_data, test_data, split))
            state['real_model_inference_completed'] = True
    except RunBlocked as exc:
        state.update(status='blocked_real_inference', reason=str(exc), real_model_inference_completed=False)
        print(json.dumps(dict(status=state['status'], reason=str(exc)), ensure_ascii=False), flush=True)
    except Exception as exc:
        state.update(status='failed_run', error_type=type(exc).__name__, reason=str(exc),
                     real_model_inference_completed=False)
        (args.output / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        state['elapsed_seconds'] = time.time() - start
        save(args.output / 'run_manifest.json', state)
    print(json.dumps(state, ensure_ascii=False), flush=True)
    return 0 if state['status'] in ('cpu_preflight_completed', 'completed_real_inference') else 2


if __name__ == '__main__':
    raise SystemExit(main())
