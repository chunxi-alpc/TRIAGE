"""Independent read-only CPU audit/export of CAD and Spotlight adapters.

Run on the research server. This imports neither torch nor experiment modules.
Original predictions, manifests, negative controls and failures remain intact.
Full-vocabulary numerical drift is checked as a recorded diagnostic: complete
logits were not saved, so this script does not claim to recompute that quantity.
"""
import argparse
from collections import defaultdict
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
import traceback


PIN = 'aab9cfed80dc012fb9e83f8bdefb8675c33f40e7'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def finite(value, label, minimum=None):
    result = float(value)
    require(math.isfinite(result) and (minimum is None or result >= minimum), 'Invalid '+label)
    return result


def metric(values):
    return dict(mean=math.fsum(values)/len(values), std=statistics.pstdev(values), values=None)


def compare_metric(actual, saved, control):
    for key in ['score', 'magnitude']:
        for field in ['mean', 'std']:
            value = finite(saved[key][field], 'saved '+key+'.'+field)
            require(math.isclose(actual[key][field], value, abs_tol=1e-12, rel_tol=1e-10),
                    f'{control}: recomputed {key}.{field} differs: {actual[key][field]} vs {value}')


def fingerprint_audit(args, inner, outer, report):
    require(inner['method'] in ['cad', 'spotlight'], 'Unknown adapter method')
    require(inner['source_revision'] == outer['source_commit'] == PIN, 'Official source revision differs')
    current = subprocess.check_output(['git','-C',str(args.seka_root),'rev-parse','HEAD'], text=True).strip()
    require(current == PIN, 'Current official source revision differs')
    require(Path(inner['checkpoint']).resolve() == Path(outer['model_path']).resolve() == args.model,
            'Inner/outer checkpoint paths differ')
    require(inner['runner_sha256'] == sha(args.adapter_root/'run_counterfact_baselines.py'), 'Adapter runner source differs')
    require(inner['adapter_sha256'] == sha(args.adapter_root/'baseline_operators.py'), 'Adapter operator source differs')
    data_hash = sha(args.data_path)
    require(data_hash == inner['data_sha256'], 'Data fingerprint differs from inner manifest')
    require(outer.get('input_files_sha256', {}).get(str(args.data_path)) == data_hash,
            'Actual data absent from outer fingerprints or differs')
    model_hashes = outer.get('model_files_sha256', {})
    require('config.json' in model_hashes and any(name.endswith(('.safetensors','.bin')) for name in model_hashes),
            'Outer checkpoint fingerprints missing')
    for name, expected in model_hashes.items():
        require(sha(args.model/name) == expected, 'Actual checkpoint fingerprint differs: '+name)
    index_path = args.model/'model.safetensors.index.json'
    if index_path.exists():
        shards = set(json.loads(index_path.read_text())['weight_map'].values())
        require(shards <= set(model_hashes), 'Checkpoint index shards missing from outer manifest')
    # This is the coordinator's documented digest construction, including the
    # default JSON separators. It is not a filename or checkpoint-name claim.
    model_digest = text_sha(json.dumps(model_hashes, sort_keys=True))
    require(inner['model_fingerprint'] == model_digest, 'Inner model digest differs from full outer hash mapping')
    source_hashes = outer.get('source_files_sha256', {})
    require({'benchmarks/counterfact/preprocess.py','benchmarks/counterfact/evaluate.py'} <= set(source_hashes),
            'Official metric/preprocessing source fingerprints missing')
    for name, expected in source_hashes.items():
        require(sha(args.seka_root/name) == expected, 'Official source changed: '+name)
    for name, expected in outer.get('input_files_sha256', {}).items():
        require(sha(name) == expected, 'Recorded input changed: '+name)
    for name in ['torch','transformers','tokenizers','datasets']:
        require(inner['versions'][name] == outer[name], 'Inner/outer library version differs: '+name)
    require(inner['model_dtype'] == outer['first_forward_parameter_dtype'], 'Inner/outer observed dtype differs')
    require(inner['max_new_tokens'] == 1 and inner['batch_size'] == 1 and inner['greedy'] is True,
            'Unexpected standard efficacy generation budget')
    require(inner['add_unmediated_fact'] is True and inner['chat'] is False and inner['candidate_filtering'] is False,
            'Unexpected information/preprocessing setting')
    require(inner['alpha'] == 1 and inner['spotlight_target'] == .1 and inner['seed'] in [0,1,2],
            'Parameters differ from frozen adapter specification')
    report['fingerprints'] = dict(model_files_sha256=model_hashes, model_mapping_digest=model_digest,
        adapter_sha256=inner['adapter_sha256'], adapter_runner_sha256=inner['runner_sha256'],
        outer_runner_sha256=outer.get('runner_sha256'), data_sha256=data_hash,
        official_source_commit=current, official_source_files_checked=len(source_hashes))
    report['identity'] = {key:inner[key] for key in ['method','method_status','checkpoint_status','official_metric','subset','seed']}
    report['identity']['is_author_original_program'] = False


def trace_evidence(args, outer, records, controls, method):
    trace_path = args.forward_trace or (Path(outer['forward_input_trace']) if outer.get('forward_input_trace') else None)
    result = dict(status='blocked', scope='Actual ordered input_ids and attention_mask hashes per completed call')
    if trace_path is None or not trace_path.is_file():
        result['reason'] = 'Actual forward trace unavailable; textual prompt hashes alone do not establish token-level equality'
        return result
    result.update(path=str(trace_path), sha256=sha(trace_path))
    require(result['sha256'] == outer.get('forward_input_trace_sha256'), 'Forward trace hash differs from manifest')
    with trace_path.open() as handle:
        calls = [json.loads(line) for line in handle]
    require(len(calls) == outer['forward_completed'] == outer['forward_input_trace_rows'], 'Forward trace total differs')
    require(all(row['completed_call_index'] == i+1 for i,row in enumerate(calls)), 'Nonsequential completed calls')
    cursor = 0
    grouped = defaultdict(dict)
    for row in records:
        count = row['actual_forward_calls']
        sequence = calls[cursor:cursor+count]
        require(len(sequence) == count, 'Trace shorter than recorded predictions')
        grouped[str(row['id'])][row['control']] = sequence
        lengths = row['metadata'].get('prompt_tokens_per_path', [row['metadata'].get('input_tokens')])
        for call, length in zip(sequence, lengths):
            value = call.get('input_ids')
            if value is not None and value.get('sha256'):
                require(value['shape'] == [1, length], 'Observed input token length differs from per-path budget')
        cursor += count
    require(cursor == len(calls), 'Unassigned model forwards beyond saved prediction controls')
    mismatches = []
    missing = []
    comparisons = 0
    def pair(left, right, case, control, path_label):
        nonlocal comparisons
        comparisons += 1
        for key in ['input_ids','attention_mask']:
            a, b = left.get(key), right.get(key)
            if not isinstance(a,dict) or not isinstance(b,dict) or not a.get('sha256') or not b.get('sha256'):
                missing.append(dict(id=case,control=control,path=path_label,tensor=key))
            elif any(a.get(field) != b.get(field) for field in ['dtype','shape','sha256']):
                mismatches.append(dict(id=case,control=control,path=path_label,tensor=key))
    for case, by_control in grouped.items():
        if 'no_hook' not in by_control:
            missing.append(dict(id=case,reason='no_hook reference unavailable'))
            continue
        for control in controls:
            if control != 'no_hook':
                pair(by_control['no_hook'][0], by_control[control][0], case, control, 'context')
        if method == 'cad':
            prior_reference = next((by_control[c][1] for c in controls if len(by_control[c]) == 2), None)
            for control in controls:
                if len(by_control[control]) == 2:
                    pair(prior_reference, by_control[control][1], case, control, 'prior')
    result.update(status='failed' if mismatches else ('blocked' if missing else 'complete_verified'),
                  completed_calls=len(calls), compared_call_pairs=comparisons,
                  mismatch_count=len(mismatches), missing_evidence_count=len(missing),
                  mismatches=mismatches, missing_evidence=missing)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--run-manifest', type=Path, required=True)
    parser.add_argument('--adapter-root', type=Path, required=True)
    parser.add_argument('--seka-root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--expected-rows', type=int, required=True)
    parser.add_argument('--forward-trace', type=Path)
    args = parser.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path):
            setattr(args,key,value.expanduser().resolve())
    args.output_dir.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    report = dict(status='running', new_gpu_forwards=0, raw_text_published=False,
                  verified_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  scope='CounterFact first-token efficacy adapter records, controls, provenance and observed cost',
                  clinical_accuracy_claim=False, official_sota_reproduction_claim=False,
                  full_vocabulary_logits_independently_recomputed=False)
    code = 0
    try:
        inner_path = args.input_dir/'manifest.json'
        predictions_path = args.input_dir/'predictions.jsonl'
        inner = json.loads(inner_path.read_text())
        outer = json.loads(args.run_manifest.read_text())
        report['raw_files_sha256'] = {str(path):sha(path) for path in [inner_path,predictions_path,args.run_manifest]}
        fingerprint_audit(args,inner,outer,report)
        method, controls = inner['method'], inner['controls']
        require(controls and len(controls) == len(set(controls)), 'Missing or duplicated controls')
        allowed = ['no_hook','matched_zero','normal','reverse']+(['wrong_span'] if method == 'spotlight' else [])
        require(controls == [name for name in allowed if name in controls], 'Unexpected control order or unknown controls')
        report['all_prespecified_controls_present'] = controls == allowed
        by_control = {control:{} for control in controls}
        records, anonymous, numerical = [], [], []
        reference_rows = {}
        with predictions_path.open() as handle:
            for line_index,line in enumerate(handle):
                row = json.loads(line)
                records.append(row)
                case, control, meta = str(row['id']), row['control'], row['metadata']
                require(control in by_control and case not in by_control[control], 'Unknown/duplicated case-control record')
                require(control == controls[line_index % len(controls)], 'Control ordering differs within case')
                target = finite(row['target_logp'],'target log probability')
                comparator = finite(row['comparator_logp'],'comparator log probability')
                require(target <= 1e-6 and comparator <= 1e-6, 'Expected log probabilities <= 0')
                p_target, p_comparator = math.exp(target), math.exp(comparator)
                win = p_target > p_comparator
                require(bool(row['efficacy_correct']) == (target > comparator), 'Saved efficacy_correct differs from scores')
                expected_calls = 2 if method == 'cad' and control != 'no_hook' else 1
                require(row['actual_forward_calls'] == meta['forward_calls'] == expected_calls, 'Recorded forward budget differs')
                require(meta['control'] == control and meta['max_new_tokens'] == 1 and meta['greedy'] is True
                        and meta['generation_seed'] == inner['seed'], 'Per-record control/generation configuration differs')
                token_ids = row['output_token_ids']
                require(isinstance(token_ids,list) and len(token_ids) == meta['output_tokens'] == 1
                        and all(isinstance(token,int) for token in token_ids), 'Expected one saved generated token')
                expected_label = 'CAD_formula_Qwen_adaptation' if method == 'cad' else 'Spotlight_paper_reimplementation_Qwen'
                require(meta['method'] == expected_label, 'Method adaptation label differs')
                if method == 'cad':
                    expected_alpha = 0 if control in ['no_hook','matched_zero'] else (-1 if control == 'reverse' else 1)
                    require(meta['alpha'] == expected_alpha and meta['candidate_filtering'] is False, 'CAD parameter differs')
                    lengths = meta['prompt_tokens_per_path']
                    require(len(lengths) == expected_calls and all(isinstance(n,int) and n>0 for n in lengths), 'CAD path lengths invalid')
                    require(meta['context_sha256'] == row['prompt_sha256'] and meta['prior_sha256'] == row['prior_sha256'], 'CAD prompt hashes differ')
                else:
                    require(meta['target_mass'] == .1 and meta['prompt_sha256'] == row['prompt_sha256'], 'Spotlight target/prompt differs')
                    require(meta['gain'] == (0 if control == 'matched_zero' else (-1 if control == 'reverse' else 1)), 'Spotlight gain differs')
                    require(isinstance(meta['input_tokens'],int) and meta['input_tokens']>0, 'Spotlight input token count invalid')
                record = dict(id=case, control=control, prompt_sha256=row['prompt_sha256'],prior_sha256=row['prior_sha256'],
                    generation_sha256=text_sha(row['prediction']), generation_token_count=len(token_ids),
                    generation_token_ids_sha256=text_sha(json.dumps(token_ids,separators=(',',':'))),
                    target_token_ids_sha256=text_sha(json.dumps(row['target_token_ids'],separators=(',',':'))),
                    comparator_token_ids_sha256=text_sha(json.dumps(row['comparator_token_ids'],separators=(',',':'))),
                    target_logp=target, comparator_logp=comparator, target_probability=p_target, comparator_probability=p_comparator,
                    target_wins=win, probability_margin=p_target-p_comparator, probability_tie=p_target==p_comparator,
                    actual_forward_calls=expected_calls, elapsed_seconds=finite(meta['elapsed_seconds'],'record elapsed',0),
                    peak_allocated_gib=finite(row['peak_allocated_gib'],'record allocated peak',0),
                    peak_reserved_gib=finite(row['peak_reserved_gib'],'record reserved peak',0),
                    prompt_tokens_per_path=meta.get('prompt_tokens_per_path',[meta.get('input_tokens')]))
                if method == 'spotlight':
                    record.update({key:meta[key] for key in ['span_mask_sha256','highlighted_tokens','attention_calls','layers_touched','rows_below_target']})
                if control == 'no_hook':
                    reference_rows[case] = row
                if control == 'matched_zero':
                    diagnostic = row['numerical_control']
                    require(str(diagnostic['id']) == case, 'Numerical diagnostic case differs')
                    finite(diagnostic['max_abs_first_token_score_difference'],'reported full-vocabulary drift',0)
                    require(case in reference_rows, 'Numerical diagnostic lacks preceding no_hook reference')
                    reference_raw = reference_rows[case]
                    require(diagnostic['argmax_changed'] == (token_ids != reference_raw['output_token_ids']), 'Saved argmax diagnostic differs from generated token IDs')
                    record['numerical_control_reported'] = diagnostic
                    record['target_logp_drift_recomputed'] = target-float(reference_raw['target_logp'])
                    record['comparator_logp_drift_recomputed'] = comparator-float(reference_raw['comparator_logp'])
                    if diagnostic['max_abs_first_token_score_difference'] == 0:
                        require(record['target_logp_drift_recomputed'] == record['comparator_logp_drift_recomputed'] == 0
                                and diagnostic['argmax_changed'] is False, 'Reported exact zero inconsistent with available scores/tokens')
                    numerical.append(diagnostic)
                by_control[control][case] = record
                anonymous.append(record)
        export_path = args.output_dir/'predictions.jsonl'
        with export_path.open('x') as handle:
            for row in anonymous:
                handle.write(json.dumps(row,sort_keys=True)+'\n')
        report['anonymous_predictions_sha256'] = sha(export_path)
        report['records_exported'] = len(anonymous)
        require(inner.get('execution_status') == 'finished' and outer.get('process_state') == 'exited' and outer.get('exit_code') == 0,
                'Run failed or is incomplete; anonymous partial records were retained')
        require(inner['expected_examples'] == args.expected_rows, 'Expected source-case count differs')
        require(inner['completed_prediction_records'] == len(records) == args.expected_rows*len(controls), 'Prediction count differs')
        ids = set(by_control[controls[0]])
        require(all(set(rows)==ids and len(rows)==args.expected_rows for rows in by_control.values()), 'Case IDs differ across controls')
        for case in ids:
            rows = [by_control[c][case] for c in controls]
            require(len({r['prompt_sha256'] for r in rows}) == len({r['prior_sha256'] for r in rows}) == 1,
                    'Textual information differs across controls within case')
            require(len({r['target_token_ids_sha256'] for r in rows}) == len({r['comparator_token_ids_sha256'] for r in rows}) == 1,
                    'Scored target token candidates differ across controls')
            if method == 'spotlight' and 'wrong_span' in controls and 'normal' in controls:
                normal, wrong = by_control['normal'][case], by_control['wrong_span'][case]
                require(normal['highlighted_tokens'] == wrong['highlighted_tokens']>0
                        and normal['span_mask_sha256'] != wrong['span_mask_sha256'], 'Wrong-span count/hash diagnostic differs')
        metrics_path = args.input_dir/'efficacy_metrics.json'
        saved_metrics = json.loads(metrics_path.read_text())
        report['raw_files_sha256'][str(metrics_path)] = sha(metrics_path)
        report['recomputed_metrics'] = {}
        for control, cases in by_control.items():
            values = list(cases.values())
            result = dict(score=metric([int(row['target_wins']) for row in values]),
                          magnitude=metric([row['probability_margin'] for row in values]))
            report['recomputed_metrics'][control] = result
            compare_metric(result,saved_metrics[control],control)
        numeric_path = args.input_dir/'numerical_controls.json'
        require(json.loads(numeric_path.read_text()) == numerical, 'Numerical-control JSON differs from per-case records')
        report['raw_files_sha256'][str(numeric_path)] = sha(numeric_path)
        report['numerical_controls'] = dict(records_checked=len(numerical),
            reported_full_vocabulary_max_abs=max((r['max_abs_first_token_score_difference'] for r in numerical),default=None),
            argmax_changes_recomputed=sum(r['argmax_changed'] for r in numerical),
            target_logp_max_abs_recomputed=max((abs(r['target_logp_drift_recomputed']) for r in by_control.get('matched_zero',{}).values()),default=None),
            comparator_logp_max_abs_recomputed=max((abs(r['comparator_logp_drift_recomputed']) for r in by_control.get('matched_zero',{}).values()),default=None),
            full_vocabulary_scope='Recorded max drift checked against both saved diagnostic files; full logits unavailable for independent recomputation',
            wrong_span_scope='Equal token count and differing mask hashes checked; disjointness follows inspected adapter source, full masks were not saved')
        comparisons = {}
        per_case_comparisons = []
        for comparator in controls:
            if comparator == 'normal' or 'normal' not in controls:
                continue
            corrected = regressed = 0
            differences = []
            for case in sorted(ids):
                a, b = by_control['normal'][case], by_control[comparator][case]
                corrected += int(a['target_wins'] and not b['target_wins'])
                regressed += int(b['target_wins'] and not a['target_wins'])
                delta = int(a['target_wins'])-int(b['target_wins'])
                differences.append(delta)
                per_case_comparisons.append(dict(id=case,primary='normal',comparator=comparator,score_difference=delta,
                                                magnitude_difference=a['probability_margin']-b['probability_margin']))
            comparisons['normal_vs_'+comparator] = dict(cases=len(ids),corrected=corrected,regressed=regressed,
                                                        score_difference=math.fsum(differences)/len(ids))
        comparisons_path = args.output_dir/'paired_case_effects.jsonl'
        with comparisons_path.open('x') as handle:
            for row in per_case_comparisons:
                handle.write(json.dumps(row,sort_keys=True)+'\n')
        report['paired_case_effects_sha256'] = sha(comparisons_path)
        report['comparisons'] = comparisons
        forwards = sum(row['actual_forward_calls'] for row in records)
        require(outer['forward_attempts'] == outer['forward_completed'] == forwards, 'Actual outer forward total differs from prediction controls')
        report['costs'] = dict(forward_attempts=outer['forward_attempts'],forward_completed=forwards,
            per_control={control:dict(forwards=sum(r['actual_forward_calls'] for r in rows.values()),
                recorded_call_seconds_sum=math.fsum(r['elapsed_seconds'] for r in rows.values()),
                max_record_peak_allocated_gib=max(r['peak_allocated_gib'] for r in rows.values()),
                max_record_peak_reserved_gib=max(r['peak_reserved_gib'] for r in rows.values())) for control,rows in by_control.items()},
            max_record_peak_allocated_gib=max(r['peak_allocated_gib'] for r in anonymous),
            max_record_peak_reserved_gib=max(r['peak_reserved_gib'] for r in anonymous),
            outer_peak_after_inner_resets_allocated_gib=outer.get('peak_allocated_gib'),
            outer_peak_after_inner_resets_reserved_gib=outer.get('peak_reserved_gib'),
            process_elapsed_seconds=outer.get('elapsed_seconds'),inner_evaluation_elapsed_seconds=inner['elapsed_seconds'],
            forward_input_hashing_seconds=outer.get('forward_input_hashing_seconds'),
            peak_scope='Inner runner resets peaks before each case-control call. Maximum saved call peak is valid for observed calls; outer final peak alone may underestimate earlier calls. Model-load peak before the first inner reset is not recoverable, so whole-process peak is not independently established.',
            elapsed_scope='Outer process time includes load/hash; inner evaluation and per-call times are nested, not additive. Input hashing/synchronization is included, not subtracted as pure model time.')
        report['actual_input_evidence'] = trace_evidence(args,outer,records,controls,method)
        report['information_and_budget'] = dict(batch_size=1,max_new_tokens_per_path=1,seed=inner['seed'],
            same_context_prompt_hash_per_case=True,same_target_candidates_per_case=True,
            context_input='Original official full CounterFact prompt, including old and alternative facts; no answer-dependent clinical selection claim',
            prior_input='Same original question without context, used only by CAD',
            inference_cost='CAD normal/zero/reverse each 2 forwards per case; no_hook 1. Spotlight every control 1. Equal output budget does not imply equal total compute.',
            matched_zero_cost='CAD zero retains both paths; Spotlight zero retains custom attention path',
            applicability='Only CounterFact efficacy interface; neither clinical decision paths nor multi-select generation evaluated')
        report['source_cases'] = len(ids)
        report['status'] = 'complete_verified'
    except BaseException as error:
        code = 1
        report.update(status='failed',error=type(error).__name__+': '+str(error))
        traceback.print_exc()
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        report['checker_sha256'] = sha(Path(__file__).resolve())
        with (args.output_dir/'audit.json').open('x') as handle:
            json.dump(report,handle,indent=2)
            handle.write('\n')
    print(json.dumps({key:report[key] for key in ['status','new_gpu_forwards','raw_text_published','full_vocabulary_logits_independently_recomputed']}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
