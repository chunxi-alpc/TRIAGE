"""Replay unchanged, metadata-matched TrialMatchAI outputs through frozen V5 NLI.

No base/adapter model is loaded and no labels are read. ``--merged`` contains the
original private precomputed rows; ``--source-status`` supplies actual generator
identity and timing provenance (repeat for the archived serial run and workers).
Per-sample allocated GPU work is distinct from the replay/parallel wall clock.
"""
from __future__ import annotations
import argparse, collections, copy, hashlib, json, math, os, pathlib, sys, time, traceback
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.module1.care_v5 import CAREModule1V5, BioLinkBERTMedNLI, _patient_text


def load_jsonl(path):
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def file_digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def public_span(row):
    return {key: row[key] for key in ('sentence_id', 'start', 'end', 'nli_relation', 'class_id', 'scores', 'evidence_source') if key in row}


def atomic_write(path, value):
    pending = path.with_name(path.name + '.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    pending.replace(path)


class StrictReplay:
    """Consume each condition in the frozen sample/condition iteration order."""
    def __init__(self, inputs, rules, merged):
        self.by_key = {}
        for row in merged:
            key = tuple(row.get(k) for k in ('sample_id', 'rule_id', 'condition_id', 'criterion_kind'))
            if any(value is None for value in key) or key in self.by_key:
                raise ValueError('Merged TrialMatchAI rows have missing or duplicate four-part identities')
            output = row.get('trialmatch')
            if not isinstance(output, dict) or not isinstance(output.get('raw_output'), str):
                raise ValueError('Merged row lacks unchanged original TrialMatchAI raw output')
            if output.get('state') not in ('T', 'F', 'U'):
                raise ValueError('Merged TrialMatchAI output has invalid state')
            for name in ('inference_seconds', 'input_tokens', 'output_tokens', 'generation_batch_size', 'timing_method', 'truncated'):
                if name not in output:
                    raise ValueError('Merged TrialMatchAI output missing original field: ' + name)
            if not isinstance(output['inference_seconds'], (int, float)) or not math.isfinite(output['inference_seconds']) or output['inference_seconds'] < 0:
                raise ValueError('Invalid original TrialMatchAI inference time')
            self.by_key[key] = row
        expected = []
        for sample in inputs:
            rule = rules[sample['rule_id']]
            for kind, group in (('inclusion', rule['conditions']), ('exclusion', rule.get('exceptions', []))):
                expected.extend((sample['sample_id'], rule['rule_id'], atom['condition_id'], kind) for atom in group)
        if len(expected) != len(set(expected)) or set(expected) != set(self.by_key):
            raise ValueError('Merged rows do not exactly cover frozen sample/rule/condition/kind requests')
        self.remaining = len(expected)
        self.queue = collections.deque()
        self.active_sample = None

    def select_sample(self, sample, rule):
        if self.queue:
            raise RuntimeError('Previous replay sample has unconsumed condition outputs')
        self.active_sample = sample['sample_id']
        patient = _patient_text(sample['sentences'])
        for kind, group in (('inclusion', rule['conditions']), ('exclusion', rule.get('exceptions', []))):
            for atom in group:
                key = sample['sample_id'], rule['rule_id'], atom['condition_id'], kind
                row = self.by_key[key]
                condition_hash=digest(json.dumps(atom,ensure_ascii=False,sort_keys=True,separators=(',',':')))
                for name, expected in (('patient_sha256', digest(patient)), ('condition_sha256', condition_hash)):
                    if name in row and row[name] != expected:
                        raise ValueError('Merged request text fingerprint mismatch: ' + name)
                if 'patient' in row and row['patient'] != patient:
                    raise ValueError('Merged original patient differs from frozen input')
                if 'condition' in row:
                    text = row['condition'].get('text') if isinstance(row['condition'], dict) else row['condition']
                    if text != atom['text']:
                        raise ValueError('Merged original criterion differs from frozen rule')
                output = row['trialmatch']
                if output.get('criterion_kind') not in (None, kind):
                    raise ValueError('Merged output criterion kind mismatch')
                parsed = output.get('parsed_response') or {}
                target = parsed.get('Inclusion_Criteria_Evaluation' if kind == 'inclusion' else 'Exclusion_Criteria_Evaluation')
                if target and (len(target) != 1 or target[0].get('Criterion', '').strip() != atom['text'].strip()):
                    raise ValueError('Parsed original criterion does not match frozen atom')
                self.queue.append((patient, atom['text'], atom['condition_id'], kind, output))

    def judge(self, patient, condition, criterion_kind='inclusion'):
        if not self.queue:
            raise RuntimeError('Replay requested absent or already consumed output; no fresh inference fallback')
        old_patient, text, cid, kind, output = self.queue.popleft()
        if (patient, condition['text'], condition['condition_id'], criterion_kind) != (old_patient, text, cid, kind):
            raise RuntimeError('Core request order differs from strict replay queue')
        self.remaining -= 1
        return copy.deepcopy(output)


def source_provenance(paths, manifest_path):
    rows = []
    for path in paths:
        value = json.loads(pathlib.Path(path).read_text())
        rows.append({'path': str(path), 'sha256': file_digest(path), 'status': value})
    manifest = json.loads(pathlib.Path(manifest_path).read_text()) if manifest_path else None
    worker_checks = []
    for value in (manifest or {}).get('parallel_workers', []):
        if value.get('status') != 'completed' or not value.get('active_adapters'):
            raise ValueError('Manifest contains incomplete worker or missing active adapter verification')
        if value.get('loaded_peft_base_model_names') and any(name != 'microsoft/phi-4' for name in value['loaded_peft_base_model_names']):
            raise ValueError('Actual worker PEFT base identity differs from required Phi-4')
        worker_checks.append({key:value[key] for key in ('rank','device','gpu_name','active_adapters',
            'base_model_id','adapter_model_id','loaded_peft_base_model_names','parameter_count',
            'trainable_parameter_count','load_seconds','elapsed_seconds','start_utc','end_utc',
            'peak_allocated_gib','peak_reserved_gib') if key in value})
    identities = []
    for row in rows:
        value = row['status']
        adapters = value.get('trialmatch_active_adapters', value.get('active_adapters'))
        count = value.get('trialmatch_lora_parameter_count', value.get('lora_parameter_count'))
        if adapters and isinstance(count, int) and count > 0:
            identities.append({'source_status': row['path'], 'active_adapters': adapters,
                               'lora_parameter_count': count, 'models': value.get('models'),
                               'device': value.get('device'), 'gpu_name': value.get('gpu_name')})
    if not identities:
        raise ValueError('No actual generator status verifies active TrialMatchAI LoRA and parameter count')
    counts = {identity['lora_parameter_count'] for identity in identities}
    if len(counts) != 1:
        raise ValueError('Generator statuses disagree on actual LoRA parameter count')
    return rows, identities, manifest, worker_checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', default='/home/gcx/care_module1_20261018_v4/data/private/inputs.jsonl')
    parser.add_argument('--rules', default=str(ROOT/'experiments/care_module1_v5/data/rules_v5.jsonl'))
    parser.add_argument('--merged', required=True)
    parser.add_argument('--source-status', action='append', required=True)
    parser.add_argument('--parallel-manifest')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:6')
    parser.add_argument('--evidence-model', default='/home/gcx/models/care_module1_v5/biolinkbert-mednli')
    parser.add_argument('--cache', default='/home/gcx/.cache/huggingface')
    parser.add_argument('--nli-batch-size', type=int, default=32)
    parser.add_argument('--expected-count', type=int, default=192)
    args = parser.parse_args()
    out = pathlib.Path(args.output); private = out/'private'
    out.mkdir(parents=True, exist_ok=True); private.mkdir(exist_ok=False)
    started = time.perf_counter()
    status = {'status': 'starting_replay', 'start_utc': datetime.now(timezone.utc).isoformat(),
        'device': args.device, 'inputs': args.inputs, 'sample_count': args.expected_count,
        'models': ['microsoft/phi-4 + majdabd33/trialmatchai-phi4-reasoning-lora (unchanged cached generation)', 'cnut1648/biolinkbert-mednli'],
        'trialmatch_loaded_in_replay': False, 'qwen_called': False,
        'timing_note': 'elapsed_seconds measures this replay stage only. Shared generation seconds are summed allocated GPU work, not parallel wall clock.',
        'merged_file': args.merged, 'merged_sha256': file_digest(args.merged)}
    atomic_write(out/'run_status.json', status)
    try:
        inputs, rule_rows, merged = load_jsonl(args.inputs), load_jsonl(args.rules), load_jsonl(args.merged)
        rules = {row['rule_id']: row for row in rule_rows}
        if len(inputs) != args.expected_count or len(rules) != len(rule_rows) or len({x['sample_id'] for x in inputs}) != len(inputs):
            raise ValueError('Frozen sample count or sample/rule identities invalid')
        replay = StrictReplay(inputs, rules, merged)
        sources, identities, manifest, worker_checks = source_provenance(args.source_status, args.parallel_manifest)
        if manifest and manifest.get('output_sha256') and manifest['output_sha256'] != status['merged_sha256']:
            raise ValueError('Merged file differs from validated parallel merge manifest')
        status.update(generator_status_sources=sources, trialmatch_generator_identities=identities,
            trialmatch_active_adapters=identities[0]['active_adapters'],
            trialmatch_lora_parameter_count=identities[0]['lora_parameter_count'],
            trialmatch_identity_source='copied from actual generator statuses; not loaded by replay',
            parallel_worker_active_adapter_checks=worker_checks,
            parallel_worker_lora_parameter_count_note='Not measured separately by worker; actual serial LoRA count retained as provenance, never inferred from total worker parameters.',
            trialmatch_condition_count=len(merged), trialmatch_conditions_completed=len(merged),
            trialmatch_shared_generation_seconds=sum(row['trialmatch']['inference_seconds'] for row in merged),
            trialmatch_generation_gpu_seconds=sum(row['trialmatch']['inference_seconds'] for row in merged),
            trialmatch_parse_errors=dict(collections.Counter(row['trialmatch']['parse_error'] for row in merged if row['trialmatch'].get('parse_error'))),
            trialmatch_truncated_conditions=sum(bool(row['trialmatch']['truncated']) for row in merged),
            trialmatch_output_tokens=sum(row['trialmatch']['output_tokens'] for row in merged),
            parallel_manifest=manifest, parallel_manifest_path=args.parallel_manifest,
            discarded_serial_batch='Only flushed completed rows enter merged output; interrupted/unflushed batch is excluded. Its elapsed work is not estimated here.')
        if manifest is not None:
            status['parallel_manifest_sha256'] = file_digest(args.parallel_manifest)
            # Preserve measured controller clocks exactly; never infer wall time
            # from summing overlapping worker process or GPU durations.
            for key in ('parallel_wall_seconds', 'parallel_wallclock_seconds', 'wall_clock_seconds',
                        'serial_completed_conditions', 'serial_reused_conditions', 'discarded_batch_elapsed_seconds',
                        'aggregate_worker_gpu_seconds', 'worker_gpu_seconds', 'start_utc', 'end_utc'):
                if key in manifest:
                    status['controller_' + key] = manifest[key]
            if worker_checks:
                starts=[datetime.fromisoformat(row['start_utc']) for row in worker_checks if row.get('start_utc')]
                ends=[datetime.fromisoformat(row['end_utc']) for row in worker_checks if row.get('end_utc')]
                if len(starts)==len(worker_checks)==len(ends):
                    status['parallel_workers_wallclock_seconds']=(max(ends)-min(starts)).total_seconds()
                    status['parallel_wallclock_definition']='Latest worker completion minus earliest worker process start; includes loading; actual recorded UTC timestamps.'
            for source_name in sorted({row.get('source','unspecified') for row in merged}):
                selected=[row for row in merged if row.get('source','unspecified')==source_name]
                status.setdefault('generation_by_source',{})[source_name]={
                    'conditions':len(selected),'allocated_gpu_seconds':sum(row['trialmatch']['inference_seconds'] for row in selected)}
        import torch
        if not torch.cuda.is_available() or not args.device.startswith('cuda:'):
            raise RuntimeError('Replay evidence stage requires the requested CUDA GPU; no CPU fallback')
        gpu_index = int(args.device.split(':', 1)[1]); torch.cuda.set_device(gpu_index)
        torch.cuda.reset_peak_memory_stats(gpu_index)
        status.update(gpu_name=torch.cuda.get_device_name(gpu_index),
            gpu_total_memory_gib=round(torch.cuda.get_device_properties(gpu_index).total_memory/1024**3, 3),
            versions={'python': sys.version.split()[0], 'torch': torch.__version__, 'transformers': __import__('transformers').__version__})
        t0 = time.perf_counter()
        nli = BioLinkBERTMedNLI.from_pretrained(args.device, cache_dir=args.cache, model_name=args.evidence_model)
        status.update(mednli_load_seconds=round(time.perf_counter()-t0, 3),
            mednli_id2label={str(k): v for k,v in nli.id2label.items()}, status='evaluating_evidence')
        engine = CAREModule1V5(replay, nli, nli_batch_size=args.nli_batch_size)
        atomic_write(out/'run_status.json', status)
        with (private/'predictions_private.jsonl').open('w') as pf, (out/'predictions_public.jsonl').open('w') as pub:
            for index, sample in enumerate(inputs, 1):
                t0 = time.perf_counter(); rule = rules[sample['rule_id']]; replay.select_sample(sample, rule)
                graph = engine.run([rule], sample['sentences'], question=rule['text'])
                prediction = graph['rules'][0]; condition_private=[]; condition_public=[]
                for condition in prediction['conditions'] + prediction['exceptions']:
                    tm = condition['trialmatch']; completion = tm['raw_output']
                    lookup = condition['trialmatch_seconds']; condition['trialmatch_cache_lookup_seconds'] = lookup
                    condition['trialmatch_seconds'] = tm['inference_seconds']; condition['trialmatch_timing_method'] = tm['timing_method']
                    condition_private.append({'condition_id': condition['condition_id'], 'raw_trialmatch_output': completion, 'nli_candidates': condition['nli_candidates']})
                    item = {'condition_id': condition['condition_id'], 'trialmatch_raw_label': tm.get('raw_label'),
                        'trialmatch_state': tm['state'], 'trialmatch_parse_error': tm.get('parse_error'),
                        'trialmatch_truncated': tm['truncated'], 'trialmatch_input_tokens': tm['input_tokens'],
                        'trialmatch_output_tokens': tm['output_tokens'], 'trialmatch_generation_batch_id': tm.get('generation_batch_id'),
                        'trialmatch_generation_batch_size': tm['generation_batch_size'], 'trialmatch_timing_method': tm['timing_method'],
                        'trialmatch_cache_lookup_seconds': lookup, 'trialmatch_seconds': tm['inference_seconds'],
                        'numeric_seconds': condition['numeric_seconds'], 'evidence_selection_seconds': condition['evidence_selection_seconds'],
                        'mednli_seconds': condition['mednli_seconds'], 'trialmatch_output_sha256': digest(completion),
                        'state': condition['status'], 'nli_evidence_state': condition['evidence_state'],
                        'resolution': condition['resolution'], 'conflict': condition['conflict'],
                        'support_spans': [public_span(x) for x in condition['support_evidence']],
                        'refute_spans': [public_span(x) for x in condition['refute_evidence']],
                        'unknown_spans': [public_span(x) for x in condition['unknown_evidence']],
                        'nli_candidate_scores': [{'sentence_id': x['sentence_id'], 'start': x['start'], 'end': x['end'],
                            'relation': x['nli_relation'], 'class_id': x['class_id'], 'scores': x['scores'],
                            'skipped_reason': x.get('skipped_reason')} for x in condition['nli_candidates']]}
                    numeric = condition.get('numeric') or {}
                    item['numeric_status'] = numeric.get('status', 'U')
                    item['numeric_evidence_spans'] = [{'sentence_id': x['sentence_id'], 'start': x['start'], 'end': x['end']} for x in numeric.get('evidence', [])]
                    condition_public.append(item)
                postprocess = time.perf_counter()-t0; allocated = sum(c['trialmatch_seconds'] for c in condition_public)
                graph['metadata'].update(trialmatch_timing_method='original shared generation GPU-time allocation; not parallel wall clock',
                    postprocess_seconds=postprocess, trialmatch_seconds=allocated, elapsed_seconds=postprocess+allocated)
                private_row = {'sample_id': sample['sample_id'], 'rule_id': sample['rule_id'],
                    'patient_sentences': sample['sentences'], 'trialmatch_and_evidence': condition_private, 'module1_output': graph}
                public_row = {'sample_id': sample['sample_id'], 'rule_id': sample['rule_id'],
                    'trialmatch_applicability': prediction['trialmatch_applicability'],
                    'trialmatch_model_only_applicability': prediction['trialmatch_model_only_applicability'],
                    'applicability': prediction['applicability'], 'uncertain': prediction['uncertain'], 'conditions': condition_public,
                    'elapsed_seconds': round(postprocess+allocated, 6), 'postprocess_seconds': postprocess,
                    'timing_method': 'postprocess wall time plus allocated original generation GPU work; not parallel wall clock',
                    'trialmatch_seconds': allocated, 'numeric_seconds': sum(c['numeric_seconds'] for c in condition_public),
                    'evidence_selection_seconds': sum(c['evidence_selection_seconds'] for c in condition_public),
                    'mednli_seconds': sum(c['mednli_seconds'] for c in condition_public)}
                pf.write(json.dumps(private_row, ensure_ascii=False)+'\n'); pub.write(json.dumps(public_row, ensure_ascii=False)+'\n')
                if index % 8 == 0:
                    pf.flush(); pub.flush(); os.fsync(pf.fileno()); os.fsync(pub.fileno())
                    status['sample_count_completed']=index; atomic_write(out/'run_status.json', status)
            if replay.queue or replay.remaining:
                raise RuntimeError('Replay finished with unconsumed original outputs')
        status.update(status='completed', sample_count_completed=len(inputs),
            peak_allocated_gib=round(torch.cuda.max_memory_allocated(gpu_index)/1024**3, 3),
            peak_reserved_gib=round(torch.cuda.max_memory_reserved(gpu_index)/1024**3, 3))
    except Exception as exc:
        status.update(status='failed', error_type=type(exc).__name__, error=str(exc)[:2000], traceback=traceback.format_exc(limit=12))
    finally:
        status.update(end_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=round(time.perf_counter()-started, 3))
        status['replay_wall_seconds'] = status['elapsed_seconds']
        atomic_write(out/'run_status.json', status)
        print(json.dumps({key:status.get(key) for key in ('status','sample_count_completed','error_type','error','replay_wall_seconds')}, ensure_ascii=False))
    if status['status'] != 'completed': raise SystemExit(2)


if __name__ == '__main__': main()
