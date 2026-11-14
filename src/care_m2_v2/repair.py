"""CPU-only repair/reuse of the archived development run; never loads a model."""
import argparse, csv, hashlib, json, os, shutil
from collections import Counter, defaultdict
from pathlib import Path
import torch
from transformers import AutoTokenizer
from care_m2_inc.io import read_json, read_jsonl, write_json, write_jsonl, sha256_file
from .data import digest
from .candidates import SCORING_CONTRACT, candidate_contract, recover_legacy_top, score_key
from .study import train, spec, controls, correct, primary, GRID, TUNED, METHODS, append_policy, policy

class CachedActivations:
    def __init__(self, tokenizer, stage, rows):
        self.tokenizer = tokenizer; self.stage = Path(stage)
        self.manifest = read_json(self.stage/'run_manifest.json')
        self.allowed = {r['prompt'] for r in rows}
        self.used = {}; self.offsets = {}
    def positions(self, row, prompt=None):
        text = prompt or row['prompt']
        if text not in self.offsets:
            self.offsets[text] = self.tokenizer(text, return_offsets_mapping=True)['offset_mapping']
        return [[i for i, (a, b) in enumerate(self.offsets[text]) if b > a and b > s[0] and a < s[1]]
                for s in row.get('spans', [])]
    def capture(self, prompt, positions=None):
        if prompt not in self.allowed: raise ValueError('offline fit attempted nontrain capture')
        key = digest([prompt, positions]); name = 'raw/activation_' + key + '.pt'
        path = self.stage/name
        if not path.is_file(): raise ValueError('missing cached activation; no model fallback permitted')
        expected = self.manifest['outputs'][name]
        if sha256_file(path) != expected: raise ValueError('cached activation hash mismatch')
        self.used[str(path)] = expected
        return torch.load(path, map_location='cpu', weights_only=True)

def model_spec(edits, biases):
    return {'edits': [{k: digest(v.detach().cpu().tolist()) if torch.is_tensor(v) else v
                       for k, v in e.items()} for e in edits if e['beta'] != 0],
            'biases': [(p, b) for p, b in biases if b != 0]}

def repaired_prompt(row, record):
    if record['method'] == 'text': return append_policy(row['prompt'], policy(row, record['control']))
    if record['method'] == 'neutral_text':
        return append_policy(row['prompt'], 'First enumerate completions of missing facts. Decide only when every completion gives the same action; otherwise ask a decisive attribute.')
    if record['method'] == 'filter':
        prompt = row['prompt']
        for s in row['signals']:
            if s['state'] == 'F': prompt = '\n'.join(line for line in prompt.splitlines() if not line.startswith(s['rule_id'] + ':'))
        return prompt
    return row['prompt']

def run(data, previous, model, out):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('run with CUDA_VISIBLE_DEVICES empty')
    torch.set_num_threads(2)
    data = Path(data); previous = Path(previous); out = Path(out); out.mkdir(parents=True, exist_ok=False)
    manifest = read_json(data/'manifest.json')
    for name, h in manifest['files'].items(): assert sha256_file(data/name) == h
    inputs = read_jsonl(data/'inputs.jsonl')
    dev_ids = {r['case_id'] for r in inputs if r['split'] == 'dev'}
    labels = {r['case_id']: r for r in read_jsonl(data/'labels.jsonl') if r['case_id'] in dev_ids}
    dev = [{**r, **labels[r['case_id']]} for r in inputs if r['split'] == 'dev']
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    token_files = {p.name: sha256_file(p) for p in Path(model).glob('*token*') if p.is_file()}
    # Establish that each recovered ASCII continuation identifies exactly one vocab ID.
    decoded = tok.batch_decode([[i] for i in range(len(tok))], skip_special_tokens=False)
    counts = Counter(decoded)
    unique = {text: i for i, text in enumerate(decoded) if counts[text] == 1 and text in [' '+x for x in 'ABCDEFGH']}
    boundary_checks = 0
    for r in inputs:
        candidate_contract(tok, r['prompt'], r['options'])
        assert tok.encode(r['prompt']) == tok.encode(r['prompt'], add_special_tokens=False)
        boundary_checks += 1
    write_json(out/'tokenizer_validation.json', {'scoring_contract': SCORING_CONTRACT,
        'base_prompts_checked': boundary_checks, 'tokenizer_files': token_files,
        'unique_candidate_decodings': unique, 'model_loaded': False, 'forward_count': 0})
    source_hashes = {}; original_snapshot = {str(p): sha256_file(p) for p in previous.rglob('*') if p.is_file()}
    tasks = ['N1', 'N2', 'N3', 'N5']; stages = {}; caches = {}; all_replays = []; grid = []
    for task in tasks:
        stage = previous/task; target = out/'reuse'/task; (target/'raw').mkdir(parents=True)
        run_manifest = read_json(stage/'run_manifest.json')
        for name in ['axes.pt', 'train_fit.json', 'dev_predictions.jsonl', 'data_manifest.json', 'raw/forward_journal.jsonl', 'raw/model_fingerprint.json']:
            path = stage/name; h = sha256_file(path)
            assert h == run_manifest['outputs'][name], ('source hash mismatch', path)
            source_hashes[str(path)] = h
        fp = read_json(stage/'raw/model_fingerprint.json')
        assert all(fp['files'][name] == h for name, h in token_files.items())
        modelhash = digest(fp['files'])
        tr = [r for r in inputs if r['task'] == task and r['split'] == 'train']
        cpu = CachedActivations(tok, stage, tr)
        if task == 'N5':
            axes = train(cpu, tr, target)
            source_hashes.update(cpu.used)
            old = torch.load(stage/'axes.pt', map_location='cpu', weights_only=True)['A1']
            new = axes['A1']
            audit = read_json(target/'n5_pair_audit.json')
            write_json(target/'offline_refit.json', {'forward_count': 0, 'model_loaded': False,
                'source_model_hash': modelhash, 'source_git': fp['git'], 'cached_activations': cpu.used,
                'source_axes_sha256': sha256_file(stage/'axes.pt'), 'new_axes_sha256': sha256_file(target/'axes.pt'),
                'matched_pairs': len(audit['pairs']), 'status_counts': audit['status_counts'],
                'old_difference_mean_centroid_error': float((old['differences'].mean(0)-old['mean']).norm()),
                'new_difference_mean_centroid_error': float((new['differences'].mean(0)-new['mean']).norm()),
                'uses_labels': False, 'evaluation_population_changed': False})
        else:
            for name in ['axes.pt', 'train_fit.json']: shutil.copyfile(stage/name, target/name)
            axes = torch.load(target/'axes.pt', map_location='cpu', weights_only=True)
        shutil.copyfile(stage/'data_manifest.json', target/'data_manifest.json')
        lookup = {r['case_id']: r for r in dev if r['task'] == task}
        old_predictions = read_jsonl(stage/'dev_predictions.jsonl')
        needed = {(p['prompt_sha256'], digest(p['spec'])) for p in old_predictions if 'spec' in p}
        sources = {}
        for record in read_jsonl(stage/'raw/forward_journal.jsonl'):
            assert record['checksum'] == digest(record['payload'])
            p = record['payload']
            if record['kind'] == 'score' and (p['prompt_sha256'], digest(p['spec'])) in needed:
                sources[(p['prompt_sha256'], digest(p['spec']))] = record
        cache = {}; stage_replays = []
        for p in old_predictions:
            r = lookup[p['case_id']]
            result = {k: p[k] for k in ('case_id', 'family_id', 'method', 'beta', 'control')}
            result.update(task=task, legacy_prediction=p['prediction'], legacy_correct=p['correct'])
            if 'diagnostic' in p:
                result.update(status='deterministic_reuse', prediction=p['prediction'], correct=p['correct'])
            elif task == 'N5' and p['spec']['edits']:
                result.update(status='pending_new_N5_axis', prediction=None, correct=None)
            else:
                prompt = repaired_prompt(r, p)
                assert hashlib.sha256(prompt.encode()).hexdigest() == p['prompt_sha256']
                candidates = candidate_contract(tok, prompt, r['options'])
                record = sources[(p['prompt_sha256'], digest(p['spec']))]; original = record['payload']
                assert original['candidate_logits'] == p['candidate_logits'] and original['top_unconstrained_token'] == p['top_unconstrained_token']
                recovered = recover_legacy_top(original, candidates, unique)
                if recovered is None:
                    result.update(status='pending_non_candidate_global_top', prediction=None, correct=None)
                else:
                    recovered['reuse_provenance'] = {'source_journal': str(stage/'raw/forward_journal.jsonl'),
                        'source_journal_sha256': source_hashes[str(stage/'raw/forward_journal.jsonl')],
                        'source_key': record['key'], 'source_payload_sha256': record['checksum'],
                        'tokenizer_validation_sha256': sha256_file(out/'tokenizer_validation.json'),
                        'new_model_forward_count': 0}
                    key = score_key(modelhash, prompt, r['options'], original['spec'], candidates)
                    cache[key] = recovered
                    result.update(status='exact_winner_reused', prediction=recovered['prediction'],
                                  correct=correct(r, p['control'], recovered['prediction']), corrected_cache_key=key)
            stage_replays.append(result)
        write_jsonl(target/'raw/forward_journal.jsonl', [{'kind': 'score', 'key': k, 'payload': v, 'checksum': digest(v)} for k, v in sorted(cache.items())])
        write_jsonl(target/'dev_replay.jsonl', stage_replays)
        for method, beta in sorted({(p['method'], p['beta']) for p in stage_replays}):
            rs = [p for p in stage_replays if p['method'] == method and p['beta'] == beta]
            known = [p for p in rs if p['correct'] is not None]
            grid.append({'task': task, 'method': method, 'beta': beta, 'total': len(rs), 'recovered': len(known),
                'correct_known': sum(p['correct'] for p in known),
                'primary': primary(task, known) if len(known) == len(rs) else None,
                'pending': len(rs)-len(known)})
        stages[task] = (cpu, axes, modelhash, list(lookup.values())); caches[task] = cache
        all_replays.extend(stage_replays)
    write_jsonl(out/'dev_replay.jsonl', all_replays)
    with (out/'dev_grid.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(grid[0])); w.writeheader(); w.writerows(grid)
    # Expand precisely the missing main-grid requests, then reserve only controls
    # whose inherited route strength changes. No test or closed-book task is used.
    pending = {}; budget = {}; proposed = {}
    for task, (cpu, axes, modelhash, rows) in stages.items():
        chosen = {}; main_keys = set(); random_by_beta = {}
        for method in METHODS[task]:
            gs = [g for g in grid if g['task'] == task and g['method'] == method]
            if gs and all(g['pending'] == 0 for g in gs):
                chosen[method] = max(gs, key=lambda g: (g['primary'], -g['beta']))['beta']
        for method in METHODS[task]:
            if method not in TUNED and method not in ('baseline', 'text', 'filter', 'neutral_text'): continue
            for beta in GRID if method in TUNED else [0.]:
                for r in rows:
                    for c in controls(r):
                        prompt, edits, biases = spec(cpu, r, c, method, beta, axes)
                        sp = model_spec(edits, biases); cs = candidate_contract(tok, prompt, r['options'])
                        key = score_key(modelhash, prompt, r['options'], sp, cs)
                        if key not in caches[task]:
                            main_keys.add(key)
                            pending[key] = {'task': task, 'case_id': r['case_id'], 'control': c, 'method': method, 'beta': beta,
                                            'reason': 'new_N5_axis' if task == 'N5' and sp['edits'] else 'missing_candidate_winner'}
        route_betas = [chosen['route']] if 'route' in chosen else GRID
        for beta in route_betas:
            keys = set()
            for r in rows:
                for c in controls(r):
                    prompt, edits, biases = spec(cpu, r, c, 'random', beta, axes)
                    sp = model_spec(edits, biases); cs = candidate_contract(tok, prompt, r['options'])
                    key = score_key(modelhash, prompt, r['options'], sp, cs)
                    if key not in caches[task] and key not in main_keys: keys.add(key)
            random_by_beta[str(beta)] = len(keys)
        budget[task] = {'main_dev_forwards': len(main_keys), 'random_control_by_route_beta': random_by_beta,
                        'maximum_total_dev_forwards': len(main_keys) + max(random_by_beta.values(), default=0),
                        'training_forwards': 0, 'test_forwards': 0}
        independent = {m: b for m, b in chosen.items() if m in TUNED or m in ('baseline', 'text', 'filter', 'neutral_text', 'oracle', 'any_unknown')}
        proposed[task] = {'complete_independent_methods': independent,
                          'dependent_control_strength': chosen.get('route'), 'formal_selection_written': False}
    write_jsonl(out/'pending_dev_requests.jsonl', list(pending.values()))
    write_json(out/'partial_selections.json', proposed)
    write_json(out/'budget.json', {'status': 'proposed_not_launched', 'by_task': budget,
        'smoke_forwards': 6, 'smoke_activation_reuse': str(previous/'N1/raw'),
        'main_dev_forwards': sum(x['main_dev_forwards'] for x in budget.values()),
        'max_dev_forwards': sum(x['maximum_total_dev_forwards'] for x in budget.values()),
        'max_total_forwards_including_smoke': 6 + sum(x['maximum_total_dev_forwards'] for x in budget.values()),
        'test_forwards': 0, 'training_forwards': 0,
        'gpu_count': 1, 'preferred_gpu': 0, 'estimated_peak_GiB': 8, 'reserve_GiB': 10,
        'wall_time_budget_minutes': 10, 'GPU_hour_budget': 1/6})
    after = {str(p): sha256_file(p) for p in previous.rglob('*') if p.is_file()}
    assert after == original_snapshot, 'archived experiment changed'
    write_json(out/'provenance.json', {'source_files': source_hashes, 'source_snapshot': original_snapshot,
        'source_code_sha256': {str(p): sha256_file(p) for p in Path('src/care_m2_v2').glob('*.py')},
        'source_snapshot_unchanged': True, 'source_file_count': len(original_snapshot),
        'forward_count': 0, 'model_loaded': False, 'test_outputs_used_for_selection': False,
        'status_counts': dict(Counter(x['status'] for x in all_replays)),
        'corrected_score_cache_counts': {t: len(c) for t, c in caches.items()},
        'new_files': {str(p.relative_to(out)): sha256_file(p) for p in out.rglob('*') if p.is_file()}})
    return {'forward_count': 0, 'status_counts': dict(Counter(x['status'] for x in all_replays)),
            'budget': read_json(out/'budget.json')}

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ['data', 'previous', 'model', 'out']: p.add_argument('--'+flag, required=True)
    args = p.parse_args()
    print(json.dumps(run(args.data, args.previous, args.model, args.out), ensure_ascii=False, indent=2))
