"""Shared frozen inputs, source-paired statistics, and resumable output helpers."""
from pathlib import Path
from collections import defaultdict
import json
import os
import numpy as np

PROJECT = Path('/home/gcx/aster-final-gap-completion-20261114')
OUT = PROJECT / 'reports/aster_final_gap_completion_20261114'
CODE = PROJECT / 'experiments/aster_final_gap_completion_20261114'
AUDIT = Path('/home/gcx/aster-final-missing-audit-20261114')
REPO6 = Path('/home/gcx/aster-e678-gaps-20261112')
REPO5 = Path('/home/gcx/aster-e5-gaps-20261112')
R = REPO6 / 'reports/aster_e678_20261112'
E5 = REPO5 / 'reports/aster_e5_gaps_20261112'
Q = Path('/home/gcx/aster-cr-retrain-paired-v2-20261111/reports/aster_cr_retrain_paired_v2_20261111')
F = Path('/home/gcx/aster-cr-balanced-publish-20261112/reports/aster_cr_final_method_10006_20261112')
D = Path('/home/gcx/aster-cr-diagnostics-e3cv-20261111/reports/aster_cr_m3_intercept_diagnostics_20261112')
EXT = Path('/home/gcx/aster-codex03-external-20261112/reports/aster_codex03_external_20261112')
HIST = Path('/home/gcx/aster_cr_repair_v2_run_20261108')

def iterrows(path):
    path = Path(path)
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def rows(path):
    return list(iterrows(path))

def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)

def lines(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temp.open('w') as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)

def append(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')

def pathkey(row):
    variant = row.get('variant', 'original')
    return row['case_id'], 'original' if variant == 'full' else variant

def paired_binary(records, baseline, key, seed=11):
    reference = {key(r): r for r in baseline}
    if len({key(r) for r in records}) != len(records):
        raise ValueError('Duplicate evaluation unit')
    if any(key(r) not in reference for r in records):
        raise ValueError('Missing paired raw evaluation unit')
    sources = sorted({r['source_id'] for r in records})
    groups = {s: np.zeros(5, float) for s in sources}
    for row in records:
        base = reference[key(row)]
        if base['source_id'] != row['source_id']:
            raise ValueError('Paired source mismatch')
        ok, raw = bool(row['correct']), bool(base['correct'])
        groups[row['source_id']] += [1, ok, raw, ok and not raw, raw and not ok]
    values = np.asarray([groups[s] for s in sources])
    total = values.sum(0)
    weights = np.random.default_rng(seed).multinomial(len(sources), np.full(len(sources), 1 / len(sources)), size=2000)
    draws = weights @ values
    def ci(v):
        return np.quantile(v, [.025, .975]).tolist()
    return dict(n=int(total[0]), sources=len(sources), correct=int(total[1]),
                accuracy=float(total[1] / total[0]), accuracy_ci=ci(draws[:, 1] / draws[:, 0]),
                raw_correct=int(total[2]), difference_vs_raw=float((total[1] - total[2]) / total[0]),
                difference_vs_raw_ci=ci((draws[:, 1] - draws[:, 2]) / draws[:, 0]),
                corrected_vs_raw=int(total[3]), new_errors_vs_raw=int(total[4]),
                corrected_count_ci=ci(draws[:, 3]), new_error_count_ci=ci(draws[:, 4]),
                bootstrap=dict(draws=2000, seed=seed, unit='source_id', paired=True, interval='percentile 95%'))

def path_stats(records, raw, seed=11):
    result = {}
    for interference, n in [(False, 806), (True, 2418)]:
        rr = [r for r in records if (pathkey(r)[1] != 'original') == interference]
        if not rr:
            continue
        if len(rr) != n or len({r['source_id'] for r in rr}) != 40:
            raise ValueError('Incomplete controlled path population')
        result['interference' if interference else 'natural'] = paired_binary(rr, raw, pathkey, seed)
    return result

def e5_source(method):
    if '_seed' in method:
        kind, seed = method.split('_seed')
        lr = '0.001' if kind == 'lora' else '0.0001'
        sub = 'loreft_e1' if kind == 'loreft' else 'native_supervised_e1/' + kind
        return E5 / sub / ('lr' + lr + '_seed' + seed)
    if method in ['CC', 'strong_evidence_prompt']:
        return E5 / 'prompt_controls_e1' / method
    return E5 / 'representation_e1_plan' / method

class E5Data:
    def __init__(self, prompts=True):
        from experiments.aster_closed_form.data import canonical_guide
        self.views = [v for v in iterrows(Q / 'data/historical_views.jsonl') if v['split'] == 'test']
        self.alias = {v['original_view_id']: v['uid'] for v in self.views if 'original_view_id' in v}
        self.path_views = [v for v in iterrows(Q / 'data/path_interference_views.jsonl') if v['split'] == 'test']
        self.ids = {v['uid'] for v in self.path_views}
        self.groups = defaultdict(dict)
        for v in self.path_views:
            self.groups[v['case_id'], v['variant']][v['gid']] = v['uid']
        self.cases = {r['case_id']: r for r in iterrows(HIST / 'data/path_cases.jsonl') if r['split'] == 'test'}
        self.trees = {}
        for cid, case in self.cases.items():
            tree = canonical_guide(case['guideline'])
            self.trees[cid] = ({n['id']: n for n in tree['nodes']}, tree['nodes'][0]['id'])
        self.prompts = {}
        if prompts:
            for row in iterrows(Q / 'data/inputs.jsonl'):
                if row['uid'] in self.ids:
                    self.prompts[row['uid']] = row['prompt']
            if set(self.prompts) != self.ids:
                raise ValueError('Missing frozen interference input')
        self.raw = {}
        for file in [Q / 'data/reused_scores.jsonl'] + sorted((Q / 'cache').glob('scores*.jsonl')):
            for r in iterrows(file):
                uid = r.get('uid', r.get('view_id'))
                if uid in self.ids and r.get('status', 'ok') == 'ok':
                    self.raw[uid] = float(r['ell'])
        if set(self.raw) != self.ids:
            raise ValueError('Incomplete existing raw cache')
        self.raw_paths = [r for r in iterrows(Q / 'evaluation/test/path_predictions.jsonl') if r['method'] == 'raw']

    def cached_scores(self, method):
        source = e5_source(method)
        values = {}
        for r in iterrows(source / 'test_input_predictions.jsonl'):
            uid = self.alias.get(r['view_id'])
            if uid in self.ids:
                values[uid] = float(r['score'])
        mapping = {text: uid for uid, text in self.prompts.items()}
        for r in iterrows(source / 'test_path_node_scores.jsonl'):
            uid = mapping.get(r['prompt'])
            if uid is not None:
                values[uid] = float(r['score'])
        for r in iterrows(OUT / 'e5' / method / 'new_node_scores.jsonl'):
            values[r['uid']] = float(r['score'])
        return values

    def complete_paths(self, method, score):
        from experiments.aster_cr_retrain_paired_v2_20261111.evaluation import _execute_prepared_path
        destination = OUT / 'e5' / method
        pathfile = destination / 'interference_path_predictions.jsonl'
        completed = {pathkey(r) for r in iterrows(pathfile)}
        for (cid, variant), uids in self.groups.items():
            if (cid, variant) in completed:
                continue
            case = self.cases[cid]
            nodes, root = self.trees[cid]
            gid, node_scores = root, {}
            while gid in nodes and gid not in node_scores:
                s = float(score(uids[gid]))
                if not np.isfinite(s):
                    raise ValueError('Nonfinite method node score')
                node_scores[gid] = s
                gid = nodes[gid]['yes' if s > 0 else 'no']
            record = dict(method=method, case_id=cid, source_id=case['source_id'], variant=variant,
                          interference=True, **_execute_prepared_path(case, node_scores, nodes, root))
            append(pathfile, record)
            completed.add((cid, variant))
            if len(completed) < 4 or len(completed) % 25 == 0:
                write(destination / 'status.json', dict(state='running', paths_completed=len(completed), expected=2418))

    def summarize(self, method):
        destination = OUT / 'e5' / method
        natural = [dict(r, method=method, variant='original', interference=False)
                   for r in iterrows(e5_source(method) / 'test_path_predictions.jsonl')]
        interference = rows(destination / 'interference_path_predictions.jsonl')
        combined = natural + interference
        stats = path_stats(combined, self.raw_paths)
        if set(stats) != {'natural', 'interference'}:
            raise ValueError('Missing natural/interference population')
        lines(destination / 'path_predictions.jsonl', combined)
        old = read(e5_source(method) / 'test_metrics.json')
        result = dict(complete=True, method=method, paths=stats, groups=old['groups'],
                      source_metrics=str(e5_source(method) / 'test_metrics.json'),
                      source_pairs=str(e5_source(method) / 'test_pair_predictions.jsonl'),
                      training_reused=True, new_training=False)
        write(destination / 'metrics.json', result)
        write(destination / 'done.json', dict(complete=True, natural_paths=806, interference_paths=2418, sources=40))
        return result
