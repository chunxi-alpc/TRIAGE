"""Independent, read-only audit of PR33/35; run on the research server.

Only the explicit --output path is created. No old result is rewritten.
No model, GPU, private patient text, or new external-test labels are used.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import time
from collections import Counter
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def softmax(values):
    top = max(values)
    exp = [math.exp(x - top) for x in values]
    total = sum(exp)
    return [x / total for x in exp]


def prediction(row, config, kind):
    base = softmax([x / config['t0'] for x in row['scores']['original']])
    support = row['option_S']
    total = sum(support)
    norm = [x / total for x in support] if total > 0 else base
    sp = softmax([math.log(max(x, 1e-30)) / config['ts'] for x in norm]) if total > 0 else base
    beta = config['beta']
    if kind == 'dynamic_fusion':
        ranked = sorted(norm)
        beta = min(1, max(0, beta + .25 * (2 * (ranked[-1] - ranked[-2]) - 1)))
    mixed = [(1-beta)*b + beta*s for b, s in zip(base, sp)]
    if kind in ('static_fusion', 'dynamic_fusion', 'no_residual', 'zero_strength'):
        delta = [0.] * len(base)
    else:
        if kind == 'legacy_attention_residual':
            numerator, zero = 'action_bias_1.0_all', 'zero_all'
        elif kind == 'numeric_drift':
            numerator, zero = 'zero_midlate4', 'original'
        else:
            geometry = {'dynamic_late4': 'late4', 'dynamic_midlate2': 'midlate2'}.get(kind, 'midlate4')
            numerator, zero = kind, 'zero_' + geometry
        delta = [x-y for x,y in zip(row['scores'][numerator], row['scores'][zero])]
    scores = [math.log(max(p, 1e-30)) + .25*math.tanh(d/2) for p,d in zip(mixed,delta)]
    return max(range(len(scores)), key=scores.__getitem__)


def paired(rows, first, second):
    by_tree = {}
    for tree in sorted({r['tree'] for r in rows}):
        ix = [i for i,r in enumerate(rows) if r['tree'] == tree]
        fix = sum(first[i] == rows[i]['gold_option'] and second[i] != rows[i]['gold_option'] for i in ix)
        loss = sum(first[i] != rows[i]['gold_option'] and second[i] == rows[i]['gold_option'] for i in ix)
        by_tree[tree] = dict(n=len(ix), corrected=fix, regressed=loss, net=fix-loss)
    groups = list(by_tree.values())
    # Exactly enumerate all 4^4 tree-cluster bootstrap outcomes; all variants stay together.
    require(len(groups) == 4, 'This legacy exact enumeration is limited to four development trees')
    samples = sorted(sum(x['net'] for x in draw)/sum(x['n'] for x in draw)
                     for draw in itertools.product(groups, repeat=4))
    def quantile(q):
        k = q*(len(samples)-1)
        low = int(k)
        high = min(low+1, len(samples)-1)
        return samples[low] + (k-low)*(samples[high]-samples[low])
    return dict(corrected=sum(x['corrected'] for x in groups), regressed=sum(x['regressed'] for x in groups),
                accuracy_delta=sum(x['net'] for x in groups)/len(rows), by_tree=by_tree,
                exact_tree_bootstrap_ci95=[quantile(.025), quantile(.975)],
                interpretation='Descriptive development interval; does not account for method-design selection')


def audit33(root):
    folder = root/'experiments/aster_dynamic_fusion_seed0_20261025'
    path = folder/'completed_predictions.csv'
    with path.open(newline='') as handle:
        all_rows = list(csv.DictReader(handle))
    require(len(all_rows) == 1600, 'PR33 must have 1600 model-record rows')
    expected = json.loads((folder/'completed_audit.json').read_text())
    output = {}
    for model in ('qwen','llama'):
        rows = [r for r in all_rows if r['model'] == model]
        keys = {(r['case_id'],r['variant'],r['seed']) for r in rows}
        require(len(rows) == len(keys) == 800, 'PR33 missing or duplicated rows')
        require({r['seed'] for r in rows} == {'0'}, 'PR33 unexpected seed')
        require(len({r['case_id'] for r in rows}) == 200, 'PR33 patient count')
        require(Counter(r['variant'] for r in rows) == dict.fromkeys(('NC','AC','IC','SW'),200), 'PR33 variant count')
        for pid in {r['case_id'] for r in rows}:
            require({r['variant'] for r in rows if r['case_id']==pid} == {'NC','AC','IC','SW'}, 'PR33 patient variants')
        counts = {method:sum(r[method] == r['gold_option'] for r in rows) for method in expected[model]['methods']}
        for method,count in counts.items():
            require(count == expected[model]['methods'][method]['metrics']['correct'], 'PR33 count mismatch: '+method)
        converted = [{**r,'gold_option':int(r['gold_option'])} for r in rows]
        diff = paired(converted, [int(r['residual']) for r in rows], [int(r['no_delta']) for r in rows])
        require(diff['corrected'] == expected[model]['residual_vs_no_delta']['corrected_vs_comparator'], 'PR33 correction mismatch')
        require(diff['regressed'] == expected[model]['residual_vs_no_delta']['regressed_vs_comparator'], 'PR33 regression mismatch')
        output[model] = dict(counts=counts, residual_vs_no_delta=diff)
    return dict(scope='Independent CSV counts and paired statistics; not fresh GPU or full logits replay',
                prediction_sha256=digest(path), models=output)


def audit35(root):
    folder = root/'experiments/aster_sparse_dynamic_qwen_seed0'
    rows = [json.loads(line) for line in (folder/'numeric/predictions.jsonl').read_text().splitlines()]
    saved = json.loads((folder/'results.json').read_text())
    require(len(rows) == len({(r['id'],r['variant'],r['seed']) for r in rows}) == 800, 'PR35 rows')
    require(len({r['id'] for r in rows}) == 200, 'PR35 patients')
    require(all(r['seed'] == 0 and r['split'] == 'dev' for r in rows), 'PR35 split/seed')
    for pid in {r['id'] for r in rows}:
        require({r['variant'] for r in rows if r['id']==pid} == {'NC','AC','IC','SW'}, 'PR35 patient variants')
    for row in rows:
        require(not {'patient','rules','options','assessments','condition_key'} & row.keys(), 'PR35 disallowed text field')
        for method,values in row['scores'].items():
            require(all(math.isfinite(x) for x in values), 'PR35 non-finite logits')
            require(max(range(len(values)), key=values.__getitem__) == row['answers'][method], 'PR35 argmax mismatch')
    trees = sorted({r['tree'] for r in rows})
    counts, checked = {}, 0
    for kind,method in saved['methods'].items():
        require(len(method['all_candidates']) == 27, 'PR35 candidate count')
        replay = []
        for candidate in method['all_candidates']:
            preds = [prediction(r,candidate['config'],kind) for r in rows]
            require(preds == candidate['predictions'], 'PR35 candidate replay mismatch '+kind+' '+str(candidate['index']))
            require(sum(p==r['gold_option'] for p,r in zip(preds,rows)) == candidate['metrics']['correct'], 'PR35 candidate count mismatch')
            replay.append(preds)
            checked += 1
        oof = [None] * len(rows)
        for tree in trees:
            training = [t for t in trees if t != tree]
            scores = [sum(sum(p==r['gold_option'] for p,r in zip(preds,rows) if r['tree']==t)/sum(r['tree']==t for r in rows) for t in training)/len(training) for preds in replay]
            best = max(range(len(scores)),key=scores.__getitem__)
            require(best == method['folds'][tree]['config_index'], 'PR35 held-out fold choice mismatch')
            for i,row in enumerate(rows):
                if row['tree'] == tree:
                    oof[i] = replay[best][i]
        require(oof == method['predictions'], 'PR35 OOF mismatch '+kind)
        counts[kind] = sum(p==r['gold_option'] for p,r in zip(oof,rows))
    primary = saved['methods'][saved['primary']]
    controls = {}
    for kind,item in saved['matched_primary_controls'].items():
        preds = [prediction(r,primary['folds'][r['tree']]['config'],kind) for r in rows]
        require(preds == item['predictions'], 'PR35 matched control mismatch '+kind)
        controls[kind] = paired(rows,primary['predictions'],preds)
    recorded = json.loads((folder/'audit.json').read_text())['artifacts_sha256']
    for name,sha in recorded.items():
        require(digest(folder/name) == sha, 'PR35 artifact hash mismatch '+name)
    require(sum(v['status']['forward_calls'] for v in saved['resources'].values()) == 8800, 'PR35 recorded GPU count')
    return dict(scope='Independent stdlib replay of all 351 configurations, fold choices, matched controls, and saved hashes',
                checked_candidates=checked, counts=counts, matched_controls=controls, artifacts_sha256=recorded,
                gpu_forward_count_scope='Historical manifest count checked; no GPU forward performed by this audit')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Output exists; use a new audit path')
    started = time.time()
    report = dict(status='running', independent_implementation=True, new_gpu_forwards=0)
    try:
        report['pr33'] = audit33(args.repo)
        report['pr35'] = audit35(args.repo)
        report['status'] = 'complete_verified'
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = type(error).__name__+': '+str(error)
        raise
    finally:
        report['elapsed_seconds'] = time.time()-started
        report['checker_sha256'] = digest(Path(__file__))
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as handle:
            json.dump(report,handle,indent=2)
            handle.write('\n')
    print(json.dumps({'status':report['status'],'new_gpu_forwards':0,'checked_candidates':report['pr35']['checked_candidates']}))


if __name__ == '__main__':
    main()
