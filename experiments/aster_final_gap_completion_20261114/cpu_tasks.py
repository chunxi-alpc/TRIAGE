"""Complete approved CPU-only gaps without loading a backbone."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
import time
import traceback
from common import *
sys.path.insert(0, str(REPO6))
from experiments.aster_e678 import evaluate as ev
from experiments.aster_cr_retrain_paired_v2_20261111 import evaluation as legacy

def controlled(model):
    source = R / 'models' / model
    target = OUT / 'e6' / model
    models = ev.load_models(source)
    store = ev.DynamicStore(source, required_heads=models['E1']['heads'])
    old = Path('/home/gcx/aster_five_models_20261109/results') / model / 'mega/paths/nodes.jsonl'
    old_records = {r['id']: r for r in iterrows(old)}
    for r in iterrows(AUDIT / 'cache_maps' / ('e6_' + model + '_additional_fullhead_nodes.jsonl')):
        store.records[r['uid']] = old_records[r['original_id']]
    spec = ev.scope_inputs(source, 'test')
    spec['path_views'] = [v for v in spec['path_views'] if v.get('variant', 'original') in ('original', 'full') and ev.uid(v) in store.records]
    result = legacy.scope_evaluate('test', store=store, models=models, out=target, **spec)
    paths = rows(target / 'evaluation/test/path_predictions.jsonl')
    raw = [r for r in paths if r['method'] == 'raw']
    stats = {m: path_stats([r for r in paths if r['method'] == m], raw, seed=legacy.SEED)
             for m in ['raw', 'calibration', 'E1']}
    if any(r.get('error') for r in paths):
        raise ValueError('Controlled path lacks a visited cached node')
    result.update(frozen_parameters=str(source / 'models/E1/full'), new_model_forwards=0,
                  source_parameters_frozen=True, dynamic_paths=stats, evaluation_scope='806 natural paths and existing paired views')
    write(target / 'evaluation/test/metrics.json', result)
    write(target / 'test_done.json', dict(complete=True, paths_per_method=806, sources=40, new_model_forwards=0))

def cc():
    data = E5Data(prompts=False)
    offset = read(E5 / 'prompt_controls_e1/cc_parameters.json')['offset']
    data.complete_paths('CC', lambda uid: data.raw[uid] + offset)
    data.summarize('CC')

def external():
    raw = [r for r in iterrows(Q / 'evaluation/test/path_predictions.jsonl') if r['method'] == 'raw']
    results = {}
    for folder in sorted((EXT / 'evaluation/test').iterdir()):
        if not folder.is_dir():
            continue
        paths = rows(folder / 'path_predictions.jsonl')
        if paths:
            results[folder.name] = dict(paths=path_stats(paths, raw), source_predictions=str(folder / 'path_predictions.jsonl'),
                                       source_statistics=str(folder / 'statistics.json'), new_model_forwards=0)
    if len(results) != 11:
        raise ValueError('Expected exactly 11 external rows')
    write(OUT / 'e5/external_paired_ci/metrics.json', dict(complete=True, methods=results))

def transfer(scope):
    if scope == 'nli4ct':
        spec = ev.scope_inputs(Q, scope)
        source = Q
        short = 'nli500'
    elif scope == 'nli4ct_official5500':
        spec = ev.scope_inputs(R / 'models/gemma3', scope)
        mapping = {ev.uid(v): v['original_view_id'] for v in spec['views']}
        spec['views'] = [dict(v, uid=mapping[ev.uid(v)], view_id=mapping[ev.uid(v)]) for v in spec['views']]
        spec['pairs'] = [dict(v, left=mapping[v['left']], right=mapping[v['right']]) for v in spec['pairs']]
        source, short = F, 'nli5500'
    else:
        spec = dict(views=legacy.data_rows(Q, 'medpic_views.jsonl'), pairs=legacy.data_rows(Q, 'medpic_pairs.jsonl'), native_task='medpic')
        source, short = F, 'medpic467'
    old = {r['view_id']: r for r in iterrows(source / 'evaluation' / scope / 'predictions.jsonl')}
    if set(old) != {ev.uid(v) for v in spec['views']}:
        raise ValueError('Existing prediction and metadata UID sets differ')
    for view in spec['views']:
        r = old[ev.uid(view)]
        if r['source_id'] != view['source_id'] or r.get('label') != view.get('label'):
            raise ValueError('Existing transfer prediction/metadata protocol mismatch')
    b0 = read(D / 'configs/independent_intercepts_reused.json')['full']['b0']
    scores = {name: {} for name in ['raw', 'calibration', 'E1']}
    for uid, r in old.items():
        scores['raw'][uid] = float(r['scores']['raw'])
        scores['E1'][uid] = float(r['scores']['E1'])
        scores['calibration'][uid] = float(r['scores']['raw'] + b0)
    model = legacy.model_parameters(Q, 'E1')
    models = {'E1': model, 'calibration': dict(model, directory=str(D / 'configs/independent_intercepts_reused.json'))}
    original = legacy.score_views
    legacy.score_views = lambda store, views, models: scores
    target = OUT / 'e7/qwen' / short
    try:
        result = legacy.scope_evaluate(scope, store=None, models=models, out=target, **spec)
    finally:
        legacy.score_views = original
    predictions = rows(target / 'evaluation' / scope / 'predictions.jsonl')
    for row in predictions:
        if row['predictions']['raw'] != old[row['view_id']]['predictions']['raw'] or row['predictions']['E1'] != old[row['view_id']]['predictions']['E1']:
            raise ValueError('Reused raw/E1 decision differs from saved prediction')
    native = {}
    if scope.startswith('nli'):
        subsets = {'all_statements': predictions, 'original_statements': [r for r in predictions if r['original']]}
        for population, rr in subsets.items():
            baseline = [dict(r, correct=r['predictions']['raw'] == r['label']) for r in rr]
            native[population] = {}
            for method in scores:
                records = [dict(r, correct=r['predictions'][method] == r['label']) for r in rr]
                stat = paired_binary(records, baseline, lambda r: r['view_id'], seed=legacy.SEED)
                y = np.array([r['label'] for r in rr]); pred = np.array([r['predictions'][method] for r in rr])
                labels = []
                for label in [0, 1]:
                    tp = int(((y == label) & (pred == label)).sum())
                    fp = int(((y != label) & (pred == label)).sum())
                    fn = int(((y == label) & (pred != label)).sum())
                    labels.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
                stat['macro_f1'] = float(np.mean(labels))
                native[population][method] = stat
        cluster = 'trial_or_trial_combination' if scope == 'nli4ct' else 'original_statement_and_variants'
    else:
        qq = [r for r in iterrows(target / 'evaluation' / scope / 'question_predictions.jsonl') if r['variant'] in ('full', 'original')]
        baseline = [r for r in qq if r['method'] == 'raw']
        for method in scores:
            native[method] = paired_binary([r for r in qq if r['method'] == method], baseline, lambda r: r['item_id'], seed=legacy.SEED)
        cluster = 'existing_source_id'
    result.update(native_population_statistics=native, independent_intercept=b0,
                  intercept_fit_nodes=761, intercept_fit_sources=80, target_labels_used_for_fitting=False,
                  clustering=cluster, clustering_user_confirmed=True, new_model_forwards=0,
                  includes_development_questions=scope == 'medpic_official467',
                  population_note='Descriptive all-question evaluation, including development questions.' if scope == 'medpic_official467' else None)
    write(target / 'evaluation' / scope / 'metrics.json', result)
    write(target / 'done.json', dict(complete=True, new_model_forwards=0, clustering=cluster))

def main():
    tasks = [('C01', lambda: controlled('gemma3')), ('C02', lambda: controlled('medgemma')),
             ('C03', cc), ('C04', external), ('C05', lambda: transfer('nli4ct')),
             ('C06', lambda: transfer('nli4ct_official5500')), ('C07', lambda: transfer('medpic_official467'))]
    status_path = OUT / 'cpu_status.json'
    status = read(status_path, {'tasks': {}})
    for task, run in tasks:
        if status['tasks'].get(task, {}).get('state') == 'complete':
            continue
        start = time.time()
        status['tasks'][task] = dict(state='running', started=start)
        write(status_path, status)
        try:
            run()
            status['tasks'][task] = dict(state='complete', seconds=time.time() - start)
        except Exception as e:
            status['tasks'][task] = dict(state='failed', error=str(e), traceback=traceback.format_exc())
            write(status_path, status)
            raise
        write(status_path, status)
    status['state'] = 'complete'
    write(status_path, status)

if __name__ == '__main__':
    main()
