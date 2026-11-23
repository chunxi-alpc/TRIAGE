"""Extract published capacity comparisons without rescoring archived inputs."""
import argparse
import csv
import json
import time
from pathlib import Path

from source_inventory import read, write_csv


def table(path):
    with path.open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, default=Path.home())
    parser.add_argument('--output', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    cpu0 = time.process_time()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[3]
    rows = []

    def add(scope, method, dim, target, path, **metrics):
        rows.append(dict(scope=scope, method=method, dimension=dim, objective=target, evidence=str(path), **metrics))

    budget_path = Path('aster_cancel_story_20261109/meeting/real_completion/metrics.json')
    history = read(args.archive/budget_path)
    for name in ['budget_0.05','linear_output','budget_0.25','budget_0.5','budget_1.0','budget_2.0','no_CF',
                 'r32_budget_0.1','r32_budget_0.25','r32_no_CF']:
        model = history['methods'][name]
        ap, cf = model['groups']['AP'], model['groups']['CF']
        add('40 previous test sources, legacy holdout', name, 32 if name.startswith('r32') else 16,
            'hard budget sweep' if 'no_CF' not in name else 'unconstrained', budget_path,
            normalized_budget=model['settings']['tau'], ap_pairs=ap['n'], cf_pairs=cf['n'],
            ap_vr=ap['variance_reduction'], cf_er=cf['mean_evidence_retention'], cf_correction_rms=cf['correction_rms'])

    trace = Path('reports/aster_cancel_story_20261109/meeting/trace_validation/summary.csv')
    for item in table(repo/trace):
        if item['question'] == 'Q2' and item['metric'] == 'R2':
            size = int(item['method'].rsplit('N', 1)[1])
            add('development A pairs', item['method'], size, 'trace prediction', trace, trace_r2=item['value'])

    full = Path('aster-full-complement-70/reports/aster_full_complement_70/cv_summary.json')
    for name, size in [('E1',16), ('full_complement',6528)]:
        model = read(args.archive/full)['models'][name]
        metrics = {k:model['metrics'][k]['estimate'] for k in ['ap_vr','cf_er','cf_correction_rms','cf_over_budget_rate']}
        add('80 sources; five folds; Dataset A', name, size, 'hard constraints', full,
            ap_pairs=model['counts']['ap_pairs'], cf_pairs=model['counts']['cf_pairs'], **metrics)

    grid = Path('aster-m3-dual-objective-20261116/reports/aster_m3_dual_objective_20261116/oof_grid_70.csv')
    targets = table(args.archive/grid)
    chosen = Path('aster-m3-pareto-knee-20261116/reports/aster_m3_pareto_knee_20261116/pareto_key_points.csv')
    choices = {r['candidate_id']:r['role'] for r in table(args.archive/chosen) if r['role'] in ('knee_A','auto_eta0')}
    for item in targets:
        if item['id'] in choices:
            add('80 sources; five folds; Dataset B', item['id'],16,choices[item['id']], grid,
                **{k:item[k] for k in ['eta','regularization','ap_pairs','cf_pairs','ap_vr','cf_er','cf_correction_rms','cf_normalized_distortion']})
    write_csv(out/'archived_capacity.csv', list(dict.fromkeys(k for r in rows for k in r)), rows)
    (out/'capacity_status.json').write_text(json.dumps(dict(status='reused', archival_comparison_rows=len(rows),
        archived_grid_size=len(targets), new_experiment_units=0, new_fits=0, new_forwards=0,
        gpu_hours=0, cpu_seconds=time.process_time()-cpu0), indent=2)+'\n')


if __name__ == '__main__':
    main()
