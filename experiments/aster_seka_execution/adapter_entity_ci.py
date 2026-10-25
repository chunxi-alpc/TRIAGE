"""Server-only paired entity intervals from verified CAD/Spotlight exports."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from counterfact_entity_ci import bootstrap_effects, digest, require, source_clusters


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--audit-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    audit_path = args.audit_dir / 'audit.json'
    predictions_path = args.audit_dir / 'predictions.jsonl'
    audit = json.loads(audit_path.read_text())
    require(audit['status'] == 'complete_verified', 'Require a completed independent export')
    require(audit['actual_input_evidence']['status'] == 'complete_verified', 'Actual input verification missing')
    require(digest(predictions_path) == audit['anonymous_predictions_sha256'], 'Prediction bytes changed')
    mapping, source_sha = source_clusters(args.data)
    require(source_sha == audit['fingerprints']['data_sha256'], 'Entity/source fingerprint differs')
    controls = list(audit['recomputed_metrics'])
    values = {name: {} for name in controls}
    for line in predictions_path.open():
        row = json.loads(line)
        name, sid = row['control'], str(row['id'])
        require(name in values and sid not in values[name], 'Unknown or duplicate case/control')
        require(row['status'] == 'ok' and isinstance(row['target_wins'], bool), 'Invalid case result')
        values[name][sid] = int(row['target_wins'])
    for name in controls:
        require(set(values[name]) == set(mapping), 'All fixed 5000 cases must remain in every control')
        require(abs(sum(values[name].values()) / 5000 - audit['recomputed_metrics'][name]['score']['mean']) < 1e-12,
                'Recomputed efficacy does not match independent audit')
    ordered = sorted(mapping, key=lambda sid: mapping[sid]['absolute_processed_row_index'])
    entities = [mapping[sid]['entity_cluster'] for sid in ordered]
    references = [name for name in controls if name != 'normal']
    columns, comparisons = [], {}
    for name in references:
        delta = np.asarray([values['normal'][sid] - values[name][sid] for sid in ordered])
        corrected, regressed = (delta > 0).astype(float), (delta < 0).astype(float)
        columns.extend([delta, corrected, regressed])
        comparisons[name] = dict(difference_percentage_points=float(delta.mean() * 100),
                                 corrected=int(corrected.sum()), regressed=int(regressed.sum()),
                                 n_cases=5000)
    intervals = bootstrap_effects(np.asarray(columns).T, entities, 10000, 0)
    for i, name in enumerate(references):
        comparisons[name].update(ci95_difference_percentage_points=[100 * v for v in intervals[3*i]],
                                 ci95_correction_rate=intervals[3*i+1], ci95_regression_rate=intervals[3*i+2])
    report = dict(status='complete_verified', new_gpu_forwards=0,
                  scope='Within-adapter standard efficacy paired statistics, not clinical evidence or cross-implementation ranking',
                  created_at_utc=datetime.now(timezone.utc).isoformat(),
                  method=audit['identity']['method'], source_sha256=source_sha,
                  audit_sha256=digest(audit_path), predictions_sha256=digest(predictions_path),
                  checker_sha256=digest(__file__), bootstrap_source_sha256=digest(Path(__file__).with_name('counterfact_entity_ci.py')),
                  group='exact CounterFact entity; not patient', n_groups=len(set(entities)),
                  bootstrap_replicates=10000, bootstrap_seed=0, all_fixed_cases_retained=True,
                  greedy_single_run=True, comparisons=comparisons,
                  numerical_drift_evidence=audit['numerical_controls'])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/'entity_ci.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'comparisons': comparisons}))


if __name__ == '__main__':
    main()
