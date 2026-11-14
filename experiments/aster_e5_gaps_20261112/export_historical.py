"""Export already-completed historical E5 results; never fit or run a model."""
import argparse
import copy
import csv
import gzip
import json
from collections import Counter
from pathlib import Path
import numpy as np

METHODS = {
    "raw": "\u539f\u6a21\u578b",
    "scale_095": "\u56fa\u5b9a\u6bd4\u4f8b\u7f29\u653e\uff08a=0.95\uff0c\u89e3\u6790\u63a7\u5236\uff09",
    "no_w": "\u4ec5\u622a\u8ddd\u6821\u51c6\uff08\u5386\u53f2\uff09",
    "ordinary_r8": "\u76d1\u7763\u7ebf\u6027\u8bfb\u51fa\uff08\u5386\u53f28\u7ef4\uff09",
    "leace": "LEACE\uff08\u5386\u53f2\uff09",
    "iti": "ITI\uff08\u5386\u53f2\uff09",
    "lofit": "LoFiT\uff08\u5386\u53f2\uff09",
    "lora": "LoRA\uff08\u5386\u53f2\uff09",
    "linear_output": "ASTER-CR\u7ebf\u6027\uff08\u65e7\u6570\u636e\u5386\u53f2\u53c2\u7167\uff0c\u975eE1\uff09"
}


def dump(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_jsonl(path, rows):
    with gzip.open(path, 'xt', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/home/gcx/aster_cancel_story_20261109/meeting/real_completion'))
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[2] / 'reports/aster_e5_gaps_20261112/historical')
    args = parser.parse_args()
    src, out = args.source, args.output
    if out.exists() and any(out.iterdir()):
        raise SystemExit('Output directory is not empty; choose a new output to preserve existing reports.')
    out.mkdir(parents=True, exist_ok=True)
    archive = json.loads((src / 'metrics.json').read_text())
    meta = json.loads((src / 'metadata.json').read_text())
    selected = {k: copy.deepcopy(archive['methods'][k]) for k in METHODS if k != 'scale_095'}
    with (src / 'test_pair_results.csv').open(newline='') as f:
        old_pairs = list(csv.DictReader(f))
    old_paths = [json.loads(line) for line in (src / 'path_results.jsonl').read_text().splitlines() if line]
    pair_meta = {p['pair_id']: p for p in meta['pairs'] if p['split'] == 'test'}
    raw_pairs = [r for r in old_pairs if r['method'] == 'raw']
    # Positive fixed scaling preserves every zero-threshold class and path.
    # All interval transforms below follow exact positive homogeneity, not a new bootstrap.
    scaled = copy.deepcopy(selected['raw'])
    scaled['settings'] = {'scale': 0.95, 'derivation': 'Analytic transformation of existing raw scores; no fit or inference.'}
    kinds = {'A': ('A',), 'P': ('P',), 'AP': ('A', 'P'), 'C': ('C',), 'F': ('F',), 'CF': ('C', 'F'), 'F0': ('F0',)}
    for group, mm in scaled['groups'].items():
        if group == 'natural':
            continue
        rr = [r for r in raw_pairs if r['kind'] in kinds[group]]
        differences = np.array([float(r['original_difference']) for r in rr])
        old_rms = mm['difference_rms']
        old_ci = mm['difference_rms_ci']
        mm['difference_rms'] = 0.95 * old_rms
        mm['difference_rms_ci'] = [0.95 * x for x in old_ci]
        mm['correction_rms'] = 0.05 * old_rms
        mm['correction_rms_ci'] = [0.05 * x for x in old_ci]
        mm['correction_rms_over_ap_rms'] = mm['correction_rms'] / archive['test_AP_raw_rms']
        mm['correction_abs_max'] = float(0.05 * np.max(np.abs(differences)))
        if 'variance_reduction' in mm:
            mm['variance_reduction'] = 1 - 0.95**2
            mm['variance_reduction_ci'] = [1 - 0.95**2, 1 - 0.95**2]
        if 'mean_evidence_retention' in mm:
            mm['mean_evidence_retention'] = 0.95
            mm['mean_evidence_retention_ci'] = [0.95, 0.95]
            n_exceed = int(np.sum(np.abs(0.05 * differences) > archive['fit_budget'] + 1e-7))
            mm['exceeds_fit_budget_count'] = n_exceed
            mm['exceeds_fit_budget_fraction'] = n_exceed / len(differences)
    selected['scale_095'] = scaled
    # Keep the original archived floating-point records intact; exact identities
    # are stated separately and displayed as mathematical zeros in the table.
    result = {
        'status': 'Historical results export only; E1 compatibility awaits Codex 01 report.',
        'source_directory': str(src),
        'model': 'Qwen3-8B',
        'evaluation_history': 'Previously used 40-source controlled evaluation; not a new independent test.',
        'denominators': {'sources': 40, 'AP_pairs': 2293, 'CF_pairs': 40, 'CF_sources': 12, 'F0_pairs_separate': 6, 'natural_nodes': 372, 'dynamic_paths': 806},
        'historical_fit_context': {'AP': archive['fit_AP'], 'CF': archive['fit_CF'], 'note': 'Describes historical ASTER fit context; it is not a verified training manifest for every external method.'},
        'target_training_protocol': 'E1 / Dataset A, 2540 pairs. Historical learned-method rows are not certified as E1-compatible.',
        'uncertainty': archive['uncertainty'],
        'interval_provenance': 'Existing intervals copied verbatim; fixed-scale intervals transformed analytically from raw intervals. No new bootstrap.',
        'analytic_identities': 'Raw and shared-intercept pair correction is exactly zero; archived floating-point residuals remain in imported records. Positive scaling leaves binary predictions and paths unchanged.',
        'model_inference_runs': 0,
        'training_runs': 0,
        'excluded': {'loreft': 'Effectiveness not established; excluded from numeric tables.'},
        'methods': {k: selected[k] for k in METHODS},
    }
    dump(out / 'common_metrics.json', result)
    table, intervals = [], []
    for name, label in METHODS.items():
        m = selected[name]
        ap, cf, path = m['groups']['AP'], m['groups']['CF'], m['path']
        correction = 0.0 if name in ('raw', 'no_w') else cf['correction_rms']
        correction_ci = [0.0, 0.0] if name in ('raw', 'no_w') else cf['correction_rms_ci']
        table.append({
            'method': name, 'method_zh': label, 'status': 'analytic_control' if name == 'scale_095' else 'historical_result_reuse',
            'E1_compatibility': 'not_determined_pending_Codex01',
            'AP_n': ap['n'], 'AP_response_rms': ap['difference_rms'],
            'AP_rms_ci_low': ap['difference_rms_ci'][0], 'AP_rms_ci_high': ap['difference_rms_ci'][1],
            'CF_n': cf['n'], 'CF_correction_rms': correction,
            'CF_correction_rms_ci_low': correction_ci[0], 'CF_correction_rms_ci_high': correction_ci[1],
            'CF_paired_correct': cf['paired_correct'], 'CF_paired_accuracy': cf['paired_accuracy'],
            'CF_paired_accuracy_ci_low': cf['paired_accuracy_ci'][0], 'CF_paired_accuracy_ci_high': cf['paired_accuracy_ci'][1],
            'path_n': path['n'], 'path_correct': path['correct'], 'path_accuracy': path['accuracy'],
            'path_accuracy_ci_low': path['accuracy_ci'][0], 'path_accuracy_ci_high': path['accuracy_ci'][1],
            'path_difference_vs_raw': path['difference_vs_raw'],
            'path_difference_ci_low': path['difference_vs_raw_ci'][0], 'path_difference_ci_high': path['difference_vs_raw_ci'][1],
            'history': 'Previously used 40 sources / 806 paths; not independent confirmation.',
        })
        for scope, metrics in list(m['groups'].items()) + [('path', path)]:
            for key, value in metrics.items():
                if key.endswith('_ci') and value is not None:
                    metric = key[:-3]
                    intervals.append({'method': name, 'scope': scope, 'metric': metric, 'estimate': metrics.get(metric), 'lower_95': value[0], 'upper_95': value[1], 'n': metrics.get('n'), 'provenance': 'analytic_transform_of_archived_CI' if name == 'scale_095' else 'archived_2000_source_bootstrap_seed11'})
    write_csv(out / 'section_4_5_historical_table.csv', table)
    write_csv(out / 'source_95ci.csv', intervals)
    predictions = []
    for r in old_pairs:
        if r['method'] not in METHODS:
            continue
        z = dict(r)
        for field in ('left_score', 'right_score', 'original_difference', 'correction_difference', 'corrected_difference'):
            z[field] = float(z[field])
        z['paired_correct'] = z['paired_correct'] == 'True'
        p = pair_meta[z['pair_id']]
        z.update({key: p[key] for key in ('left', 'right', 'left_label', 'right_label')})
        z['prediction_left'] = int(z['left_score'] > 0)
        z['prediction_right'] = int(z['right_score'] > 0)
        z['provenance'] = 'historical_prediction'
        predictions.append(z)
        if z['method'] == 'raw':
            s = copy.deepcopy(z)
            s.update(method='scale_095', provenance='analytic_transform_of_historical_raw')
            s['left_score'] *= 0.95
            s['right_score'] *= 0.95
            s['corrected_difference'] = 0.95 * s['original_difference']
            s['correction_difference'] = -0.05 * s['original_difference']
            predictions.append(s)
    write_jsonl(out / 'pair_predictions.jsonl.gz', predictions)
    paths = []
    for r in old_paths:
        if r['method'] in METHODS:
            paths.append(dict(r, provenance='historical_prediction'))
            if r['method'] == 'raw':
                paths.append(dict(r, method='scale_095', provenance='analytic_identity_of_historical_raw_path'))
    write_jsonl(out / 'path_predictions.jsonl.gz', paths)
    arrays = np.load(src / 'evaluated_scores.npz')
    index = {v: i for i, v in enumerate(arrays['ids'].tolist())}
    views = [v for v in meta['views'] if v['split'] == 'test']
    def node_rows():
        for name in METHODS:
            score = arrays['raw'] * 0.95 if name == 'scale_095' else arrays[name]
            for view in views:
                ell = float(score[index[view['view_id']]])
                if np.isfinite(ell):
                    yield dict(view, method=name, score=ell, prediction=int(ell > 0), provenance='analytic_transform_of_historical_raw' if name == 'scale_095' else 'historical_prediction')
    write_jsonl(out / 'input_predictions.jsonl.gz', node_rows())
    md = ["# \u7b2c4.5\u8282\u5386\u53f2\u7ed3\u679c\u6570\u636e\u5305", "", "\u672c\u6587\u4ef6\u4ec5\u5bfc\u51fa\u5df2\u5b8c\u6210\u7ed3\u679c\u3002\u5386\u53f240\u6765\u6e90\u30012,293\u4e2aA/P\u5bf9\u300140\u4e2aC/F\u5bf9\u3001806\u6761\u52a8\u6001\u8def\u5f84\u5747\u6709\u65e2\u5f80\u4f7f\u7528\u8bb0\u5f55\uff0c\u4e0d\u80fd\u79f0\u4e3a\u5168\u65b0\u72ec\u7acb\u6d4b\u8bd5\u3002C/F\u5b9e\u9645\u8986\u76d612\u4e2a\u6765\u6e90\uff1b\u53e6\u67096\u4e2aF0\u5bf9\uff0c\u5355\u5217\u4e8e\u5b8c\u6574\u6307\u6807\u3002", "", "\u672a\u91cd\u65b0\u8bad\u7ec3\u6216\u6267\u884c\u6a21\u578b\u524d\u5411\uff0c\u672a\u590d\u6d4bNLI4CT\u6216MedPIC\u3002\u4e0b\u8868\u4e0d\u662fE1\u517c\u5bb9\u6027\u7ed3\u8bba\uff1b\u6b63\u5f0fDataset A 2,540\u5bf9\u8865\u8dd1\u8303\u56f4\u7b49\u5f85Codex 01\u62a5\u544a\u3002\u65e7ASTER-CR\u7ebf\u6027\u884c\u4ec5\u662f\u5386\u53f2\u53c2\u7167\u3002", "", "|\u65b9\u6cd5|A/P\u54cd\u5e94RMS\u53ca95%\u533a\u95f4|C/F\u4fee\u6b63RMS\u53ca95%\u533a\u95f4|C/F\u5168\u5bf9|\u8def\u5f84\u5168\u5bf9|\u8def\u5f84\u6b63\u786e\u738795%\u533a\u95f4|", "|---|---:|---:|---:|---:|---:|"]
    for row in table:
        md.append('|{}|{:.4f} [{:.4f}, {:.4f}]|{:.4f} [{:.4f}, {:.4f}]|{}/{}|{}/{}|[{:.2f}%, {:.2f}%]|'.format(
            row['method_zh'], row['AP_response_rms'], row['AP_rms_ci_low'], row['AP_rms_ci_high'],
            row['CF_correction_rms'], row['CF_correction_rms_ci_low'], row['CF_correction_rms_ci_high'],
            row['CF_paired_correct'], row['CF_n'], row['path_correct'], row['path_n'],
            100*row['path_accuracy_ci_low'], 100*row['path_accuracy_ci_high']))
    md += ["", "\u533a\u95f4\u590d\u7528\u5f52\u6863\u76842,000\u6b21\u6765\u6e90\u7c07\u91cd\u91c7\u6837\uff08\u79cd\u5b5011\uff0c\u767e\u5206\u4f4d95%\u533a\u95f4\uff09\uff1b\u540c\u4e00\u8f6e\u6765\u6e90\u91c7\u6837\u7528\u4e8e\u6240\u6709\u65b9\u6cd5\uff0c\u7a00\u758f\u7ec4\u672a\u5b9a\u4e49\u7684\u91cd\u91c7\u6837\u503c\u6309\u539f\u8bb0\u5f55\u6392\u9664\u3002\u56fa\u5b9a\u7f29\u653e\u7684\u533a\u95f4\u7531\u539f\u6a21\u578b\u533a\u95f4\u89e3\u6790\u8f6c\u6362\uff0c\u65e0\u65b0\u589e\u62df\u5408\uff1b\u65b9\u5dee\u964d\u4f4e\u73879.75%\u548cER 95%\u7684\u9000\u5316\u533a\u95f4\u6765\u81ea\u6570\u5b66\u6052\u7b49\u5f0f\u3002", "", "\u539f\u6a21\u578b\u53ca\u4ec5\u622a\u8ddd\u7684\u914d\u5bf9\u4fee\u6b63\u4e25\u683c\u4e3a\u96f6\uff1b\u4e3b\u8868\u6309\u89e3\u6790\u96f6\u663e\u793a\uff0ccommon_metrics.json\u548csource_95ci.csv\u4fdd\u7559\u5f52\u6863\u4e2d\u7684\u6d6e\u70b9\u6b8b\u5dee\u3002ITI\u5386\u53f2\u5b9e\u6d4bC/F\u53cc\u7aef\u5168\u5bf9\u4e3a0/40\uff0c\u4e0d\u80fd\u89e3\u91ca\u6210\u672a\u8fd0\u884c\u3002LoReFT\u6709\u6548\u6027\u672a\u786e\u8ba4\uff0c\u672a\u8fdb\u5165\u6570\u503c\u8868\u3002", "", "- section_4_5_historical_table.csv\uff1a\u7b2c4.5\u8282\u6570\u636e\u53ca\u660e\u786e\u5206\u6bcd\u3002", "- common_metrics.json\uff1a\u5404\u7c7b\u522b\u3001\u81ea\u7136\u8282\u70b9\u3001\u5b8c\u6574\u8def\u5f84\u6307\u6807\u4e0e\u5df2\u6709\u533a\u95f4\u3002", "- source_95ci.csv\uff1a\u5168\u90e8\u5df2\u5f52\u6863\u533a\u95f4\u4e0e\u89e3\u6790\u7f29\u653e\u533a\u95f4\u3002", "- pair_predictions.jsonl.gz\uff1a\u4e24\u7aef\u5206\u6570\u3001\u6807\u7b7e\u3001\u9884\u6d4b\u53ca\u6b63\u786e\u6027\u3002", "- path_predictions.jsonl.gz\uff1a\u9010\u4f8b\u5b9e\u9645\u8def\u5f84\u3001\u53f6\u8282\u70b9\u3001\u9996\u9519\u4e0e\u6b63\u786e\u6027\uff1b\u7f29\u653e\u8def\u5f84\u89e3\u6790\u590d\u7528\u3002", "- input_predictions.jsonl.gz\uff1a\u5386\u53f2\u6d4b\u8bd5\u8f93\u5165\u5206\u6570\u4e0e\u6807\u7b7e\uff1b\u53ea\u5bfc\u51fa\u6709\u9650\u5206\u6570\u3002", "", "\u5386\u53f2\u8bad\u7ec3\u4e0e\u5f53\u524dE1\u4e0d\u540c\uff1b\u5916\u90e8\u65b9\u6cd5\u7684\u8bad\u7ec3\u6e05\u5355\u548c\u76ee\u6807\u517c\u5bb9\u6027\u7531Codex 01\u62a5\u544a\u786e\u5b9a\uff0c\u672c\u5bfc\u51fa\u4e0d\u91cd\u65b0\u5ba1\u8ba1\u5b9e\u73b0\u3002"]
    md.append("Source directory: " + str(src))
    with (out / 'README.zh.md').open('x', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
    counts = {'methods': list(METHODS), 'pair_predictions': dict(Counter(r['method'] for r in predictions)), 'path_predictions': dict(Counter(r['method'] for r in paths)), 'interval_rows': len(intervals), 'new_training_runs': 0, 'new_model_inference_runs': 0}
    dump(out / 'export_summary.json', counts)
    print(json.dumps({'output': str(out), 'methods': len(METHODS), 'pair_rows': len(predictions), 'path_rows': len(paths), 'interval_rows': len(intervals)}))


if __name__ == '__main__':
    main()
