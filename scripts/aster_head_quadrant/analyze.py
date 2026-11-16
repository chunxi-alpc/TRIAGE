#!/usr/bin/env python3
"""Issue #72: CPU-only, read-only analysis of cached E1 attention heads.

No model/GPU packages are imported. All feature rows used in a fold belong to
that fold's training pair endpoints. Percentiles use average midranks on 0-100.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

KINDS = ('A', 'P', 'C', 'F')
QUADRANTS = ('AP_selective', 'CF_selective', 'mixed', 'low')
VIEWS = ('activation', 'gradient', 'patch')


def percentile_ranks(values):
    """Finite observations receive 100*(average_rank-0.5)/n; NaN stays missing."""
    values = np.asarray(values, dtype=np.float64)
    result = np.full(values.shape, np.nan)
    valid = np.isfinite(values)
    if valid.any():
        result[valid] = 100.0 * (rankdata(values[valid], method='average') - 0.5) / valid.sum()
    return result


def pair_weights(pairs):
    """E1 category/source/pair balancing, identical to training.pair_weights."""
    counts = Counter((p['kind'], p['source_id']) for p in pairs)
    sources = defaultdict(set)
    for kind, source in counts:
        sources[kind].add(source)
    if not pairs:
        return np.empty(0, dtype=np.float64)
    return np.array([1.0 / len(sources) / len(sources[p['kind']]) /
                     counts[p['kind'], p['source_id']] for p in pairs], dtype=np.float64)


def endpoint_weights(pairs, uid_to_index, size):
    """Each weighted pair contributes half of its weight to either endpoint."""
    weights = np.zeros(size, dtype=np.float64)
    for pair, weight in zip(pairs, pair_weights(pairs)):
        weights[uid_to_index[pair['left']]] += weight / 2
        weights[uid_to_index[pair['right']]] += weight / 2
    return weights


def weighted_sigma(activations, weights, batch_size=64, min_scale=1e-5):
    """Two-pass weighted population SD; only positive-weight rows are read.

    Returns (scale, floored_mask); the scale floor matches the existing M2 code.
    We accumulate in float64 to avoid cancellation in almost-constant channels.
    """
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.flatnonzero(weights > 0)
    if not len(valid) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Positive finite endpoint weights are required')
    normalized = weights / weights.sum()
    shape = activations.shape[1:]
    mean = np.zeros(shape, dtype=np.float64)
    for start in range(0, len(valid), batch_size):
        idx = valid[start:start + batch_size]
        block = np.asarray(activations[idx], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError('Nonfinite training activation')
        mean += np.einsum('i,i...->...', normalized[idx], block, optimize=False)
    variance = np.zeros(shape, dtype=np.float64)
    for start in range(0, len(valid), batch_size):
        idx = valid[start:start + batch_size]
        centered = np.asarray(activations[idx], dtype=np.float64) - mean
        variance += np.einsum('i,i...->...', normalized[idx], centered * centered, optimize=False)
    scale = np.sqrt(variance)
    floor_mask = scale < min_scale
    return np.maximum(scale, min_scale), floor_mask


def movement_rms(activations, pairs, uid_to_index, sigma, batch_size=64):
    """Category-conditional, source-balanced RMS of standardized pair changes."""
    if not pairs:
        return np.full(activations.shape[1], np.nan)
    weights = pair_weights(pairs)
    total = np.zeros(activations.shape[1], dtype=np.float64)
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        li = [uid_to_index[p['left']] for p in chunk]
        ri = [uid_to_index[p['right']] for p in chunk]
        delta = (np.asarray(activations[ri], np.float64) - np.asarray(activations[li], np.float64)) / sigma
        per_head = np.mean(delta * delta, axis=-1)
        total += np.einsum('i,ih->h', weights[start:start + len(chunk)], per_head)
    return np.sqrt(total)


def quadrants(x, y, quantile=.75):
    """Jointly finite rows set axis thresholds; equality is assigned high."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    labels = np.full(x.shape, 'missing', dtype='<U12')
    valid = np.isfinite(x) & np.isfinite(y)
    if not valid.any():
        return labels, {'x': None, 'y': None}
    tx, ty = float(np.quantile(x[valid], quantile)), float(np.quantile(y[valid], quantile))
    high_x, high_y = x >= tx, y >= ty
    labels[valid & ~high_x & ~high_y] = 'low'
    labels[valid & high_x & ~high_y] = 'AP_selective'
    labels[valid & ~high_x & high_y] = 'CF_selective'
    labels[valid & high_x & high_y] = 'mixed'
    return labels, {'x': tx, 'y': ty}


def combine_patch(exact_scores, n_heads):
    """Rank within each class's observed candidates before combining classes.

    A/P and C/F averages each require both classes. Missing values are never zero.
    Both axes are therefore present only on the four-class complete intersection.
    """
    result = {}
    for kind in KINDS:
        raw = np.full(n_heads, np.nan)
        for head, value in exact_scores.get(kind, {}).items():
            raw[int(head)] = float(value)
        result[kind + '_raw'] = raw
        result[kind + '_pct'] = percentile_ranks(raw)
    for group, kinds in (('AP', ('A', 'P')), ('CF', ('C', 'F'))):
        for suffix in ('raw', 'pct'):
            result[group + '_' + suffix] = (result[kinds[0] + '_' + suffix] +
                                            result[kinds[1] + '_' + suffix]) / 2
    result['both_available'] = np.isfinite(result['AP_pct']) & np.isfinite(result['CF_pct'])
    return result


def fold_consensus(x, y, min_folds=1):
    """Average both axes over the exact same jointly observed folds per head."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    counts = valid.sum(axis=0)
    mean_x = np.divide(np.where(valid, x, 0).sum(axis=0), counts,
                       out=np.full(counts.shape, np.nan), where=counts > 0)
    mean_y = np.divide(np.where(valid, y, 0).sum(axis=0), counts,
                       out=np.full(counts.shape, np.nan), where=counts > 0)
    mean_x[counts < min_folds] = np.nan
    mean_y[counts < min_folds] = np.nan
    return mean_x, mean_y, counts


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    result = {'n': int(valid.sum()), 'spearman': None, 'pearson': None}
    if valid.sum() >= 3 and np.ptp(x[valid]) > 0 and np.ptp(y[valid]) > 0:
        result['spearman'] = float(spearmanr(x[valid], y[valid]).statistic)
        result['pearson'] = float(np.corrcoef(x[valid], y[valid])[0, 1])
    return result


def view_summary(x, y, quantile=.75):
    labels, thresholds = quadrants(x, y, quantile)
    n = int(np.count_nonzero(labels != 'missing'))
    counts = {q: int(np.count_nonzero(labels == q)) for q in QUADRANTS}
    active = n - counts['low']
    return dict(n=n, thresholds=thresholds,
                quadrants={q: {'count': counts[q], 'fraction': counts[q] / n if n else None} for q in QUADRANTS},
                high_on_either_axis=active,
                mixed_fraction_among_high_on_either_axis=counts['mixed'] / active if active else None,
                axis_correlation=correlation(x, y))


def json_clean(value):
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def dump_json(path, value):
    path.write_text(json.dumps(json_clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(json_clean(row) for row in rows)


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def patch_source_coverage(m1, train_sources):
    result = {}
    for kind in KINDS:
        representatives = m1.get('exact_representatives', {}).get(kind, [])
        sources = {r['source_id'] for r in representatives}
        if sources - train_sources:
            raise ValueError('Cached M1 representatives include nontraining sources')
        result[kind] = {'candidate_heads': len(m1['exact_scores'].get(kind, {})),
                        'representative_pairs': len(representatives), 'sources': len(sources),
                        'source_ids': sorted(sources)}
    return result


def summarize_selectivity(labels, raw_ap, raw_cf):
    result = {}
    for q in QUADRANTS:
        mask = (labels == q) & np.isfinite(raw_ap) & np.isfinite(raw_cf)
        delta = raw_ap[mask] - raw_cf[mask]
        result[q] = dict(n=int(mask.sum()), mean_AP_effect=float(np.mean(raw_ap[mask])) if mask.any() else None,
                         mean_CF_effect=float(np.mean(raw_cf[mask])) if mask.any() else None,
                         median_AP_effect=float(np.median(raw_ap[mask])) if mask.any() else None,
                         median_CF_effect=float(np.median(raw_cf[mask])) if mask.any() else None,
                         AP_greater_count=int(np.sum(delta > 0)), CF_greater_count=int(np.sum(delta < 0)),
                         equal_effect_count=int(np.sum(delta == 0)),
                         mean_AP_minus_CF=float(np.mean(delta)) if mask.any() else None,
                         median_AP_minus_CF=float(np.median(delta)) if mask.any() else None,
                         AP_greater_fraction=float(np.mean(delta > 0)) if mask.any() else None,
                         CF_greater_fraction=float(np.mean(delta < 0)) if mask.any() else None)
    return result


def run(input_root, output_dir, batch_size=64):
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_all = [json.loads(line) for line in (input_root / 'data/E1_pairs.jsonl').read_text().splitlines() if line.strip()]
    folds = load_json(input_root / 'folds.json')['folds']
    uids = load_json(input_root / 'runtime/fit_uids.json')
    index = {uid: i for i, uid in enumerate(uids)}
    heads_path = input_root / 'runtime/fit_heads.npy'
    present_path = input_root / 'runtime/fit_present.npy'
    activations = np.load(heads_path, mmap_mode='r') if heads_path.exists() else None
    present = np.load(present_path, mmap_mode='r') if present_path.exists() else None
    n_heads = 1152
    if activations is not None and activations.shape[1] != n_heads:
        raise ValueError('Unexpected attention-head dimension')
    fold_rows, details, stored = [], [], []
    for fold in folds:
        fold_id = int(fold['fold'])
        train_sources = set(fold['train_sources'])
        pairs = [p for p in pairs_all if p['source_id'] in train_sources]
        wanted = sorted({p[key] for p in pairs for key in ('left', 'right')})
        available = {uid for uid in wanted if uid in index and activations is not None and
                     (present is None or int(present[index[uid]]) & 1)}
        complete_pairs = [p for p in pairs if p['left'] in available and p['right'] in available]
        coverage = dict(training_sources=len(train_sources), total_pairs=len(pairs), available_pairs=len(complete_pairs),
                        requested_endpoints=len(wanted), available_endpoints=len(available),
                        endpoint_coverage=len(available) / len(wanted) if wanted else 0,
                        missing_endpoint_uids=sorted(set(wanted) - available),
                        missing_reason='none' if len(available) == len(wanted) else
                        'fit_heads.npy unavailable, UID absent, or activation-present bit unset; no inference attempted')
        # A complete training-endpoint set is needed for the requested fixed estimand.
        # Incomplete cache coverage is reported and this activation view remains missing.
        movements = {q: np.full(n_heads, np.nan) for q in KINDS}
        if len(complete_pairs) == len(pairs) and pairs:
            weights = endpoint_weights(pairs, index, len(uids))
            sigma, floored = weighted_sigma(activations, weights, batch_size=batch_size)
            coverage['sigma_floor_channels'] = int(floored.sum())
            coverage['sigma_channels'] = int(sigma.size)
            coverage['sigma_min'] = float(sigma.min())
            coverage['sigma_max'] = float(sigma.max())
            coverage['endpoint_weight_sum'] = float(weights.sum())
            for q in KINDS:
                ps = [p for p in pairs if p['kind'] == q]
                movements[q] = movement_rms(activations, ps, index, sigma, batch_size=batch_size)
        for q in KINDS:
            ps = [p for p in pairs if p['kind'] == q]
            coverage[q] = dict(pairs=len(ps), sources=len({p['source_id'] for p in ps}),
                               available_pairs=sum(p['left'] in available and p['right'] in available for p in ps))
        m1 = load_json(input_root / f'models/E1/fold{fold_id}/m1.json')
        patch = combine_patch(m1['exact_scores'], n_heads)
        gradient = {q: np.asarray(m1['attribution'][q], float) for q in KINDS}
        gradient_ranks = {q: percentile_ranks(gradient[q]) for q in KINDS}
        ap_rms = np.sqrt((movements['A'] ** 2 + movements['P'] ** 2) / 2)
        cf_rms = np.sqrt((movements['C'] ** 2 + movements['F'] ** 2) / 2)
        coords = dict(activation=(percentile_ranks(ap_rms), percentile_ranks(cf_rms)),
                      gradient=((gradient_ranks['A'] + gradient_ranks['P']) / 2,
                                (gradient_ranks['C'] + gradient_ranks['F']) / 2),
                      patch=(patch['AP_pct'], patch['CF_pct']))
        selected = np.zeros(n_heads, dtype=bool)
        selected[m1['heads']] = True
        labels = {view: quadrants(*xy)[0] for view, xy in coords.items()}
        fold_detail = dict(fold=fold_id, activation_cache=coverage,
                           patch_sources=patch_source_coverage(m1, train_sources),
                           patch_complete_four_class=int(patch['both_available'].sum()),
                           patch_group_union_intersection=int(np.count_nonzero(
                               (np.isfinite(patch['A_raw']) | np.isfinite(patch['P_raw'])) &
                               (np.isfinite(patch['C_raw']) | np.isfinite(patch['F_raw'])))),
                           m1_selected=int(selected.sum()),
                           views={view: view_summary(*xy) for view, xy in coords.items()},
                           attribution_patch_by_category={q: correlation(gradient[q], patch[q + '_raw']) for q in KINDS},
                           attribution_patch_by_group={group: correlation(coords['gradient'][i], patch[group + '_pct'])
                                                       for i, group in enumerate(('AP', 'CF'))})
        for view in VIEWS:
            fold_detail['views'][view]['m1_selected_quadrants'] = dict(Counter(labels[view][selected]))
            fold_detail['views'][view]['m1_selected_with_view'] = int(np.sum(selected & (labels[view] != 'missing')))
        details.append(fold_detail)
        stored.append(dict(fold=fold_id, coords=coords, labels=labels, selected=selected, patch=patch,
                           movements=movements, gradient=gradient, ap_rms=ap_rms, cf_rms=cf_rms))
        for h in range(n_heads):
            row = dict(fold=fold_id, head=h, layer=h // 32, head_in_layer=h % 32,
                       activation_AP_rms=ap_rms[h], activation_CF_rms=cf_rms[h], m1_selected=int(selected[h]))
            for q in KINDS:
                row.update({f'activation_{q}_rms': movements[q][h], f'gradient_{q}_raw': gradient[q][h],
                            f'gradient_{q}_pct': gradient_ranks[q][h], f'patch_{q}_raw': patch[q + '_raw'][h],
                            f'patch_{q}_pct': patch[q + '_pct'][h]})
            for view in VIEWS:
                row.update({f'{view}_AP_pct': coords[view][0][h], f'{view}_CF_pct': coords[view][1][h],
                            f'{view}_quadrant': labels[view][h]})
            row.update(patch_AP_raw=patch['AP_raw'][h], patch_CF_raw=patch['CF_raw'][h],
                       patch_both_available=int(patch['both_available'][h]))
            fold_rows.append(row)
        print(json.dumps({'fold_completed': fold_id, 'endpoints': coverage['available_endpoints'],
                          'patch_complete': fold_detail['patch_complete_four_class'],
                          'seconds': round(time.monotonic() - started, 2)}), flush=True)

    consensus, all_labels, view_details = {}, {}, {}
    selected_matrix = np.stack([fold['selected'] for fold in stored])
    selected_counts = selected_matrix.sum(axis=0)
    for view in VIEWS:
        xs = np.stack([fold['coords'][view][0] for fold in stored])
        ys = np.stack([fold['coords'][view][1] for fold in stored])
        mx, my, count = fold_consensus(xs, ys, min_folds=3 if view == 'patch' else len(folds))
        labels, _ = quadrants(mx, my)
        fold_labels = np.stack([fold['labels'][view] for fold in stored])
        stable = ((fold_labels == labels) & (fold_labels != 'missing')).sum(axis=0)
        eligible = labels != 'missing'
        modal = np.array([max(Counter(row[row != 'missing']).values(), default=0) for row in fold_labels.T])
        consensus[view] = dict(x=mx, y=my, count=count, stable=stable, modal=modal)
        all_labels[view] = labels
        vd = view_summary(mx, my)
        vd['sensitivity_median'] = view_summary(mx, my, quantile=.5)
        vd['stability'] = dict(denominator=int(eligible.sum()),
                              consensus_agreement_fold_counts=dict(Counter(stable[eligible])),
                              modal_agreement_fold_counts=dict(Counter(modal[eligible])),
                              all_five_same_quadrant=int(np.sum(eligible & (modal == 5))),
                              all_five_match_consensus=int(np.sum(eligible & (stable == 5))),
                              at_least_four_match_consensus=int(np.sum(eligible & (stable >= 4))))
        vd['stability']['by_consensus_quadrant'] = {
            q: dict(n=int(np.sum(labels == q)),
                    all_five_match=int(np.sum((labels == q) & (stable == 5))),
                    at_least_four_match=int(np.sum((labels == q) & (stable >= 4))))
            for q in QUADRANTS}
        vd['m1_selected_any'] = {'n': int(np.sum((selected_counts > 0) & eligible)),
                                 'quadrants': dict(Counter(labels[(selected_counts > 0) & eligible]))}
        vd['m1_selected_all'] = {'n': int(np.sum((selected_counts == len(folds)) & eligible)),
                                 'quadrants': dict(Counter(labels[(selected_counts == len(folds)) & eligible]))}
        vd['m1_selection_events'] = {'n': int(selected_counts[eligible].sum()),
                                    'quadrants': {q: int(selected_counts[labels == q].sum()) for q in QUADRANTS}}
        vd['layer_distribution'] = {str(layer): dict(Counter(labels[layer * 32:(layer + 1) * 32])) for layer in range(36)}
        fold_correlations = []
        for i in range(len(folds)):
            for j in range(i + 1, len(folds)):
                valid = (fold_labels[i] != 'missing') & (fold_labels[j] != 'missing')
                fold_correlations.append(dict(fold_a=stored[i]['fold'], fold_b=stored[j]['fold'],
                                              n=int(valid.sum()), quadrant_agreement=float(np.mean(fold_labels[i][valid] == fold_labels[j][valid])) if valid.any() else None,
                                              AP=correlation(xs[i], xs[j]), CF=correlation(ys[i], ys[j])))
        vd['between_folds'] = fold_correlations
        view_details[view] = vd

    raw_ap, raw_cf, raw_counts = fold_consensus(np.stack([f['patch']['AP_raw'] for f in stored]),
                                               np.stack([f['patch']['CF_raw'] for f in stored]), min_folds=3)
    consensus_rows = []
    for h in range(n_heads):
        row = dict(head=h, layer=h // 32, head_in_layer=h % 32,
                   m1_selected_count=int(selected_counts[h]), m1_selected_any=int(selected_counts[h] > 0),
                   m1_selected_all=int(selected_counts[h] == len(folds)),
                   patch_AP_raw=raw_ap[h], patch_CF_raw=raw_cf[h], patch_covered_folds=int(raw_counts[h]))
        for view in VIEWS:
            c = consensus[view]
            row.update({f'{view}_AP_pct': c['x'][h], f'{view}_CF_pct': c['y'][h],
                        f'{view}_quadrant': all_labels[view][h], f'{view}_stable_folds': int(c['stable'][h]),
                        f'{view}_modal_folds': int(c['modal'][h]), f'{view}_covered_folds': int(c['count'][h])})
        consensus_rows.append(row)
    # Repeat consensus membership on fold rows to make the main metric table self-contained.
    lookup = {row['head']: row for row in consensus_rows}
    for row in fold_rows:
        for key, value in lookup[row['head']].items():
            if key not in ('head', 'layer', 'head_in_layer'):
                row['consensus_' + key] = value
    correlations = {}
    for i, first in enumerate(VIEWS):
        for second in VIEWS[i + 1:]:
            valid = (all_labels[first] != 'missing') & (all_labels[second] != 'missing')
            correlations[first + '_vs_' + second] = dict(
                AP=correlation(consensus[first]['x'], consensus[second]['x']),
                CF=correlation(consensus[first]['y'], consensus[second]['y']),
                quadrant_agreement=float(np.mean(all_labels[first][valid] == all_labels[second][valid])) if valid.any() else None,
                n=int(valid.sum()),
                quadrant_crosstab={q: dict(Counter(all_labels[second][valid & (all_labels[first] == q)])) for q in QUADRANTS})
    summary = dict(
        task='GitHub issue #72', cpu_only=True, gpu_used=False, model_forward_calls=0, m1_reruns=0,
        input_root=str(input_root), execution=dict(python=platform.python_version(), numpy=np.__version__,
                                                   cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES', 'not set'),
                                                   seconds=time.monotonic() - started),
        protocol=dict(number_heads=n_heads, layers=36, heads_per_layer=32, folds=len(folds),
                      percentile='100 * (average_rank - 0.5) / n finite heads; ties receive average rank',
                      threshold_quantile=.75, sensitivity_quantile=.5, boundary='>= is high',
                      weights='equal categories, equal sources within category, equal pairs within source',
                      sigma='weighted population SD of unique training endpoints; each pair weight split equally across endpoints',
                      sigma_floor=1e-5, precision='float64 accumulation from cached float16 activations',
                      class_rms_weights='category-conditional pair_weights: weights in each category sum to 1',
                      patch_complete='A and P and C and F must all have exact effects in the same fold',
                      patch_consensus_minimum_folds=3,
                      patch_percentile='per category, within the 64 gradient-screened candidates before complete-case intersection',
                      patch_raw_consensus='two-category means then mean across the same complete-coverage folds as percentile axes',
                      source_bootstrap=False,
                      incomplete_activation_cache='report coverage and leave fold activation axes missing; never impute or infer'),
        folds=details, views=view_details, correlations=correlations,
        patch_coverage=dict(per_fold_complete_four_class=[d['patch_complete_four_class'] for d in details],
                            per_fold_group_union_intersection=[d['patch_group_union_intersection'] for d in details],
                            complete_fold_count_distribution=dict(Counter(raw_counts)),
                            at_least_three_folds=int(np.sum(raw_counts >= 3)),
                            all_five_folds=int(np.sum(raw_counts == 5)),
                            candidate_bias='gradient-screened candidate set; not an unbiased distribution over 1152 heads'),
        m1_selection=dict(per_fold=[int(x.sum()) for x in selected_matrix], unique_any_fold=int(np.sum(selected_counts > 0)),
                          selected_all_folds=int(np.sum(selected_counts == 5)), frequency_distribution=dict(Counter(selected_counts))),
        exact_effect_by_consensus_quadrant={view: summarize_selectivity(all_labels[view], raw_ap, raw_cf) for view in VIEWS},
        score_interpretation='gradient attribution is a local first-order score-sensitivity proxy; cached exact patches measure intervention magnitude')
    write_csv(output_dir / 'head_quadrant_metrics.csv', fold_rows)
    write_csv(output_dir / 'head_quadrant_consensus.csv', consensus_rows)
    dump_json(output_dir / 'quadrant_summary.json', summary)
    print(json.dumps({'complete': True, 'seconds': round(time.monotonic() - started, 2),
                      'output_dir': str(output_dir), 'patch_consensus_heads': summary['patch_coverage']['at_least_three_folds']}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    run(args.input_root, args.output_dir, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
