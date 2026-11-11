"""Fresh M1/M2 and evidence-constrained M3 for the paired-data comparison.

No model loading or GPU work occurs on import. The caller supplies a disk-backed
feature Store and a cached exact-patching callback. Historical files stay read-only.
"""
from __future__ import annotations
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import LinearConstraint, brentq, minimize
from scipy.special import expit
from sklearn.utils.extmath import randomized_svd

SEED = 20261108
KINDS = ('A', 'P', 'C', 'F')
DELTA = 0.2923349
REGULARIZATION = 0.01
WEIGHT_TOTAL = 4887.0


def _dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + str(__import__("os").getpid()))
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def pair_weights(pairs, group, total=1.0):
    """Category and within-category source weights, or natural pair weights."""
    if not pairs:
        return np.empty(0, dtype=np.float64)
    if group in ('E0', 'E3'):
        return np.full(len(pairs), total / len(pairs), dtype=np.float64)
    counts = Counter((p['kind'], p['source_id']) for p in pairs)
    sources = defaultdict(set)
    for kind, source in counts:
        sources[kind].add(source)
    classes = len(sources)
    return np.array([total / classes / len(sources[p['kind']]) / counts[(p['kind'], p['source_id'])]
                     for p in pairs], dtype=np.float64)


def _old_difficulty(pair, features):
    a, b = features.ell(pair['left']), features.ell(pair['right'])
    wrong = int((a > 0) != bool(pair['left_label']) or (b > 0) != bool(pair['right_label']))
    return 1 + wrong + int(min(abs(a), abs(b)) < 1)


def _flat_heads(feature):
    value = np.asarray(feature['heads'], dtype=np.float32)
    return value.reshape(-1, value.shape[-1])


def _m1(group, pairs, features, patcher, outdir, heads_to_select=51):
    if patcher is None:
        raise ValueError('Fresh M1 requires the exact-patching callback')
    start = time.monotonic()
    one = features.get(pairs[0]['left'])
    number_heads = _flat_heads(one).shape[0]
    layer_heads = 32
    ranks, attributes, exact_scores, representatives = {}, {}, {}, {}
    forward_count = 0
    counts = Counter(p['kind'] for p in pairs)
    for kind in KINDS:
        ps = [p for p in pairs if p['kind'] == kind]
        if not ps:
            raise ValueError(f'{group}: no training pairs for category {kind}')
        if group == 'E0':
            weights = np.array([_old_difficulty(p, features) for p in ps], dtype=np.float64)
            weights /= weights.sum()
        else:
            weights = pair_weights(ps, group)
        attribution = np.zeros(number_heads, dtype=np.float64)
        for pair, weight in zip(ps, weights):
            left, right = features.get(pair['left']), features.get(pair['right'])
            if 'grads' not in left:
                raise ValueError('Missing fit attribution gradient for ' + pair['left'])
            gradients = np.asarray(left['grads'], dtype=np.float32).reshape(number_heads, -1)
            diff = _flat_heads(right) - _flat_heads(left)
            attribution += weight * np.abs((diff * gradients).sum(axis=-1))
        candidates = np.argsort(-attribution)[:min(64, number_heads)].tolist()
        if group == 'E0':
            ordered = sorted(ps, key=lambda p: (-_old_difficulty(p, features), p['pair_id']))
            chosen, seen = [], set()
            for p in ordered:
                if p['source_id'] in seen or abs(features.ell(p['right']) - features.ell(p['left'])) < 1e-8:
                    continue
                chosen.append(p)
                seen.add(p['source_id'])
                if len(chosen) >= 12:
                    break
            representative_weights = np.array([_old_difficulty(p, features) for p in chosen], dtype=np.float64)
        else:
            by_source = defaultdict(list)
            for p in ps:
                by_source[p['source_id']].append(p)
            chosen = [min(by_source[source], key=lambda p: p['pair_id']) for source in sorted(by_source)]
            representative_weights = np.array([len(by_source[p['source_id']]) if group == 'E3' else 1.0
                                               for p in chosen], dtype=np.float64)
        if len(chosen):
            representative_weights /= representative_weights.sum()
        aggregate = {h: 0.0 for h in candidates}
        exact_rows = []
        for pair, weight in zip(chosen, representative_weights):
            measured = patcher(pair, candidates)
            if isinstance(measured, dict):
                measured = [(int(h), row['forward_ell'], row['reverse_ell']) for h, row in measured.items()]
            returned = set()
            left_score, right_score = features.ell(pair['left']), features.ell(pair['right'])
            for head, forward, reverse in measured:
                head = int(head)
                if head not in aggregate or head in returned:
                    raise ValueError('Unexpected/duplicate exact patch head')
                returned.add(head)
                effect = 0.5 * (abs(float(forward) - left_score) + abs(float(reverse) - right_score))
                aggregate[head] += weight * effect
                exact_rows.append(dict(pair_id=pair['pair_id'], source_id=pair['source_id'], head=head,
                                       weight=float(weight), forward_ell=float(forward), reverse_ell=float(reverse),
                                       absolute_bidirectional_effect=float(effect)))
                forward_count += 2
            if returned != set(candidates):
                raise ValueError('Incomplete exact patch candidate results')
        ranking = sorted(candidates, key=lambda h: (-aggregate[h], h))
        ranks[kind] = ranking
        attributes[kind] = attribution.tolist()
        exact_scores[kind] = {str(h): float(aggregate[h]) for h in ranking}
        representatives[kind] = [dict(pair_id=p['pair_id'], source_id=p['source_id'], weight=float(w))
                                 for p, w in zip(chosen, representative_weights)]
        _dump(Path(outdir) / ('m1_exact_' + kind + '.json'), exact_rows)
    def merge_rank(first, second):
        union = set(ranks[first] + ranks[second])
        if group == 'E3':
            first_weight = counts[first] / (counts[first] + counts[second])
        else:
            first_weight = 0.5
        return sorted(union, key=lambda h: (
            first_weight * (ranks[first].index(h) if h in ranks[first] else 64)
            + (1 - first_weight) * (ranks[second].index(h) if h in ranks[second] else 64), h))
    nuisance = merge_rank('A', 'P')
    signal = merge_rank('C', 'F')
    selected = []
    for i in range(max(len(nuisance), len(signal))):
        for ranking in (nuisance, signal):
            if i < len(ranking) and ranking[i] not in selected:
                selected.append(ranking[i])
        if len(selected) >= heads_to_select:
            break
    selected = selected[:heads_to_select]
    if len(selected) != heads_to_select:
        raise ValueError('Insufficient distinct M1 heads')
    result = dict(group=group, heads=selected, ranks=ranks, attribution=attributes,
                  exact_scores=exact_scores, exact_representatives=representatives,
                  layer_distribution={str(k): v for k, v in sorted(Counter(h // layer_heads for h in selected).items())},
                  fit_pair_counts=dict(counts), seconds=time.monotonic() - start,
                  exact_forward_requests=forward_count,
                  weighting='historical 1+wrong+boundary' if group == 'E0' else
                            'natural category and source frequencies' if group == 'E3' else
                            'equal categories; equal sources within each category',
                  exact_sampling='up to 12 historical difficulty-prioritized sources per category' if group == 'E0'
                                 else 'one fixed pair per category and training source',
                  ranking_split='fit', selection='alternate nuisance and evidence rankings; fixed 51 heads')
    _dump(Path(outdir) / 'm1.json', result)
    return selected, result


def _basis(matrix, rank=16):
    matrix = np.asarray(matrix, dtype=np.float32)
    if not len(matrix):
        return np.empty((matrix.shape[1], 0), np.float32), np.empty(0, np.float32)
    _, singulars, vt = randomized_svd(matrix, n_components=min(rank, min(matrix.shape)), random_state=SEED)
    keep = singulars > max(1e-7, float(singulars.max()) * 1e-6)
    return vt[keep].T.astype(np.float32), singulars.astype(np.float32)


def _orth_nuisance(signal, nuisance):
    if nuisance.shape[1] == 0:
        return np.empty((nuisance.shape[0], 0), np.float32), np.empty(0, np.float64)
    q = np.linalg.qr(signal.astype(np.float64), mode='reduced')[0] if signal.shape[1] else np.empty((signal.shape[0], 0))
    residual = nuisance.astype(np.float64) - q @ (q.T @ nuisance.astype(np.float64))
    residual -= q @ (q.T @ residual)
    u, singulars, _ = np.linalg.svd(residual, full_matrices=False)
    keep = singulars > max(1e-10, (float(singulars.max()) if len(singulars) else 0) * 1e-6)
    return u[:, keep].astype(np.float32), singulars


def _effective_rank(singulars):
    eigenvalues = np.square(np.asarray(singulars, np.float64))
    if not eigenvalues.size or eigenvalues.sum() <= 0:
        return 0.0
    probability = eigenvalues / eigenvalues.sum()
    positive = probability > 0
    return float(np.exp(-np.sum(probability[positive] * np.log(probability[positive]))))


def _m2(group, heads, pairs, features, views, outdir):
    start = time.monotonic()
    if group == 'E0':
        standard_views = [v for v in views if v.get('variant') == 'full']
        ids = [v['view_id'] for v in standard_views]
        node_weights = np.full(len(ids), 1 / max(len(ids), 1), np.float64)
    else:
        node = defaultdict(float)
        for pair, weight in zip(pairs, pair_weights(pairs, group)):
            node[pair['left']] += weight * 0.5
            node[pair['right']] += weight * 0.5
        ids = sorted(node)
        node_weights = np.array([node[uid] for uid in ids], np.float64)
    if not ids:
        raise ValueError('No fit-only standardization inputs')
    x = np.stack([features.heads(uid, heads).reshape(-1) for uid in ids]).astype(np.float32)
    if group == 'E0':
        mu, scale = x.mean(0), x.std(0)
    else:
        mu = np.sum(x * node_weights[:, None], axis=0).astype(np.float32)
        scale = np.sqrt(np.sum(np.square(x.astype(np.float64) - mu) * node_weights[:, None], axis=0)).astype(np.float32)
    scale = np.maximum(scale, 1e-5)
    standardized_inputs = (x - mu) / scale
    del x
    difference = {}
    svd_weights = {}
    for name, kinds in [('S', ('C', 'F')), ('U', ('A', 'P'))]:
        ps = [p for p in pairs if p['kind'] in kinds]
        matrix = np.stack([(features.heads(p['right'], heads).reshape(-1)
                            - features.heads(p['left'], heads).reshape(-1)) / scale for p in ps]).astype(np.float32)
        weights = pair_weights(ps, group, total=float(len(ps)))
        difference[name] = matrix if group in ('E0', 'E3') else matrix * np.sqrt(weights).astype(np.float32)[:, None]
        svd_weights[name] = weights
    signal, s_singulars = _basis(difference['S'])
    nuisance, u_singulars = _basis(difference['U'])
    orthogonal, n_orth_singulars = _orth_nuisance(signal, nuisance)
    projected = standardized_inputs @ orthogonal
    projected_singulars = np.linalg.svd(projected, compute_uv=False).astype(np.float32)
    del standardized_inputs, difference
    output = dict(heads=np.array(heads, np.int64), mu=mu, scale=scale, S=signal, U=nuisance, N=orthogonal,
                  S_singular_values=s_singulars, U_singular_values=u_singulars,
                  N_orthogonalization_singular_values=n_orth_singulars,
                  projected_feature_singular_values=projected_singulars,
                  S_effective_rank=np.array(signal.shape[1]), U_effective_rank=np.array(nuisance.shape[1]),
                  N_effective_rank=np.array(orthogonal.shape[1]),
                  projected_feature_entropy_effective_rank=np.array(_effective_rank(projected_singulars)),
                  standardization_ids=np.array(ids), standardization_weights=node_weights,
                  S_pair_weights=svd_weights['S'], U_pair_weights=svd_weights['U'])
    np.savez_compressed(Path(outdir) / 'm2.npz', **output)
    summary = dict(rank=16, dimension=len(mu), S_rank=signal.shape[1], U_rank=nuisance.shape[1],
                   N_rank=orthogonal.shape[1], S_singular_values=s_singulars.tolist(),
                   U_singular_values=u_singulars.tolist(), projected_feature_singular_values=projected_singulars.tolist(),
                   projected_feature_entropy_effective_rank=_effective_rank(projected_singulars),
                   standardization_input_count=len(ids), seconds=time.monotonic() - start,
                   standardization='historical full fit inputs, unweighted' if group == 'E0' else
                                   'pair endpoint weights aggregated by unique input',
                   weighting='natural pairs' if group in ('E0', 'E3') else 'equal categories and within-category sources')
    return output, summary


def coordinates(uid, features, support):
    raw = features.heads(uid, support['heads']).reshape(-1).astype(np.float32)
    return (((raw - support['mu']) / support['scale']) @ support['N']).astype(np.float64)


def refit_m3(group, pairs, features, natural_views, support, outdir, delta=DELTA, regularization=REGULARIZATION):
    """Reuse freshly fitted M1/M2 for an explicitly requested evidence-budget fit."""
    start = time.monotonic()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache = {}
    def coord(uid):
        if uid not in cache:
            cache[uid] = coordinates(uid, features, support)
        return cache[uid]
    ap = [p for p in pairs if p['kind'] in ('A', 'P')]
    cf = [p for p in pairs if p['kind'] in ('C', 'F')]
    X = np.stack([coord(p['right']) - coord(p['left']) for p in ap])
    y = np.array([features.ell(p['right']) - features.ell(p['left']) for p in ap], np.float64)
    E = np.stack([coord(p['right']) - coord(p['left']) for p in cf])
    weights = pair_weights(ap, group, total=WEIGHT_TOTAL)
    Q = X.T @ (weights[:, None] * X) + regularization * np.eye(X.shape[1])
    q = X.T @ (weights * y)
    unconstrained = -np.linalg.solve(Q, q)
    objective_scale = max(float(np.average(y * y, weights=weights)), 1.0) * WEIGHT_TOTAL
    Qn, qn = Q / objective_scale, q / objective_scale
    maximum = float(np.max(np.abs(E @ unconstrained))) if len(E) else 0.0
    initial = unconstrained * min(1.0, delta / max(maximum, 1e-15))
    result = minimize(lambda w: float(0.5 * w @ Qn @ w + qn @ w), initial,
                      jac=lambda w: Qn @ w + qn, method='SLSQP',
                      constraints=[LinearConstraint(E, -delta, delta)] if len(E) else [],
                      options=dict(ftol=1e-13, maxiter=3000, disp=False))
    w = result.x
    corrections = E @ w
    violation = max(0.0, float(np.max(np.abs(corrections))) - delta) if len(E) else 0.0
    if not result.success or violation > 1e-7:
        raise RuntimeError(f'M3 optimization failed: {result.message}; maximum violation={violation}')
    original = [v for v in natural_views if v.get('variant', 'full') == 'full']
    if not original:
        raise ValueError('No training-source natural calibration nodes')
    natural_base = np.array([features.ell(v['view_id']) + coord(v['view_id']) @ w for v in original])
    natural_labels = np.array([v['label'] for v in original], np.float64)
    derivative = lambda b: float(np.mean(expit(natural_base + b) - natural_labels))
    lo, hi = -1.0, 1.0
    while derivative(lo) > 0:
        lo *= 2
    while derivative(hi) < 0:
        hi *= 2
    b = float(brentq(derivative, lo, hi, xtol=1e-12))
    np.savez_compressed(outdir / 'm3.npz', w=w, b=np.array(b), delta=np.array(delta),
                        regularization=np.array(regularization), AP_weights=weights,
                        AP_pair_ids=np.array([p['pair_id'] for p in ap]),
                        CF_pair_ids=np.array([p['pair_id'] for p in cf]),
                        calibration_ids=np.array([v['view_id'] for v in original]))
    replay = y + X @ w
    summary = dict(success=bool(result.success), solver='SLSQP', message=str(result.message),
                   iterations=int(result.nit), rank=int(len(w)), regularization=float(regularization), delta=float(delta),
                   AP_pairs=len(ap), CF_pairs=len(cf), AP_weight_sum=float(weights.sum()),
                   AP_weight_by_kind={kind: float(sum(weight for p, weight in zip(ap, weights) if p['kind'] == kind)) for kind in ('A', 'P')},
                   calibration_nodes=len(original), b=b, w=w.tolist(),
                   constraint_max_violation=violation, constraint_satisfaction_rate=float(np.mean(np.abs(corrections) <= delta + 1e-7)),
                   max_CF_correction=float(np.max(np.abs(corrections))),
                   fit_AP_original_RMS=float(np.sqrt(np.mean(y * y))),
                   fit_AP_corrected_RMS=float(np.sqrt(np.mean(replay * replay))),
                   fit_CF_correction_RMS=float(np.sqrt(np.mean(corrections * corrections))),
                   weighted_regression_objective=float(np.sum(weights * replay * replay) + regularization * (w @ w)),
                   seconds=time.monotonic() - start)
    _dump(outdir / 'm3.json', summary)
    return summary


def fit_group(group, pairs, features, views, natural_views, outdir, train_sources=None, patcher=None, delta=DELTA):
    """Fit all three modules using only the selected training sources."""
    started = time.monotonic()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    all_fit_sources = {p['source_id'] for p in pairs if p.get('split', 'fit') == 'fit'}
    allowed = set(train_sources) if train_sources is not None else None
    usable = lambda row: row.get('split', 'fit') == 'fit' and (allowed is None or row['source_id'] in allowed)
    pairs = [p for p in pairs if p['kind'] in KINDS and usable(p)]
    views = [v for v in views if usable(v)]
    natural_views = [v for v in natural_views if usable(v)]
    if not pairs:
        raise ValueError('Empty training-source pair set')
    heads, m1 = _m1(group, pairs, features, patcher, outdir)
    support, m2 = _m2(group, heads, pairs, features, views, outdir)
    m3 = refit_m3(group, pairs, features, natural_views, support, outdir, delta=delta)
    fitted_sources = sorted({p['source_id'] for p in pairs})
    summary = dict(group=group, training_sources=fitted_sources, train_sources=fitted_sources,
                   validation_sources=sorted(all_fit_sources - set(fitted_sources)),
                   training_pair_count=len(pairs), heads=heads, M1=m1, M2=m2, M3=m3,
                   delta=float(delta), rank=16, regularization=REGULARIZATION,
                   prediction='ell + (((heads_selected-mu)/scale) @ N) @ w + b',
                   fresh_modules=['M1', 'M2', 'M3'], seconds=time.monotonic() - started)
    _dump(outdir / 'model.json', summary)
    return summary
