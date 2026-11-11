"""Separate fold-specific CPU attribution from the shared exact-patch queue.

The fitting arithmetic follows training._m1. Only raw pair/head interventions
are shared. Each plan retains its own categories, sources and representative
weights, and finalization reads exact scores without starting model forwards.
"""
from __future__ import annotations
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from .common import write
from .training import KINDS, _flat_heads, pair_weights


def make_plan(group, pairs, features, outdir, train_sources=None):
    """Return and save m1_plan.json; plan['kinds'][kind] contains the queue inputs."""
    if group not in ('E1', 'E2', 'E3'):
        raise ValueError('E0 already has historical M1 parameters; only new groups need plans')
    started = time.monotonic()
    all_fit_sources = {p['source_id'] for p in pairs if p.get('split', 'fit') == 'fit'}
    allowed = None if train_sources is None else set(train_sources)
    pairs = [p for p in pairs if p['kind'] in KINDS and p.get('split', 'fit') == 'fit'
             and (allowed is None or p['source_id'] in allowed)]
    if not pairs:
        raise ValueError('Empty training-source M1 pair set')
    number_heads = _flat_heads(features.get(pairs[0]['left'])).shape[0]
    kinds = {}
    for kind in KINDS:
        ps = [p for p in pairs if p['kind'] == kind]
        if not ps:
            raise ValueError(f'{group}: no training pairs for category {kind}')
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
        by_source = defaultdict(list)
        for pair in ps:
            by_source[pair['source_id']].append(pair)
        chosen = [min(by_source[source], key=lambda pair: pair['pair_id']) for source in sorted(by_source)]
        representative_weights = np.array([len(by_source[pair['source_id']]) if group == 'E3' else 1.0
                                           for pair in chosen], dtype=np.float64)
        if len(chosen):
            representative_weights /= representative_weights.sum()
        representatives = [dict(pair, weight=float(weight)) for pair, weight in zip(chosen, representative_weights)]
        kinds[kind] = dict(attribution=attribution.tolist(), candidates=candidates,
                           representatives=representatives, fit_pair_count=len(ps))
    fitted_sources = sorted({p['source_id'] for p in pairs})
    plan = dict(group=group, kinds=kinds, number_heads=number_heads, heads_to_select=51,
                fit_pair_counts=dict(Counter(p['kind'] for p in pairs)),
                training_sources=fitted_sources, train_sources=fitted_sources,
                validation_sources=sorted(all_fit_sources - set(fitted_sources)),
                plan_seconds=time.monotonic() - started,
                attribution_split='training sources only',
                queue_contract='Union raw (left input, right input, candidate head) interventions across plans; retain plan weights separately.')
    target = Path(outdir)
    target.mkdir(parents=True, exist_ok=True)
    write(target / 'm1_plan.json', plan)
    return plan


def finish_plan(plan, features, exact_accessor, outdir):
    """Finalize this plan from a read-only raw exact-score accessor."""
    started = time.monotonic()
    target = Path(outdir)
    target.mkdir(parents=True, exist_ok=True)
    group = plan['group']
    counts = plan['fit_pair_counts']
    ranks, attributes, exact_scores, representatives = {}, {}, {}, {}
    forward_count = 0
    for kind in KINDS:
        item = plan['kinds'][kind]
        candidates = item['candidates']
        aggregate = {head: 0.0 for head in candidates}
        exact_rows = []
        for pair in item['representatives']:
            weight = np.float64(pair['weight'])
            measured = exact_accessor(pair, candidates)
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
        ranking = sorted(candidates, key=lambda head: (-aggregate[head], head))
        ranks[kind] = ranking
        attributes[kind] = item['attribution']
        exact_scores[kind] = {str(head): float(aggregate[head]) for head in ranking}
        representatives[kind] = [dict(pair_id=pair['pair_id'], source_id=pair['source_id'], weight=float(pair['weight']))
                                 for pair in item['representatives']]
        write(target / ('m1_exact_' + kind + '.json'), exact_rows)
    def merge_rank(first, second):
        union = set(ranks[first] + ranks[second])
        first_weight = counts[first] / (counts[first] + counts[second]) if group == 'E3' else 0.5
        return sorted(union, key=lambda head: (
            first_weight * (ranks[first].index(head) if head in ranks[first] else 64)
            + (1 - first_weight) * (ranks[second].index(head) if head in ranks[second] else 64), head))
    nuisance = merge_rank('A', 'P')
    signal = merge_rank('C', 'F')
    selected = []
    for i in range(max(len(nuisance), len(signal))):
        for ranking in (nuisance, signal):
            if i < len(ranking) and ranking[i] not in selected:
                selected.append(ranking[i])
        if len(selected) >= plan.get('heads_to_select', 51):
            break
    selected = selected[:plan.get('heads_to_select', 51)]
    if len(selected) != plan.get('heads_to_select', 51):
        raise ValueError('Insufficient distinct M1 heads')
    result = dict(group=group, heads=selected, ranks=ranks, attribution=attributes,
                  exact_scores=exact_scores, exact_representatives=representatives,
                  layer_distribution={str(k): v for k, v in sorted(Counter(h // 32 for h in selected).items())},
                  fit_pair_counts=counts, seconds=plan.get('plan_seconds', 0) + time.monotonic() - started,
                  exact_forward_requests=forward_count,
                  weighting='natural category and source frequencies' if group == 'E3' else
                            'equal categories; equal sources within each category',
                  exact_sampling='one fixed pair per category and training source',
                  ranking_split='fit', selection='alternate nuisance and evidence rankings; fixed 51 heads',
                  exact_execution='shared global raw pair/head cache; aggregation remains specific to this training plan',
                  plan_seconds=plan.get('plan_seconds', 0), finalization_seconds=time.monotonic() - started)
    write(target / 'm1.json', result)
    return selected, result
