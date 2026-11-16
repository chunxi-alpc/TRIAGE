"""Adaptive M1: source-cross-fitted gradient screening and exact patch confirmation.

The candidate set and final head set are data-dependent. No Top-K head budget is
used. Exact interventions are evaluated on source folds not used for that
fold's gradient screen, with within-source donor-swapped controls.
"""
from __future__ import annotations

import hashlib
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .common import read, write
from .training import KINDS, _flat_heads, pair_weights

M1_CONFIG = read(Path(__file__).with_name("m1_adaptive_v2_config.json"))
METHOD_VERSION = M1_CONFIG["method_version"]


def _stable_seed(*parts):
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _source_folds(sources, group):
    n_folds = int(M1_CONFIG["inner_source_folds"])
    sources = sorted(set(sources))
    if len(sources) < n_folds:
        raise ValueError(f"M1 requires at least {n_folds} training sources; found {len(sources)}")
    rng = np.random.default_rng(_stable_seed(METHOD_VERSION, group, *sources))
    shuffled = np.asarray(sources, dtype=object)[rng.permutation(len(sources))]
    return [sorted(chunk.tolist()) for chunk in np.array_split(shuffled, n_folds)]


def _gradient_attribution(pairs, features, group, number_heads):
    if not pairs:
        return np.zeros(number_heads, dtype=np.float64)
    weights = pair_weights(pairs, group)
    attribution = np.zeros(number_heads, dtype=np.float64)
    for pair, weight in zip(pairs, weights):
        left = features.get(pair["left"])
        right = features.get(pair["right"])
        if "grads" not in left or "grads" not in right:
            raise ValueError("M1 v2 requires endpoint gradients for " + pair["pair_id"])
        grad_left = np.asarray(left["grads"], dtype=np.float32).reshape(number_heads, -1)
        grad_right = np.asarray(right["grads"], dtype=np.float32).reshape(number_heads, -1)
        difference = _flat_heads(right) - _flat_heads(left)
        forward = np.abs(np.sum(difference * grad_left, axis=-1))
        reverse = np.abs(np.sum(-difference * grad_right, axis=-1))
        attribution += float(weight) * 0.5 * (forward + reverse)
    return attribution


def _robust_screen(attribution):
    values = np.asarray(attribution, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        scale = float(np.std(values))
    if scale <= 1e-12:
        return [], np.zeros_like(values)
    robust_z = (values - median) / scale
    threshold = float(M1_CONFIG["gradient_robust_z_threshold"])
    candidates = np.flatnonzero(robust_z >= threshold).astype(int).tolist()
    return candidates, robust_z


def _validation_representatives(pairs, validation_sources, kind, fold_index):
    """Build observed pairs and source-/label-matched donor-swapped controls."""
    validation_sources = set(validation_sources)
    grouped = defaultdict(list)
    for pair in pairs:
        if pair["source_id"] not in validation_sources:
            continue
        pattern = (int(pair["left_label"]), int(pair["right_label"]))
        grouped[(pair["source_id"], pattern)].append(pair)

    eligible_by_source = defaultdict(list)
    sample_size = int(M1_CONFIG["pairs_per_source_label_pattern"])
    skipped_patterns = 0
    for (source_id, pattern), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        rows = sorted(rows, key=lambda row: row["pair_id"])
        if len(rows) < 2:
            skipped_patterns += 1
            continue
        rng = np.random.default_rng(_stable_seed(METHOD_VERSION, "representative", kind, fold_index, source_id, pattern))
        count = min(sample_size, len(rows))
        indices = sorted(rng.choice(len(rows), size=count, replace=False).tolist())
        chosen = [rows[index] for index in indices]
        donors = chosen[1:] + chosen[:1]
        observed_edges = {(row["left"], row["right"]) for row in chosen}
        null_edges = [(row["left"], donor["right"]) for row, donor in zip(chosen, donors)]
        if any(left == right for left, right in null_edges):
            skipped_patterns += 1
            continue
        if len(set(null_edges)) != len(null_edges) or any(edge in observed_edges for edge in null_edges):
            skipped_patterns += 1
            continue
        eligible_by_source[source_id].append((pattern, chosen, donors))

    representatives = []
    used_patterns = 0
    for source_id in sorted(eligible_by_source):
        pattern_groups = eligible_by_source[source_id]
        if not pattern_groups:
            continue
        used_patterns += len(pattern_groups)
        pattern_weight = 1.0 / len(pattern_groups)
        for pattern, chosen, donors in pattern_groups:
            pair_weight = pattern_weight / len(chosen)
            for index, (pair, donor) in enumerate(zip(chosen, donors)):
                representatives.append(dict(
                    pair,
                    m1_role="observed",
                    m1_fold=int(fold_index),
                    m1_weight=float(pair_weight),
                ))
                null_pair = dict(
                    pair,
                    right=donor["right"],
                    right_label=int(donor["right_label"]),
                    pair_id=f"m1null:{kind}:{fold_index}:{source_id}:{pair['pair_id']}:{donor['pair_id']}",
                    m1_role="null",
                    m1_fold=int(fold_index),
                    m1_weight=float(pair_weight),
                    m1_null_donor_pair=str(donor["pair_id"]),
                )
                representatives.append(null_pair)

    return representatives, dict(
        validation_sources=len(validation_sources),
        source_patterns_available=len(grouped),
        source_patterns_used=used_patterns,
        source_patterns_skipped=skipped_patterns,
        representative_pairs=sum(row["m1_role"] == "observed" for row in representatives),
        matched_null_pairs=sum(row["m1_role"] == "null" for row in representatives),
    )


def make_plan(group, pairs, features, outdir, train_sources=None):
    """Create a five-fold source-cross-fitted adaptive M1 plan."""
    if group not in ("E1", "E2", "E3"):
        raise ValueError("Adaptive M1 v2 is defined for E1/E2/E3; E0 remains a historical baseline")
    started = time.monotonic()
    all_fit_sources = {pair["source_id"] for pair in pairs if pair.get("split", "fit") == "fit"}
    allowed = None if train_sources is None else set(train_sources)
    pairs = [
        pair for pair in pairs
        if pair["kind"] in KINDS
        and pair.get("split", "fit") == "fit"
        and (allowed is None or pair["source_id"] in allowed)
    ]
    if not pairs:
        raise ValueError("Empty training-source M1 pair set")

    first = features.get(pairs[0]["left"])
    number_heads = _flat_heads(first).shape[0]
    fitted_sources = sorted({pair["source_id"] for pair in pairs})
    source_folds = _source_folds(fitted_sources, group)
    kinds = {}

    for kind in KINDS:
        kind_pairs = [pair for pair in pairs if pair["kind"] == kind]
        if not kind_pairs:
            raise ValueError(f"{group}: no training pairs for category {kind}")
        fold_plans = []
        for fold_index, validation_sources in enumerate(source_folds, start=1):
            validation_set = set(validation_sources)
            discovery_pairs = [pair for pair in kind_pairs if pair["source_id"] not in validation_set]
            attribution = _gradient_attribution(discovery_pairs, features, group, number_heads)
            candidates, robust_z = _robust_screen(attribution)
            representatives, representative_diagnostics = _validation_representatives(
                kind_pairs, validation_sources, kind, fold_index
            )
            fold_plans.append(dict(
                fold=fold_index,
                discovery_sources=sorted({pair["source_id"] for pair in discovery_pairs}),
                validation_sources=validation_sources,
                attribution=attribution.tolist(),
                robust_z=robust_z.tolist(),
                candidates=candidates,
                representatives=representatives,
                representative_diagnostics=representative_diagnostics,
            ))
        kinds[kind] = dict(fit_pair_count=len(kind_pairs), folds=fold_plans)

    target = Path(outdir)
    target.mkdir(parents=True, exist_ok=True)
    plan = dict(
        method_version=METHOD_VERSION,
        method_config=M1_CONFIG,
        group=group,
        kinds=kinds,
        number_heads=int(number_heads),
        inner_source_folds=len(source_folds),
        fit_pair_counts=dict(Counter(pair["kind"] for pair in pairs)),
        training_sources=fitted_sources,
        train_sources=fitted_sources,
        validation_sources=sorted(all_fit_sources - set(fitted_sources)),
        plan_seconds=time.monotonic() - started,
        attribution_split="within each inner fold, discovery sources only",
        patch_split="out-of-fold validation sources only",
        queue_contract="union raw pair/head interventions across plans; preserve each plan's fold-specific candidate set and source weights",
    )
    write(target / "m1_plan.json", plan)
    return plan


def _bh_adjust(pvalues):
    if not pvalues:
        return {}
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    adjusted = {}
    running = 1.0
    for rank in range(count, 0, -1):
        head, pvalue = ordered[rank - 1]
        running = min(running, float(pvalue) * count / rank)
        adjusted[head] = min(1.0, running)
    return adjusted


def _paired_signflip_p(values, alternative, draws, seed, minimum_sources):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n < minimum_sources:
        return 1.0
    observed = float(np.mean(values))
    if alternative == "greater" and observed <= 0:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(int(draws), n))
    null_means = (signs @ values) / n
    if alternative == "greater":
        exceedances = int(np.count_nonzero(null_means >= observed))
    else:
        exceedances = int(np.count_nonzero(np.abs(null_means) >= abs(observed)))
    return float((exceedances + 1) / (int(draws) + 1))


def _source_bootstrap(values, alternative, draws, seed, minimum_sources):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n < minimum_sources:
        return None, None, 0.0
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(int(draws), n))
    bootstrap_means = values[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975]).tolist()
    if alternative == "greater":
        stability = float(np.mean(bootstrap_means > 0))
    else:
        stability = float(max(np.mean(bootstrap_means > 0), np.mean(bootstrap_means < 0)))
    return float(low), float(high), stability


def finish_plan(plan, features, exact_accessor, outdir):
    """Confirm adaptive candidates using held-out exact patches and matched nulls."""
    if plan.get("method_version") != METHOD_VERSION:
        raise ValueError("M1 plan version mismatch; regenerate the plan before exact patching")
    if exact_accessor is None:
        raise ValueError("Adaptive M1 requires an exact bidirectional patch accessor")
    started = time.monotonic()
    target = Path(outdir)
    target.mkdir(parents=True, exist_ok=True)
    group = plan["group"]
    fold_count = int(plan["inner_source_folds"])
    source_values = defaultdict(lambda: defaultdict(lambda: {"observed": [], "null": []}))
    candidate_counts = {kind: Counter() for kind in KINDS}
    candidate_unions = {kind: set() for kind in KINDS}
    attributes = {}
    representatives_summary = {}
    exact_forward_requests = 0
    exact_validation_pair_count = 0

    for kind in KINDS:
        exact_rows = []
        attributes[kind] = []
        representatives_summary[kind] = []
        for fold_plan in plan["kinds"][kind]["folds"]:
            fold_index = int(fold_plan["fold"])
            candidates = [int(head) for head in fold_plan["candidates"]]
            candidate_counts[kind].update(candidates)
            candidate_unions[kind].update(candidates)
            attributes[kind].append(dict(
                fold=fold_index,
                attribution=fold_plan["attribution"],
                robust_z=fold_plan["robust_z"],
                candidates=candidates,
                discovery_sources=fold_plan["discovery_sources"],
                validation_sources=fold_plan["validation_sources"],
            ))
            representatives_summary[kind].append(dict(
                fold=fold_index,
                validation_sources=fold_plan["validation_sources"],
                representatives=[
                    dict(
                        pair_id=pair["pair_id"],
                        source_id=pair["source_id"],
                        role=pair["m1_role"],
                        weight=float(pair["m1_weight"]),
                    )
                    for pair in fold_plan["representatives"]
                ],
                diagnostics=fold_plan["representative_diagnostics"],
            ))
            if not candidates:
                continue
            for pair in fold_plan["representatives"]:
                measured = exact_accessor(pair, candidates)
                if isinstance(measured, dict):
                    measured = [
                        (int(head), row["forward_ell"], row["reverse_ell"])
                        for head, row in measured.items()
                    ]
                returned = set()
                left_score = features.ell(pair["left"])
                right_score = features.ell(pair["right"])
                for head, forward, reverse in measured:
                    head = int(head)
                    if head not in candidates or head in returned:
                        raise ValueError("Unexpected or duplicate exact-patch head")
                    returned.add(head)
                    forward_effect = float(forward) - left_score
                    reverse_effect = float(reverse) - right_score
                    symmetric_contribution = 0.5 * (forward_effect - reverse_effect)
                    if kind in ("A", "P"):
                        selection_value = abs(symmetric_contribution)
                    else:
                        label_direction = int(pair["right_label"]) - int(pair["left_label"])
                        if label_direction not in (-1, 1):
                            raise ValueError("C/F M1 pairs must have opposite endpoint labels")
                        selection_value = symmetric_contribution * label_direction
                    role = pair["m1_role"]
                    if role not in ("observed", "null"):
                        raise ValueError("Unknown M1 validation-pair role")
                    weight = float(pair["m1_weight"])
                    source_values[kind][head][pair["source_id"]][role].append((weight, float(selection_value)))
                    exact_rows.append(dict(
                        pair_id=pair["pair_id"],
                        source_id=pair["source_id"],
                        fold=fold_index,
                        role=role,
                        head=head,
                        weight=weight,
                        forward_ell=float(forward),
                        reverse_ell=float(reverse),
                        forward_effect=float(forward_effect),
                        reverse_effect=float(reverse_effect),
                        symmetric_contribution=float(symmetric_contribution),
                        selection_value=float(selection_value),
                    ))
                    exact_forward_requests += 2
                if returned != set(candidates):
                    raise ValueError("Incomplete exact patch candidate results")
                exact_validation_pair_count += 1
        write(target / ("m1_exact_" + kind + ".json"), exact_rows)

    selected_by_kind = {}
    ranks = {}
    exact_scores = {}
    selection_metrics = {}
    candidate_frequency = {}
    source_effect_records = {}

    for kind in KINDS:
        candidate_heads = sorted(candidate_unions[kind])
        alternative = "greater" if kind in ("A", "P") else "two-sided"
        per_head = {}
        pvalues = {}
        raw_source_effects = {}

        for head in candidate_heads:
            source_effects = {}
            for source_id, roles in source_values[kind][head].items():
                observed_rows = roles["observed"]
                null_rows = roles["null"]
                observed_weight = sum(weight for weight, _ in observed_rows)
                null_weight = sum(weight for weight, _ in null_rows)
                if observed_weight <= 0 or null_weight <= 0:
                    continue
                observed_mean = sum(weight * value for weight, value in observed_rows) / observed_weight
                null_mean = sum(weight * value for weight, value in null_rows) / null_weight
                source_effects[source_id] = float(observed_mean - null_mean)

            ordered_sources = sorted(source_effects)
            values = np.asarray([source_effects[source] for source in ordered_sources], dtype=np.float64)
            frequency = float(candidate_counts[kind][head] / fold_count)
            candidate_frequency.setdefault(kind, {})[str(head)] = frequency
            seed = _stable_seed(METHOD_VERSION, group, kind, head)
            pvalue = _paired_signflip_p(
                values, alternative,
                M1_CONFIG["permutation_draws"], seed,
                int(M1_CONFIG["min_validation_sources"]),
            )
            low, high, direction_stability = _source_bootstrap(
                values, alternative,
                M1_CONFIG["bootstrap_draws"], seed + 1,
                int(M1_CONFIG["min_validation_sources"]),
            )
            mean_effect = float(np.mean(values)) if len(values) else 0.0
            pvalues[head] = pvalue
            per_head[head] = dict(
                candidate_frequency=frequency,
                validation_source_count=len(values),
                effect_mean=mean_effect,
                ci95=None if low is None else [low, high],
                permutation_p=pvalue,
                bootstrap_direction_stability=direction_stability,
                alternative=alternative,
            )
            raw_source_effects[head] = source_effects

        qvalues = _bh_adjust(pvalues)
        selected = []
        for head in candidate_heads:
            metric = per_head[head]
            qvalue = float(qvalues.get(head, 1.0))
            metric["bh_q"] = qvalue
            ci = metric["ci95"]
            if ci is None:
                ci_pass = False
            elif alternative == "greater":
                ci_pass = ci[0] > 0
            else:
                ci_pass = ci[0] > 0 or ci[1] < 0
            passed = (
                metric["candidate_frequency"] >= float(M1_CONFIG["candidate_fold_frequency_min"])
                and metric["validation_source_count"] >= int(M1_CONFIG["min_validation_sources"])
                and qvalue < float(M1_CONFIG["fdr_q"])
                and ci_pass
                and metric["bootstrap_direction_stability"] >= float(M1_CONFIG["bootstrap_direction_stability_min"])
            )
            metric["selected"] = bool(passed)
            if passed:
                selected.append(head)

        selected.sort(key=lambda head: (-abs(per_head[head]["effect_mean"]), head))
        ranks[kind] = sorted(candidate_heads, key=lambda head: (-abs(per_head[head]["effect_mean"]), head))
        exact_scores[kind] = {str(head): float(per_head[head]["effect_mean"]) for head in candidate_heads}
        selection_metrics[kind] = {str(head): per_head[head] for head in candidate_heads}
        selected_by_kind[kind] = selected
        source_effect_records[kind] = {
            str(head): [
                dict(source_id=source, difference=float(value))
                for source, value in sorted(raw_source_effects[head].items())
            ]
            for head in candidate_heads
        }

    nuisance_heads = sorted(set(selected_by_kind["A"]) | set(selected_by_kind["P"]))
    evidence_heads = sorted(set(selected_by_kind["C"]) | set(selected_by_kind["F"]))
    shared_heads = sorted(set(nuisance_heads) & set(evidence_heads))
    selected_union = set(nuisance_heads) | set(evidence_heads)

    def head_strength(head):
        strengths = [
            abs(selection_metrics[kind][str(head)]["effect_mean"])
            for kind in KINDS if str(head) in selection_metrics[kind]
            and selection_metrics[kind][str(head)]["selected"]
        ]
        return max(strengths) if strengths else 0.0

    selected = sorted(selected_union, key=lambda head: (-head_strength(head), head))
    result = dict(
        method_version=METHOD_VERSION,
        method_config=M1_CONFIG,
        group=group,
        heads=selected,
        nuisance_heads=nuisance_heads,
        evidence_heads=evidence_heads,
        shared_heads=shared_heads,
        selected_by_kind=selected_by_kind,
        ranks=ranks,
        attribution=attributes,
        exact_scores=exact_scores,
        selection_metrics=selection_metrics,
        candidate_frequency=candidate_frequency,
        source_effects=source_effect_records,
        exact_representatives=representatives_summary,
        fit_pair_counts=plan["fit_pair_counts"],
        fit_source_count=len(plan["training_sources"]),
        inner_source_folds=fold_count,
        candidate_counts_by_kind={kind: len(candidate_unions[kind]) for kind in KINDS},
        selected_counts_by_kind={kind: len(selected_by_kind[kind]) for kind in KINDS},
        validation_source_counts_by_kind={
            kind: sorted({
                source
                for fold_plan in plan["kinds"][kind]["folds"]
                for source in fold_plan["validation_sources"]
            })
            for kind in KINDS
        },
        exact_validation_pair_count=exact_validation_pair_count,
        exact_forward_requests=exact_forward_requests,
        weighting="category-specific pair weights for gradient screening; equal source weight for out-of-fold patch confirmation",
        exact_sampling="up to two pairs per source and ordered label-transition pattern; right-side donor cyclic permutation supplies matched null pairs",
        selection="robust gradient screen; out-of-fold bidirectional patching; source-level matched-null tests; BH-FDR; source bootstrap; no fixed head count",
        selection_split="five-fold source-level cross-fitting; each fold's candidate screen excludes its patch-validation sources",
        seconds=plan.get("plan_seconds", 0.0) + time.monotonic() - started,
    )
    write(target / "m1.json", result)
    if not selected:
        raise RuntimeError(
            "Adaptive M1 selected no heads. Inspect m1.json and the per-category diagnostics; "
            "do not backfill with a fixed Top-K selection."
        )
    return selected, result
