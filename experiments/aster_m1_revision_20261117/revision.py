"""Prediction-oriented M1 revision; no FDR or functional-module claim.

The caller supplies an immutable feature store and the existing model engine.
Imports do not load models or start GPU work. Historical outputs are never edited.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence
import hashlib
import json
import numpy as np

KINDS = ("A", "P", "C", "F")
VERSION = "m1-antisymmetric-paired-effect-v3-20261117"
BASE_COMMIT = "2558da64513bcbc74b224d63aedda1e66c3d9dfb"


@dataclass(frozen=True)
class Config:
    candidate_mass: float = .95
    retained_mass: float = .90
    min_sources: int = 16
    seed: int = 20261117
    identity_atol: float = 1e-4
    identity_rtol: float = 1e-5
    max_patch_forwards: int | None = None

    def __post_init__(self):
        if not 0 < self.candidate_mass <= 1 or not 0 < self.retained_mass <= 1:
            raise ValueError("Mass fractions must be in (0, 1]")
        if self.min_sources < 2 or (self.max_patch_forwards is not None and self.max_patch_forwards < 1):
            raise ValueError("Invalid source count or resource ceiling")
        if not np.isfinite([self.identity_atol, self.identity_rtol]).all():
            raise ValueError("Nonfinite numerical tolerances")
        if self.identity_atol <= 0 or self.identity_rtol < 0:
            raise ValueError("Invalid numerical tolerances")

    def permits_patch_forwards(self, count):
        """None means the user-authorized full confirmation has no forward ceiling."""
        return self.max_patch_forwards is None or count <= self.max_patch_forwards


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def validate_pairs(pairs: Sequence[Mapping]) -> None:
    if not pairs:
        raise ValueError("Empty training pairs")
    ids, endpoint_sources = set(), {}
    for p in pairs:
        if p.get("split") != "fit" or p.get("kind") not in KINDS:
            raise ValueError("M1 accepts fit-only A/P/C/F pairs")
        if p["pair_id"] in ids:
            raise ValueError("Duplicate pair_id")
        ids.add(p["pair_id"])
        l, r = p["left_label"], p["right_label"]
        if l not in (0, 1) or r not in (0, 1):
            raise ValueError("Binary endpoint labels required")
        if (l == r) != (p["kind"] in ("A", "P")):
            raise ValueError("Pair kind and label transition disagree")
        if p["left"] == p["right"]:
            raise ValueError("Counterfactual endpoints must be distinct")
        for uid in (p["left"], p["right"]):
            if uid in endpoint_sources and endpoint_sources[uid] != p["source_id"]:
                raise ValueError("An endpoint crosses source boundaries")
            endpoint_sources[uid] = p["source_id"]


def partition_sources(pairs, config=Config()):
    validate_pairs(pairs)
    sources = sorted({p["source_id"] for p in pairs})
    order = np.random.default_rng(config.seed).permutation(len(sources))
    cut = len(sources) // 2
    screening = {sources[i] for i in order[:cut]}
    confirmation = set(sources) - screening
    for selected in (screening, confirmation):
        for kind in KINDS:
            count = len({p["source_id"] for p in pairs if p["source_id"] in selected and p["kind"] == kind})
            if count < config.min_sources:
                raise ValueError("Insufficient training-source coverage; do not import dev/test sources")
    return sorted(screening), sorted(confirmation)


def balanced_mean(values, sources):
    a = np.asarray(values, dtype=float)
    if a.ndim != 2 or len(a) != len(sources) or not len(a) or not np.isfinite(a).all():
        raise ValueError("Expected finite [pair, head] values with source IDs")
    groups = defaultdict(list)
    for i, source in enumerate(sources):
        groups[source].append(i)
    return np.mean([a[ix].mean(axis=0) for ix in groups.values()], axis=0)


def mass_select(scores, mass):
    """Adaptive count, including all exact ties at the boundary; no fixed Top-K.

The sum is a ranking statistic, NOT an additive decomposition of causal effects.
"""
    a = np.asarray(scores, dtype=float)
    if a.ndim != 1 or not np.isfinite(a).all() or np.any(a < 0) or not 0 < mass <= 1:
        raise ValueError("Invalid nonnegative scores or mass")
    positive = np.flatnonzero(a > 0)
    if not len(positive):
        return []
    # Rescaling prevents overflow without changing ranks or retained fractions.
    norm = a / a.max()
    order = positive[np.argsort(-norm[positive], kind="stable")]
    target = mass * norm.sum()
    index = min(int(np.searchsorted(np.cumsum(norm[order]), target, side="left")), len(order) - 1)
    threshold = a[order[index]]
    return np.flatnonzero((a >= threshold) & (a > 0)).tolist()


def head_array(record, field):
    a = np.asarray(record[field], dtype=float)
    if a.ndim < 2 or not np.isfinite(a).all():
        raise ValueError("Invalid head feature: " + field)
    return a.reshape(-1, a.shape[-1])


def symmetric_attribution(left, right):
    if "grads" not in left or "grads" not in right:
        raise ValueError("Both endpoint gradients are required; left-only cache is not sufficient")
    l, r = head_array(left, "heads"), head_array(right, "heads")
    gl, gr = head_array(left, "grads"), head_array(right, "grads")
    if not (l.shape == r.shape == gl.shape == gr.shape):
        raise ValueError("Head/gradient geometry mismatch")
    diff = r - l
    return .5 * (np.abs(np.sum(diff * gl, axis=1)) + np.abs(np.sum(diff * gr, axis=1)))


def screen(pairs, store, source_ids, config=Config()):
    validate_pairs(pairs)
    allowed = set(source_ids)
    result = {}
    for kind in KINDS:
        chosen = [p for p in pairs if p["kind"] == kind and p["source_id"] in allowed]
        sources = [p["source_id"] for p in chosen]
        if len(set(sources)) < config.min_sources:
            raise ValueError("Insufficient screening sources")
        a = [symmetric_attribution(store.get(p["left"]), store.get(p["right"])) for p in chosen]
        score = balanced_mean(a, sources)
        result[kind] = dict(heads=mass_select(score, config.candidate_mass), scores=score.tolist())
    return result


def paired_effect(row, config=Config()):
    """Signed response plus magnitude above a measured numerical floor.

Self-patching is a numerical control, NOT a counterfactual semantic null.
Negative C/F direction is retained for auditing, never silently discarded.
"""
    names = ("base_left", "base_right", "paired_forward", "paired_reverse", "self_left", "self_right")
    b_l, b_r, f, r, s_l, s_r = [float(row[k]) for k in names]
    if not np.isfinite([b_l, b_r, f, r, s_l, s_r]).all():
        raise ValueError("Nonfinite patch result")
    t_l = config.identity_atol + config.identity_rtol * abs(b_l)
    t_r = config.identity_atol + config.identity_rtol * abs(b_r)
    e_l, e_r = abs(s_l - b_l), abs(s_r - b_r)
    if e_l > t_l or e_r > t_r:
        raise ValueError("Self-patch identity failed; stop before head selection")
    u_l, u_r = f - s_l, r - s_r
    signed = .5 * (u_l - u_r)
    floor = .5 * (e_l + e_r + t_l + t_r)
    magnitude = max(abs(signed) - floor, 0.)
    direction = int(row["right_label"]) - int(row["left_label"])
    return dict(magnitude=magnitude, signed=signed, c=signed,
                u_L=u_l, u_R=u_r, common_mode=.5 * (u_l + u_r),
                absolute_side_mean=.5 * (abs(u_l) + abs(u_r)),
                label_aligned=direction * signed, numerical_floor=floor,
                identity_error_left=e_l, identity_error_right=e_r,
                tolerance_left=t_l, tolerance_right=t_r,
                identity_error=max(e_l, e_r))


def confirm(pairs, candidates, rows, total_heads, config=Config()):
    validate_pairs(pairs)
    by_id = {p["pair_id"]: p for p in pairs}
    expected = {(p["pair_id"], int(h)) for p in pairs for h in candidates[p["kind"]]["heads"]}
    measured = {}
    for row in rows:
        key = (row["pair_id"], int(row["head"]))
        if key not in expected or key in measured:
            raise ValueError("Unexpected or duplicate patch result")
        p = by_id[row["pair_id"]]
        for field in ("source_id", "kind", "left_label", "right_label"):
            if row[field] != p[field]:
                raise ValueError("Patch/pair metadata mismatch: " + field)
        measured[key] = paired_effect(row, config)
    if set(measured) != expected:
        raise ValueError("Missing patch results; incomplete coverage is not a negative result")
    selected, diagnostics = {}, []
    for kind in KINDS:
        ps = [p for p in pairs if p["kind"] == kind]
        sources = [p["source_id"] for p in ps]
        if len(set(sources)) < config.min_sources:
            raise ValueError("Insufficient confirmation sources")
        heads = candidates[kind]["heads"]
        if len(set(heads)) != len(heads) or any(h < 0 or h >= total_heads for h in heads):
            raise ValueError("Invalid candidate head IDs")
        if not heads:
            selected[kind] = []
            continue
        mags = [[measured[(p["pair_id"], h)]["magnitude"] for h in heads] for p in ps]
        aligned = [[measured[(p["pair_id"], h)]["label_aligned"] for h in heads] for p in ps]
        scores, directions = balanced_mean(mags, sources), balanced_mean(aligned, sources)
        selected[kind] = [heads[i] for i in mass_select(scores, config.retained_mass)]
        diagnostics.extend(dict(kind=kind, head=int(h), effect=float(scores[i]),
                                label_aligned=float(directions[i]), sources=len(set(sources)))
                           for i, h in enumerate(heads))
    nuisance, evidence = set(selected["A"] + selected["P"]), set(selected["C"] + selected["F"])
    union = sorted(nuisance | evidence)
    return dict(version=VERSION, heads=union, by_kind=selected,
                nuisance_response_heads=sorted(nuisance), evidence_response_heads=sorted(evidence),
                shared_response_heads=sorted(nuisance & evidence), diagnostics=diagnostics,
                status="features_selected" if union else "intercept_only_fallback",
                inference="predictive feature selection; no FDR or functional specialization claim")


class ExactPatcher:
    """Adapter for the repository Engine API, using only within-pair donors.

The caller must enforce checkpoint/tokenizer/prompt/cache identity before calling.
Fresh unpatched scores are compared with cached scores before each patch batch.
"""
    def __init__(self, store, inputs, engine, config=Config()):
        self.store, self.inputs, self.engine, self.config = store, inputs, engine, config

    def measure(self, pair, heads, on_record=None):
        if pair.get("split") != "fit":
            raise ValueError("Training-source exact confirmation only")
        if not heads:
            return []
        left = head_array(self.store.get(pair["left"]), "heads")
        right = head_array(self.store.get(pair["right"]), "heads")
        if left.shape != right.shape or any(h < 0 or h >= len(left) for h in heads):
            raise ValueError("Invalid model-local head geometry")
        contexts = [self.engine.prepare(self.inputs[pair[k]]["prompt"]) for k in ("left", "right")]
        bases = [float(self.engine.run(c, capture=False)["ell"]) for c in contexts]
        for uid, base in zip((pair["left"], pair["right"]), bases):
            cached = float(self.store.ell(uid))
            if not np.isfinite(cached) or not np.isfinite(base) or abs(cached-base) > self.config.identity_atol + self.config.identity_rtol*abs(base):
                raise ValueError("Cached and fresh baseline disagree; cache reuse refused")
        records = []
        for h in heads:
            scores = [float(self.engine.run(c, replacements={int(h): value}, capture=False)["ell"])
                      for c, value in ((contexts[0], right[h]), (contexts[1], left[h]),
                                       (contexts[0], left[h]), (contexts[1], right[h]))]
            row = {k: pair[k] for k in ("pair_id", "source_id", "kind", "left_label", "right_label")}
            row.update(head=int(h), base_left=bases[0], base_right=bases[1],
                       paired_forward=scores[0], paired_reverse=scores[1], self_left=scores[2], self_right=scores[3])
            if on_record is not None:
                on_record(dict(row))
            row.update(paired_effect(row, self.config))
            records.append(row)
        return records


def select_and_fit(pairs, features, views, natural_views, inputs, engine, outdir,
                   identity: Mapping, config=Config()):
    """New experiment entry; caller handles model loading under the existing queue.

No dev/test features or labels are read here. Old M2/M3 are intentionally reused.
Output directory must be new. No automatic GPU scheduling or historical overwrite.
"""
    validate_pairs(pairs)
    for key in ("checkpoint_sha256", "tokenizer_sha256", "inputs_sha256", "scoring_sha256"):
        if not identity.get(key):
            raise ValueError("Missing immutable cache identity: " + key)
    source_ids = {p["source_id"] for p in pairs}
    for rows in (views, natural_views):
        if any(v.get("split") != "fit" or v["source_id"] not in source_ids for v in rows):
            raise ValueError("M2/M3/calibration received non-training inputs")
    original = [v for v in natural_views if v.get("variant", "full") == "full"]
    if {v["label"] for v in original} != {0, 1}:
        raise ValueError("Both labels required for finite unregularized intercept calibration")
    screen_ids, confirm_ids = partition_sources(pairs, config)
    candidates = screen(pairs, features, screen_ids, config)
    confirmation = [p for p in pairs if p["source_id"] in set(confirm_ids)]
    # Four patch forwards per head plus two fresh baseline forwards per nonempty pair.
    count = sum(4*len(candidates[p["kind"]]["heads"])+2 for p in confirmation if candidates[p["kind"]]["heads"])
    if not config.permits_patch_forwards(count):
        raise RuntimeError(f"Resource preflight: {count} forwards; no truncation or partial run allowed")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    manifest = dict(version=VERSION, base_commit=BASE_COMMIT, config=asdict(config), identity=dict(identity),
                    screening_sources=screen_ids, confirmation_sources=confirm_ids,
                    estimated_confirmation_forwards=count, candidates=candidates,
                    scope="revised exploratory protocol; not independent confirmatory validation")
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest["protocol_sha256"] = fingerprint
    dump(outdir/"protocol.json", manifest)
    patcher, records = ExactPatcher(features, inputs, engine, config), []
    try:
        with (outdir/"paired_effects.jsonl").open("x", encoding="utf-8") as log:
            for p in confirmation:
                batch = patcher.measure(p, candidates[p["kind"]]["heads"])
                for row in batch:
                    log.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
                log.flush()
                records.extend(batch)
        total_heads = head_array(features.get(pairs[0]["left"]), "heads").shape[0]
        result = confirm(confirmation, candidates, records, total_heads, config)
        dump(outdir/"m1.json", result)
        return fit_selected(pairs, features, views, original, outdir, result, fingerprint)

    except Exception as exc:
        dump(outdir/"failure.json", dict(status="failed", error=str(exc), performance_evaluated=False))
        raise


def fit_selected(pairs, features, views, natural_views, outdir, result, fingerprint):
    """Fit unchanged E1 M2/M3 to a completed M1 result."""
    from experiments.aster_cr_retrain_paired_v2_20261111 import training
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    validate_pairs(pairs)
    sources = {p["source_id"] for p in pairs}
    if any(v.get("split") != "fit" or v["source_id"] not in sources for v in list(views)+list(natural_views)):
        raise ValueError("Non-fit M2/M3/calibration input")
    original = [v for v in natural_views if v.get("variant", "full") == "full"]
    if {v["label"] for v in original} != {0, 1}:
        raise ValueError("Both natural fit labels required")
    heads = result["heads"]
    if heads:
        support, m2 = training._m2("E1", heads, pairs, features, views, outdir)
    else:
        support = dict(heads=np.array([], dtype=int), mu=np.empty(0), scale=np.empty(0), N=np.empty((0, 0)))
        np.savez_compressed(outdir/"m2.npz", **support)
        m2 = dict(rank=0, N_rank=0, status="intercept_only_fallback")
    m3 = training.refit_m3("E1", pairs, features, original, support, outdir)
    if not m3.get("success") or m3.get("constraint_max_violation", 0) > 1e-7:
        raise RuntimeError("M3 solver or hard-budget failure")
    status = result["status"] if int(m3["rank"]) else "intercept_only_fallback"
    summary = dict(complete=True, status=status, heads=heads, M1=result, M2=m2, M3=m3,
                   protocol_sha256=fingerprint, performance_evaluated=False,
                   training_sources=sorted(sources), train_sources=sorted(sources),
                   prediction="ell + (((heads_selected-mu)/scale) @ N) @ w + b")
    dump(outdir/"model.json", summary)
    return summary
