"""Official INLP, adapted to visibility of disposition text in E1 guidelines.

The erasure target is a property of the input text, never the node truth value,
pair type, source identity, or left/right side. Evidence labels are used only
for separately trained diagnostic probes and development utility measurement.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np
from .data import PREPARED, RAW_SOURCE_OUT, get_prompt, load_bundle, read_rows
from .evaluate import write_json, write_jsonl

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports/aster_codex03_external_20261112/inlp"
OFFICIAL = Path("/home/gcx/aster_codex03_external_assets_20261112/official/nullspace_projection")
SHARED = REPO / "reports/aster_codex03_external_20261112/caa/raw_prompt_residuals"
MODEL = Path("/home/gcx/TRIAGE/models/modelscope/Qwen3-8B")
METHOD = "INLP-task-adapted"
ITERATIONS = (1, 4, 8)
DEFINITION = (
    "Whether all leaf disposition/conclusion descriptions remain visible in the guideline. "
    "0: at least one leaf is replaced by an existing deletion or neutral placeholder; "
    "1: leaf descriptions contain none of those placeholders. Protected leaves may remain. "
    "Swapped substantive leaf descriptions still receive 1. This is disposition-text "
    "visibility, not a claim to label every form of interference."
)


def concept_from_prompt(prompt):
    import re
    separator = "\u6307\u5357\u539f\u6587\u5b9a\u4e49\u4e0e\u8bf4\u660e\uff1a"
    guide = prompt.split(separator, 1)[0]
    leaves = re.findall(r"^L\d+:\s*(.*)$", guide, re.MULTILINE)
    if not leaves:
        return None
    units = (
        "[\u5904\u7f6e\u5df2\u5220\u9664]",
        "\u8bb0\u5f55\u5185\u5bb9\u3002",
        "\u8bb0\u5f55\u5185\u5bb9\u4e0d\u5f71\u54cd\u5f53\u524d\u6761\u4ef6\u5224\u65ad\u3002",
        "Administrative information.",
    )
    for leaf in leaves:
        for unit in units:
            if leaf.strip() and not leaf.replace(unit, "").strip():
                return 0
    return 1


def label_rows(split):
    by_uid = {}
    for view in load_bundle(split)["views"]:
        key = view["uid"]
        if key in by_uid:
            if by_uid[key]["evidence_label"] != view.get("label"):
                raise ValueError("Conflicting evidence labels for one exact prompt UID: " + key)
            continue
        by_uid[key] = dict(uid=key, source_id=view["source_id"], split=split,
                          concept=concept_from_prompt(get_prompt(key)),
                          evidence_label=view.get("label"))
    return list(by_uid.values())


class ResidualCache:
    def __init__(self, shared=SHARED):
        from experiments.aster_cr_retrain_paired_v2_20261111.common import load_records
        self.shared = Path(shared)
        self.records = load_records(RAW_SOURCE_OUT)

    def path(self, uid):
        shared = self.shared / (uid + ".npy")
        if shared.exists():
            return shared, "fresh_shared_original_prompt"
        record = self.records.get(uid)
        return (Path(record["feature"]), "historical_original_activation") if record else (None, "absent")

    def get(self, uid, require_shared=False):
        path, origin = self.path(uid)
        if path is None or (require_shared and origin != "fresh_shared_original_prompt"):
            return None
        if path.suffix == ".npy":
            result = np.load(path, allow_pickle=False)
        else:
            with np.load(path, allow_pickle=False) as data:
                if "residual" not in data:
                    return None
                result = data["residual"]
        result = np.asarray(result, np.float32).reshape(-1)
        if result.shape != (4096,) or not np.all(np.isfinite(result)):
            raise ValueError("Invalid final-layer original activation: " + uid)
        return result

    def matrix(self, rows, require_shared=False):
        features, missing = [], []
        for row in rows:
            value = self.get(row["uid"], require_shared=require_shared)
            if value is None:
                missing.append(row["uid"])
            else:
                features.append(value)
        if missing:
            raise RuntimeError(f"Missing original pre-final-norm residuals for {len(missing)} inputs; first {missing[0]}")
        return np.stack(features)


def official_module(root=OFFICIAL):
    """Load untouched author files without colliding with this repository's src."""
    root = Path(root)
    def module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        return value
    classifier = module("aster_inlp_author_classifier", root / "src/classifier.py")
    package = types.ModuleType("src")
    package.classifier = classifier
    old = sys.modules.get("src")
    try:
        sys.modules["src"] = package
        return module("aster_inlp_author_debias", root / "src/debias.py")
    finally:
        if old is None:
            sys.modules.pop("src", None)
        else:
            sys.modules["src"] = old


class OriginalReadout:
    """Original Qwen3 RMSNorm and FP32 two-label projection, without loading LM."""
    def __init__(self, model=MODEL):
        import torch
        from safetensors import safe_open
        from transformers import AutoTokenizer
        self.torch = torch
        model = Path(model)
        index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
        config = json.loads((model / "config.json").read_text())
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        ids = [tokenizer.encode(s, add_special_tokens=False) for s in ("\u662f", "\u5426")]
        if any(len(value) != 1 for value in ids):
            raise ValueError("The original answer labels must each be one token")
        self.label_ids = [value[0] for value in ids]
        with safe_open(model / index["model.norm.weight"], framework="pt", device="cpu") as handle:
            self.weight = handle.get_tensor("model.norm.weight").to(torch.bfloat16)
        head = "lm_head.weight" if "lm_head.weight" in index else "model.embed_tokens.weight"
        with safe_open(model / index[head], framework="pt", device="cpu") as handle:
            sl = handle.get_slice(head)
            self.label_weight = torch.cat([sl[k:k+1] for k in self.label_ids]).float()
        self.eps = float(config["rms_norm_eps"])

    def score(self, residuals):
        torch = self.torch
        x = torch.as_tensor(np.asarray(residuals, np.float32)).to(torch.bfloat16)
        with torch.no_grad():
            f = x.float()
            h = (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + self.eps)).to(torch.bfloat16)
            h = self.weight * h
            logits = torch.nn.functional.linear(h.float(), self.label_weight)
        return (logits[..., 0] - logits[..., 1]).float().numpy()


def prepare(output=OUT, shared=SHARED):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cache = ResidualCache(shared)
    summary, missing_fit = {}, []
    for split in ("fit", "dev"):
        records = label_rows(split)
        write_jsonl(output / ("labels_" + split + ".jsonl"), records)
        available, fresh, missing = 0, 0, []
        for row in records:
            key = row["uid"]
            path, origin = cache.path(key)
            value = cache.get(key)
            available += value is not None
            fresh += origin == "fresh_shared_original_prompt"
            if value is None:
                missing.append(key)
        if split == "fit":
            missing_fit = [dict(uid=key, location="last_decoder_layer_output_pre_final_norm",
                                position="last_prompt_token", required_shape=[4096]) for key in missing]
        summary[split] = dict(unique_inputs=len(records), concepts=dict(Counter(str(r["concept"]) for r in records)),
                              available_original_residuals=available, fresh_shared_residuals=fresh,
                              missing_original_residuals=len(missing),
                              sources=len({r["source_id"] for r in records}))
    write_jsonl(output / "missing_fit_residuals.jsonl", missing_fit)
    commit = subprocess.check_output(["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"], text=True).strip()
    result = dict(status="prepared", method=METHOD, concept_definition=DEFINITION,
                  label_source="Exact prompt leaf descriptions; no pair type, side, source ID, or truth label is used",
                  neutral_marker_source="experiments/aster_cr/repair/build.py::action_guides and experiments/aster_cr/build.py",
                  official_repository="https://github.com/shauli-ravfogel/nullspace_projection",
                  official_commit=commit, official_path=str(OFFICIAL),
                  shared_original_features=str(shared), training_pairs_pool=2540,
                  training_scope="All concept-labeled unique E1 endpoint prompts; repeated UIDs counted once",
                  planned_iterations=list(ITERATIONS), deterministic=True,
                  random_seed_policy="Deterministic lbfgs classifier; no random subsampling or dropout, no manufactured seed repeats",
                  readout="Fixed original Qwen3 RMSNorm and original FP32 yes/no output rows",
                  intervention="Final prompt token, final decoder-layer output before final RMSNorm",
                  data=summary)
    write_json(output / "preparation.json", result)
    return result


def classifier():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1.0, solver="lbfgs", class_weight="balanced", max_iter=1000, tol=1e-4)


def probe_summary(model, X, y):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    prediction = model.predict(X)
    return dict(count=len(y), accuracy=float(accuracy_score(y, prediction)),
                balanced_accuracy=float(balanced_accuracy_score(y, prediction)),
                class_counts={str(k): int(v) for k, v in Counter(y.tolist()).items()})


def save_probe(path, model):
    np.savez(path, coef=model.coef_, intercept=model.intercept_, classes=model.classes_)


def fit(output=OUT, shared=SHARED, threads=8):
    from scipy.linalg import orth
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits
    import torch
    output = Path(output)
    if not (output / "preparation.json").exists():
        prepare(output, shared)
    preparation = json.loads((output / "preparation.json").read_text())
    train = [r for r in read_rows(output / "labels_fit.jsonl") if r["concept"] is not None]
    dev = [r for r in read_rows(output / "labels_dev.jsonl") if r["concept"] is not None]
    cache = ResidualCache(shared)
    X = cache.matrix(train, require_shared=True)
    D = cache.matrix(dev)
    y = np.asarray([r["concept"] for r in train], int)
    z = np.asarray([r["concept"] for r in dev], int)
    e = np.asarray([r["evidence_label"] for r in train], int)
    de = np.asarray([r["evidence_label"] for r in dev], int)
    readout = OriginalReadout()
    torch.set_num_threads(threads)
    replay = readout.score(D[:8])
    cached = np.asarray([cache.records[r["uid"]]["ell"] for r in dev[:8]])
    replay_error = float(np.max(np.abs(replay - cached)))
    write_json(output / "original_readout_replay.json", dict(count=8, max_abs_error=replay_error,
               original_scores=cached.tolist(), replay_scores=replay.tolist()))
    if not np.allclose(replay, cached, atol=0.1, rtol=0.001):
        raise RuntimeError("Original activation/readout replay differs from the canonical raw scores")
    official = official_module()
    started, cpu_started = time.perf_counter(), time.process_time()
    fit_config = dict(classifier="sklearn.linear_model.LogisticRegression", C=1.0, solver="lbfgs",
                      class_weight="balanced", max_iter=1000, tol=1e-4,
                      num_classifiers=8, min_accuracy=0.0, is_autoregressive=True,
                      by_class=False, dropout_rate=0.0, input_dim=4096,
                      candidate_iterations=list(ITERATIONS), objective="Iteratively fit a linear concept classifier and remove its row space",
                      selection="Maximize dev endpoint truth accuracy; tie-break by lower dev disposition-concept balanced accuracy, then fewer iterations",
                      selection_scope="All unique labeled dev pair-endpoint prompts, not test data",
                      extra_AP_CF_loss=False, evidence_labels_used_in_erasure_training=False)
    write_json(output / "training_config.json", fit_config)
    with threadpool_limits(limits=threads):
        P8, rowspaces, weights = official.get_debiasing_projection(
            LogisticRegression, {k: fit_config[k] for k in ("C", "solver", "class_weight", "max_iter", "tol")},
            8, X.shape[1], True, 0.0, X, y, D, z, by_class=False, dropout_rate=0)
        erasure_seconds = time.perf_counter() - started
        if len(weights) != 8:
            raise RuntimeError("The official iteration trajectory did not return eight classifiers")
        raw_concept = classifier().fit(X, y)
        raw_evidence = classifier().fit(X, e)
        save_probe(output / "probe_raw_concept.npz", raw_concept)
        save_probe(output / "probe_raw_evidence.npz", raw_evidence)
        raw_scores = readout.score(D)
        raw_dev = dict(concept=probe_summary(raw_concept, D, z),
                       evidence=probe_summary(raw_evidence, D, de),
                       node_accuracy=float(np.mean((raw_scores > 0) == de)))
        candidates = []
        for count in ITERATIONS:
            folder = output / ("iterations_" + str(count))
            folder.mkdir(exist_ok=True)
            P = P8 if count == 8 else official.get_projection_to_intersection_of_nullspaces(rowspaces[:count], X.shape[1])
            B = orth(np.concatenate(weights[:count], axis=0).T)
            equivalence_error = float(np.max(np.abs(P - (np.eye(len(P)) - B @ B.T))))
            if equivalence_error > 1e-5:
                raise RuntimeError("Low-rank storage is not numerically equivalent to the official projection")
            # Keep the official dense projection as a reference; inference stores the exact removed subspace.
            np.savez_compressed(folder / "projection.npz", basis=B.astype(np.float32),
                                official_projection=P.astype(np.float32),
                                classifier_weights=np.concatenate(weights[:count], axis=0))
            XP = X @ P
            DP = D @ P
            concept_probe = classifier().fit(XP, y)
            evidence_probe = classifier().fit(XP, e)
            save_probe(folder / "probe_concept.npz", concept_probe)
            save_probe(folder / "probe_evidence.npz", evidence_probe)
            scores = readout.score(DP)
            node_accuracy = float(np.mean((scores > 0) == de))
            probes = dict(concept=probe_summary(concept_probe, DP, z),
                          evidence=probe_summary(evidence_probe, DP, de))
            record = dict(iterations=count, erased_rank=int(B.shape[1]), dev_node_accuracy=node_accuracy,
                          dev_concept_balanced_accuracy=probes["concept"]["balanced_accuracy"],
                          dev_evidence_balanced_accuracy=probes["evidence"]["balanced_accuracy"],
                          official_projection_equivalence_max_abs_error=equivalence_error,
                          parameter_count=int(B.size), projection=str(folder / "projection.npz"))
            write_json(folder / "development.json", dict(**record, probes=probes, raw=raw_dev))
            write_jsonl(folder / "development_endpoint_scores.jsonl", [dict(uid=r["uid"], ell=float(s),
                         prediction=int(s > 0), evidence_label=r["evidence_label"], concept=r["concept"])
                         for r, s in zip(dev, scores)])
            candidates.append(record)
    selected = dict(sorted(candidates, key=lambda r: (-r["dev_node_accuracy"], r["dev_concept_balanced_accuracy"], r["iterations"]))[0])
    feature_index = Path(shared) / "index.jsonl"
    feature_records = {r["uid"]: r for r in read_rows(feature_index)} if feature_index.exists() else {}
    feature_times = [feature_records.get(r["uid"], {}).get("feature_seconds") for r in train]
    feature_seconds = sum(feature_times) if all(isinstance(v, (int, float)) for v in feature_times) else None
    direct_cpu_wall = time.perf_counter() - started
    selected.update(method=METHOD, selected_on="dev", frozen_before_test=True,
                    concept_definition=DEFINITION, train_unique_inputs=len(train), dev_unique_inputs=len(dev),
                    training_config=fit_config, candidate_results=candidates,
                    original_readout=preparation["readout"], official_commit=preparation["official_commit"],
                    official_repository=preparation["official_repository"], deterministic=True,
                    training_wall_seconds=None if feature_seconds is None else direct_cpu_wall + feature_seconds,
                    direct_cpu_training_wall_seconds=direct_cpu_wall,
                    shared_original_feature_seconds=feature_seconds,
                    shared_original_feature_inputs=len(train),
                    training_cpu_seconds=time.process_time() - cpu_started,
                    erasure_fit_seconds=erasure_seconds, direct_fit_gpu_hours=0.0,
                    training_gpu_hours=None if feature_seconds is None else feature_seconds / 3600,
                    training_cost_scope="CPU erasure fitting/probes/dev selection plus measured original-prompt feature extraction when available. Original features are shared with CAA; this accounting charges their measured extraction cost once to an independent INLP run. No CAA target-completion work is charged to INLP.",
                    trainable_parameter_count=int(selected["parameter_count"]),
                    stored_parameter_count=int(selected["parameter_count"]),
                    frozen_backbone_parameters_trained=0,
                    implementation_differences=["Author src/debias.py and src/classifier.py unchanged; isolated import namespace",
                      "New text-defined disposition visibility labels and E1 unique-prompt data adapter",
                      "Deterministic supported logistic classifier; no pairwise task loss",
                      "Fixed final-prompt-token residual location and fixed original RMSNorm/head",
                      "Exact low-rank basis storage with author dense projection retained"])
    write_json(output / "selection.json", selected)
    return selected


class OnlineScorer:
    def __init__(self, selection, device="cuda:0"):
        import torch
        from experiments.aster_cr.engine import Engine
        self.torch = torch
        self.engine = Engine("qwen", device)
        with np.load(selection["projection"], allow_pickle=False) as data:
            self.basis = torch.from_numpy(data["basis"]).to(self.engine.device)
        self.selection = selection
        self.capture_uids = set()
        self.residual_output = None
        self.raw_last = None
        torch.cuda.reset_peak_memory_stats(self.engine.device)

    def _hook(self, module, args, output):
        value = output[0] if isinstance(output, tuple) else output
        result = value.clone()
        last = value[0, -1].float()
        if self.residual_output is not None:
            self.raw_last = last.detach().cpu().numpy()
        result[0, -1] = (last - (last @ self.basis) @ self.basis.T).to(value.dtype)
        return (result,) + output[1:] if isinstance(output, tuple) else result

    def __call__(self, prompt, uid):
        context = self.engine.prepare(prompt)
        handle = self.engine.layers[-1].register_forward_hook(self._hook)
        try:
            result = self.engine.run(context, capture=False)
        finally:
            handle.remove()
        self.torch.cuda.synchronize(self.engine.device)
        if uid in self.capture_uids and self.residual_output is not None:
            np.save(self.residual_output / (uid + ".npy"), self.raw_last)
        return dict(ell=result["ell"], tokens=result["tokens"],
                    forward_count=math.ceil((result["tokens"] - 1) / self.engine.prefill_chunk) + 1,
                    peak_memory_gib=self.torch.cuda.max_memory_allocated(self.engine.device) / 2**30)


def smoke(scorer, output=OUT):
    output = Path(output)
    readout = OriginalReadout()
    selected = scorer.selection
    with np.load(selected["projection"], allow_pickle=False) as data:
        P = data["official_projection"]
    checks = []
    for view in load_bundle("dev")["views"][:4]:
        key = view["uid"]
        raw = scorer.engine.score(get_prompt(key), capture=True)
        residual = raw["residual"].numpy()
        replay = float(readout.score(residual))
        projected = float(readout.score(residual @ P))
        live = scorer(get_prompt(key), key)["ell"]
        checks.append(dict(uid=key, raw_ell=raw["ell"], raw_replay=replay,
                           projected_replay=projected, online_hook_ell=live,
                           projection_changes_score=abs(live - raw["ell"]) > 1e-5))
    passed = all(np.isclose(r["raw_ell"], r["raw_replay"], atol=0.1, rtol=0.001)
                 and np.isclose(r["projected_replay"], r["online_hook_ell"], atol=0.1, rtol=0.001)
                 for r in checks) and any(r["projection_changes_score"] for r in checks)
    report = dict(passed=bool(passed), actual_gpu_smoke=True, count=len(checks), checks=checks,
                  projection=selected["projection"], model_path=str(MODEL))
    write_json(output / "online_smoke.json", report)
    if not passed:
        raise RuntimeError("INLP online hook does not agree with original-readout projection replay")
    return report


def frozen_probe_evaluation(split, output=OUT, shared=SHARED):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    output = Path(output)
    selected = json.loads((output / "selection.json").read_text())
    rows = [r for r in label_rows(split) if r["concept"] is not None]
    live_features = output / split / "raw_residuals"
    cache = ResidualCache(live_features if live_features.exists() else shared)
    X = cache.matrix(rows)
    folder = Path(selected["projection"]).parent
    with np.load(selected["projection"], allow_pickle=False) as data:
        B = data["basis"]
    XP = X - (X @ B) @ B.T
    result = dict(split=split, count=len(rows), probes={}, selected_iterations=selected["iterations"],
                  concept_definition=DEFINITION, no_probe_fit_on_this_split=True)
    for kind, label in (("concept", "concept"), ("evidence", "evidence_label")):
        target = np.asarray([r[label] for r in rows], int)
        for mode, values, path in (("raw", X, output / ("probe_raw_" + kind + ".npz")),
                                   ("erased", XP, folder / ("probe_" + kind + ".npz"))):
            with np.load(path, allow_pickle=False) as probe:
                score = (values @ probe["coef"].T + probe["intercept"]).reshape(-1)
                prediction = probe["classes"][(score > 0).astype(int)]
            result["probes"][mode + "_" + kind] = dict(accuracy=float(accuracy_score(target, prediction)),
               balanced_accuracy=float(balanced_accuracy_score(target, prediction)))
    write_json(output / ("probes_" + split + ".json"), result)
    return result


def score(split, output=OUT, device="cuda:0", scorer=None):
    from .inference import score_split
    from .evaluate import evaluate
    output = Path(output)
    selected = json.loads((output / "selection.json").read_text())
    if not selected.get("frozen_before_test"):
        raise ValueError("Development selection is not frozen")
    scorer = scorer or OnlineScorer(selected, device)
    if not (output / "online_smoke.json").exists():
        smoke(scorer, output)
    scores_path = output / split / "scores.jsonl"
    scorer.capture_uids = {row["uid"] for row in load_bundle(split)["views"]}
    scorer.residual_output = output / split / "raw_residuals"
    scorer.residual_output.mkdir(parents=True, exist_ok=True)
    metadata = dict(selected, model_path=str(MODEL), run_method=METHOD,
                    layer=35, position="last_prompt_token", strength=1.0,
                    official_code_version=selected["official_commit"],
                    readout_protocol="Fixed original Qwen3 RMSNorm and original output rows")
    measured = score_split(METHOD, split, scorer, scores_path, metadata=metadata)
    measured["inference_gpu_hours"] = measured["inference_wall_seconds"] / 3600
    measured["physical_gpu_visibility"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    write_json(scores_path.with_suffix(".metadata.json"), measured)
    result = evaluate(METHOD, scores_path, split, output.parent, metadata=measured)
    try:
        frozen_probe_evaluation(split, output)
    except RuntimeError as error:
        write_json(output / ("probes_" + split + ".json"), dict(complete=False, reason=str(error)))
    return result


def run(output=OUT, shared=SHARED, device="cuda:0", threads=8):
    output = Path(output)
    if not (output / "selection.json").exists():
        prepare(output, shared)
        fit(output, shared, threads)
    selection = json.loads((output / "selection.json").read_text())
    scorer = OnlineScorer(selection, device)
    smoke(scorer, output)
    for split in ("dev", "test"):
        score(split, output, device, scorer)
    write_json(output / "status.json", dict(complete=True, method=METHOD, splits=["dev", "test"],
               selected_iterations=selection["iterations"], actual_experiment_results=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "fit", "smoke", "score", "run"))
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--shared", type=Path, default=SHARED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", choices=("dev", "test"))
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.stage == "prepare":
        result = prepare(args.output, args.shared)
    elif args.stage == "fit":
        result = fit(args.output, args.shared, args.threads)
    elif args.stage == "smoke":
        selected = json.loads((args.output / "selection.json").read_text())
        result = smoke(OnlineScorer(selected, args.device), args.output)
    elif args.stage == "score":
        if args.split is None:
            parser.error("score requires --split")
        result = score(args.split, args.output, args.device)
    else:
        result = run(args.output, args.shared, args.device, args.threads)
    print(json.dumps(dict(stage=args.stage, output=str(args.output), finished=True)))


if __name__ == "__main__":
    main()
