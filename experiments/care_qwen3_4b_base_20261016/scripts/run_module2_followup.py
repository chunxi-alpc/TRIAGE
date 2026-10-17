#!/usr/bin/env python3
"""Run on the server only. Saves real per-case outputs; never fabricates missing runs."""
from __future__ import annotations
import argparse
import contextlib
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
# run_experiments adds the pinned external/SEKA path before importing its src package.
from run_experiments import tokenize_span, causal_inputs
from module2_subspaces import (
    LOCATIONS, MODEL_REVISION, SEKA_COMMIT, attention_bias, bias_support, collect,
    contrastive_basis, fit_basis, key_edit, logit_delta, normalized_delta,
    o2_cache, official_projectors, paired_rows, save_json, sha256, training_matrices,
)

def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def prepare_rows(artifacts, dataset):
    rows = read_jsonl(artifacts / ("data/clinical/candidates.jsonl" if dataset == "clinical" else "data/derived/pilot.jsonl"))
    if dataset == "clinical":
        return rows
    for r in rows:
        r.update(pair_id=r["group_id"], rule_id=r["group_id"], value=r["patient_value"],
                 gold_label=r["option_map"][r["gold_answer"]])
    return rows

def matched_o2_controls(rows):
    controls = []
    for row in rows:
        if row["state"] not in ("RA", "RN"):
            continue
        r = copy.deepcopy(row)
        value = r["patient_value"] + (-4 if r["applicable"] else 4)
        patient = r["patient"].replace(f'= {r["patient_value"]} units', f'= {value} units')
        assert patient != r["patient"]
        r["prompt"] = r["prompt"].replace(r["patient"], patient)
        r["patient"], r["value"], r["patient_value"] = patient, value, value
        action = row["prompt"][slice(*row["spans"]["action"])]
        start = r["prompt"].index(action)
        r["spans"]["action"] = [start, start + len(action)]
        r["case_id"] += "-matched-control"
        controls.append(r)
    return controls

def config_id(method, rank, location, seed):
    return f"{method}-r{rank}-L{location[0]}-H{location[1]}-s{seed}"

def make_operators(train, pool, keys, official, locations, ranks):
    operators, metadata, blockers = {}, {}, []
    for location in locations:
        raw, signal, background, audit = training_matrices(train, keys, location, control_rows=pool)
        for method in ("B1", "B2", "B3", "B4", "random42", "random43", "random44"):
            for rank in ([0] if method == "B2" else ranks):
                seed = int(method[6:]) if method.startswith("random") else 42
                cid = config_id(method, rank, location, seed)
                try:
                    if method == "B2":
                        op = official[location].float()
                    else:
                        if method == "B1":
                            basis = fit_basis(raw, rank, center=True)
                        elif method == "B3":
                            basis = fit_basis(signal, rank)
                        elif method == "B4":
                            if background is None:
                                raise ValueError("no same-rule same-label magnitude-matched numeric controls")
                            basis = contrastive_basis(signal, background, rank)
                        else:
                            basis, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(128, rank)))
                        op = torch.from_numpy(basis @ basis.T).float()
                    assert torch.isfinite(op).all()
                    operators[cid] = op
                    metadata[cid] = dict(method=method, rank=rank, location=list(location), seed=seed,
                        trace=float(op.trace()), frobenius=float(op.norm()),
                        controls=len(audit) if method == "B4" else None,
                        control_audit=audit if method == "B4" else None,
                        operator="official (Ppos + .25 Pneg)/2" if method == "B2" else "orthogonal projector")
                except ValueError as error:
                    blockers.append(dict(config=cid, reason=str(error)))
    return operators, metadata, blockers

def score(model, tok, row, context=None):
    enc, _, _ = tokenize_span(tok, row["prompt"], row["spans"]["action"])
    ids = []
    for letter in "ABCD":
        tokens = tok.encode(" " + letter, add_special_tokens=False)
        assert len(tokens) == 1
        ids.append(tokens[0])
    with context if context is not None else contextlib.nullcontext():
        logits = model(**causal_inputs(enc, model.device)).logits[0, -1].float()
    chosen = logits[ids]
    return chosen.softmax(-1).cpu().tolist()

def record(row, probs, fold, role, cid, epsilon, mode, report=None):
    pred = "ABCD"[int(np.argmax(probs))]
    g = "ABCD".index(row["gold_answer"])
    return dict(case_id=row["case_id"], pair_id=row["pair_id"], group_id=row["group_id"],
        applicable=row["applicable"], gold_answer=row["gold_answer"], prediction=pred,
        correct=int(pred == row["gold_answer"]),
        misapplied=int(not row["applicable"] and pred == row["inapplicable_action"]),
        candidate_probabilities=probs,
        gold_margin=probs[g] - max(v for i, v in enumerate(probs) if i != g),
        fold=fold, role=role, config=cid, epsilon=epsilon, mode=mode, disturbance=report or {})

def choose(records, metadata, strengths):
    selected = {}
    for method in sorted({m["method"] for m in metadata.values()}):
        choices = []
        for cid, m in metadata.items():
            if m["method"] != method:
                continue
            for epsilon in strengths:
                rows = [r for r in records if r["config"] == cid and r["epsilon"] == epsilon]
                pairs = {}
                for r in rows:
                    pairs.setdefault(r["pair_id"], []).append(r)
                assert rows and all(len(p) == 2 for p in pairs.values())
                pa = np.mean([all(r["correct"] for r in p) for p in pairs.values()])
                acc = np.mean([r["correct"] for r in rows])
                choices.append((float(pa), float(acc), -epsilon, -m["rank"], cid, epsilon))
        best = max(choices)
        selected[method] = dict(config=best[4], epsilon=best[5], dev_pair_accuracy=best[0], dev_accuracy=best[1])
    return selected

def run_fold(model, tok, all_rows, keys, train, dev, test, output, args, fold):
    output.mkdir(parents=True, exist_ok=True)
    split = {role: sorted({r["group_id"] for r in values}) for role, values in [("train", train), ("dev", dev), ("test", test)]}
    assert not (set(split["train"]) & set(split["dev"]) or set(split["train"]) & set(split["test"]) or set(split["dev"]) & set(split["test"]))
    save_json(output / "split.json", split)
    pool = train
    if args.dataset == "o2":
        extra = matched_o2_controls(train)
        for row in extra:
            keys[row["case_id"]] = collect(model, tok, row, args.locations)[0]
        pool = train + extra
    official_dir = output / "official"
    # O2 archived operators are reused only with exact train-group and hash checks.
    if args.dataset == "o2":
        old = json.loads((args.artifacts / "results/manifest.json").read_text())
        assert sorted(old["projection_training_groups"]) == split["train"]
        archive_dir = args.artifacts / "results/projections"
        for name in ("pos", "neg"):
            path = archive_dir / f"Qwen3-4B-Base_{name}_proj.pt"
            expected = old["artifact_hashes"][f"results/projections/{path.name}"]
            assert sha256(path) == expected
        pos = torch.load(archive_dir / "Qwen3-4B-Base_pos_proj.pt", weights_only=True)
        neg = torch.load(archive_dir / "Qwen3-4B-Base_neg_proj.pt", weights_only=True)
        assert pos["layers"] == neg["layers"] == list(range(36))
        official = {location: (pos["proj"][location[0], location[1]] + .25 * neg["proj"][location[0], location[1]]) / 2 for location in args.locations}
    else:
        official = official_projectors(model, tok, train, sorted({L for L, _ in args.locations}), official_dir, True)
    operators, metadata, blockers = make_operators(train, pool, keys, official, args.locations, args.ranks)
    torch.save(operators, output / "operators.pt")
    save_json(output / "operators.json", metadata)
    save_json(output / "blocked_configurations.json", blockers)
    # Persist immediately. A partial run is not silently promoted to complete.
    journal = output / "predictions.jsonl"
    prior = read_jsonl(journal) if journal.exists() else []
    done = {(r["role"], r["case_id"], r["config"], r["epsilon"], r["mode"]) for r in prior}
    records = list(prior)
    stream = journal.open("a", buffering=1)
    def append(rec):
        key = (rec["role"], rec["case_id"], rec["config"], rec["epsilon"], rec["mode"])
        if key not in done:
            stream.write(json.dumps(rec, allow_nan=False) + "\n")
            done.add(key)
            records.append(rec)
    try:
        for row in dev:
            _, capture, idx, baseline = collect(model, tok, row, args.locations)
            append(record(row, baseline, fold, "dev", "baseline", 0., "baseline"))
            for cid, op in operators.items():
                location = tuple(metadata[cid]["location"])
                for epsilon in args.strengths:
                    if ("dev", row["case_id"], cid, epsilon, "amplify") in done:
                        continue
                    if epsilon == 0:
                        probs, report = baseline, dict(relative_norm=0.)
                    else:
                        with key_edit(model, idx, location, op, epsilon) as report:
                            probs = score(model, tok, row)
                    append(record(row, probs, fold, "dev", cid, epsilon, "amplify", report))
        dev_records = [r for r in records if r["role"] == "dev" and r["mode"] == "amplify"]
        selection = choose(dev_records, metadata, args.strengths)
        save_json(output / "selection.json", selection)
        test_controls = matched_o2_controls(test) if args.dataset == "o2" else []
        for control in test_controls:
            keys[control["case_id"]] = collect(model, tok, control, args.locations)[0]
        for row in test:
            _, capture, idx, baseline = collect(model, tok, row, args.locations, capture_q=True)
            append(record(row, baseline, fold, "test", "baseline", 0., "baseline"))
            other = next((b if a["case_id"] == row["case_id"] else a) for a, b in paired_rows(test) if row["case_id"] in (a["case_id"], b["case_id"]))
            for method, selected in selection.items():
                cid, epsilon = selected["config"], selected["epsilon"]
                op, location = operators[cid], tuple(metadata[cid]["location"])
                L, h = location
                for mode in ("amplify", "uniform_bias", "ablate", "patch", "patch_same_label"):
                    if ("test", row["case_id"], cid, epsilon, mode) in done:
                        continue
                    report = {}
                    if mode == "uniform_bias":
                        delta_mask = logit_delta(model, capture[L], idx, location, op, epsilon)
                        support = bias_support(delta_mask.shape[-1], idx, location, delta_mask.device)
                        attention_bias._support = support
                        rms = delta_mask[support].square().mean().sqrt()
                        report = dict(logit_rms=float(rms), calibration="per-input baseline QK; no gold label")
                        probs = score(model, tok, row, attention_bias(model, location, delta_mask, uniform=True))
                    else:
                        donor_row = other
                        if mode == "patch_same_label":
                            pool = test + test_controls
                            choices = [r for r in pool if r["case_id"] != row["case_id"]
                                and (r["group_id"], r["rule_id"], r["gold_label"], r["applicable"]) ==
                                    (row["group_id"], row["rule_id"], row["gold_label"], row["applicable"])]
                            distance = abs(float(row["value"]) - float(other["value"]))
                            donor_row = min(choices, key=lambda r: (abs(abs(float(row["value"])-float(r["value"]))-distance), r["case_id"])) if choices else None
                            if donor_row is None:
                                raise ValueError("No same-label patch control in held-out family")
                        donor = torch.from_numpy(keys[donor_row["case_id"]][location]) if mode.startswith("patch") else None
                        with key_edit(model, idx, location, op, epsilon, "patch" if mode.startswith("patch") else mode, donor) as report:
                            probs = score(model, tok, row)
                        if mode.startswith("patch"):
                            report.update(donor_case_id=donor_row["case_id"], donor_gold=donor_row["gold_answer"],
                                numeric_distance=abs(float(row["value"])-float(donor_row["value"])),
                                opposite_pair_distance=abs(float(row["value"])-float(other["value"])))
                    append(record(row, probs, fold, "test", cid, epsilon, mode, report))
            del capture
        save_json(output / "complete.json", dict(
            status="completed_real_model_inference", records=len(records),
            selection=selection, blocked=blockers, diagnostics="patch uses opposite-applicability donor; oracle mechanistic test only"))
    finally:
        stream.close()

def smoke(model, tok, rows, output, locations):
    """Real model checks: zero edit, BF16 QK/direct-mask equivalence, patch alignment."""
    pair = paired_rows(rows)[0]
    a, b = pair
    ka, full, idx, base = collect(model, tok, a, locations, capture_q=True)
    kb, _, ib, _ = collect(model, tok, b, locations)
    ia = tokenize_span(tok, a["prompt"], a["spans"]["action"])[0]["input_ids"][0, idx]
    it = tokenize_span(tok, b["prompt"], b["spans"]["action"])[0]["input_ids"][0, ib]
    assert torch.equal(ia, it), "paired action token mismatch"
    location = locations[0]
    op = torch.eye(128)
    p0 = score(model, tok, a, key_edit(model, idx, location, op, 0.))
    assert max(abs(x-y) for x, y in zip(base, p0)) < 1e-7
    pkey = score(model, tok, a, key_edit(model, idx, location, op, .05))
    delta = logit_delta(model, full[location[0]], idx, location, op, .05)
    pscore = score(model, tok, a, attention_bias(model, location, delta))
    error = max(abs(x-y) for x, y in zip(pkey, pscore))
    assert error < .02, ("BF16 end-to-end key/mask equivalence", error)
    assert max(abs(x-y) for x, y in zip(base, pkey)) > 0, "intervention had no measurable effect"
    save_json(output / "smoke.json", dict(status="passed", zero_edit_max_error=max(abs(x-y) for x,y in zip(base,p0)),
        key_score_probability_max_error=error, tolerance=.02, paired_token_ids=ia.tolist()))
    return dict(a=a["case_id"], b=b["case_id"])

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", type=Path, default=Path("/home/gcx/care_20261016/CARE_Codex_Pilot"))
    p.add_argument("--model", type=Path)
    p.add_argument("--dataset", choices=["o2", "clinical"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ranks", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--strengths", type=float, nargs="+", default=[0., .01, .05, .1])
    p.add_argument("--locations", type=str, nargs="+", default=["2:0", "20:5", "9:0", "35:0"])
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    args.locations = [tuple(map(int, s.split(":"))) for s in args.locations]
    assert args.locations and len(set(args.locations)) == len(args.locations)
    assert all(0 <= L < 36 and 0 <= h < 8 for L, h in args.locations)
    assert all(0 < r <= 128 for r in args.ranks)
    assert all(0 <= s <= 1 for s in args.strengths) and 0. in args.strengths
    assert all(f in [0, 1, 2] for f in args.folds)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    assert len(visible.split(",")) == 1 and visible in {"0", "1", "2", "3", "6", "7"}, "pin one authorized physical GPU"
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model_path = args.model or args.artifacts / "models/Qwen3-4B-Base"
    assert model_path.is_dir(), "local fixed model weights required"
    config = json.loads((model_path / "config.json").read_text())
    assert config.get("_name_or_path", "Qwen/Qwen3-4B-Base").rstrip("/").split("/")[-1] == "Qwen3-4B-Base"
    official_root = ROOT / "external/SEKA"
    actual_seka = subprocess.check_output(["git", "-C", str(official_root), "rev-parse", "HEAD"], text=True).strip()
    assert actual_seka == SEKA_COMMIT
    assert not subprocess.check_output(["git", "-C", str(official_root), "status", "--porcelain"], text=True).strip(), "official source must be unmodified"
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    fingerprint = dict(dataset=args.dataset, data_sha256=sha256(args.artifacts / ("data/clinical/candidates.jsonl" if args.dataset == "clinical" else "data/derived/pilot.jsonl")),
        model_files={str(x.relative_to(model_path)): sha256(x) for x in sorted(model_path.glob("*.safetensors"))},
        config_sha256=sha256(model_path / "config.json"), seka_commit=actual_seka,
        ranks=args.ranks, strengths=args.strengths, locations=[list(x) for x in args.locations], folds=args.folds,
        source_sha256={x.name: sha256(x) for x in [Path(__file__), ROOT/"scripts/module2_subspaces.py"]})
    assert fingerprint["model_files"], "no safetensors weights found"
    archived = json.loads((args.artifacts / "results/manifest.json").read_text())
    assert archived["model_revision"] == MODEL_REVISION
    assert fingerprint["model_files"] == {name: info["sha256"] for name, info in archived["weights"].items()}, "weight hashes differ from fixed official revision"
    if args.dataset == "clinical":
        clinical_manifest = json.loads((args.artifacts / "results/obs1_clinical/manifest.json").read_text())
        assert fingerprint["data_sha256"] == clinical_manifest["data_sha256"]
    fingerprint["tokenizer_files"] = {x.name: sha256(x) for x in sorted(model_path.glob("*token*")) if x.is_file()}
    settings = args.output / "run_fingerprint.json"
    if settings.exists():
        assert json.loads(settings.read_text()) == fingerprint, "resume configuration/provenance mismatch"
    save_json(settings, fingerprint)
    save_json(args.output / "status.json", dict(status="running", planned_model_revision=MODEL_REVISION, model_revision_verification="all model shards match fixed archived revision"))
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    c = model.config
    assert (c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads, c.head_dim) == (36, 32, 8, 128)
    rows = prepare_rows(args.artifacts, args.dataset)
    relevant = rows if args.dataset == "clinical" else [r for r in rows if r["state"] in ("RA", "RN")]
    token_audit = []
    for a, b in paired_rows(relevant):
        pair_ids = []
        for row in (a, b):
            enc, indices, _ = tokenize_span(tok, row["prompt"], row["spans"]["action"])
            assert row["spans"]["action"][1] <= row["prompt"].index("Options")
            pair_ids.append(enc["input_ids"][0, indices].tolist())
            token_audit.append(dict(case_id=row["case_id"], indices=indices, token_ids=pair_ids[-1]))
        assert pair_ids[0] == pair_ids[1], ("paired action tokens differ", a["pair_id"])
    save_json(args.output / "target_token_audit.json", token_audit)
    smoke(model, tok, relevant, args.output, args.locations)
    if args.smoke:
        return
    if args.dataset == "o2":
        keys = o2_cache(args.artifacts, rows, args.locations)
        split_list = [(0, *[[r for r in relevant if r["split"] == role] for role in ("train", "dev", "test")])]
    else:
        # Every clinical family is tested once; reuse the exact archived family split.
        split_list = []
        for fold in args.folds:
            info = json.loads((args.artifacts / f"results/obs1_clinical/fold{fold}/manifest.json").read_text())
            split_list.append((fold, *[[r for r in relevant if r["group_id"] in info[k]] for k in ("train_families", "dev_families", "test_families")]))
        keys = {row["case_id"]: collect(model, tok, row, args.locations)[0] for row in relevant}
        torch.save(keys, args.output / "baseline_target_keys.pt")
    for fold, train, dev, test in split_list:
        run_fold(model, tok, rows, keys, train, dev, test, args.output / f"fold{fold}", args, fold)
    save_json(args.output / "status.json", dict(status="completed_real_model_inference",
        seconds=time.time()-start, physical_gpu=visible, gpu=torch.cuda.get_device_name(),
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
        model_id="Qwen/Qwen3-4B-Base", claimed_revision=MODEL_REVISION, outer_folds=args.folds if args.dataset=="clinical" else [0],
        model_provenance="all model shard digests matched archived official revision",
        seka_commit=actual_seka, seed=42, patient_data="simulated",
        generalization="held-out clinical families" if args.dataset=="clinical" else "one fictional grammar; held-out identifier groups",
        versions={k:importlib.metadata.version(k) for k in ("torch","transformers","numpy")}))
if __name__ == "__main__":
    with torch.inference_mode():
        main()
