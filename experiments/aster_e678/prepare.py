"""Prepare model-specific E1 jobs without allocating a GPU or altering history."""
import argparse, json, time
from pathlib import Path
from collections import Counter
from experiments.aster_cr_retrain_paired_v2_20261111.common import rows, read, write, lines, load_inputs

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports/aster_e678_20261112"
SOURCE = Path("/home/gcx/aster-cr-retrain-paired-v2-20261111/reports/aster_cr_retrain_paired_v2_20261111")
OLD = Path("/home/gcx/aster_five_models_20261109")
KEYS = ("gemma2", "gemma2_base", "llama31", "gemma3", "medgemma", "qwen_base")
DATA_FILES = ("E1_pairs.jsonl", "E1_views.jsonl", "natural_views.jsonl", "historical_pairs.jsonl",
              "historical_views.jsonl", "path_features.jsonl", "path_interference_views.jsonl",
              "nli4ct_views.jsonl", "nli4ct_pairs.jsonl", "medpic_views.jsonl", "medpic_pairs.jsonl")
def view_id(v):
    return v.get("uid", v.get("view_id"))
def prepare(refresh=False):
    specs = read(REPO / "experiments/aster_e678/models.json")
    data = {name: rows(SOURCE / "data" / name) for name in DATA_FILES}
    pairs = data["E1_pairs.jsonl"]
    counts = Counter(p["kind"] for p in pairs)
    if counts != Counter(dict(A=635, P=635, C=635, F=635)):
        raise ValueError("Expected current E1 2540-pair balanced source dataset")
    stage_ids = {}
    stage_ids["fit"] = sorted({p[k] for p in pairs for k in ("left", "right")} |
                             {view_id(v) for v in data["natural_views.jsonl"]})
    for split in ("dev", "test"):
        stage_ids[split] = sorted({view_id(v) for name in ("historical_views.jsonl",
            "path_features.jsonl", "path_interference_views.jsonl") for v in data[name] if v.get("split") == split})
    for task in ("nli4ct", "medpic"):
        stage_ids[task] = sorted({view_id(v) for v in data[task + "_views.jsonl"] if v.get("split") == "test"})
    grad_ids = sorted({p["left"] for p in pairs})
    inputs = load_inputs(SOURCE)
    from .official_targets import build
    official = read(OUT / "shared/official_targets/manifest.json")
    if not official:
        official = build(inputs, SOURCE / "data", OUT / "shared")
    stage_ids.update(official["stage_ids"])
    report = {}
    for key in KEYS:
        root = OUT / "models" / key
        root.mkdir(parents=True, exist_ok=True)
        if (root / "prepared.json").exists() and not refresh:
            report[key] = read(root / "prepared.json")
            continue
        (root / "data").mkdir(exist_ok=True)
        for name in DATA_FILES:
            link = root / "data" / name
            if not link.exists():
                link.symlink_to(SOURCE / "data" / name)
        for scope, paths in official["metadata"].items():
            for kind, path in paths.items():
                link = root / "data" / (scope + "_" + kind + ".jsonl")
                if not link.exists():
                    link.symlink_to(path)
        write(root / "extra_inputs.json", {"path": official["extra_inputs"]})
        write(root / "stage_ids.json", stage_ids)
        write(root / "gradient_ids.json", grad_ids)
        # Old learned parameters and corrected predictions are never imported.
        # Only this same checkpoint's unchanged raw scores and full head arrays qualify.
        reused = {}
        for task, source_views in (("mega", data["historical_views.jsonl"]), ("nli4ct", data["nli4ct_views.jsonl"])):
            old_views_path = OLD / "data" / task / "views.jsonl"
            if not old_views_path.exists() or not (OLD / "results" / key / task).exists():
                continue
            old_views = {v["view_id"]: v for v in rows(old_views_path)}
            old_scores = {r["view_id"]: r for p in sorted((OLD / "results" / key / task / "scores").glob("*.jsonl"))
                          for r in rows(p) if r.get("status") == "ok"}
            for v in source_views:
                uid = view_id(v)
                old_id = v.get("original_view_id")
                r, old_v = old_scores.get(old_id), old_views.get(old_id)
                if not r or not old_v or not Path(r.get("feature", "")).is_file():
                    continue
                current = inputs[uid]
                if old_v["prompt"] != current["prompt"]:
                    continue
                if tuple(old_v.get("labels") or [chr(26159), chr(21542)]) != tuple(current.get("labels") or [chr(26159), chr(21542)]):
                    continue
                reused[uid] = dict(r, uid=uid, view_id=uid, available_heads="all",
                    has_grads=bool(r.get("has_grads", r.get("split") == "fit")),
                    cache_model=key, origin="same-checkpoint unchanged native template and canonical final-label score",
                    original_view_id=old_id)
        lines(root / "data/reused_scores.jsonl", list(reused.values()))
        configuration = dict(model_key=key, model=specs[key], source_protocol="E1-balanced-2540",
            source_report=str(SOURCE), fit_pairs=2540, pair_counts=dict(counts), fit_sources=80,
            dev_sources=20, head_count=51, trace_rank=16, regularization=.01, delta=.2923349,
            AP_weight_sum=4887, seed=20261108, calibration_nodes=761,
            calibration_split="source fit natural nodes, same as completed Qwen E1",
            freeze_split="corresponding checkpoint source development set",
            score_protocol="native checkpoint template; last-token BF16 backbone and FP32 YES-NO projection; native softcap",
            module_independence=["M1 ranking and exact patches", "M2 evidence and nuisance spaces", "M3 correction and intercept"],
            target_labels_used_for_fit=False, old_corrected_results_reused=False,
            gpus=[], state="prepared_waiting_codex00", stage_counts={k:len(v) for k,v in stage_ids.items()})
        write(root / "configuration.json", configuration)
        result = dict(state="prepared_waiting_codex00", model_key=key, raw_cache_reused=len(reused),
            cache_reused_by_stage={s:len(set(ids) & set(reused)) for s,ids in stage_ids.items()},
            stage_counts=configuration["stage_counts"], source_inputs=str(SOURCE / "data/inputs.jsonl"))
        write(root / "prepared.json", result)
        report[key] = result
    write(OUT / "preparation.json", dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), models=report,
        qwen_reused=str(SOURCE), gpu_started=False, historical_results_untouched=True))
    return report
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args=parser.parse_args()
    print(json.dumps(prepare(refresh=args.refresh), ensure_ascii=False))
