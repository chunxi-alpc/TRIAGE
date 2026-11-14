"""CPU-only publication tables from completed predictions and frozen metrics."""
from pathlib import Path
from collections import defaultdict
import argparse, csv, gzip, json
import numpy as np

E5 = "reports/aster_e5_gaps_20261112"
E678 = "reports/aster_e678_20261112"
OUT = "reports/aster_completion_20261114"
SERVER678 = Path("/home/gcx/aster-e678-gaps-20261112")
METHODS = ["raw", "bias_only", "aster_cr"]
TRANSFER_METHODS = ["raw", "calibration", "E1"]
DRAWS, SEED = 2000, 11

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def read_csv(path):
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))

def records(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def write_gz(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as handle:
            for row in rows:
                handle.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))

def portable(value):
    if isinstance(value, str) and "/reports/" in value:
        return "reports/" + value.split("/reports/", 1)[1]
    return value

def e5_table(root):
    base = root / E5
    rows = read_csv(base / "section_4_5_new_e1.csv")
    boot = {(r["scope"], r["comparison"], r["metric"]): r for r in
            read_csv(base / "output_controls_e1/all_source_bootstrap.csv")}
    for row in rows:
        method = row["method"]
        for key in ["source", "ci_source"]:
            row[key] = portable(row[key])
        row["result_role"] = "new E1 baseline result"
        row["history"] = "Previously used 40 sources / 806 paths"
        if "endpoint_nll" in method or method == "linear_readout_e1_last_attention_ce":
            for metric, column in [
                ("ap_score_change_rms", "AP_rms"),
                ("cf_correction_rms", "CF_correction_rms"),
                ("cf_paired_accuracy", "CF_paired_accuracy"),
                ("natural_path_accuracy", "path_accuracy")]:
                ci = boot["test", method, metric]
                row[column + "_ci_low"] = ci["low"]
                row[column + "_ci_high"] = ci["high"]
            ci = boot["test", method + " minus raw", "natural_path_accuracy"]
            row["path_difference_vs_raw"] = ci["estimate"]
            row["path_difference_ci_low"] = ci["low"]
            row["path_difference_ci_high"] = ci["high"]
            row["CF_paired_correct"] = round(float(row["CF_paired_accuracy"]) * int(row["CF_n"]))
        else:
            source = root / row["source"]
            metrics = read_json(source)
            ci = metrics["groups"]["CF"]["paired_accuracy_ci"]
            row["CF_paired_accuracy_ci_low"], row["CF_paired_accuracy_ci_high"] = ci
    historical = read_csv(base / "section_4_5_data.csv")
    refs = []
    for name in ["raw", "scale_095"]:
        row = next(r.copy() for r in historical if r["method"] == name)
        row["result_role"] = "reused raw or fixed analytic reference"
        row["E1_compatibility"] = "no learned parameters"
        row["training_protocol"] = "none (raw)" if name == "raw" else "fixed a=0.95, no fit"
        row["source"] = E5 + "/historical/section_4_5_historical_table.csv"
        row["ci_source"] = row["source"]
        refs.append(row)
    ref = read_csv(base / "e1_reference/section_4_5_e1_reference.csv")[0]
    ref["result_role"] = "reused frozen ASTER-CR E1"
    ref["training_protocol"] = "E1 / Dataset A 2540 pairs; existing frozen parameters"
    ref["CF_paired_accuracy"] = int(ref["CF_paired_correct"]) / int(ref["CF_n"])
    detail = read_json(base / "e1_reference/archived_test_metrics_and_ci.json")
    ref["interference_path_n"] = detail["metrics"]["interference_path_count"]
    ref["interference_path_correct"] = detail["metrics"]["interference_path_correct"]
    ref["source"] = E5 + "/e1_reference/archived_test_metrics_and_ci.json"
    ref["ci_source"] = ref["source"]
    refs.append(ref)
    all_rows = refs + rows
    write_csv(root / OUT / "section_4_5_E1_with_95ci.csv", all_rows)
    seeds = []
    for family in ["lora_SFT_E1", "lofit_SFT_E1", "LoReFT_binary_task_adaptation_E1"]:
        selected = [r for r in rows if r["method"].startswith(family)]
        assert len(selected) == 3
        x = np.array([float(r["path_accuracy"]) for r in selected])
        seeds.append(dict(method=family, seeds="11;22;33", n_seeds=3,
                          path_accuracy_mean=float(x.mean()), path_accuracy_sample_sd=float(x.std(ddof=1)),
                          path_accuracy_min=float(x.min()), path_accuracy_max=float(x.max()),
                          interpretation="Descriptive seed variation; not a confidence interval"))
    write_csv(root / OUT / "E5_seed_variation.csv", seeds)
    return all_rows

def cluster_statistics(grouped, methods):
    """Paired percentile source-cluster bootstrap; all methods share draws."""
    values = np.array([grouped[s] for s in sorted(grouped)], dtype=float)
    rng = np.random.default_rng(SEED)
    ids = rng.integers(0, len(values), size=(DRAWS, len(values)))
    sampled = values[ids].sum(axis=1)
    rates = sampled[:, 1:] / sampled[:, 0, None]
    total = values.sum(axis=0)
    result = {}
    for i, method in enumerate(methods):
        ci = np.quantile(rates[:, i], [0.025, 0.975])
        raw = np.quantile(rates[:, i] - rates[:, 0], [0.025, 0.975])
        bias = np.quantile(rates[:, i] - rates[:, 1], [0.025, 0.975])
        result[method] = dict(n=int(total[0]), correct=int(total[i + 1]),
            accuracy=float(total[i + 1] / total[0]), accuracy_ci_low=float(ci[0]), accuracy_ci_high=float(ci[1]),
            delta_vs_raw=float((total[i + 1] - total[1]) / total[0]),
            delta_vs_raw_ci_low=float(raw[0]), delta_vs_raw_ci_high=float(raw[1]),
            delta_vs_bias=float((total[i + 1] - total[2]) / total[0]),
            delta_vs_bias_ci_low=float(bias[0]), delta_vs_bias_ci_high=float(bias[1]),
            source_clusters=len(values), bootstrap_draws=DRAWS, bootstrap_seed=SEED)
    return result

def official_mega(root):
    result, exports = [], []
    for model in ["qwen_base", "gemma2_base", "llama31"]:
        relative = E678 + "/models/" + model + "/evaluation/mega_official"
        source = SERVER678 / relative
        meta = read_json(root / relative / "metrics.json")
        assert meta["state"] == "complete"
        paths, natural = {}, {}
        published = root / relative / "published"
        if (published / "path_predictions.jsonl.gz").exists():
            for row in records(published / "path_predictions.jsonl.gz"):
                paths[row["case_id"], row["method"]] = row
            for row in records(published / "natural_node_predictions.jsonl.gz"):
                natural[row["id"]] = row
        else:
            for shard in sorted(source.glob("shard*")):
                for row in records(shard / "path_predictions.jsonl"):
                    paths[row["case_id"], row["method"]] = row
                for row in records(shard / "natural_nodes.jsonl"):
                    natural[row["id"]] = row
        assert len(paths) == meta["expected"]["paths"] * len(METHODS)
        assert len(natural) == meta["expected"]["gold_path_nodes"]
        out = root / relative / "published"
        write_gz(out / "path_predictions.jsonl.gz", [paths[k] for k in sorted(paths)])
        write_gz(out / "natural_node_predictions.jsonl.gz", [natural[k] for k in sorted(natural)])
        grouped = defaultdict(lambda: np.zeros(4))
        cases = sorted({case for case, method in paths if method == "raw"})
        for case in cases:
            group = grouped[paths[case, "raw"]["source_id"]]
            group[0] += 1
            for i, method in enumerate(METHODS):
                group[i + 1] += int(paths[case, method]["correct"])
        stats = cluster_statistics(grouped, METHODS)
        for method in METHODS:
            recorded = next(v for v in meta["metrics"] if v["method"] == method and v["metric"] == "dynamic_path")
            row = dict(model=model, method=method, metric="dynamic_path", **stats[method])
            assert row["correct"] == recorded["correct"] and row["n"] == recorded["n"]
            row["reported_delta_vs_raw_ci_low"] = recorded["paired_source_ci_low"]
            row["reported_delta_vs_raw_ci_high"] = recorded["paired_source_ci_high"]
            row["source"] = relative + "/metrics.json"
            row["predictions"] = relative + "/published/path_predictions.jsonl.gz"
            row["history"] = "Official 7458 paths; earlier source/input use retained per record"
            result.append(row)
        exports.append(dict(model=model, paths=len(cases), path_method_records=len(paths),
                            natural_node_records=len(natural), directory=str(out.relative_to(root))))
    write_csv(root / OUT / "official_MEGA_7458.csv", result)
    write_json(root / OUT / "official_prediction_exports.json", exports)
    return result

def transfers(root):
    result = []
    for model in ["llama31", "gemma2", "gemma3", "medgemma"]:
        for scope in ["nli4ct", "nli4ct_official5500", "medpic", "medpic_official467"]:
            relative = E678 + "/models/" + model + "/evaluation/" + scope
            base = root / relative
            meta = read_json(base / "metrics.json")
            assert meta["complete"] is True
            grouped = defaultdict(lambda: np.zeros(4))
            if scope.startswith("nli"):
                for row in records(base / "predictions.jsonl.gz"):
                    if scope == "nli4ct" and not row["original"]:
                        continue
                    group = grouped[row["source_id"]]
                    group[0] += 1
                    for i, method in enumerate(TRANSFER_METHODS):
                        group[i + 1] += row["predictions"][method] == row["label"]
                metric = "all_statement_accuracy" if scope == "nli4ct_official5500" else "natural_native_accuracy"
                count_key = "all_statement_count" if scope == "nli4ct_official5500" else "natural_native_count"
                correct_key = "all_statement_correct" if scope == "nli4ct_official5500" else "natural_native_correct"
            else:
                full = {}
                for row in records(base / "question_predictions.jsonl.gz"):
                    if row["variant"] == "full":
                        full[row["item_id"], row["method"]] = row
                cases = sorted({key[0] for key in full if key[1] == "raw"})
                for case in cases:
                    group = grouped[full[case, "raw"]["source_id"]]
                    group[0] += 1
                    for i, method in enumerate(TRANSFER_METHODS):
                        group[i + 1] += full[case, method]["correct"]
                metric, count_key, correct_key = "natural_native_accuracy", "natural_native_count", "natural_native_correct"
            stats = cluster_statistics(grouped, TRANSFER_METHODS)
            for method in TRANSFER_METHODS:
                row = dict(model=model, population=scope, method=method, metric=metric, **stats[method])
                assert row["n"] == meta["models"][method][count_key]
                assert row["correct"] == meta["models"][method][correct_key]
                row["source"] = relative + "/metrics.json"
                row["history"] = "All 467 questions include development questions; descriptive" if scope == "medpic_official467" else "Existing designated evaluation population"
                row["frozen_source_parameters"] = True
                row["M4_target_calibration"] = False
                result.append(row)
    write_csv(root / OUT / "frozen_transfer_results_95ci.csv", result)
    return result

def source_test(root):
    result = []
    for model in ["qwen_base", "gemma2_base", "llama31", "gemma2"]:
        relative = E678 + "/models/" + model + "/evaluation/test/metrics.json"
        d = read_json(root / relative)
        if not d.get("complete"):
            continue
        for method, metrics in d["models"].items():
            row = dict(model=model, method=method, source=relative, **metrics)
            row["history"] = "Previously used 40 sources, 806 natural paths, 2418 interference paths"
            result.append(row)
    write_csv(root / OUT / "E6_controlled_source_test.csv", result)

def base_it(root):
    result = []
    for model in ["qwen", "gemma2"]:
        relative = E678 + "/base_it/" + model + "/comparison.json"
        d = read_json(root / relative)
        a, b = d["base"], d["instruction_tuned"]
        result.append(dict(family=model, base_name=d["base_name"], it_name=d["it_name"],
            base_selected_heads=a["selected_head_count"], it_selected_heads=b["selected_head_count"],
            shared_heads=d["head_selection"]["intersection_count"], jaccard=d["head_selection"]["jaccard"],
            source=relative, interpretation=d["interpretation"]))
    write_csv(root / OUT / "Base_IT_head_comparison.csv", result)

def figure(root, official):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(8.4, 4.2), layout="constrained")
    models = ["qwen_base", "gemma2_base", "llama31"]
    colors = ["#9A9A9A", "#E69F00", "#0072B2"]
    names = ["Raw", "Intercept only", "ASTER-CR E1"]
    x = np.arange(len(models))
    for i, method in enumerate(METHODS):
        rows = [next(r for r in official if r["model"] == m and r["method"] == method) for m in models]
        y = np.array([r["accuracy"] * 100 for r in rows])
        err = np.array([[r["accuracy"] - r["accuracy_ci_low"] for r in rows],
                        [r["accuracy_ci_high"] - r["accuracy"] for r in rows]]) * 100
        bars = ax.bar(x + (i - 1) * .25, y, .23, yerr=err, capsize=3, color=colors[i], label=names[i])
        for bar, value in zip(bars, y):
            ax.text(bar.get_x() + bar.get_width()/2, value + 3.5, f"{value:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, ["Qwen3-8B-Base", "Gemma-2-9B-Base", "Llama-3.1-8B-Instruct"])
    ax.set_ylabel("Complete-path accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Official MEGA: 7,458 paths / 418 source clusters")
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    folder = root / OUT / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    for extension in ["svg", "pdf", "png"]:
        fig.savefig(folder / ("official_MEGA_accuracy." + extension), dpi=220)
    plt.close(fig)

def main(root):
    e5 = e5_table(root)
    official = official_mega(root)
    transfer = transfers(root)
    source_test(root)
    base_it(root)
    figure(root, official)
    write_json(root / OUT / "publication_validation.json", dict(
        state="complete", e5_new_result_rows=18, e5_reference_rows=3, e5_table_rows=len(e5),
        official_model_method_rows=len(official), transfer_rows=len(transfer),
        official_populations_verified_against_existing_metrics=True,
        transfer_counts_verified_against_existing_metrics=True,
        bootstrap=dict(draws=DRAWS, seed=SEED, unit="source cluster", paired=True),
        new_training_runs=0, new_model_forward_passes=0))
    print(json.dumps({"state":"complete", "e5_rows":len(e5), "official_rows":len(official), "transfer_rows":len(transfer)}))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path.cwd())
    main(p.parse_args().root.resolve())
