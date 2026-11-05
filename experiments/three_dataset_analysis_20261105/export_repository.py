"""Copy this completed run's aggregate artifacts into the repository (CPU only)."""
import argparse
import json
import re
import shutil
from pathlib import Path

SUMMARY_FILES = """
medpair_deletion.json medpair_new_accuracy.json medpair_new_comparisons.json
medpair_addback_lengths.json medpair_fusion.json medpair_fusion_parameters.json
medguide_accuracy.json medguide_swaps.json medguide_threshold.json
medguide_threshold_within_tree_supplement.json medguide_fusion.json
medguide_fusion_parameters.json medeinst_m3a.json
medeinst_m1_supplement.json medeinst_work3_supplement.json
medpair_pcd_supplement.json medguide_historical_supplement.json
historical_source_notes.json completion.json verification.json
medpair_completion.json medguide_pairing_checks.json offline_status.json
scoring_pilot_0_status.json scoring_full_0_status.json scoring_full_1_status.json
table1_medpair_deletion.csv table1_medpair_addback.csv
table1_medpair_new_accuracy.csv table2_medguide_accuracy.csv
table2_medguide_swaps.csv table3_medeinst_fact_response.csv
table3_medguide_threshold.csv table4_fusion.csv
""".split()
REPORT_FILES = ["REPORT.zh.md", "table1.md", "table2.md", "table3.md", "table4.md"]
RUN_NAME = "three_dataset_analysis_20261105"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("/home/gcx") / RUN_NAME)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    source = args.run_root / "results"
    destination = args.repo_root / "results" / RUN_NAME
    report = args.repo_root / "reports" / RUN_NAME
    required = [source / name for name in SUMMARY_FILES + REPORT_FILES]
    required.append(args.run_root / "config.json")
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing completed-run artifacts: " + ", ".join(missing))
    destination.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    for name in SUMMARY_FILES:
        if name.endswith('.csv'):
            (destination / name).write_text((source / name).read_text(encoding='utf-8'), encoding='utf-8')
        else:
            shutil.copy2(source / name, destination / name)
    shutil.copy2(args.run_root / "config.json", destination / "config.json")
    for name in REPORT_FILES:
        content = (source / name).read_text(encoding="utf-8")
        # Runtime reports refer to CSV/JSON alongside themselves. Link their
        # repository copies in the dedicated aggregate-results directory.
        for artifact in SUMMARY_FILES:
            content = re.sub(
                r"(?<![\w/])" + re.escape(artifact) + r"(?![\w])",
                "[" + artifact + "](../../results/" + RUN_NAME + "/" + artifact + ")",
                content,
            )
        content += (
            "\n[实验与结果入口](README.md) · "
            "[实验说明](EXPERIMENTS.zh.md) · [结果分析](ANALYSIS.zh.md)\n"
        )
        (report / name).write_text(content, encoding="utf-8")
    entries = [
        ("input_inventory", "/home/gcx/experiment_data_reuse_20261105/data_inventory.json"),
        ("source_inventory_snapshot", str(args.run_root / "data/source_inventory.json")),
        ("medguide_fixed_split", str(args.run_root / "data/medguide_split.json")),
        ("medguide_inputs_and_scores", str(args.run_root / "data/medguide_scores_and_inputs.jsonl")),
        ("medpair_inputs", str(args.run_root / "data/medpair_inputs.jsonl")),
        ("medpair_view_mapping", str(args.run_root / "data/medpair_views.jsonl")),
        ("medpair_addback_selection", str(args.run_root / "data/medpair_addback_pairs.jsonl")),
        ("medpair_complete_views", str(source / "medpair_complete_views.jsonl")),
        ("medpair_pilot_scores", str(source / "medpair_scores_pilot_0.jsonl")),
        ("medpair_full_gpu0_scores", str(source / "medpair_scores_full_0.jsonl")),
        ("medpair_full_gpu1_scores", str(source / "medpair_scores_full_1.jsonl")),
        ("medpair_historical_pairs", str(source / "medpair_deletion_pairs.jsonl")),
        ("medpair_new_pairs", str(source / "medpair_new_pairs.jsonl")),
        ("medguide_swap_pairs", str(source / "medguide_swap_pairs.jsonl")),
        ("medguide_threshold_pairs", str(source / "medguide_threshold_pairs.jsonl")),
        ("medeinst_independent_predictions", str(source / "medeinst_independent_predictions.jsonl")),
        ("medpair_fusion_predictions", str(source / "medpair_fusion_predictions.jsonl")),
        ("medguide_fusion_predictions", str(source / "medguide_fusion_predictions.jsonl")),
        ("cebu_historical_report", str(source / "cebu_historical_report.json")),
    ]
    locations = {
        "run": RUN_NAME,
        "repository_url": "https://github.com/chunxi-alpc/TRIAGE",
        "repository_checkout": str(args.repo_root),
        "server": "gcx@10.29.128.115",
        "environment": "/home/gcx/gcx261014",
        "run_root": str(args.run_root),
        "status": "complete",
        "new_annotations": 0,
        "aggregate_results": "results/" + RUN_NAME,
        "reports": "reports/" + RUN_NAME,
        "code": "experiments/" + RUN_NAME,
        "copied_summary_files": SUMMARY_FILES + ["config.json"],
        "server_only_artifacts": [
            {"role": role, "path": path} for role, path in entries
        ],
        "logs": str(args.run_root / "logs"),
    }
    (destination / "artifact_locations.json").write_text(
        json.dumps(locations, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "aggregate_files": len(SUMMARY_FILES) + 1,
                      "report_files": len(REPORT_FILES), "new_gpu_inference": False}))


if __name__ == "__main__":
    main()
