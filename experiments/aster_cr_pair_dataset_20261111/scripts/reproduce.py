"""Replay frozen decisions into a fresh server directory without AI calls or training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from runtime_paths import DEFAULT_ROOT, PACKAGE, SCRIPTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.frozen_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Choose a separate output directory outside the frozen experiment")
    checkout = next((p for p in [PACKAGE, *PACKAGE.parents] if (p / ".git").exists()), PACKAGE)
    if output.is_relative_to(checkout):
        raise ValueError("Keep all data and reproduced outputs outside the Git checkout")
    if output.exists():
        raise FileExistsError("Output directory must be new; existing experiments are preserved")
    required = ["historical_pairs.jsonl", "generated_candidates.jsonl",
                "review_results.jsonl", "source_splits.json"]
    for name in required:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    output.mkdir(parents=True)
    (output / "reports").mkdir()
    (output / "logs").mkdir()
    for name in required + ["f0_diagnostic_pairs.jsonl"]:
        path = source / name
        if path.exists():
            (output / name).symlink_to(path)
    metadata = ["inventory.json", "completion_historical.json", "experiment_protocol.json",
                "execution_scope.json", "balanced_selection_revision.json"]
    for name in metadata:
        path = source / "reports" / name
        if path.exists():
            (output / "reports" / name).symlink_to(path)
    manifest = source / "dataset_manifest.json"
    if manifest.exists():
        frozen_metadata = json.loads(manifest.read_text(encoding="utf-8"))
        (output / "reports/frozen_build_metadata.json").write_text(
            json.dumps(frozen_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    context = dict(frozen_root=str(source), output_root=str(output),
                   scripts=str(SCRIPTS), semantic_reviews_reused=True,
                   new_ai_reviews=0, new_generation=0, training_performed=False,
                   gpus=[], hash_checks=False)
    (output / "reports/reproduction_context.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    env = dict(os.environ, ASTER_PAIR_ROOT=str(output))
    with (output / "logs/reproduction.log").open("w", encoding="utf-8") as log:
        subprocess.run([sys.executable, str(SCRIPTS / "validate.py"),
                        "--stage", "all", "--finalize"], env=env, cwd=output,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([sys.executable, str(SCRIPTS / "final_dataset_report.py")],
                       env=env, cwd=output, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    report = json.loads((output / "reports/final_dataset_summary.json").read_text())
    expected_counts = dict(A=4368, P=635, C=676, F=666)
    expected_cases = dict(A=635, P=622, C=635, F=635)
    actual_counts = {k: report["all_classes"][k]["count"] for k in expected_counts}
    balanced = report["balanced_coverage"]
    if actual_counts != expected_counts or report["total_accepted"] != 6345:
        raise RuntimeError("Reproduction differs from the recorded accepted pool")
    if report["balanced_total"] != 2540 or any(
        balanced[k]["count"] != 635 or balanced[k]["sources"] != 80
        or balanced[k]["cases"] != expected_cases[k] for k in expected_counts
    ):
        raise RuntimeError("Reproduction differs from the recorded v2 balanced selection")
    if not (output / "dataset_manifest.json").is_file():
        raise RuntimeError("Manifest generation failed; inspect logs/report_manifest.log")
    result = dict(context, status="complete", accepted=report["total_accepted"],
                  counts=actual_counts, balanced_total=report["balanced_total"],
                  balanced_coverage=balanced, version="20261111-sol-agents-v2")
    (output / "reports/reproduction_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status="complete", accepted=6345, balanced_total=2540,
                         balanced_cases=expected_cases, output_root=str(output))))


if __name__ == "__main__":
    main()
