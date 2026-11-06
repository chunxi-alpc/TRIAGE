"""Recompute result tables from the published saved scores, using CPU only."""
import argparse
import gzip
import json
import shutil
import tempfile
from pathlib import Path

from . import pipeline
from .evaluation import evaluate
from .existing_report import history_report, path_report, metric_row, table_csv
from .common import write_json


def materialize(source, target):
    """Restore only the input and score files needed by these table calculations."""
    for folder in ("data", "scores", "paths", "diagnostics", "results"):
        (target / folder).mkdir(parents=True, exist_ok=True)
        for path in (source / folder).iterdir():
            if not path.is_file():
                continue
            if folder == "results" and path.name not in ("fusion.json", "amplification_selection.json"):
                continue
            dest = target / folder / path.name.removesuffix(".gz")
            if path.suffix == ".gz":
                with gzip.open(path, "rb") as src, dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            else:
                shutil.copy2(path, dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2] / "results/aster_existing_data_20261106")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="aster-saved-tables-") as temp:
        root = Path(temp)
        materialize(args.root, root)
        views, pairs = pipeline.load_data(root)
        scores = pipeline.all_scores(root)
        amp = json.loads((root / "results/amplification_selection.json").read_text())
        scores = {m: s for m, s in scores.items() if not m.startswith("amplify_") or m == amp["selected"]}
        scores["erase_plus_selected_amplification"] = scores["aster"] if amp["selected"] == "aster" else scores.pop(amp["selected"])
        flags = {"semantic": "semantic_available", "boundary_guideline": "boundary_annotation_available", "boundary_facts": "decisive_annotation_available"}
        limited = {m: ("semantic_available" if "semantic" in m else flags[m]) for m in scores if "semantic" in m or m in flags}
        full = {m: s for m, s in scores.items() if m not in limited}
        main_result = evaluate(views, pairs, full, bootstrap=2000)
        write_json(root / "results/main_effects.json", main_result)
        table_csv(root / "results/main_effects.csv", [metric_row(m, result) for m, result in main_result["methods"].items()])
        subsets = {}
        for method, flag in limited.items():
            eligible = {key: v for key, v in views.items() if v.get(flag, False)}
            ps = [p for p in pairs if p["left"] in eligible and p["right"] in eligible]
            subsets[method] = evaluate(eligible, ps, {m: scores[m] for m in dict.fromkeys(("full", "aster", method))}, bootstrap=2000)
        write_json(root / "results/available_subsets.json", subsets)
        history_report(root)
        path_report(root)
        args.output.mkdir(parents=True, exist_ok=True)
        names = ["main_effects.json", "main_effects.csv", "available_subsets.json", "historical_fact_preference.json", "historical_fact_preference.csv", "historical_deletion_restore.json", "historical_deletion_restore.csv", "path_results.json", "path_results.csv"]
        for name in names:
            shutil.copy2(root / "results" / name, args.output / name)
        print(json.dumps({"state": "complete", "model_calls": 0, "output": str(args.output), "tables": names}))


if __name__ == "__main__":
    main()
