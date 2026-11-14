"""Publish completed artifacts without rerunning any model.
Run on the experiment server. JSONL files are preserved byte-for-byte inside gzip.
Large activation caches and pretrained weights stay at the recorded server paths.
"""
from pathlib import Path
import argparse, csv, gzip, json, os, shutil

SOURCES = [
    (Path("/home/gcx/aster-e5-gaps-20261112"), "experiments/aster_e5_gaps_20261112", "reports/aster_e5_gaps_20261112"),
    (Path("/home/gcx/aster-e678-gaps-20261112"), "experiments/aster_e678", "reports/aster_e678_20261112"),
]
PRUNE = {"__pycache__", "checkpoints", "cache", "data", "runtime", "shared", "logs",
         "truth_features", "iti_cache", "gpu_locks", "features", ".git"}
SUFFIXES = {".py", ".sh", ".json", ".jsonl", ".csv", ".md", ".gz", ".npz", ".pt", ".txt"}

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def copy_artifacts(destination):
    inventory, omitted = [], []
    for root, code, report in SOURCES:
        for prefix in [code, report]:
            for directory, dirs, files in os.walk(root / prefix):
                dirs[:] = sorted(d for d in dirs if d not in PRUNE and not d.startswith("shard"))
                for name in sorted(files):
                    src = Path(directory) / name
                    relative = src.relative_to(root)
                    if src.is_symlink():
                        omitted.append({"source": str(src), "reason": "server symlink"})
                        continue
                    if name == "remaining_cases.jsonl":
                        omitted.append({"source": str(src), "reason": "input text; plan.json records the continuation"})
                        continue
                    if src.suffix not in SUFFIXES and name != ".gitignore":
                        continue
                    if src.stat().st_size > 90 * 1024**2 and src.suffix != ".jsonl":
                        omitted.append({"source": str(src), "bytes": src.stat().st_size, "reason": "large server artifact"})
                        continue
                    target = destination / relative
                    if src.suffix == ".jsonl":
                        target = Path(str(target) + ".gz")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        if src.suffix == ".jsonl":
                            with src.open("rb") as reader, target.open("wb") as raw:
                                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as writer:
                                    shutil.copyfileobj(reader, writer)
                        else:
                            shutil.copy2(src, target)
                    inventory.append({
                        "source": str(src), "repository_path": str(target.relative_to(destination)),
                        "source_bytes": src.stat().st_size, "published_bytes": target.stat().st_size,
                        "format": "gzip (original bytes)" if src.suffix == ".jsonl" else "original",
                    })
    q = Path("/home/gcx/aster-gpu-queue-20261113")
    target = destination / "reports/aster_completion_20261114/queue"
    target.mkdir(parents=True, exist_ok=True)
    for src in sorted(q.iterdir()):
        if src.is_file() and src.suffix in {".json", ".py"} and src.stat().st_size < 2 * 1024**2:
            shutil.copy2(src, target / src.name)
    out = destination / "reports/aster_completion_20261114"
    with (out / "artifact_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)
    dump(out / "server_artifacts.json", {
        "source_worktrees": [str(r) for r, _, _ in SOURCES],
        "excluded_directory_names": sorted(PRUNE), "excluded_directory_prefixes": ["shard"],
        "official_predictions": "Merged completed shard predictions are exported by summarize_results.py.",
        "other_server_artifacts": omitted,
        "original_reports_unchanged": True, "new_model_forward_passes": 0,
    })
    print(json.dumps({"copied_files": len(inventory),
                      "published_MiB": round(sum(x["published_bytes"] for x in inventory) / 1024**2, 2),
                      "server_only_individual_files": len(omitted)}))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=Path.cwd())
    copy_artifacts(parser.parse_args().destination.resolve())
