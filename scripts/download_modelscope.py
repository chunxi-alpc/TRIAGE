#!/usr/bin/env python3
"""Download original model weights through the official ModelScope SDK.
Public platform mirrors are recorded explicitly, without replacing Hugging Face metadata.
"""
import argparse
import concurrent.futures
import fcntl
import json
import os
import shutil
from pathlib import Path
import time
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/models.json"
MODELS = {
    "qwen3_1_7b": ("Qwen/Qwen3-1.7B", "Qwen/Qwen3-1.7B"),
    "qwen3_8b": ("Qwen/Qwen3-8B", "Qwen/Qwen3-8B"),
    "llama3_1_8b": ("LLM-Research/Meta-Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"),
    "deberta_nli": ("cross-encoder/nli-deberta-v3-base", "cross-encoder/nli-deberta-v3-base"),
    "medgemma_27b": ("google/medgemma-27b-text-it", "google/medgemma-27b-text-it"),
}

def save(key, result):
    with (ROOT / "configs/models.json.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        config = json.loads(CONFIG.read_text()) if CONFIG.exists() else {"models": {}}
        config.setdefault("models", {})[key] = result
        if key in ("llama3_1_8b", "medgemma_27b"):
            config["optional_models"] = [m for m in config.get("optional_models", []) if m.get("repo_id") != result["repo_id"]]
        temporary = CONFIG.with_suffix(".tmp")
        temporary.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(CONFIG)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["llama3_1_8b", "medgemma_27b"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--parallel-workers", type=int, default=8)
    args = parser.parse_args()
    os.environ["MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS"] = str(args.parallel_workers)
    from download_network import install_resolution_cache
    install_resolution_cache()
    from modelscope import snapshot_download
    from modelscope.hub.api import HubApi
    from modelscope_hub.constants import DOWNLOAD_PART_SIZE, DOWNLOAD_PARALLEL_THRESHOLD
    def seed_parts(destination, files):
        # Preserve bytes already received in single-stream mode when enabling SDK ranges.
        for file in files:
            path = destination / file.get("Path", "")
            partial = path.with_suffix(path.suffix + ".incomplete")
            size = file.get("Size", 0)
            if size <= DOWNLOAD_PARALLEL_THRESHOLD or not partial.is_file():
                continue
            with partial.open("rb") as source:
                for start in range(0, partial.stat().st_size, DOWNLOAD_PART_SIZE):
                    end = min(start + DOWNLOAD_PART_SIZE, size) - 1
                    part = Path(f"{path}_{start}_{end}")
                    received = min(end - start + 1, partial.stat().st_size - start)
                    if not part.exists():
                        source.seek(start)
                        with part.open("wb") as target:
                            remaining = received
                            while remaining:
                                chunk = source.read(min(16 * 1024 * 1024, remaining))
                                target.write(chunk)
                                remaining -= len(chunk)
    def download(key):
        model_id, original = MODELS[key]
        destination = ROOT / "models/modelscope" / model_id.split("/")[-1]
        started = time.time()
        result = {"repo_id": original, "modelscope_id": model_id, "source": f"https://modelscope.cn/models/{model_id}", "endpoint": "https://modelscope.cn", "path": str(destination), "task": "nli" if key == "deberta_nli" else "causal_lm", "revision": args.revision, "status": "downloading"}
        save(key, result)
        try:
            files = HubApi().get_model_files(model_id, revision=args.revision, recursive=True)
            manifest = [{k: f.get(k) for k in ("Path", "Name", "Size", "Sha256", "Revision", "Type")} for f in files]
            result["expected_bytes"] = sum(f.get("Size",0) for f in files if str(f.get("Path", "")).endswith(".safetensors"))
            (ROOT / f"configs/{key}_modelscope_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n")
            result["file_revisions"] = sorted({f.get("Revision") for f in files if f.get("Revision")})
            save(key, result)
            seed_parts(destination, files)
            patterns = ["*.safetensors", "*.json", "*.txt", "*.model", "*.tiktoken", "README.md", "*.jinja", "LICENSE*"]
            if not any(str(f.get("Path", "")).endswith(".safetensors") for f in files):
                patterns.append("pytorch_model*.bin")
            snapshot_download(model_id, revision=args.revision, local_dir=str(destination), allow_file_pattern=patterns, max_workers=2)
            result["status"] = "downloaded"
            if result["task"] == "nli":
                result["id2label"] = json.loads((destination / "config.json").read_text()).get("id2label", {})
            result["bytes"] = sum(p.stat().st_size for p in destination.rglob("*") if p.is_file() and not p.name.endswith(".incomplete") and ".cache" not in p.parts)
            result["storage_bytes"] = sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
        except Exception as exc:
            result["status"] = "failed"
            result["reason"] = f"{type(exc).__name__}: {exc}"
        result["download_seconds"] = round(time.time() - started, 1)
        save(key, result)
        print(json.dumps({key: result},ensure_ascii=False), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(download,args.models))
if __name__ == "__main__":
    main()
