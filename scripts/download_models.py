#!/usr/bin/env python3
"""Download public TRIAGE checkpoints through the official Hugging Face Hub API.
Existing local/cache files are reused; rerun this script to resume downloads.
"""
import argparse
import fcntl
import socket
import concurrent.futures
import json
import os
from pathlib import Path
import time

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "20")
ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    ("qwen3_1_7b", "Qwen/Qwen3-1.7B", "causal_lm"),
    ("qwen3_8b", "Qwen/Qwen3-8B", "causal_lm"),
    ("medcpt", "ncbi/MedCPT-Cross-Encoder", "relevance"),
    ("deberta_nli", "cross-encoder/nli-deberta-v3-base", "nli"),
    ("medical_nli", "pritamdeka/PubMedBERT-MNLI-MedNLI", "nli"),
]
CONFIG = ROOT / "configs/models.json"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", choices=[m[0] for m in MODELS])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    args = parser.parse_args()
    os.environ["HF_ENDPOINT"] = args.endpoint
    from huggingface_hub import HfApi, snapshot_download, configure_http_backend
    import requests
    class FreshResolveSession(requests.Session):
        def request(self, method, url, *a, **kw):
            if url.startswith(args.endpoint) and "/resolve/" in url:
                url += ("&" if "?" in url else "?") + f"triage_refresh={time.time_ns()}"
            return super().request(method, url, *a, **kw)
    configure_http_backend(backend_factory=FreshResolveSession)
    from download_network import install_resolution_cache
    install_resolution_cache()
    api = HfApi()
    existing = json.loads(CONFIG.read_text()) if CONFIG.exists() else {}
    existing.setdefault("models", {})
    existing["endpoint"] = args.endpoint
    existing["optional_models"] = [
        {"repo_id": "meta-llama/Llama-3.1-8B-Instruct", "status": "skipped", "reason": "Gated model; optional and no authorization supplied."},
        {"repo_id": "google/medgemma-27b-text-it", "status": "skipped", "reason": "Gated model; optional and no authorization supplied."},
    ]
    def download(item):
        key, repo, task = item
        destination = ROOT / "models" / repo.split("/")[-1]
        result = {"repo_id": repo, "path": str(destination), "task": task, "source": f"https://huggingface.co/{repo}", "endpoint": args.endpoint}
        started = time.time()
        try:
            info = api.model_info(repo, files_metadata=True)
            result["revision"] = info.sha
            names = [s.rfilename for s in info.siblings]
            has_safe = any(n.endswith(".safetensors") for n in names)
            patterns = ["*.json", "*.txt", "*.model", "*.tiktoken", "README.md", "*.jinja", "LICENSE*"]
            patterns += ["*.safetensors"] if has_safe else ["pytorch_model*.bin"]
            snapshot_download(repo_id=repo, revision=info.sha, local_dir=str(destination), allow_patterns=patterns, max_workers=4)
            result["status"] = "downloaded"
            result["bytes"] = sum(p.stat().st_size for p in destination.rglob("*") if p.is_file() and ".cache" not in p.parts)
            if task == "nli":
                result["id2label"] = json.loads((destination / "config.json").read_text()).get("id2label", {})
        except Exception as exc:
            result["status"] = "failed"
            result["reason"] = f"{type(exc).__name__}: {exc}"
        result["download_seconds"] = round(time.time() - started, 1)
        return key, result
    selected = [m for m in MODELS if not args.models or m[0] in args.models]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download, item) for item in selected]
        for future in concurrent.futures.as_completed(futures):
            key, result = future.result()
            with (ROOT / "configs/models.json.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                current = json.loads(CONFIG.read_text()) if CONFIG.exists() else existing
                current.setdefault("models", {})[key] = result
                current["endpoint"] = args.endpoint
                temporary = CONFIG.with_suffix(".tmp")
                temporary.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n")
                temporary.replace(CONFIG)
            print(json.dumps({key: result}, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
