"""Download only the pinned CARE V5 adapter and MedNLI on the GPU server.

The loopback HTTP proxy resolves HF metadata and signed redirects. Weight bytes
come directly from HF's signed CDN, with resumable ranges and final SHA256 checks.
Signed URLs are never saved or logged. No model loading or GPU work occurs here.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import math
import os
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

MODELS = (
    {"repo": "majdabd33/trialmatchai-phi4-reasoning-lora", "revision": "9eaddaf048c8d4266a291884ff89db1cf05b07fc",
     "directory": "trialmatchai-phi4-reasoning-lora", "workers": 1,
     "files": ("README.md", "adapter_config.json", "adapter_model.safetensors", "merges.txt",
               "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json")},
    {"repo": "cnut1648/biolinkbert-mednli", "revision": "f9f4a9d1e046a0e6574feef938f1308cecaebbd5",
     "directory": "biolinkbert-mednli", "workers": 3,
     "files": ("README.md", "config.json", "pytorch_model.bin", "special_tokens_map.json",
               "tokenizer.json", "tokenizer_config.json", "vocab.txt", "mednli.py")},
)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def digest(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while data := handle.read(4 * 1024 * 1024):
            result.update(data)
    return result.hexdigest()


def safe_error(exc):
    # Exception strings can contain signed redirect URLs, so never emit them.
    return f"{type(exc).__name__}:HTTP{exc.code}" if isinstance(exc, urllib.error.HTTPError) else type(exc).__name__


def request_json(opener, url):
    for attempt in range(5):
        try:
            with opener.open(url, timeout=45) as response:
                return json.load(response)
        except Exception as exc:
            logging.warning("metadata retry=%d reason=%s", attempt + 1, safe_error(exc))
            if attempt == 4:
                raise RuntimeError("HF metadata could not be read") from None
            time.sleep(2 + 2 * attempt)


def validate(path, metadata):
    if not path.is_file() or path.stat().st_size != metadata["size"]:
        return None
    sha256 = digest(path)
    if "lfs" in metadata:
        return sha256 if sha256 == metadata["lfs"]["oid"] else None
    data = path.read_bytes()
    blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    return sha256 if blob == metadata["oid"] else None


def signed_location(redirect_opener, url):
    try:
        response = redirect_opener.open(urllib.request.Request(url, method="HEAD"), timeout=45)
    except urllib.error.HTTPError as exc:
        if exc.code not in (301, 302, 303, 307, 308):
            raise
        response = exc
    try:
        location = response.headers.get("Location")
        if not location:
            raise RuntimeError("HF weight resolve did not provide signed CDN redirect")
        return urllib.parse.urljoin(url, location)
    finally:
        response.close()


def download_weight(root, metadata, resolve_url, redirect_opener, direct_opener, workers):
    destination = root / metadata["path"]
    parts = root / ("." + metadata["path"] + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    total = metadata["size"]
    chunk_size = 32 * 1024 * 1024
    count = math.ceil(total / chunk_size)

    def get_part(index):
        begin, finish = index * chunk_size, min(total, (index + 1) * chunk_size)
        part = parts / f"{index:05d}.part"
        expected = finish - begin
        if part.exists() and part.stat().st_size > expected:
            raise RuntimeError("oversized partial range")
        for attempt in range(8):
            current = part.stat().st_size if part.exists() else 0
            if current == expected:
                return
            try:
                location = signed_location(redirect_opener, resolve_url)
                request = urllib.request.Request(location, headers={"Range": f"bytes={begin + current}-{finish - 1}"})
                with direct_opener.open(request, timeout=45) as response:
                    if response.status != 206:
                        raise RuntimeError("CDN ignored byte range")
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {begin + current}-"):
                        raise RuntimeError("CDN returned different byte range")
                    with part.open("ab") as handle:
                        remaining = expected - current
                        while remaining:
                            data = response.read(min(1024 * 1024, remaining))
                            if not data:
                                raise EOFError("incomplete CDN range")
                            handle.write(data)
                            remaining -= len(data)
                logging.info("range complete file=%s chunk=%d/%d bytes=%d", metadata["path"], index + 1, count, expected)
                return
            except Exception as exc:
                logging.warning("range retry file=%s chunk=%d attempt=%d reason=%s", metadata["path"], index + 1, attempt + 1, safe_error(exc))
                if attempt == 7:
                    raise RuntimeError(f"CDN range download exhausted retries for {metadata['path']} chunk {index + 1}") from None
                time.sleep(min(10, 1 + attempt))

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(get_part, range(count)))
    pending = destination.with_name(destination.name + ".assembling")
    with pending.open("wb") as handle:
        for index in range(count):
            with (parts / f"{index:05d}.part").open("rb") as source:
                while data := source.read(4 * 1024 * 1024):
                    handle.write(data)
    sha256 = validate(pending, metadata)
    if sha256 is None:
        raise RuntimeError(f"SHA256 or size verification failed for {metadata['path']}")
    pending.replace(destination)
    for part in parts.glob("*.part"):
        part.unlink()
    parts.rmdir()
    return sha256


def download_model(spec, args):
    proxy_handler = urllib.request.ProxyHandler({"http": args.proxy, "https": args.proxy})
    proxy = urllib.request.build_opener(proxy_handler)
    redirects = urllib.request.build_opener(proxy_handler, NoRedirect)
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    metadata_url = f"https://huggingface.co/api/models/{spec['repo']}/tree/{spec['revision']}?recursive=true&expand=false"
    entries = {row["path"]: row for row in request_json(proxy, metadata_url) if row.get("type") == "file"}
    root = pathlib.Path(args.root) / spec["directory"]
    root.mkdir(parents=True, exist_ok=True)
    result = {"repo": spec["repo"], "revision": spec["revision"], "directory": str(root), "files": []}
    for name in spec["files"]:
        metadata = entries[name]
        path = root / name
        file_start = time.perf_counter()
        sha256 = validate(path, metadata)
        reused = sha256 is not None
        if not reused:
            resolve = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{name}"
            if "lfs" in metadata:
                sha256 = download_weight(root, metadata, resolve, redirects, direct, spec["workers"])
            else:
                pending = path.with_name(path.name + ".partial")
                for attempt in range(5):
                    try:
                        with proxy.open(resolve, timeout=45) as response:
                            pending.write_bytes(response.read())
                        sha256 = validate(pending, metadata)
                        if sha256 is None:
                            raise RuntimeError("small file hash/size mismatch")
                        pending.replace(path)
                        break
                    except Exception as exc:
                        logging.warning("file retry file=%s attempt=%d reason=%s", name, attempt + 1, safe_error(exc))
                        if attempt == 4:
                            raise RuntimeError(f"small file download failed for {name}") from None
                        time.sleep(2 + attempt)
        record = {"path": name, "size": path.stat().st_size, "sha256": sha256,
                  "git_blob_sha1": metadata["oid"], "reused": reused,
                  "elapsed_seconds": round(time.perf_counter() - file_start, 3)}
        if "lfs" in metadata:
            record["expected_lfs_sha256"] = metadata["lfs"]["oid"]
        result["files"].append(record)
        logging.info("verified repo=%s file=%s bytes=%d sha256=%s reused=%s", spec["repo"], name, record["size"], sha256, reused)
        (root / "download_receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    result["status"] = "verified"
    (root / "download_receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/gcx/models/care_module1_v5")
    parser.add_argument("--proxy", default="http://127.0.0.1:20819")
    parser.add_argument("--log-dir", default="/home/gcx/care_module1_v5")
    args = parser.parse_args()
    logs = pathlib.Path(args.log_dir)
    logs.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=logs / "download_small_models.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    start = time.perf_counter()
    status = {"status": "running", "start_utc": datetime.now(timezone.utc).isoformat(), "models": []}
    status_path = logs / "download_small_models_status.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(download_model, spec, args) for spec in MODELS]
            for future in concurrent.futures.as_completed(futures):
                status["models"].append(future.result())
                status_path.write_text(json.dumps(status, indent=2) + "\n")
        status["status"] = "verified"
    except Exception as exc:
        status["status"] = "failed"
        status["error"] = safe_error(exc)
        logging.error("download failed reason=%s", safe_error(exc))
    finally:
        status["elapsed_seconds"] = round(time.perf_counter() - start, 3)
        status["end_utc"] = datetime.now(timezone.utc).isoformat()
        status_path.write_text(json.dumps(status, indent=2) + "\n")
    raise SystemExit(0 if status["status"] == "verified" else 2)


if __name__ == "__main__":
    main()
