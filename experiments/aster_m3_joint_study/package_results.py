"""Compress new prediction artifacts for delivery, retaining server originals."""
from __future__ import annotations
import argparse
import gzip
import json
from pathlib import Path
import shutil


def package(root):
    root=Path(root).resolve()
    files=[]
    for source in sorted((root/"evaluation").glob("*.jsonl")):
        target=source.with_suffix(source.suffix+".gz")
        with source.open("rb") as fin, gzip.open(target,"wb") as fout:
            shutil.copyfileobj(fin,fout)
        files.append(dict(original=str(source.relative_to(root)),
                          compressed=str(target.relative_to(root)), bytes=target.stat().st_size))
    for name in ("REPORT.zh.md","README.zh.md"):
        path=root/name
        if path.exists():
            text=path.read_text(encoding="utf-8")
            for row in files:
                text=text.replace("("+row["original"]+")","("+row["compressed"]+")")
                text=text.replace("`"+row["original"]+"`","`"+row["compressed"]+"`")
            path.write_text(text,encoding="utf-8")
    manifest=dict(prediction_files=files, lossless_compression=True, hashes_used=False,
                  server_originals_retained=True,
                  server_projection_caches="models/foldK/projected_cache.npz",
                  source_root=str(root), training_commit=json.loads((root/"status.json").read_text())["code_commit"])
    (root/"delivery_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return manifest


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("output")
    args=p.parse_args()
    package(args.output)
