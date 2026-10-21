#!/usr/bin/env python3
"""Verify the existing Phi-4/LoRA bytes against the recorded model fingerprint."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
expected=json.loads((HERE/"model_fingerprints.json").read_text())
for model in expected["models"].values():
    for name, meta in model["files"].items():
        path=Path(model["path"])/name
        h=hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda:f.read(8*1024*1024),b""):
                h.update(block)
        assert path.stat().st_size == meta["bytes"] and h.hexdigest() == meta["sha256"], str(path)
print("Phi-4 and TrialMatchAI file fingerprints verified")
