#!/usr/bin/env python3
"""Restore exact completed journals from tracked gzip copies; no GPU needed."""
import gzip
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for p in (ROOT/'results').rglob('*.jsonl.gz'):
    target=p.with_suffix('')
    data=gzip.decompress(p.read_bytes())
    if target.exists():
        assert target.read_bytes()==data,f"existing journal differs: {target}"
    else:target.write_bytes(data)
print('Saved journals are ready for analysis.')

