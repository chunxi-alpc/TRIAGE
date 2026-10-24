"""Fetch the pinned research dataset; run on server, or as a network-only relay."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
REV='5dcfe380f49c6f17f3c6ba5e25e0e958f9642f0e'
p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,default=Path(__file__).resolve().parents[1]/'sources');args=p.parse_args()
args.destination.mkdir(parents=True,exist_ok=True)
manifest={}
for remote,local in [('advice_pairs.jsonl','advice_pairs.jsonl'),('ConflictMed.jsonl','ConflictMed.jsonl'),('ConflictMed_v2.jsonl','ConflictMed_v2.jsonl'),('README.md','DriftMed_README.md')]:
    url=f'https://huggingface.co/datasets/RDBH/DriftMed/resolve/{REV}/{remote}'
    with urllib.request.urlopen(url,timeout=60) as r:body=r.read()
    (args.destination/local).write_bytes(body)
    manifest[local]=dict(url=url,bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
# Revision-specific metadata avoids drifting provenance on a future replay.
with urllib.request.urlopen(f'https://huggingface.co/api/datasets/RDBH/DriftMed/revision/{REV}',timeout=60) as r:
    (args.destination/'driftmed_api.json').write_bytes(r.read())
(args.destination/'download_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
