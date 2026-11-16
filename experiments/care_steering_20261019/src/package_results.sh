#!/usr/bin/env bash
set -euo pipefail
cd /home/gcx/care_steering_20261019
source /home/gcx/gcx261014/bin/activate
python src/build_report.py
python - <<'PY'
from pathlib import Path
import hashlib,json
root=Path.cwd();files={}
for sub in ["src","data","results","audit","sources"]:
 for p in sorted((root/sub).rglob("*")):
  if not p.is_file() or any(x in p.parts for x in ["__pycache__","official_knowledge_selection_steering","official_raw"]):continue
  if p.suffix==".html" and p.name.startswith("paper_"):continue
  files[str(p.relative_to(root))]={"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
(root/"ARTIFACT_MANIFEST.json").write_text(json.dumps(files,indent=2,ensure_ascii=False))
PY
tar --exclude="__pycache__" --exclude="sources/official_knowledge_selection_steering" --exclude="sources/official_raw" --exclude="sources/paper_2610.20067v1.html" -czf CARE_steering_reproduction_20261019.tar.gz src data results audit sources logs REPRODUCE.txt REPORT.html REPORT.txt ARTIFACT_MANIFEST.json run_facts.sh run_rules.sh run_free.sh run_c_full.sh
sha256sum CARE_steering_reproduction_20261019.tar.gz > CARE_steering_reproduction_20261019.sha256
