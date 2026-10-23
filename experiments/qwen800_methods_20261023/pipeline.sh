#!/usr/bin/env bash
set -euo pipefail
cd /home/gcx/TRIAGE-worktrees/qwen800-methods
source /home/gcx/gcx261014/bin/activate
EXP=experiments/qwen800_methods_20261023
PRIVATE=/home/gcx/qwen800_methods_20261023/private
exec 9>"$PRIVATE/pipeline.lock"
flock -n 9 || exit 1
STAGE=inference
publish() {
 git add "$EXP/results"
 if ! git diff --cached --quiet; then git commit -m "$1"; fi
 git push origin HEAD:codex/qwen800-methods-20261023
}
failed() {
 CODE=$?
 trap - ERR
 python - "$STAGE" "$CODE" <<'PY'
import json,sys,datetime
from pathlib import Path
Path('experiments/qwen800_methods_20261023/results/STATUS.json').write_text(json.dumps({'status':'failed','stage':sys.argv[1],'exit_code':int(sys.argv[2]),'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'logs':'/home/gcx/qwen800_methods_20261023/private'},indent=2))
PY
 publish "Record Qwen800 pipeline failure status" || true
 exit "$CODE"
}
trap failed ERR
CUDA_VISIBLE_DEVICES=0 python "$EXP/run.py" --input "$PRIVATE/shard0.jsonl" --output "$PRIVATE/predictions0.jsonl" > "$PRIVATE/gpu0.log" 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 python "$EXP/run.py" --input "$PRIVATE/shard1.jsonl" --output "$PRIVATE/predictions1.jsonl" > "$PRIVATE/gpu1.log" 2>&1 &
P1=$!
python - "$P0" "$P1" <<'PY'
import json,sys,datetime
from pathlib import Path
Path('experiments/qwen800_methods_20261023/results/STATUS.json').write_text(json.dumps({'status':'running','gpu0_pid':int(sys.argv[1]),'gpu1_pid':int(sys.argv[2]),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'expected_cases':800,'expected_rows':52496,'automatic_final_analysis_and_push':True},indent=2))
PY
S0=0; S1=0
wait "$P0" || S0=$?
wait "$P1" || S1=$?
if ((S0!=0 || S1!=0)); then false; fi
STAGE=analysis
cp "$PRIVATE/predictions0.manifest.json" "$EXP/results/"
cp "$PRIVATE/predictions1.manifest.json" "$EXP/results/"
python "$EXP/analyze.py" > "$PRIVATE/analysis.log" 2>&1
STAGE=publish
publish "Report complete Qwen800 A-B-C-D-CS method comparisons and K-U results"
trap - ERR
