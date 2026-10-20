#!/usr/bin/env bash
set -u
cd /home/gcx/TRIAGE-worktrees/care-module2-n1-n5-20261020
source /home/gcx/gcx261014/bin/activate
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN_DIR="$1"
python -u -m care_m2_v2.cli run --out "$RUN_DIR" --push > "$RUN_DIR.console.log" 2>&1
run_exit=$?
printf '%s\n' "$run_exit" > "$RUN_DIR.exitcode"
if [ "$run_exit" -ne 0 ]; then
    git add "$RUN_DIR" "$RUN_DIR.console.log" "$RUN_DIR.exitcode"
    git commit -m 'results(care-m2): preserve v2 execution failure and raw journal'
    git push origin HEAD:refs/heads/codex/care-module2-n1-n5-only-20261020
else
    git add "$RUN_DIR.console.log" "$RUN_DIR.exitcode"
    git commit -m 'results(care-m2): preserve completed launcher log and exit code'
    git push origin HEAD:refs/heads/codex/care-module2-n1-n5-only-20261020
fi
exit "$run_exit"
