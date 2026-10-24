#!/usr/bin/env bash
set -euo pipefail
cd /home/gcx/TRIAGE-worktrees/aster-p0-p3-20261025
source /home/gcx/gcx261014/bin/activate
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
run_dir=/home/gcx/aster_p0_p3_20261025
exp_dir=experiments/aster_p0_p3_20261025
exec 9>"$run_dir/test.lock"
flock -n 9 || exit 1
test -s "$exp_dir/results/frozen.json"
CUDA_VISIBLE_DEVICES=0 python -u "$exp_dir/infer.py" --name qwen --phase test > "$run_dir/test_qwen.log" 2>&1 &
p0=$!
CUDA_VISIBLE_DEVICES=1 python -u "$exp_dir/infer.py" --name llama --phase test > "$run_dir/test_llama.log" 2>&1 &
p1=$!
printf '{"qwen":%s,"llama":%s}\n' "$p0" "$p1" > "$run_dir/test_pids.json"
result=0
wait "$p0" || result=1
wait "$p1" || result=1
python "$exp_dir/analysis.py" --test > "$run_dir/test_analysis.log" 2>&1 || result=1
python "$exp_dir/human_audit.py" > "$run_dir/human_audit.log" 2>&1 || result=1
python "$exp_dir/report.py" > "$run_dir/final_report.log" 2>&1 || result=1
printf '{"exit_code":%s}\n' "$result" > "$exp_dir/results/execution_exit.json"
git add "$exp_dir/results"
if ! git diff --cached --quiet; then
 git commit -m 'Report actual ASTER P3 inference coverage, risk and clustered comparisons'
 git push origin HEAD:codex/aster-p0-p3-20261025
fi
exit "$result"
