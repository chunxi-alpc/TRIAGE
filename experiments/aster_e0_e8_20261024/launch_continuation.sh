#!/usr/bin/env bash
set -euo pipefail
cd /home/gcx/TRIAGE-worktrees/aster-e0-e8-20261024
source /home/gcx/gcx261014/bin/activate
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
run_root=/home/gcx/aster_e0_e8_20261024/private/continuation_v2
script=experiments/aster_e0_e8_20261024
mkdir -p "$run_root"
exec 9>"$run_root/launch.lock"
flock -n 9 || { echo 'Continuation already running'; exit 1; }
CUDA_VISIBLE_DEVICES=6 python -u "$script/continue_run.py" --role evidence --model /home/gcx/TRIAGE/models/modelscope/Qwen3-1.7B --out "$run_root/evidence" > "$run_root/evidence.log" 2>&1 &
evidence_pid=$!
CUDA_VISIBLE_DEVICES=0 python -u "$script/continue_run.py" --model /home/gcx/TRIAGE/models/modelscope/Qwen3-8B --out "$run_root/qwen" --evidence "$run_root/evidence" > "$run_root/qwen.log" 2>&1 &
qwen_pid=$!
CUDA_VISIBLE_DEVICES=1 python -u "$script/continue_run.py" --model /home/gcx/TRIAGE/models/modelscope/Meta-Llama-3.1-8B-Instruct --out "$run_root/llama" --evidence "$run_root/evidence" > "$run_root/llama.log" 2>&1 &
llama_pid=$!
printf '{"evidence":%s,"qwen":%s,"llama":%s}\n' "$evidence_pid" "$qwen_pid" "$llama_pid" > "$run_root/pids.json"
result=0
wait "$qwen_pid" || result=1
wait "$llama_pid" || result=1
wait "$evidence_pid" || result=1
python "$script/continuation_report.py" --root "$run_root" --out "$script/results/continuation_v2"
git add "$script/results/continuation_v2"
if ! git diff --cached --quiet; then
 git commit -m 'Report actual ASTER continuation results and paired tree intervals'
 git push origin HEAD:codex/aster-e0-e8-20261024
fi
exit "$result"
