#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
source /home/gcx/gcx261014/bin/activate
export PYTHONPATH=/home/gcx/care_steering_20261019/.deps${PYTHONPATH:+:$PYTHONPATH}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
E=experiments/medguide_full_20261021
exec 9>"$E/private/run.lock"
flock -n 9 || exit 1
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv > "$E/private/gpu_before.csv"
python -m pip freeze > "$E/private/pip_freeze.txt"
python "$E/prepare.py" || exit 1
worker() {
 local k="$1"
 export CUDA_VISIBLE_DEVICES="$k"
 python experiments/care_module1_semantic_v8/semantic_runner.py --cases "$E/private/input_${k}.jsonl" --arm A1 --output "$E/private/a1_${k}" --batch-size 8 --max-new-tokens 384 > "$E/private/a1_${k}.log" 2>&1 || return 1
 python "$E/qwen_run.py" --shard "$k" > "$E/private/qwen_${k}.log" 2>&1
}
pids=()
for k in 0 1 2 3; do worker "$k" & pids+=("$!"); done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
python "$E/summarize.py" || failed=1
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv > "$E/private/gpu_after.csv"
cp "$E/private/gpu_before.csv" "$E/gpu_before.csv"
cp "$E/private/pip_freeze.txt" "$E/environment.txt"
printf '%s\n' "$failed" > "$E/exit_status.txt"
# Only derived, text-free results are versioned. Raw data and raw outputs stay private.
git add "$E/results" "$E/gpu_before.csv" "$E/environment.txt" "$E/exit_status.txt"
git commit -m "Record MedGUIDE full-run results and complete-denominator coverage"
git push origin HEAD
exit "$failed"
