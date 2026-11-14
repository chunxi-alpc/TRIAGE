#!/usr/bin/env bash
set -euo pipefail
cd /home/gcx/TRIAGE-worktrees/aster-e0-e8-20261024
source /home/gcx/gcx261014/bin/activate
export CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
run_root=/home/gcx/aster_e0_e8_20261024/private/improvement_seed0_v1
script=experiments/aster_e0_e8_20261024
mkdir -p "$run_root"
exec 9>"$run_root/launch.lock"
flock -n 9 || { echo 'Improvement already running'; exit 1; }
result=0
for model_name in qwen llama; do
 if [[ "$model_name" == qwen ]]; then
  model_path=/home/gcx/TRIAGE/models/modelscope/Qwen3-8B
 else
  model_path=/home/gcx/TRIAGE/models/modelscope/Meta-Llama-3.1-8B-Instruct
 fi
 python -u "$script/improve_branch.py" --model "$model_path" --name "$model_name" --out "$run_root/$model_name" > "$run_root/$model_name.log" 2>&1 || result=1
 python "$script/improvement_report.py" --root "$run_root" --out "$script/results/improvement_seed0_v1"
 git add "$script/results/improvement_seed0_v1"
 if ! git diff --cached --quiet -- "$script/results/improvement_seed0_v1"; then
  git commit -m "Report actual seed-0 ASTER improvement: $model_name" --only -- "$script/results/improvement_seed0_v1"
  git push origin HEAD:codex/aster-e0-e8-20261024
 fi
done
exit "$result"
