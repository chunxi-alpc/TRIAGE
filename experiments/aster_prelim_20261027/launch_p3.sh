#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source /home/gcx/gcx261014/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
D=experiments/aster_prelim_20261027
CUDA_VISIBLE_DEVICES=0 python -u "$D/run_p3.py" --name qwen --model /home/gcx/TRIAGE/models/modelscope/Qwen3-8B > "$D/private/p3_qwen.log" 2>&1 &
QWEN_PID=$!
CUDA_VISIBLE_DEVICES=1 python -u "$D/run_p3.py" --name llama --model /home/gcx/TRIAGE/models/modelscope/Meta-Llama-3.1-8B-Instruct > "$D/private/p3_llama.log" 2>&1 &
LLAMA_PID=$!
printf '{"qwen":%s,"llama":%s}\n' "$QWEN_PID" "$LLAMA_PID" > "$D/private/p3_pids.json"
FAIL=0
wait "$QWEN_PID" || FAIL=1
wait "$LLAMA_PID" || FAIL=1
python "$D/summarize_p3.py"
exit "$FAIL"
