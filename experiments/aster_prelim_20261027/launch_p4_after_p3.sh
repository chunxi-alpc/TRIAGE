#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source /home/gcx/gcx261014/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
D=experiments/aster_prelim_20261027
for EXP_PID in $(python -c "import json;print(*json.load(open('$D/private/p3_pids.json')).values())"); do
 if kill -0 "$EXP_PID" 2>/dev/null; then tail --pid="$EXP_PID" -f /dev/null; fi
done
python "$D/summarize_p3.py"
python -c "import json;assert json.load(open('$D/results/p3/gate_status.json'))['P3']=='COMPLETED'"
CUDA_VISIBLE_DEVICES=0 python -u "$D/run_p4.py" --name qwen --model /home/gcx/TRIAGE/models/modelscope/Qwen3-8B > "$D/private/p4_qwen.log" 2>&1 &
QWEN_PID=$!
CUDA_VISIBLE_DEVICES=1 python -u "$D/run_p4.py" --name llama --model /home/gcx/TRIAGE/models/modelscope/Meta-Llama-3.1-8B-Instruct > "$D/private/p4_llama.log" 2>&1 &
LLAMA_PID=$!
printf '{"qwen":%s,"llama":%s}\n' "$QWEN_PID" "$LLAMA_PID" > "$D/private/p4_pids.json"
FAIL=0
wait "$QWEN_PID" || FAIL=1
wait "$LLAMA_PID" || FAIL=1
python "$D/summarize_p4.py"
exit "$FAIL"
