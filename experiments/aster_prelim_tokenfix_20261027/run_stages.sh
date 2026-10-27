#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source /home/gcx/gcx261014/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
D=experiments/aster_prelim_tokenfix_20261027
exec 9>"$D/private/pipeline.lock"
flock -n 9 || exit 0
for stage in 1 2 3 4 5; do
 if [[ "$stage" == 1 || "$stage" == 2 ]]; then python "$D/import_qwen_cache.py" --stage "$stage"; fi
 printf '%s\n' "$stage" > "$D/private/current_stage"
 if [[ "$stage" == 1 ]]; then
  SUMMARY=summarize.py;GATE="$D/results/gate_status.json";EXPECTED=PASS_SW_GATE
 else
  SUMMARY="summarize_p${stage}.py";GATE="$D/results/p${stage}/gate_status.json"
  case "$stage" in 2) EXPECTED=PASS_NEUTRAL_GATE;;3) EXPECTED=COMPLETED;;4) EXPECTED=PASS;;5) EXPECTED=PASS_PRUNE_GATE;;esac
 fi
 if [[ ! -f "$GATE" ]]; then
  PIDS=()
  for pair in 'qwen:0:Qwen3-8B' 'llama:1:Meta-Llama-3.1-8B-Instruct'; do
   IFS=: read -r NAME CARD MODEL <<< "$pair"
   PREFIX="p${stage}_${NAME}";[[ "$stage" == 1 ]] && PREFIX="$NAME"
   if [[ -f "$D/private/$PREFIX/DONE.json" ]];then continue;fi
   if [[ -d "$D/private/$PREFIX" ]]; then
    printf '{"stage":"P%s","reason":"existing incomplete directory; inspect before resume"}\n' "$stage" > "$D/private/runner_failure.json"
    python "$D/publish.py" --push
    exit 1
   fi
   CUDA_VISIBLE_DEVICES="$CARD" python -u "$D/run_p${stage}.py" --name "$NAME" --model "/home/gcx/TRIAGE/models/modelscope/$MODEL" > "$D/private/$PREFIX.log" 2>&1 &
   PIDS+=("$!")
  done
  printf '%s\n' "${PIDS[@]}" > "$D/private/p${stage}_pids"
  FAIL=0
  for PID in "${PIDS[@]}";do wait "$PID" || FAIL=1;done
  python "$D/$SUMMARY"
  if [[ "$FAIL" != 0 ]]; then
   printf '{"stage":"P%s","reason":"worker exited nonzero; raw logs retained"}\n' "$stage" > "$D/private/runner_failure.json"
   python "$D/publish.py" --push
   exit 1
  fi
 fi
 python "$D/publish.py" --push
 if ! python -c "import json;assert json.load(open('$GATE'))['P$stage']=='$EXPECTED'";then exit 0;fi
done
python "$D/publish.py" --push
