#!/usr/bin/env bash
set -euo pipefail
source /home/gcx/gcx261014/bin/activate
triage_root=/home/gcx/TRIAGE
baseline="$1"
shift
case "$baseline" in
  ProbeRAG) entry=scripts/evaluate_qa.py ;;
  SEKA) entry=benchmarks/eval_fact_gen.py ;;
  JUICE) entry=src/main_intervene.py ;;
  *) exit 2 ;;
esac
cd "$triage_root/third_party/$baseline"
export PYTHONPATH="$PWD:$PWD/src"
export CUDA_VISIBLE_DEVICES=0
exec python "$entry" "$@"
