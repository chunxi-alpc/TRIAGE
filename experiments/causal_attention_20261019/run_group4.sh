#!/usr/bin/env bash
set -euo pipefail
cd /home/gcx/TRIAGE-worktrees/causal-attention-20261019
source /home/gcx/gcx261014/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ROOT=experiments/causal_attention_20261019
CUDA_VISIBLE_DEVICES=0 python "$ROOT/run_attention.py" --split confirmation --configs "$ROOT/configs/group4.json" --structures "$ROOT/structure/results/original_v7_ungated.jsonl" --source original_v7_ungated --baseline "$ROOT/results/group1_confirmation_main/baseline.jsonl" --output "$ROOT/results/group4_original_v7_ungated" > "$ROOT/logs/group4_original_v7_ungated.log" 2>&1 &
old_pid=$!
CUDA_VISIBLE_DEVICES=1 python "$ROOT/run_attention.py" --split confirmation --configs "$ROOT/configs/group4.json" --structures "$ROOT/structure/results/fixed.jsonl" --source fixed --baseline "$ROOT/results/group1_confirmation_main/baseline.jsonl" --output "$ROOT/results/group4_fixed" > "$ROOT/logs/group4_fixed.log" 2>&1 &
fixed_pid=$!
wait "$old_pid"
wait "$fixed_pid"
