#!/usr/bin/env bash
set -euo pipefail
cd /home/gcx/TRIAGE-worktrees/causal-attention-20261019
source /home/gcx/gcx261014/bin/activate
export CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python -u experiments/causal_attention_20261019/run_key_edit.py \
  --config experiments/causal_attention_20261019/configs/group3_fixed.json \
  --output experiments/causal_attention_20261019/results/group3_key_methods \
  --baseline-dev experiments/causal_attention_20261019/results/group1_gpu0/baseline.jsonl \
  --baseline-confirmation experiments/causal_attention_20261019/results/group1_confirmation_main/baseline.jsonl \
  --direct-dev experiments/causal_attention_20261019/results/group1_gpu1/C_L18-23_s1_q1.jsonl \
  --direct-confirmation experiments/causal_attention_20261019/results/group1_confirmation_main/C_L18-23_s1_q1.jsonl \
  --trigger-evidence 'Group1 actual Qwen dev: layers18-23 C lambda1 final-query changed attention and predictions; 35 to 37 of96, corrected3 damaged1, primary0of12 unchanged. Weak exploratory signal.'

