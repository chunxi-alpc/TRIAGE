#!/usr/bin/env bash
source /home/gcx/gcx261014/bin/activate
export TRIAGE_ROOT=/home/gcx/TRIAGE
export HF_HOME=/home/gcx/.cache/huggingface
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,6,7}"
cd "$TRIAGE_ROOT"
