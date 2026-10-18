#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
HF="${HF_CLI:-/home/gcx/n/.venv/bin/hf}"
ROOT="${MODEL_DIR:-/home/gcx/models/care_module1_v5}"
"$HF" download microsoft/phi-4 --local-dir "$ROOT/phi-4"
"$HF" download majdabd33/trialmatchai-phi4-reasoning-lora --local-dir "$ROOT/trialmatchai-phi4-reasoning-lora"
"$HF" download cnut1648/biolinkbert-mednli --local-dir "$ROOT/biolinkbert-mednli"
