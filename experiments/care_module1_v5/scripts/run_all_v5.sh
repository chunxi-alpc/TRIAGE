#!/usr/bin/env bash
set -u -o pipefail
ROOT="/home/gcx/TRIAGE-v5-pr4"
EXP="$ROOT/experiments/care_module1_v5"
OUT="$EXP/runs/v5_20261018"
VENV="/home/gcx/n/.venv/bin/python"
V4="/home/gcx/care_module1_20261018_v4/data"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
cd "$ROOT"
mkdir -p "$OUT"
write_failure() {
  local kind="$1" message="$2"
  "$VENV" - "$OUT/run_status.json" "$kind" "$message" <<'PY'
import json,sys,datetime
path,kind,message=sys.argv[1:]
json.dump({"status":"failed","error_type":kind,"error":message,
          "start_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "device":"cuda:6","sample_count":192},open(path,"w"),indent=2)
open(path,"a").write("\n")
PY
  "$VENV" "$EXP/scripts/evaluate.py" --labels "$EXP/data/labels_public.jsonl" \
    --pairs "$EXP/data/pairs_public.jsonl" --predictions "$OUT/predictions_public.jsonl" \
    --qwen-reference "$EXP/data/v4_qwen_direct_public.jsonl" \
    --output "$OUT/metrics.json" --errors "$OUT/errors_public.jsonl" > "$OUT/evaluate.log" 2>&1 || true
  "$VENV" "$EXP/scripts/generate_report.py" --run-status "$OUT/run_status.json" \
    --metrics "$OUT/metrics.json" --errors "$OUT/errors_public.jsonl" --output "$OUT/REPORT.zh.md" > "$OUT/report.log" 2>&1 || true
  exit 2
}
cat > "$OUT/run_status.json" <<'JSON'
{"status":"preparing_data","device":"cuda:6","sample_count":192,"rule_count":16}
JSON
for REQUIRED in "$V4/labels.jsonl" "$V4/model/predictions.jsonl" "$V4/pairs.jsonl" "$V4/rules_manifest.jsonl" \
  "/home/gcx/care_module1_20261018_v4/data/private/inputs.jsonl"; do
  if [[ ! -s "$REQUIRED" ]]; then
    write_failure "MissingExperimentInput" "Required frozen V4 source file is missing: $REQUIRED"
  fi
done
"$VENV" "$EXP/scripts/prepare_public_data.py" \
  --v4-labels "$V4/labels.jsonl" \
  --v4-predictions "$V4/model/predictions.jsonl" \
  --v4-pairs "$V4/pairs.jsonl" \
  --labels-out "$EXP/data/labels_public.jsonl" \
  --qwen-reference-out "$EXP/data/v4_qwen_direct_public.jsonl" \
  --pairs-out "$EXP/data/pairs_public.jsonl" > "$OUT/prepare_public_data.log" 2>&1 || \
  write_failure "PublicDataPreparationFailed" "Frozen labels, pairs, or historical predictions could not be prepared."
"$VENV" "$EXP/scripts/prepare_rules.py" \
  --manifest "$V4/rules_manifest.jsonl" \
  --output "$EXP/data/rules_v5.jsonl" > "$OUT/prepare_rules.log" 2>&1 || \
  write_failure "RulePreparationFailed" "Offline V5 rule preparation failed."
cat > "$OUT/run_status.json" <<'JSON'
{"status":"preparing_models","device":"cuda:6","sample_count":192,"rule_count":16}
JSON
if ! "$VENV" "$EXP/scripts/validate_model_configs.py" \
  --base /home/gcx/models/care_module1_v5/phi-4 \
  --adapter /home/gcx/models/care_module1_v5/trialmatchai-phi4-reasoning-lora \
  --evidence /home/gcx/models/care_module1_v5/biolinkbert-mednli \
  --output "$EXP/data/model_validation.json" > "$OUT/model_precheck.log" 2>&1; then
  bash "$EXP/scripts/download_models.sh" > "$OUT/download.log" 2>&1 || write_failure "ModelDownloadFailed" "Model download failed; details are in download.log."
fi
"$VENV" "$EXP/scripts/validate_model_configs.py" \
  --base /home/gcx/models/care_module1_v5/phi-4 \
  --adapter /home/gcx/models/care_module1_v5/trialmatchai-phi4-reasoning-lora \
  --evidence /home/gcx/models/care_module1_v5/biolinkbert-mednli \
  --output "$EXP/data/model_validation.json" > "$OUT/model_validation.log" 2>&1 || write_failure "ModelIdentityValidationFailed" "Requested model identity/config check failed; GPU inference was not started."
"$VENV" -m pytest "$EXP/tests" -q > "$OUT/tests.log" 2>&1 || write_failure "ServerTestsFailed" "V5 tests failed before inference; details are in tests.log."
"$VENV" "$EXP/scripts/run_experiment.py" --device cuda:6 --output "$OUT" \
  --base-model /home/gcx/models/care_module1_v5/phi-4 \
  --adapter-model /home/gcx/models/care_module1_v5/trialmatchai-phi4-reasoning-lora \
  --evidence-model /home/gcx/models/care_module1_v5/biolinkbert-mednli > "$OUT/run.log" 2>&1
RUN_RC=$?
"$VENV" "$EXP/scripts/evaluate.py" --labels "$EXP/data/labels_public.jsonl" \
  --pairs "$EXP/data/pairs_public.jsonl" --predictions "$OUT/predictions_public.jsonl" \
  --qwen-reference "$EXP/data/v4_qwen_direct_public.jsonl" \
  --output "$OUT/metrics.json" --errors "$OUT/errors_public.jsonl" > "$OUT/evaluate.log" 2>&1 || true
"$VENV" "$EXP/scripts/generate_report.py" --run-status "$OUT/run_status.json" \
  --metrics "$OUT/metrics.json" --errors "$OUT/errors_public.jsonl" --output "$OUT/REPORT.zh.md" > "$OUT/report.log" 2>&1 || true
exit "$RUN_RC"
