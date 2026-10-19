#!/usr/bin/env bash
# Run on gcx@10.29.128.115. Existing completed predictions are resumed.
set -eo pipefail
source /home/gcx/gcx261014/bin/activate
set -u
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
python_bin=/home/gcx/n/.venv/bin/python
experiment=experiments/care_module1_v8_1
historical=experiments/care_module1_semantic_v8

run_arm() {
    local gpu="$1" split="$2" arm="$3" cases="$4"
    shift 4
    local output="$experiment/results/${split}_${arm}"
    mkdir -p "$output"
    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" "$experiment/runner.py" \
        --arm "$arm" --cases "$cases" --output "$output" \
        --batch-size 8 --max-new-tokens 384 "$@" >> "$output/shell.log" 2>&1
}
wait_pair() {
    local failed=0
    wait "$1" || failed=1
    wait "$2" || failed=1
    return "$failed"
}

# A1 historical predictions are reused directly by the analyzer and B2.
run_arm 0 dev B1 "$historical/data/semantic_dev_inputs.jsonl" & first=$!
run_arm 1 dev B2 "$historical/data/semantic_dev_inputs.jsonl" \
    --first-pass-cache "$historical/results/dev_A1/predictions.jsonl" & second=$!
wait_pair "$first" "$second"
"$python_bin" "$experiment/analyze.py" --select

# Selection is frozen before comparison datasets are run.
for split in confirmation new72; do
    if [[ "$split" == confirmation ]]; then
        cases="$historical/data/semantic_confirmation_inputs.jsonl"
    else
        cases="$historical/data/new72_cases.jsonl"
    fi
    run_arm 0 "$split" B1 "$cases" & first=$!
    run_arm 1 "$split" B2 "$cases" \
        --first-pass-cache "$historical/results/${split}_A1/predictions.jsonl" & second=$!
    wait_pair "$first" "$second"
done

# Natural-language cases need a new A1 first pass; reuse it for B2.
run_arm 2 natural72 A1 "$experiment/data/natural72_cases.jsonl" & first=$!
run_arm 0 natural72 B1 "$experiment/data/natural72_cases.jsonl" & second=$!
wait_pair "$first" "$second"
run_arm 1 natural72 B2 "$experiment/data/natural72_cases.jsonl" \
    --first-pass-cache "$experiment/results/natural72_A1/predictions.jsonl"
"$python_bin" "$experiment/analyze.py"
