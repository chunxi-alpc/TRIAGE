#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source /home/gcx/gcx261014/bin/activate
D=experiments/aster_prelim_20261027
for name in qwen llama; do
 pid=$(cat "$D/private/p4_fp32_${name}.pid")
 if kill -0 "$pid" 2>/dev/null; then tail --pid="$pid" -f /dev/null; fi
done
python "$D/summarize_p4.py"
python "$D/finalize.py" --publish
