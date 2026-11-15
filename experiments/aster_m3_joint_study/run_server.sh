#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source /home/gcx/gcx261014/bin/activate
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
output=reports/aster_m3_joint_study_20261115
mkdir -p "$output"
python - <<'PY' > "$output/tests.log" 2>&1
import json, unittest, pathlib, time
start=time.monotonic()
suite=unittest.defaultTestLoader.loadTestsFromNames([
    'experiments.aster_m3_joint_study.test_training',
    'experiments.aster_m3_joint_study.test_evaluation'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
pathlib.Path('reports/aster_m3_joint_study_20261115/tests.json').write_text(json.dumps({
    'tests_run':result.testsRun, 'failures':len(result.failures), 'errors':len(result.errors),
    'successful':result.wasSuccessful(), 'seconds':time.monotonic()-start,
    'environment':'gcx261014 on 10.29.128.115', 'synthetic':True,
    'real_archive_replay':'verification.json produced by full run'},indent=2)+'\n')
raise SystemExit(not result.wasSuccessful())
PY
python -u -m experiments.aster_m3_joint_study.run run \
  --e1-root /home/gcx/aster-cr-retrain-paired-v2-20261111/reports/aster_cr_retrain_paired_v2_20261111 \
  --cases-file /home/gcx/aster_cr_repair_v2_run_20261108/data/path_cases.jsonl \
  --diagnostic-root /home/gcx/aster-cr-diagnostics-e3cv-20261111/reports/aster_cr_diagnostics_e3cv_20261111 \
  --output "$output" > "$output/run.log" 2>&1
