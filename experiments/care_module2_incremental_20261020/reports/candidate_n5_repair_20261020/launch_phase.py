import json, os, subprocess, sys, time
from pathlib import Path
phase, out = sys.argv[1:]
report = Path(__file__).parent
args = [sys.executable, '-m', 'care_m2_v2.repaired_run', phase,
        '--data', 'experiments/care_module2_incremental_20261020/data_v2_ready',
        '--previous', 'experiments/care_module2_incremental_20261020/runs/v2_20261020T134304Z',
        '--model', '/home/gcx/care_20261016/CARE_Codex_Pilot/models/Qwen3-4B-Base',
        '--reuse', str(report/'offline_verified'), '--out', out]
start = time.time()
with (report/(phase+'_gpu.log')).open('w') as log:
    result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT)
(report/(phase+'_process.json')).write_text(json.dumps({'exit_code': result.returncode, 'elapsed_seconds': time.time()-start,
    'phase': phase, 'out': out, 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES')}, indent=2)+'\n')
sys.exit(result.returncode)
