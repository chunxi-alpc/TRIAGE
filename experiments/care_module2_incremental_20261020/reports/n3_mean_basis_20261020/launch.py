import json,os,subprocess,sys,time
from pathlib import Path
report=Path(__file__).parent;out=sys.argv[1];started=time.time()
with (report/'gpu.log').open('w') as log:
    result=subprocess.run([sys.executable,'-m','care_m2_v2.n3_mean_basis','run','--out',out],stdout=log,stderr=subprocess.STDOUT)
(report/'process.json').write_text(json.dumps({'exit_code':result.returncode,'elapsed_seconds':time.time()-started,
    'out':out,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')},indent=2)+'\n')
sys.exit(result.returncode)
