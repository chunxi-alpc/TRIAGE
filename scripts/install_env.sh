#!/usr/bin/env bash
set -euo pipefail
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/TRIAGE
python -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple "huggingface_hub>=0.34,<1" "datasets>=3.6,<5"
touch results/hub_datasets_ready
python -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
CUDA_VISIBLE_DEVICES=0,1,2,3,6,7 python - <<"PY"
import torch, transformers, datasets, accelerate, peft, huggingface_hub, json, sys
result={"python":sys.version,"torch":torch.__version__,"cuda":torch.version.cuda,"cuda_available":torch.cuda.is_available(),"device_count":torch.cuda.device_count(),"transformers":transformers.__version__,"datasets":datasets.__version__,"accelerate":accelerate.__version__,"peft":peft.__version__,"huggingface_hub":huggingface_hub.__version__,"environment":"/home/gcx/gcx261014","project":"/home/gcx/TRIAGE"}
assert result["cuda_available"], "CUDA is unavailable"
x=torch.ones(1,device="cuda"); assert x.item()==1
open("results/environment.json","w").write(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
PY
python -m pip freeze > configs/environment.lock.txt
touch results/environment_ready
