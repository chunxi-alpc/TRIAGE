import subprocess,zipfile,json,hashlib
from pathlib import Path
p=Path('/home/gcx/TRIAGE-worktrees/aster-seka-execution-20261025/experiments/aster_seka_execution');v=p/'runtime/vendors/nltk_data'
subprocess.run(['git','clone','--depth','1','--filter=blob:none','--no-checkout','--branch','gh-pages','git@github.com:nltk/nltk_data.git',str(v)],check=True)
subprocess.run(['git','checkout','HEAD','--','packages/tokenizers/punkt_tab.zip'],cwd=v,check=True)
f=v/'packages/tokenizers/punkt_tab.zip';out=Path('/home/gcx/nltk_data/tokenizers');out.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(f) as z:z.extractall(out)
(p/'nltk_data_resource_manifest.json').write_text(json.dumps({'repo':'nltk/nltk_data','commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=v,text=True).strip(),'file':'packages/tokenizers/punkt_tab.zip','sha256':hashlib.sha256(f.read_bytes()).hexdigest()},indent=2))
import nltk
print(nltk.sent_tokenize('First. Second.'))
