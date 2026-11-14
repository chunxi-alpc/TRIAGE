import json,os,time
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[2]
OUT=REPO/'reports/aster_e9_e11_20261112'
E1=Path('/home/gcx/aster-cr-retrain-paired-v2-20261111/reports/aster_cr_retrain_paired_v2_20261111')
MODEL=E1/'models/E1/full'
HIST=Path('/home/gcx/aster_cr_repair_v2_run_20261108/qwen')
def read(p):return json.loads(Path(p).read_text())
def rows(p):
 with Path(p).open() as f:return [json.loads(s) for s in f if s.strip()]
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def append(p,x):
 with Path(p).open('a') as f:f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n')
def linefile(p,rr):
 Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr))
def inputs():
 from experiments.aster_cr_retrain_paired_v2_20261111.common import load_inputs
 return load_inputs(E1)
class Readout:
 def __init__(self,device):
  import torch
  m2=np.load(OUT/'checkpoints/e1_m2.npz' if (OUT/'checkpoints/e1_m2.npz').exists() else MODEL/'m2.npz')
  m3=np.load(OUT/'checkpoints/e1_m3.npz' if (OUT/'checkpoints/e1_m3.npz').exists() else MODEL/'m3.npz')
  self.heads=m2['heads'].tolist()
  self.mu=torch.tensor(m2['mu'],device=device,dtype=torch.float64)
  self.scale=torch.tensor(m2['scale'],device=device,dtype=torch.float64)
  self.N=torch.tensor(m2['N'],device=device,dtype=torch.float64)
  self.w=torch.tensor(m3['w'],device=device,dtype=torch.float64)
  self.b=float(m3['b'])
 def score(self,ell,heads):
  import torch
  h=torch.as_tensor(heads,device=self.mu.device,dtype=torch.float64).reshape(-1,128)[self.heads].reshape(-1)
  return float(ell+self.b+(((h-self.mu)/self.scale)@self.N)@self.w)
