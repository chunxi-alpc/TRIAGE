"""Small Qwen dev-only LoReFT activation/optimization/reload test; no test evaluation."""
import os,sys,json,time,math
from pathlib import Path
import torch
import torch.nn.functional as F
os.environ["ASTER_CR_ROOT"]="/home/gcx/aster_cr_repair_v2_run_20261108"
from experiments.aster_cr.baselines import Adapter
from experiments.aster_cr.engine import Engine
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"reports/aster_e5_gaps_20261112/loreft_effectiveness"
OLD=Path(os.environ["ASTER_CR_ROOT"])
def readrows(p):
 with p.open() as f:return [json.loads(s) for s in f if s.strip()]
class Observed(Adapter):
 def __init__(self,*args,**kwargs):self.observations=[];super().__init__(*args,**kwargs)
 def residual_hook(self,module,args,out):
  value=out[0] if isinstance(out,tuple) else out
  before=value[:,-1].detach().float().clone()
  result=super().residual_hook(module,args,out)
  if self.active:
   after=result[0] if isinstance(result,tuple) else result
   self.observations.append(float((after[:,-1].detach().float()-before).abs().max()))
  return result
def main():
 torch.set_num_threads(4);torch.manual_seed(11);started=time.time()
 OUT.mkdir(exist_ok=True,parents=True)
 allv=readrows(OLD/"data/views.jsonl")
 dev=[]
 for kind in ["original","A","P","C","F"]:
  dev.extend(sorted((v for v in allv if v["split"]=="dev" and v["kind"]==kind),key=lambda v:len(v["prompt"]))[:2])
 pairs=readrows(Path("/home/gcx/aster_cr_pair_dataset_20261111/balanced_train.jsonl"))
 endpoints={v["prompt"]:v for p in pairs for v in [p["original"],p["modified"]]}
 pool=sorted(endpoints.values(),key=lambda v:v["view_id"])
 import random
 train_candidates=random.Random(11).sample(pool,64)
 checkpoint=OLD/"qwen/baselines/loreft_lr0.01.pt"
 obj=torch.load(checkpoint,map_location="cpu",weights_only=False)
 eng=Engine("qwen")
 candidate_losses=[]
 for v in train_candidates:
  raw=eng.score(v["prompt"],capture=False)["ell"]
  loss=float(F.binary_cross_entropy_with_logits(torch.tensor(raw),torch.tensor(float(v["label"]))))
  candidate_losses.append((loss,v))
 train=[v for _,v in sorted(candidate_losses,key=lambda x:-x[0])[:4]]
 results=[]
 old=Observed(eng,"loreft",obj["budget"],load=checkpoint)
 for v in dev:
  old.active=False;ctx=eng.prepare(v["prompt"])
  raw=eng.run(ctx,capture=False)["ell"]
  old.active=True;before=len(old.observations)
  edited=eng.run(ctx,capture=False)["ell"]
  results.append(dict(view_id=v["view_id"],split="dev",kind=v["kind"],old_raw=raw,old_active=edited,old_delta=edited-raw,old_hook_calls=len(old.observations)-before,old_residual_change=old.observations[-1]))
 old.close()
 torch.manual_seed(11);ad=Observed(eng,"loreft",obj["budget"])
 rotation=list(ad.module.rotate_layer.parameters());rotids={id(p) for p in rotation}
 others=[p for p in ad.parameters() if id(p) not in rotids]
 opt=torch.optim.AdamW([{"params":rotation,"weight_decay":0.0},{"params":others,"weight_decay":0.01}],lr=.001)
 before_params=[p.detach().clone() for p in ad.parameters()]
 train_log=[]
 for v in train:
  ad.active=True;opt.zero_grad(set_to_none=True)
  score=ad.score_train(v["prompt"]);loss=F.binary_cross_entropy_with_logits(score,torch.tensor(float(v["label"]),device=score.device))
  loss.backward();grad=math.sqrt(sum(float(p.grad.square().sum()) for p in ad.parameters() if p.grad is not None))
  torch.nn.utils.clip_grad_norm_(ad.parameters(),1.0);opt.step()
  train_log.append(dict(view_id=v["view_id"],loss=float(loss.detach()),gradient_norm=grad,rotation_norm=float(ad.module.rotate_layer.weight.detach().norm())))
 change=max(float((p.detach()-b).abs().max()) for p,b in zip(ad.parameters(),before_params))
 smoke=OUT/"repaired_smoke_only_v2.pt";ad.dump(smoke)
 for r,v in zip(results,dev):
  ad.active=False;ctx=eng.prepare(v["prompt"]);raw=eng.run(ctx,capture=False)["ell"]
  ad.active=True;before=len(ad.observations);edited=eng.run(ctx,capture=False)["ell"]
  r.update(repaired_raw=raw,repaired_active=edited,repaired_delta=edited-raw,repaired_hook_calls=len(ad.observations)-before,repaired_residual_change=ad.observations[-1])
 ad.close()
 loaded=Observed(eng,"loreft",obj["budget"],load=smoke)
 for r,v in zip(results,dev):
  ell=loaded.score(v["prompt"]);r["reloaded_score"]=ell;r["reload_difference"]=ell-r["repaired_active"]
 loaded.close()
 passed=all(r["old_delta"]==0 and r["old_hook_calls"]==1 for r in results) and any(r["repaired_delta"]!=0 and r["repaired_residual_change"]>0 for r in results) and all(abs(r["reload_difference"])<1e-6 for r in results) and change>0 and any(t["gradient_norm"]>1e-8 for t in train_log) and all(math.isfinite(t["gradient_norm"]) and t["rotation_norm"]>.99 for t in train_log)
 summary=dict(passed=passed,old_checkpoint=str(checkpoint),old_rotate_norm=float(obj["state"]["rotate_layer"].norm()),train_smoke_examples=len(train),dev_examples=len(dev),layer=ad.layer,rank=ad.rank,changed_parameter_max_abs=change,training_steps=train_log,predictions=results,seconds=time.time()-started,peak_gib=torch.cuda.max_memory_allocated()/2**30,formal_training=False,training_candidate_count=64,training_smoke_selection="largest native BCE within 64 fixed-seed fit-only examples; functionality test only",heldout_test_access=False,cause="AdamW weight decay changes Householder integer sign diagonal; rotate parameters must not decay",fix="weight_decay=0 only for orthogonally parametrized rotation; native supervised BCE unchanged")
 (OUT/"gpu_effectiveness_final.json").write_text(json.dumps(summary,indent=2)+"\n")
 print(json.dumps({k:v for k,v in summary.items() if k not in ["predictions","training_steps"]}),flush=True)
 if not passed:raise RuntimeError("LoReFT effectiveness test failed")
if __name__=="__main__":main()
