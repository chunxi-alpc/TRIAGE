import argparse,json,os,traceback
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description="Gemma 3 27B selected-head pilot")
 p.add_argument("stage",choices=["prepare","smoke","collect","scan","train","predict","confirm","report","all"])
 p.add_argument("--root",type=Path,default=Path("/home/gcx/gemma3_27b_head_pilot_20261103"))
 p.add_argument("--model",type=Path)
 p.add_argument("--prefill-chunk",type=int,default=0,help="0 uses native whole prefill")
 p.add_argument("--devices",default="0,1",help="Physical GPU mapping; this run uses 0,1")
 p.add_argument("--layer",type=int)
 p.add_argument("--groups",help="Comma-separated KV-group indices; each covers two query heads")
 p.add_argument("--kind",choices=["whole","heads"],default="whole")
 p.add_argument("--checkpoint",type=Path)
 p.add_argument("--split",choices=["development","confirmation"],default="confirmation")
 p.add_argument("--mode",choices=["zero","blind","edited"],default="edited")
 p.add_argument("--output",type=Path)
 a=p.parse_args()
 os.environ["CUDA_VISIBLE_DEVICES"]=a.devices
 from .pipeline import run,prepare,train_job,predict
 from .core import write_json,rows,make_module
 from .engine import Engine
 try:
  if a.stage=="prepare":prepare(a.root)
  elif a.stage=="predict" or (a.stage=="train" and a.layer is not None):
   prepare(a.root);engine=Engine.load(a.model or a.root/"model",a.root,a.prefill_chunk)
   if a.mode=="edited" or a.stage=="train":
    if a.layer is None:p.error("--layer is required for an explicit edited run")
    spec=dict(name=f"manual_{a.kind}_layer{a.layer}",kind=a.kind,layer=a.layer)
    if a.kind=="heads":
     if not a.groups:p.error("--groups is required for head intervention")
     spec["groups"]=[int(x) for x in a.groups.split(",")]
   else:spec=None
   if a.stage=="train":train_job(engine,a.root,spec)
   else:
    module=None
    if a.mode=="edited":
     if a.checkpoint is None:p.error("--checkpoint is required for edited prediction")
     from safetensors.torch import load_file
     module=make_module(spec,engine.config,engine.layer_device(a.layer))
     module.load_state_dict(load_file(str(a.checkpoint)))
    output=a.output or a.root/"manual_predictions"/f"{a.split}_{a.mode}.jsonl"
    summary=predict(engine,rows(a.root/"data"/f"{a.split}.jsonl"),output,spec,module,a.mode)
    write_json(output.with_suffix(".summary.json"),summary)
  else:run(a.root,a.model or a.root/"model",a.stage,a.prefill_chunk)
  print(json.dumps({"stage":a.stage,"status":"complete","root":str(a.root)},ensure_ascii=False),flush=True)
 except Exception as exc:
  write_json(a.root/"failure.json",dict(stage=a.stage,error=type(exc).__name__+": "+str(exc)))
  traceback.print_exc();raise

if __name__=="__main__":main()
