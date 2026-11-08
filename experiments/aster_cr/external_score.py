"""Native external tasks, deterministic generation, no MEGA correction or tuning."""
import argparse,json,re,time
from .common import *
EXTERNAL_SYSTEM="Follow the task instructions and output only the requested answer."
def parse_answer(dataset,text):
 text=text.strip()
 if dataset=="nli4ct":
  matches=re.findall(r"\b(Entailment|Contradiction)\b",text,re.I)
  return matches[0].capitalize() if matches and len({m.lower() for m in matches})==1 else None
 try:
  match=re.search(r"\[[^\[\]]*\]",text)
  value=json.loads(match.group()) if match else None
  return value if isinstance(value,list) else None
 except (ValueError,AttributeError):return None
def run(model,dataset,limit=None):
 if not read(ROOT/model/"frozen.json").get("frozen"):raise RuntimeError("External evaluation follows frozen MEGA configuration")
 import torch
 from .engine import Engine
 torch.set_num_threads(4)
 folder=ROOT/model/"external"/dataset;target=folder/"predictions.jsonl"
 items=rows(ROOT/"external"/dataset/"items.jsonl")
 if limit is not None:items=items[:limit]
 done={r["id"] for r in rows(target)};todo=[r for r in items if r["id"] not in done]
 if not todo:return
 eng=Engine(model);start=time.time();eng.model.generation_config.do_sample=False
 for i,r in enumerate(todo):
  rec=dict(id=r["id"],model=model,dataset=dataset)
  try:
   text=eng.tokenizer.apply_chat_template([dict(role="system",content=EXTERNAL_SYSTEM),dict(role="user",content=r["prompt"])],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   encoded=eng.tokenizer(text,return_tensors="pt",add_special_tokens=False).to(eng.device)
   length=encoded["input_ids"].shape[1];new_tokens=32 if dataset=="nli4ct" else 128
   if length+new_tokens>eng.limit:raise ValueError("Input plus generation exceeds context length; no truncation")
   with torch.inference_mode():
    generated=eng.model.generate(**encoded,max_new_tokens=new_tokens,do_sample=False,logits_to_keep=1,pad_token_id=eng.tokenizer.eos_token_id)
   answer=eng.tokenizer.decode(generated[0,length:],skip_special_tokens=True)
   prediction=parse_answer(dataset,answer)
   rec.update(prediction=prediction,raw_answer=answer,status="ok" if prediction is not None else "invalid",tokens=length)
  except ValueError as exc:rec.update(prediction=None,status="unavailable",error=str(exc))
  append(target,rec)
  if i%20==0:status(model+"_external_"+dataset,state="running",done=len(done)+i+1,total=len(items),seconds=time.time()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30)
 write(folder/"cost.json",dict(seconds=time.time()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30,items=len(todo),
  protocol="native task prompt; deterministic generation; published labels; no MEGA fitting",max_new_tokens=32 if dataset=="nli4ct" else 128))
 status(model+"_external_"+dataset,state="complete",items=len(items))
def summarize(model):
 from .repair import external
 external.ROOT=ROOT
 result={}
 for dataset in ("nli4ct","medpic"):
  folder=ROOT/model/"external"/dataset
  result[dataset]=external.evaluate_external(dataset,folder/"predictions.jsonl")
  write(folder/"metrics.json",result[dataset])
 write(ROOT/model/"external/summary.json",result)
 return result
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--model",default="qwen");p.add_argument("--dataset",choices=["nli4ct","medpic"],required=True);p.add_argument("--limit",type=int)
 run(**vars(p.parse_args()))
