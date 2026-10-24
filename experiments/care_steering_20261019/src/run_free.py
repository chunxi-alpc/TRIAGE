"""Exploratory free-generation follow-up after a dev-only output-format diagnostic.
No prompt, direction, alpha, dataset, or gold labels used to repair predictions.
"""
from engine import *
import re,sys,collections
sys.path.insert(0,str(ROOT))
from data.care_gate import predict_item
MAX_NEW_TOKENS=768
BATCH_SIZE=8

def parse_answer(raw,options):
 lines=[x.strip() for x in raw.strip().splitlines() if x.strip()]
 if not lines:return "INVALID","empty"
 line=lines[-1].replace("**","").replace("`","").strip()
 line=re.sub(r"^#{1,6}\s*","",line)
 line=re.sub(r"^(?:(?:therefore|thus|so)[,:]?\s*)?(?:(?:the\s+)?(?:correct\s+)?(?:final\s+)?answer(?:\s+is)?|option|choice)\s*[:=\-]?\s*","",line,flags=re.I)
 m=re.fullmatch(r"\(?([A-D])\)?[.。]?",line)
 if m:return m.group(1),"last_line_label"
 m=re.fullmatch(r"\(?([A-D])\)?[.):\-]\s*(.+?)[.。]?",line)
 if m:
  label,action=m.groups()
  if action.rstrip(".。 ")==options[label]:return label,"last_line_label_and_action_consistent"
  return "INVALID","label_action_mismatch"
 hits=[k for k,v in options.items() if line.rstrip(".。 ")==v]
 if len(hits)==1:return hits[0],"last_line_exact_action"
 return "INVALID","no_unambiguous_final_answer"

@torch.inference_mode()
def generate_batch(e,items,alpha):
 tok=e.tokenizer;tok.padding_side="left"
 texts=[e.text(r["prompt"]) for r in items]
 started=time.perf_counter();x=tok(texts,return_tensors="pt",padding=True).to("cuda");width=x.input_ids.shape[1]
 torch.cuda.synchronize();t=time.perf_counter()
 with e.steer(alpha,width) as trace:
  out=e.model.generate(**x,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id)
 torch.cuda.synchronize();generation_seconds=time.perf_counter()-t;elapsed=time.perf_counter()-started
 eos=e.model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
 values=[]
 for j,r in enumerate(items):
  ids=out[j,width:].tolist();end=next((i+1 for i,v in enumerate(ids) if v in eos),len(ids));ids=ids[:end]
  raw=tok.decode(ids,skip_special_tokens=True);pred,reason=parse_answer(raw,r["options"])
  values.append(dict(item_id=r["item_id"],alpha=float(alpha),raw_output=raw,generated_token_ids=ids,prediction=pred,parse_reason=reason,truncated=len(ids)>=MAX_NEW_TOKENS and ids[-1] not in eos,input_tokens=int(x.attention_mask[j].sum()),output_tokens=len(ids),batch_size=len(items),batch_generation_seconds=generation_seconds,batch_end_to_end_seconds=elapsed,total_seconds=generation_seconds/len(items),end_to_end_allocated_seconds=elapsed/len(items),timing_method="batch_wall_time_divided_by_batch_size_not_isolated_latency",steering=dict(trace),peak_memory_bytes=torch.cuda.max_memory_allocated()))
 return values

def main():
 start=time.perf_counter();e=Engine();d=torch.load(ROOT/"results/direction.pt",weights_only=False);sel=json.loads((ROOT/"results/selected.json").read_text());e.direction=d["direction"];e.layer=d["layer"];alpha=sel["alpha"]
 cases=rows(ROOT/"data/cases.jsonl");targets=[r for r in cases if r["split"] in ("test","transfer")];byid={r["item_id"]:r for r in targets}
 gates={r["item_id"]:predict_item(r) for r in targets}
 required=collections.defaultdict(list)
 for r in targets:
  # Zero epsilon D equals A by algebra; zero gate C equals A, +gate C equals B.
  for a in sorted({0.0,alpha,alpha*gates[r["item_id"]]["signed_gate"]}):required[a].append(r)
 dump(ROOT/"audit/free_generation_protocol.json",dict(phase="exploratory_followup",reason="dev outputs were long reasoning, constrained first-token scores are not generated answers",max_new_tokens=MAX_NEW_TOKENS,batch_size=BATCH_SIZE,target_count=len(targets),unique_completions=sum(map(len,required.values())),prompt_changed=False,coefficients_frozen=True,source_direction_frozen=True,invalid_scored_incorrect=True,parser="last nonempty line, exact letter or exact complete option; no gold access",D_zero_epsilon_reuses_A=True,identical_C_alpha_reuses_completion=True))
 cache={}
 for a,items in sorted(required.items()):
  for offset in range(0,len(items),BATCH_SIZE):
   outputs=generate_batch(e,items[offset:offset+BATCH_SIZE],a)
   for out in outputs:append(ROOT/"results/free_completions.jsonl",out);cache[(out["item_id"],a)]=out
 for r in targets:
  g=gates[r["item_id"]]
  for method,a in [("A",0.0),("B",alpha),("C_reduced",alpha*g["signed_gate"]),("D_adapt",0.0)]:
   out=dict(cache[(r["item_id"],a)]);pred=out["prediction"];valid=pred in r["options"];action=r["options"].get(pred)
   row=dict(**{k:r[k] for k in ("pair_id","rule_family_id","split","task","condition","applicable","old_valid","conflict","gold","requires_joint_actions")},**out,method=method,seed=SEED,correct=pred==r["gold"],new_action_selected=(r["new_action"] in action) if valid else None,old_action_selected=(r["options"][r["old_action_label"]] in action) if valid else None,valid_answer=valid,scoring="free_generation_final_line_exact",signed_gate=g["signed_gate"],gate_applicability_state=g["new_state"],epsilon=0.0,reused_completion=method in ("C_reduced","D_adapt"),gate_seconds=0.0)
   append(ROOT/"results/free_predictions.jsonl",row)
 dump(ROOT/"results/free_complete.json",dict(status="complete",total_seconds=time.perf_counter()-start,unique_completions=len(cache),target_items=len(targets),peak_memory_bytes=torch.cuda.max_memory_allocated()))
 print("FREE_COMPLETE",flush=True)
if __name__=="__main__":main()
