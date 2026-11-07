"""Extract structured facts from original cases; no model answer determines variant labels."""
import argparse,time
from .common import *
from experiments.aster_closed_form.data import canonical_guide,condition
from experiments.aster_closed_form.common import evaluate,fields
INSTRUCTION="""Extract a complete executable rule and observed facts for an existing benchmark case.
Do NOT solve the case by inserting the answer as a fact. Do NOT invent observations.
Return a single JSON object. If the full rule or a required observation is ambiguous, return
{"available":false,"reason":"..."}.
Schema:
{"available":true,
"facts":{"age_years":65,"diagnosis_confirmed":true},
"descriptions":{"age_years":"age in years","diagnosis_confirmed":"diagnosis confirmed"},
"predicate":{"op":"and","args":[{"op":">=","field":"age_years","value":65},{"op":"==","field":"diagnosis_confirmed","value":true}]},
"ancestor_predicates":[{"gid":"G0","predicate":{"op":"==","field":"diagnosis_confirmed","value":true}}],
"evidence":{"age_years":["65-year-old"],"diagnosis_confirmed":["exact sentence from patient"]},
"edits":[{"field":"age_years","spans":[{"old":"65-year-old","template":"{value}-year-old"}],"step":1,"unit":"years","minimum":0,"maximum":120,"legal_values":null}],
"decisive_sentences":["exact patient sentences needed to decide current and ancestor conditions"],
"background":{"old":"exact irrelevant patient sentence","new":"a same-language, similarly sized neutral replacement","field":"unrelated background field","before":"old background value","after":"new background value"},
"protected_action_ids":["L4"],
"reason":"short explanation"}
Operators: and/or with args, not with arg, < <= > >= == != in not_in with field and value.
Include EVERY part of the current condition and every ancestor. Facts must cover all expressions.
Evidence must quote the supplied patient exactly. Variable edits must replace ALL statements about the
changed fact, including categorical paraphrases; otherwise omit that edit. Numeric templates contain
exactly one {value}; choose step from units and rule precision, never predictions.
Discrete edits use legal_values [{"value":false,"replacements":["exact new sentence for first span"]},...].
Do not add free rewritten cases. Background may be null; it must be irrelevant to current AND ancestor
rules. Decisive sentences must preserve all required observations. Protect any leaf whose wording
defines a term needed by the target or ancestors. No fabricated external definitions.
"""
def validate(g,a):
 if not a.get("available"):return dict(state="unavailable",reason=a.get("reason"))
 expr=a["predicate"];facts=a["facts"];required=fields(expr)
 anc={x["gid"]:x["predicate"] for x in a["ancestor_predicates"]}
 for r in g["ancestors"]:
  if r["gid"] not in anc:raise ValueError("missing ancestor "+r["gid"])
  required |= fields(anc[r["gid"]])
  if int(evaluate(anc[r["gid"]],facts))!=r["label"]:raise ValueError("ancestor truth disagrees with original path")
 for key in required:
  if key not in a["descriptions"] or not a["evidence"].get(key):raise ValueError("missing evidence/description "+key)
  if any(s not in g["patient"] or not s for s in a["evidence"][key]):raise ValueError("evidence is not an exact original span")
 label=int(evaluate(expr,facts))
 if label!=g["original_label"]:return dict(state="E4_candidate",program_label=label,original_label=g["original_label"],reason="Extracted rule and released label disagree; unresolved, not relabelled")
 for s in a["decisive_sentences"]:
  if not s or s not in g["patient"]:raise ValueError("decisive sentence is not an original span")
 for ed in a.get("edits",[]):
  if ed["field"] not in required:raise ValueError("edited field is not in rule")
  for s in ed["spans"]:
   if not s["old"] or s["old"] not in g["patient"]:raise ValueError("fact edit is not an original span")
  if not ed.get("legal_values") and (ed.get("step",0)<=0 or any(s.get("template","").count("{value}")!=1 for s in ed["spans"])):raise ValueError("invalid numeric edit template")
 bg=a.get("background")
 if bg and (bg["field"] in required or not bg["old"] or bg["old"] not in g["patient"]):a["background"]=None
 return dict(state="complete",program_label=label)
def run(shard,shards=3,limit=None):
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM
 torch.set_num_threads(4)
 target=ROOT/"data"/f"annotations_{shard}.jsonl";done={r["group_id"] for r in rows(target)}
 groups=[g for g in rows(ROOT/"data/nodes.jsonl")[shard::shards] if g["group_id"] not in done]
 if limit is not None:groups=groups[:limit]
 if not groups:return
 tok=AutoTokenizer.from_pretrained(MODELS["qwen"],local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(MODELS["qwen"],local_files_only=True,dtype=torch.bfloat16,device_map={"":"cuda:0"},attn_implementation="sdpa").eval()
 started=time.time()
 for i,g in enumerate(groups):
  payload=dict(tree=canonical_guide(g["guideline"]),current_condition=condition(g),ancestors=g["ancestors"],patient=g["patient"])
  prompt=INSTRUCTION+"\nINPUT:\n"+json.dumps(payload,ensure_ascii=False)
  txt=tok.apply_chat_template([dict(role="user",content=prompt)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  inp=tok(txt,return_tensors="pt",add_special_tokens=False).to(model.device)
  result=dict(group_id=g["group_id"],split=g["split"],source_id=g["source_id"],annotation_model="qwen",input_tokens=int(inp.input_ids.shape[1]))
  try:
   with torch.inference_mode():out=model.generate(**inp,max_new_tokens=2400,do_sample=False,pad_token_id=tok.eos_token_id)
   answer=tok.decode(out[0,inp.input_ids.shape[1]:],skip_special_tokens=True)
   result["raw"]=answer
   a,_=json.JSONDecoder().raw_decode(answer[answer.index("{"):])
   result.update(value=a,validation=validate(g,a))
  except (ValueError,KeyError,TypeError,IndexError) as e:result.update(validation=dict(state="unavailable",reason=str(e)))
  append(target,result)
  status(f"annotate_{shard}",state="running",done=len(done)+i+1,total=len(done)+len(groups),seconds=time.time()-started,
     peak_gib=torch.cuda.max_memory_allocated()/2**30)
 status(f"annotate_{shard}",state="complete",done=len(done)+len(groups),seconds=time.time()-started)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--shard",type=int,required=True);p.add_argument("--limit",type=int)
 a=p.parse_args();run(a.shard,limit=a.limit)
