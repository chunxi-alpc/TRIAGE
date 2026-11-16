import re,time,hashlib
import torch
from engine import *

def norm(s):return re.sub(r"[^a-z0-9]+"," ",s.lower()).strip()
def matches(output,answer):
 # Exact answer or answer surrounded by ordinary answer boilerplate; no fuzzy judge.
 a,b=norm(answer),norm(output)
 return bool(a) and (a==b or (len(a)>=4 and re.search(r"(?:^| )"+re.escape(a)+r"(?: |$)",b) is not None))
def clean(q,j=0):
 wrappers=["{q}\nGive only the short answer.","Answer this question briefly: {q}\nReturn only the answer.","Question: {q}\nProvide the short factual answer without explanation.","Please answer: {q}\nUse only a name, number, or short phrase.","{q}\nRespond with the answer alone, no explanation."]
 return wrappers[j].format(q=q)
def conflict(r):return "Passage:\n"+r["context_mod"]+"\n\nQuestion: "+r["question"]+"\nGive only the short answer."

def main():
 start=time.perf_counter();e=Engine();data=rows(ROOT/"data/facts.jsonl");train=[r for r in data if r["split"]=="train"]
 sums_clean=[];sums_conf=[];eligible=[]
 for r in train:
  checks=[]
  for j in range(5):
   out=e.generate(clean(r["question"],j));ok=matches(out["raw_output"],r["answer_original"])
   append(ROOT/"results/fact_filter.jsonl",dict(item_id=r["item_id"],stage=f"clean{j}",expected=r["answer_original"],correct=ok,**out));checks.append(ok)
   if not ok:break
  if not all(checks) or len(checks)!=5:continue
  out=e.generate(conflict(r));ok=matches(out["raw_output"],r["answer_mod"])
  append(ROOT/"results/fact_filter.jsonl",dict(item_id=r["item_id"],stage="conflict",expected=r["answer_mod"],correct=ok,**out))
  if ok:
   sums_clean.append(e.capture(clean(r["question"])));sums_conf.append(e.capture(conflict(r)));eligible.append(r["item_id"])
 if len(eligible)<4:raise RuntimeError(f"Only {len(eligible)} eligible train pairs; direction not estimated")
 c=torch.stack(sums_clean).mean(0);f=torch.stack(sums_conf).mean(0);cos=torch.nn.functional.cosine_similarity(c,f,dim=1)
 # Exclude last transformer block, which feeds output normalization/unembedding.
 layer=int(cos[:-1].argmin());direction=c[layer]-f[layer];e.layer=layer;e.direction=direction
 torch.save(dict(clean_activations=torch.stack(sums_clean),conflict_activations=torch.stack(sums_conf),direction=direction,layer=layer,cosines=cos,eligible_ids=eligible),ROOT/"results/direction.pt")
 dump(ROOT/"results/direction.json",dict(layer_zero_based=layer,eligible_train=len(eligible),candidate_train=len(train),eligible_ids=eligible,direction_norm=float(direction.norm()),cosines=cos.tolist(),source="ClashEval",train_filter="five instruction variants all old-answer correct, conflict modified-answer correct",limitation="instruction variants are not independently authored question paraphrases"))
 dev=[r for r in data if r["split"]=="dev"];alpha_grid=[-3.0,-2.0,-1.0,-0.5,0.0];selection=[]
 for alpha in alpha_grid:
  values=[]
  for r in dev:
   out=e.generate(conflict(r),alpha);ok=matches(out["raw_output"],r["answer_mod"]);values.append(ok)
   append(ROOT/"results/source_dev.jsonl",dict(item_id=r["item_id"],alpha=alpha,context_match=ok,parametric_match=matches(out["raw_output"],r["answer_original"]),**out))
  selection.append(dict(alpha=alpha,context_accuracy=sum(values)/len(values),n=len(values)))
 chosen=max(selection,key=lambda x:(x["context_accuracy"],-abs(x["alpha"])))
 dump(ROOT/"results/selected.json",dict(**chosen,selection=selection,layer=layer,selection_source="source_dev_only",target_test_accessed=False,alpha_frozen=True,source_dev_population="all presplit dev questions, no clean/conflict eligibility selection"))
 # Freeze metadata before any factual held-out or target outcomes are accessed.
 for r in [r for r in data if r["split"]=="test"]:
  prior=e.generate(clean(r["question"]))
  append(ROOT/"results/source_test.jsonl",dict(item_id=r["item_id"],method="closed_book",correct=matches(prior["raw_output"],r["answer_original"]),**prior))
  for method,alpha in [("A",0),("B",chosen["alpha"])]:
   out=e.generate(conflict(r),alpha)
   append(ROOT/"results/source_test.jsonl",dict(item_id=r["item_id"],method=method,context_match=matches(out["raw_output"],r["answer_mod"]),parametric_match=matches(out["raw_output"],r["answer_original"]),prior_correct=matches(prior["raw_output"],r["answer_original"]),**out))
 dump(ROOT/"results/facts_complete.json",dict(status="complete",total_seconds=time.perf_counter()-start,eligible_train=len(eligible),selected=chosen,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated()))
 print("FACTS_COMPLETE",flush=True)

if __name__=="__main__":main()
