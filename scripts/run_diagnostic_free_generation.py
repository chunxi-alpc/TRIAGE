#!/usr/bin/env python3
"""Two preselected historical cases, free-text output with cached repaired graphs."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.pipeline import TriagePipeline
from src.module1.trialgpt_matching import parse_json_response
from src.pipeline.tokens import SYSTEM
from src.module1.trialgpt_matching import parse_json_response
OUT=ROOT/"results/algorithm_diagnostic"
def generate(pipe,messages):
    tok=pipe.backend.tokenizer
    prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    encoded=tok(prompt,add_special_tokens=False,return_tensors="pt").to(pipe.backend.model.device)
    return pipe._decode(encoded)
class CachedDiagnosticPipeline(TriagePipeline):
    def _decode(self,encoded):
        model,tokenizer=self.backend.model,self.backend.tokenizer
        with torch.inference_mode():
            generated=model.generate(**encoded,do_sample=False,use_cache=True,
                max_new_tokens=self.max_new_tokens,pad_token_id=tokenizer.eos_token_id)
        text=tokenizer.decode(generated[0,encoded["input_ids"].shape[1]:],skip_special_tokens=True)
        try:
            decision=parse_json_response(text)
            valid=(isinstance(decision.get("decision"),str) and bool(decision["decision"].strip())
              and isinstance(decision.get("actions"),list) and all(isinstance(a,str) for a in decision["actions"])
              and decision.get("status") in {"decided","insufficient_evidence"} and isinstance(decision.get("rationale"),str))
        except (ValueError,TypeError,json.JSONDecodeError):
            decision,valid=None,False
        return {"text":text,"decision":decision,"json_valid":valid,
                "output_tokens":int(generated.shape[1]-encoded["input_ids"].shape[1]),"use_cache":True}
def main():
    torch.manual_seed(20261014);torch.set_num_threads(4)
    items={r["case_id"]:r for r in json.loads((OUT/"manifest.json").read_text())["samples"]}
    pipe=CachedDiagnosticPipeline(max_new_tokens=384);pipe.load()
    result={"schema_version":"triage_free_generation.v1","cases":[],"reference_answers_used":False,"generation_use_cache":True,
      "selection":"Prespecified historical cases 5:1 and 9:1; references only read for later offline interpretation"}
    for case_id in ("5:1","9:1"):
        item=items[case_id]
        saved=json.loads((OUT/("case_"+case_id.replace(":","_")+".json")).read_text())
        content="Question:\n"+item["question"]+"\n\nPatient note:\n"+item["patient"]+"\n\nCurrent guideline rules:\n"+"\n\n".join(r["text"] for r in item["guideline"]["rules"])
        raw=generate(pipe,[{"role":"system","content":SYSTEM},{"role":"user","content":content}])
        follow=pipe.run(item["guideline"],item["patient"],item["question"],module1_output=saved["module1"],check_zero_gate=False,input_source=item["input_source"])
        result["cases"].append({"case_id":case_id,"raw_full_guideline":raw,"prompt_only":follow["baseline"],
           "full":follow["intervened"],"module1_guided":saved["module1"]["guideline_decision"],
           "intervention_stats":follow["intervention_stats"],"comparison":follow["comparison"],
           "elapsed_seconds":follow["elapsed_seconds"],"peak_allocated_gib":follow["gpu_memory"]["peak_allocated_gib"]})
        (OUT/"free_generation.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print("Free generation verification complete",flush=True)
if __name__=="__main__":main()

