"""Shared-model module1 -> selected heads/projection -> gated inference."""
import copy
import hashlib
import json
import time
from pathlib import Path
from .tokens import build_prompt_and_mapping
from src.module1 import Module1
from src.module1.backend import LocalBackend,ROOT

class TriagePipeline:
    def __init__(self,components_path=None,projection_path=None,device="cuda:0",top_k=12,
                 max_new_tokens=384,backend=None):
        self.backend=backend or LocalBackend(device=device)
        self.module1=Module1(backend=self.backend,top_k=top_k)
        self.components_path=Path(components_path or ROOT/"results/module2/components.json")
        self.projection_path=Path(projection_path or ROOT/"results/module3/projections.pt")
        self.components=None
        self.max_new_tokens=max_new_tokens
    def load(self):
        self.backend._load_qwen()
        if self.components is None: self.components=json.loads(self.components_path.read_text())
        return self
    def build_projections(self,records,rank=4):
        self.load()
        from src.module3 import build_projections
        return build_projections(self.backend.model,self.backend.tokenizer,records,self.components,
                                 self.projection_path,rank=rank)
    def _decode(self,encoded):
        import torch
        from src.module1.trialgpt_matching import parse_json_response
        model,tokenizer=self.backend.model,self.backend.tokenizer
        with torch.inference_mode():
            generated=model.generate(**encoded,do_sample=False,use_cache=False,
                max_new_tokens=self.max_new_tokens,pad_token_id=tokenizer.eos_token_id)
        text=tokenizer.decode(generated[0,encoded["input_ids"].shape[1]:],skip_special_tokens=True)
        try:
            decision=parse_json_response(text)
            valid=(isinstance(decision.get("decision"),str) and bool(decision["decision"].strip())
                   and isinstance(decision.get("actions"),list)
                   and all(isinstance(a,str) for a in decision["actions"])
                   and decision.get("status") in {"decided","insufficient_evidence"}
                   and isinstance(decision.get("rationale"),str))
        except (ValueError,TypeError,json.JSONDecodeError):
            decision,valid=None,False
        return {"text":text,"decision":decision,"json_valid":valid,
                "output_tokens":int(generated.shape[1]-encoded["input_ids"].shape[1])}
    def _logits(self,encoded):
        import torch
        with torch.inference_mode():
            return self.backend.model(**encoded,use_cache=False).logits[0,-1].float().detach().cpu()
    def run(self,guideline,patient,question,module1_output=None,check_zero_gate=True,
            check_prior_control=False,input_source=None):
        import torch
        from src.module3 import GatedAttentionIntervention
        started=time.time()
        self.load()
        graph=module1_output or self.module1.run(guideline=guideline,patient=patient,question=question)
        prompt,encoded,mapping=build_prompt_and_mapping(self.backend.tokenizer,patient,question,graph)
        if encoded["input_ids"].shape[1]>self.backend.max_input_tokens:
            raise ValueError("Pipeline prompt exceeds configured input budget")
        encoded=encoded.to(self.backend.model.device)
        if not self.projection_path.exists():
            raise FileNotFoundError("Build projections from separate non-test calibration records before inference")
        base_logits=self._logits(encoded)
        baseline=self._decode(encoded)
        with GatedAttentionIntervention(self.backend.model,self.components,self.projection_path,graph,mapping,
                                        context_strength=.1,prior_strength=.1) as hook:
            changed_logits=self._logits(encoded)
            intervened=self._decode(encoded)
        diff=changed_logits-base_logits
        result={"schema_version":"triage_pipeline.v1","input_source":input_source,
            "patient":patient,"question":question,"guideline":guideline,"module1":graph,
            "module1_source":"provided_output" if module1_output else "computed_live",
            "baseline":baseline,"intervened":intervened,"intervention_stats":hook.stats,
            "gates":{"binding_gate":graph["binding_gate"],"action_gate":graph["action_gate"],
                     "prior_weights":[r.get("prior_competition_weight",0) for r in graph["relation_weights"]]},
            "comparison":{"max_abs_next_token_logit_delta":float(diff.abs().max()),
                "l2_next_token_logit_delta":float(diff.norm()),
                "baseline_argmax":int(base_logits.argmax()),"intervened_argmax":int(changed_logits.argmax()),
                "decision_text_changed":baseline["text"]!=intervened["text"]},
            "token_mapping":mapping,"prompt":prompt,"reference_answers_used":False,
            "artifact_identity":{"components_sha256":hashlib.sha256(self.components_path.read_bytes()).hexdigest(),
                "projection_sha256":hashlib.sha256(self.projection_path.read_bytes()).hexdigest(),
                "context_strength":.1,"prior_strength":.1},
            "inference_settings":{"do_sample":False,"use_cache":False,"max_new_tokens":self.max_new_tokens,
                                  "same_prompt_for_baseline_and_intervention":True,"shared_qwen_instance":True}}
        if check_zero_gate:
            zero=copy.deepcopy(graph)
            zero["binding_gate"]={rid:{cid:0.0 for cid in v} for rid,v in graph["binding_gate"].items()}
            zero["action_gate"]={rid:0.0 for rid in graph["action_gate"]}
            for row in zero["relation_weights"]:
                for key in ("weight","signed_weight","binding_gate","action_gate","prior_competition_weight"):
                    row[key]=0.0
            with GatedAttentionIntervention(self.backend.model,self.components,self.projection_path,zero,mapping,
                                            context_strength=.1,prior_strength=.1) as disabled:
                zero_logits=self._logits(encoded)
            result["zero_gate_control"]={"synthetic_control":True,"all_equal":torch.equal(zero_logits,base_logits),
                "max_abs_next_token_logit_delta":float((zero_logits-base_logits).abs().max()),"stats":disabled.stats}
            if not result["zero_gate_control"]["all_equal"]:
                raise AssertionError("Zero-gate intervention changed model computation")
        if check_prior_control:
            positive=copy.deepcopy(graph)
            allowed={r["rule_id"] for r in graph["applicability"] if r["status"]=="applicable"}
            for row in positive["conflict_scores"]:
                if row["rule_id"] in allowed:
                    row.update(score=1.0,status="disagreement",prior_support="oppose",guided_support="support")
            for row in positive["relation_weights"]:
                if row["rule_id"] in allowed: row["prior_competition_weight"]=row["action_gate"]
            with GatedAttentionIntervention(self.backend.model,self.components,self.projection_path,positive,mapping,
                                            context_strength=.1,prior_strength=.1) as prior:
                prior_logits=self._logits(encoded)
            result["prior_path_positive_control"]={"synthetic_conflict_control":True,"forward_only":True,
                "max_abs_next_token_logit_delta":float((prior_logits-base_logits).abs().max()),
                "stats":prior.stats,"applicable_rule_ids":sorted(allowed)}
        result["elapsed_seconds"]=round(time.time()-started,3)
        result["gpu_memory"]={"peak_allocated_gib":torch.cuda.max_memory_allocated()/1024**3,
                             "peak_reserved_gib":torch.cuda.max_memory_reserved()/1024**3}
        json.dumps(result,ensure_ascii=False,allow_nan=False)
        return result
