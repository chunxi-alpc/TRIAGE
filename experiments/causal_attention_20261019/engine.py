"""Real Qwen3 post-RoPE attention score interventions, restricted to selected query rows."""
from __future__ import annotations
import copy, json, os, re, sys, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import ALL_ATTENTION_FUNCTIONS, repeat_kv
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/care_reliability_20261019"))
from joint.context import full_context

MODEL = "/home/gcx/care_20261016/CARE_Codex_Pilot/models/Qwen3-4B-Base"
DATA = ROOT / "experiments/care_reliability_20261019/data/mechanism_cases.jsonl"

def read_cases(split):
    return [c for c in map(json.loads, DATA.read_text().splitlines()) if c["split"] == split]

def target_spans(case, prompt, action_spans, structure, variant="C", control=None):
    """No confidence/evidence admission gates. Shared tokens receive one bias, not duplicated."""
    structure = copy.deepcopy(structure)
    rules = structure.get("rules", [])
    states = [r.get("state", "U") for r in rules]
    if control == "mismatch" and states:
        for i,r in enumerate(rules): r["state"] = states[(i+1) % len(states)]
    states = {r["rule_id"]: r.get("state", "U") for r in rules}
    targets = []
    for rule in case["rules"]:
        rid = rule["rule_id"]; state = states.get(rid, "U")
        sign = 1 if state == "T" else -1 if state == "F" else 0
        if not sign or (variant == "A" and sign < 0) or (variant == "B" and sign > 0): continue
        if control == "reverse": sign *= -1
        targets.append(dict(rule_id=rid, span=action_spans[rid], sign=sign, kind="action"))
        if variant == "D" and state == "T":
            start = prompt.rfind("\n", 0, action_spans[rid][0]) + 1
            targets.append(dict(rule_id=rid, span=[start, action_spans[rid][0]], sign=sign, kind="rule_condition"))
            for atom in rule["conditions"] + rule.get("exceptions", []):
                text = atom["condition_id"] + ": " + atom["text"]
                pos = prompt.find(text, prompt.index("Guidance:\n"))
                if pos >= 0: targets.append(dict(rule_id=rid, span=[pos, pos+len(text)], sign=sign, kind="atomic_condition"))
    if variant == "E":
        evidence_ids = {e for r in rules for k in ("proof_evidence_ids", "support_evidence_ids", "refute_evidence_ids") for e in r.get(k, [])}
        # Generator renders one sentence per fact in this exact order; decimals do not split.
        sentences = list(re.finditer(r".+?(?:[.!?](?=\s+[A-Z]|$))", case["patient"]))
        patient_start = prompt.index("\nPatient: ") + len("\nPatient: ")
        for fact, match in zip(case["structured_facts"], sentences):
            if fact["evidence_id"] in evidence_ids:
                targets.append(dict(rule_id="evidence", span=[patient_start+match.start(),patient_start+match.end()], sign=-1 if control=="reverse" else 1, kind="patient_evidence"))
    return targets

class QwenEngine:
    def __init__(self, model=MODEL, device="cuda:0"):
        self.device=device
        self.tokenizer=AutoTokenizer.from_pretrained(model, local_files_only=True, use_fast=True)
        self.model=AutoModelForCausalLM.from_pretrained(model, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        self.option_ids=[self.tokenizer.encode(" "+c, add_special_tokens=False)[0] for c in "ABCD"]
        assert all(len(self.tokenizer.encode(" "+c, add_special_tokens=False))==1 for c in "ABCD")
        self.original_sdpa=ALL_ATTENTION_FUNCTIONS["sdpa"]
        ALL_ATTENTION_FUNCTIONS.register("sdpa", self._attention)
        self.current=None
        self.key_editor=None
        self.forward_calls=0
        self.provenance=dict(model_path=model, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), device=device,
            gpu=torch.cuda.get_device_name(), torch=torch.__version__, transformers=__import__("transformers").__version__,
            dtype=str(self.model.dtype), num_layers=self.model.config.num_hidden_layers,
            query_heads=self.model.config.num_attention_heads, kv_heads=self.model.config.num_key_value_heads)

    def _attention(self, module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
        output, weights = self.original_sdpa(module,query,key,value,attention_mask,scaling=scaling,dropout=dropout,**kwargs)
        cfg=self.current
        if cfg is None or module.layer_idx not in cfg.get("layers", []): return output, weights
        if not self.has_targets and cfg.get("method", "direct") != "capture": return output, weights
        n=query.shape[-2]; window=min(int(cfg.get("query_window",1)),n)
        q=query[:,:,-window:,:].float()
        k=repeat_kv(key,module.num_key_value_groups).float()
        v=repeat_kv(value,module.num_key_value_groups)
        before=torch.matmul(q,k.transpose(-2,-1))*scaling
        if attention_mask is not None:
            before=before+attention_mask[:,:,-window:,:key.shape[-2]].float()
        else:
            visible=torch.arange(key.shape[-2],device=query.device)[None,:] <= torch.arange(n-window,n,device=query.device)[:,None]
            before=before.masked_fill(~visible[None,None],float("-inf"))
        method=cfg.get("method","direct")
        callback_diagnostics={}
        if method == "direct":
            after=before + float(cfg.get("strength",0))*self.token_signs[None,None,None,:]
        else:
            if self.key_editor is None: raise RuntimeError("key_editor callback required for "+method)
            edited_key, callback_diagnostics=self.key_editor(module,query,key,value,attention_mask,scaling,self)
            if method == "capture": return output, weights
            after=torch.matmul(q,repeat_kv(edited_key,module.num_key_value_groups).float().transpose(-2,-1))*scaling
            after=after+(before-torch.matmul(q,k.transpose(-2,-1))*scaling)
        p0=before.softmax(-1); p1=after.softmax(-1)
        edited_output=torch.matmul(p1.to(value.dtype),v).transpose(1,2).contiguous()
        if method == "direct":
            output=output.clone(); output[:,-window:,:,:]=edited_output
        else:
            output, weights=self.original_sdpa(module,query,edited_key,value,attention_mask,scaling=scaling,dropout=dropout,**kwargs)
        diag=dict(layer=module.layer_idx, edited_query_rows=window if method=="direct" else n, diagnostic_query_rows=window, effect_scope="selected_query_rows" if method=="direct" else "all_causal_queries", query_heads=query.shape[1], callback=callback_diagnostics)
        for label,mask in (("positive",self.token_signs>0),("negative",self.token_signs<0)):
            count=int(mask.sum())
            if count:
                raw0=before[:,:,:,mask]; raw1=after[:,:,:,mask]; valid=torch.isfinite(raw0)&torch.isfinite(raw1)
                deltas=(raw1-raw0)[valid]
                diag[label]=dict(key_tokens=count, finite_score_entries=int(valid.sum()),
                    score_before_mean=float(raw0[valid].mean()), score_after_mean=float(raw1[valid].mean()),
                    score_delta_mean=float(deltas.mean()),score_delta_min=float(deltas.min()),score_delta_max=float(deltas.max()),
                    attention_mass_before=float(p0[:,:,:,mask].sum(-1).mean()),
                    attention_mass_after=float(p1[:,:,:,mask].sum(-1).mean()))
        diag["attention_absolute_change_mean"]=float((p1-p0).abs().sum(-1).mean())
        self.layer_diagnostics.append(diag)
        return output, weights

    def encode_case(self,case,config,structure=None):
        structure=case["reference_structure"] if structure is None else structure
        prompt,spans=full_context(case,structure if config.get("with_structure") else None)
        targets=target_spans(case,prompt,spans,structure,config.get("variant","C"),config.get("control"))
        encoded=self.tokenizer(prompt,return_tensors="pt",return_offsets_mapping=True,add_special_tokens=False)
        offsets=encoded.pop("offset_mapping")[0].tolist()
        signs=torch.zeros(len(offsets),device=self.device,dtype=torch.float32)
        for target in targets:
            start,end=target["span"]
            inds=[i for i,(a,b) in enumerate(offsets) if b>start and a<end and b>a]
            target["token_indices"]=inds
            signs[inds]=target["sign"]
        return prompt, {k:v.to(self.device) for k,v in encoded.items()},targets,signs

    def score(self,case,config,structure=None):
        prompt,inputs,targets,signs=self.encode_case(case,config,structure)
        self.current=config; self.targets=targets; self.token_signs=signs; self.has_targets=any(t["token_indices"] for t in targets)
        self.layer_diagnostics=[]
        started=time.monotonic()
        try:
            with torch.inference_mode():
                logits=self.model(**inputs,use_cache=False,logits_to_keep=1).logits[0,-1].float()
            self.forward_calls+=1
            candidate=logits[self.option_ids].cpu().tolist()
            answer=max(range(4),key=lambda i:candidate[i]); gold="ABCD".index(case["gold_answer"])
            return dict(status="completed",is_model_prediction=True,prediction="ABCD"[answer],
                candidate_logits=candidate, candidate_probabilities=torch.tensor(candidate).softmax(-1).tolist(),
                margin=candidate[gold]-max(v for i,v in enumerate(candidate) if i!=gold),
                prompt=prompt,prompt_tokens=int(inputs["input_ids"].shape[-1]),targets=targets,
                target_token_count=int((signs!=0).sum()),noop=not bool((signs!=0).any()) or not config.get("layers"),
                attention_diagnostics=self.layer_diagnostics,structure_status=(structure or case["reference_structure"]).get("status"),structure_parse_failure=bool((structure or {}).get("parse_failure",False)),elapsed_seconds=time.monotonic()-started)
        finally: self.current=None

