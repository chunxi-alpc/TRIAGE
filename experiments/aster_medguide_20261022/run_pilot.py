#!/usr/bin/env python3
"""Run closed-book, A/B/C and oracle attention-steering arms for ASTER."""
import argparse, json, os, re, contextlib
from pathlib import Path

_ACTIVE_MASKS = None
_ACTIVE_STRENGTHS = (0.0, 0.0)

@contextlib.contextmanager
def signed_attention_patch(modeling_llama, strength_app, strength_sibling, app_mask, sibling_mask):
    """Add signed key-position biases to eager attention probabilities at every decode step."""
    global _ACTIVE_MASKS, _ACTIVE_STRENGTHS
    original = modeling_llama.eager_attention_forward
    _ACTIVE_MASKS = (app_mask, sibling_mask)
    _ACTIVE_STRENGTHS = (float(strength_app), float(strength_sibling))
    def patched(query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        out, weights = original(query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)
        if weights is None or _ACTIVE_MASKS is None:
            return out, weights
        app, sib = _ACTIVE_MASKS
        klen = weights.shape[-1]
        bias = weights.new_zeros((1, 1, 1, klen))
        if app is not None:
            n=min(klen, app.numel())
            bias[..., :n] += app[:n].to(device=weights.device,dtype=weights.dtype).view(1,1,1,n) * _ACTIVE_STRENGTHS[0]
        if sib is not None:
            n=min(klen, sib.numel())
            bias[..., :n] += sib[:n].to(device=weights.device,dtype=weights.dtype).view(1,1,1,n) * _ACTIVE_STRENGTHS[1]
        # Equivalent to adding key-specific bias to pre-softmax scores, for nonzero weights.
        adjusted = weights * torch.exp(bias.clamp(min=-20,max=20))
        adjusted = adjusted / adjusted.sum(dim=-1,keepdim=True).clamp_min(1e-20)
        if dropout:
            adjusted = torch.nn.functional.dropout(adjusted,p=dropout,training=False)
        head_out = torch.matmul(adjusted, value)
        head_out = head_out.transpose(1,2).contiguous()
        head_out = head_out.reshape(head_out.shape[0],head_out.shape[1],-1)
        return head_out, adjusted
    modeling_llama.eager_attention_forward = patched
    try:
        yield
    finally:
        modeling_llama.eager_attention_forward = original
        _ACTIVE_MASKS = None
        _ACTIVE_STRENGTHS = (0.0,0.0)

def parse_answer(text):
    m=re.search(r"(?<![A-Z])([A-E])(?![A-Z])",text.strip().upper())
    return m.group(1) if m else None

def span_mask(offsets, span):
    import torch
    mask=torch.zeros(len(offsets),dtype=torch.float32)
    if not span: return mask
    start,end=span
    for i,(s,e) in enumerate(offsets):
        if e>s and s<end and e>start: mask[i]=1.0
    return mask

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True); p.add_argument("--model",default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--output",required=True); p.add_argument("--device-map",default="auto")
    p.add_argument("--dtype",choices=["bfloat16","float16","float32"],default="bfloat16")
    p.add_argument("--max-new-tokens",type=int,default=8)
    p.add_argument("--strength",type=float,default=1.0)
    p.add_argument("--no-intervention",action="store_true")
    a=p.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype={"bfloat16":torch.bfloat16,"float16":torch.float16,"float32":torch.float32}[a.dtype]
    tok=AutoTokenizer.from_pretrained(a.model,use_fast=True)
    if not tok.is_fast: raise SystemExit("需要 fast tokenizer 才能映射规则字符区间。")
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=dtype,device_map=a.device_map,attn_implementation="eager")
    model.eval()
    import transformers.models.llama.modeling_llama as modeling_llama
    rows=[json.loads(s) for s in open(a.input,encoding="utf-8") if s.strip()]
    groups={}
    for r in rows: groups.setdefault(r["id"],{})[r["variant"]]=r
    results=[]
    for exid, variants in groups.items():
        arow=variants.get("A_original")
        if not arow: continue
        base_prompt=arow["closed_book_prompt"]
        closed=tok(base_prompt,return_tensors="pt",return_offsets_mapping=False)
        closed={k:v.to(model.device) for k,v in closed.items()}
        with torch.inference_mode():
            out=model.generate(**closed,do_sample=False,max_new_tokens=a.max_new_tokens)
        closed_text=tok.decode(out[0][closed["input_ids"].shape[1]:],skip_special_tokens=True)
        closed_ans=parse_answer(closed_text)
        for variant,row in variants.items():
            encoded=tok(row["prompt_text"],return_tensors="pt",return_offsets_mapping=True)
            offsets=encoded.pop("offset_mapping")[0].tolist()
            model_inputs={k:v.to(model.device) for k,v in encoded.items()}
            # All unsteered and steered conditions use exactly the same input and greedy decoding.
            conditions=[("baseline",0.0,0.0)]
            if variant!="A_original":
                conditions += [("applicable_only",a.strength,0.0),
                               ("unsigned",a.strength,a.strength),
                               ("signed",a.strength,-a.strength)]
            app=span_mask(offsets,row["applicable_char_span"])
            sib=span_mask(offsets,row["sibling_char_span"])
            for condition,sa,ss in conditions:
                if a.no_intervention and condition!="baseline": continue
                patch= signed_attention_patch(modeling_llama,sa,ss,app,sib) if condition!="baseline" else contextlib.nullcontext()
                try:
                    with patch, torch.inference_mode():
                        gen=model.generate(**model_inputs,do_sample=False,max_new_tokens=a.max_new_tokens)
                except (AttributeError,TypeError,ValueError) as e:
                    if condition!="baseline":
                        raise RuntimeError("当前 Transformers 的 Llama eager-attention 接口不兼容，未将该条件伪记为干预结果。") from e
                    raise
                completion=tok.decode(gen[0][model_inputs["input_ids"].shape[1]:],skip_special_tokens=True)
                results.append({"id":exid,"variant":variant,"condition":condition,"prediction":parse_answer(completion),
                                "completion":completion,"gold_answer":row["gold_answer"],"closed_book_prediction":closed_ans,
                                "original_answer":row["original_answer"],"sibling_action":row["sibling_action"],
                                "sibling_action_label":next((k for k,v in row["options"].items() if str(v).strip().casefold()==row["sibling_action"].strip().casefold()),None),
                                "strength_applicable":sa,"strength_sibling":ss,"model":a.model})
        print(f"completed {exid}",flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.output)),exist_ok=True)
    with open(a.output,"w",encoding="utf-8") as f:
        for r in results: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    print(json.dumps({"output":a.output,"predictions":len(results),"examples":len(groups)},ensure_ascii=False))

if __name__=="__main__": main()
