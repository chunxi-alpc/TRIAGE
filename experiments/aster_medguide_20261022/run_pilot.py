#!/usr/bin/env python3
"""Run real greedy inference and oracle rule steering for Llama and Qwen3."""
import argparse
import contextlib
import importlib
import inspect
import json
import re
import time
from pathlib import Path

CONDITIONS=('baseline','direct_rule','applicable_only','unsigned','signed')

@contextlib.contextmanager
def signed_attention_patch(modeling, strength_app, strength_sibling, app_mask, sibling_mask):
    """Pass additive logit bias into native eager attention BEFORE softmax.

    Native eager handles GQA expansion, dropout and [B,Q,H,D] output. The
    existing causal/padding mask remains in force during prefill and KV decode.
    This serial, process-local context restores the native function on failure.
    """
    import torch
    original=modeling.eager_attention_forward
    if next(iter(inspect.signature(original).parameters))!='module':
        raise RuntimeError('unsupported eager interface: expected module first')
    state={'calls':0}
    def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        klen=key.shape[-2]; qlen=query.shape[-2]
        bias=torch.zeros((1,1,1,klen),device=query.device,dtype=query.dtype)
        for mask,strength in ((app_mask,strength_app),(sibling_mask,strength_sibling)):
            n=min(klen,mask.numel())
            bias[...,:n]+=mask[:n].to(device=query.device,dtype=query.dtype).reshape(1,1,1,n)*strength
        if attention_mask is None:
            # Native eager expects callers to provide causality. Preserve it
            # explicitly even for direct calls and multi-token cached forwards.
            positions=torch.arange(klen,device=query.device)
            queries=torch.arange(klen-qlen,klen,device=query.device)
            causal=positions[None,:]>queries[:,None]
            attention_mask=torch.zeros((1,1,qlen,klen),device=query.device,dtype=query.dtype)
            attention_mask.masked_fill_(causal,torch.finfo(query.dtype).min)
        elif attention_mask.dtype==torch.bool:
            raise RuntimeError('eager requires additive causal mask, received boolean mask')
        adjusted=attention_mask[...,:klen]+bias
        state['calls']+=1
        return original(module,query,key,value,adjusted,scaling=scaling,dropout=dropout,**kwargs)
    modeling.eager_attention_forward=patched
    try:
        yield state
    finally:
        modeling.eager_attention_forward=original

def parse_answer(text, labels='ABCDEFGH'):
    # Accept a leading explicit answer, including ordinary Markdown and option
    # text. Never search arbitrary prose for an English article such as "A".
    clean=text.strip().replace('**','').replace('`','')
    prefix=r'(?:(?:the\s+)?(?:correct\s+)?answer\s*(?:is\s*)?:?\s*)?'
    match=re.match(r'^'+prefix+r'\(?([A-H])\)?\s*(?:\n|$)',clean,re.IGNORECASE)
    if match is None:
        match=re.match(r'^'+prefix+r'\(?([A-H])[.):](?=\s|$)',clean,re.IGNORECASE)
    return match.group(1).upper() if match and match.group(1).upper() in labels else None

def span_mask(offsets,span):
    import torch
    start,end=span
    return torch.tensor([float(e>s and s<end and e>start) for s,e in offsets])

def encode_prompt(tok,content,spans=()):
    rendered=tok.apply_chat_template([{'role':'user','content':content}],tokenize=False,
                                     add_generation_prompt=True,enable_thinking=False)
    if rendered.count(content)!=1:raise ValueError('cannot uniquely map user content in chat template')
    shift=rendered.index(content)
    encoded=tok(rendered,add_special_tokens=False,return_tensors='pt',return_offsets_mapping=True)
    offsets=encoded.pop('offset_mapping')[0].tolist()
    masks=[span_mask(offsets,[s+shift,e+shift]) for s,e in spans]
    if any(m.sum()==0 for m in masks):raise ValueError('empty rule token span')
    if len(masks)==2 and (masks[0]*masks[1]).sum():raise ValueError('overlapping rule token spans')
    return encoded,masks

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True);p.add_argument('--model',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device-map',default='auto');p.add_argument('--dtype',choices=['bfloat16','float16','float32'],default='bfloat16')
    p.add_argument('--strength',type=float,default=1.0);p.add_argument('--max-new-tokens',type=int,default=8)
    p.add_argument('--no-intervention',action='store_true')
    a=p.parse_args()
    import torch,transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer
    started=time.time();torch.manual_seed(20261022)
    tok=AutoTokenizer.from_pretrained(a.model,use_fast=True,local_files_only=True)
    if not tok.is_fast:raise ValueError('fast tokenizer required')
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=getattr(torch,a.dtype),
                                             device_map=a.device_map,attn_implementation='eager',local_files_only=True).eval()
    if model.config.model_type not in ('llama','qwen3'):raise ValueError('validated architectures: llama, qwen3')
    modeling=importlib.import_module('transformers.models.'+model.config.model_type+'.modeling_'+model.config.model_type)
    rows=[json.loads(s) for s in Path(a.input).read_text().splitlines() if s.strip()]
    groups={}
    for row in rows:
        key=(row['id'],row['variant'])
        if key[1] in groups.setdefault(key[0],{}):raise ValueError('duplicate id/variant')
        groups[key[0]][key[1]]=row
    for v in groups.values():
        if set(v)!={'A_original','B_applicable_rule_changed','C_sibling_rule_changed'}:raise ValueError('incomplete A/B/C group')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    n=0;expected=len(groups)*(1+3*(2 if a.no_intervention else 5))
    def generate(content,spans=(),strengths=None):
        encoded,masks=encode_prompt(tok,content,spans)
        encoded={k:v.to(model.device) for k,v in encoded.items()}
        patch=signed_attention_patch(modeling,*strengths,*masks) if strengths else contextlib.nullcontext({'calls':0})
        with patch as state,torch.inference_mode():
            generated=model.generate(**encoded,do_sample=False,max_new_tokens=a.max_new_tokens,
                                     pad_token_id=tok.eos_token_id,use_cache=True)
        if strengths and not state['calls']:raise RuntimeError('attention intervention was never called')
        text=tok.decode(generated[0][encoded['input_ids'].shape[1]:],skip_special_tokens=True)
        return text,{'attention_calls':state['calls'],'input_tokens':encoded['input_ids'].shape[1],
                     'rule_token_counts':[int(m.sum()) for m in masks]}
    with out.open('w',encoding='utf-8') as f:
        for exid,variants in groups.items():
            original=variants['A_original']
            text,meta=generate(original['closed_book_prompt'])
            closed=parse_answer(text,original['options'])
            def record(row,condition,text,meta,sa=0.0,ss=0.0):
                nonlocal n
                r={k:row[k] for k in ['id','source_row','disease','variant','gold_answer','original_answer',
                                      'sibling_action_label','original_sibling_action_label','replacement_action_label']}
                r.update(condition=condition,prediction=parse_answer(text,row['options']),completion=text,
                         closed_book_prediction=closed,strength_applicable=sa,strength_sibling=ss,
                         model=a.model,oracle=condition in ('direct_rule','applicable_only','unsigned','signed'),**meta)
                f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();n+=1
            record(original,'closed_book',text,meta)
            for row in variants.values():
                for condition in CONDITIONS:
                    if a.no_intervention and condition not in ('baseline','direct_rule'):continue
                    sa=ss=0.0
                    if condition=='direct_rule':text,meta=generate(row['direct_rule_prompt'])
                    elif condition=='baseline':text,meta=generate(row['prompt_text'])
                    else:
                        sa=a.strength;ss=0.0 if condition=='applicable_only' else a.strength if condition=='unsigned' else -a.strength
                        text,meta=generate(row['prompt_text'],[row['applicable_char_span'],row['sibling_char_span']],(sa,ss))
                    record(row,condition,text,meta,sa,ss)
            print('completed',exid,flush=True)
    manifest={'model':a.model,'torch':torch.__version__,'transformers':transformers.__version__,
              'examples':len(groups),'predictions':n,'expected_predictions':expected,'complete':n==expected,
              'seconds':time.time()-started,'dtype':a.dtype,'strength':a.strength,'max_new_tokens':a.max_new_tokens,
              'chat_template':True,'qwen_enable_thinking':False,'steering':'all layers/heads, rule text keys, all query positions',
              'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else None}
    out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)

if __name__=='__main__':main()
