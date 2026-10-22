from pathlib import Path
import torch
from .engine import LAYER
from .study import spec,target_positions
from care_m2_inc.io import write_json

def run(engine,rows,out):
    r=next(r for r in rows if r['task']=='N1' and r['split']=='train')
    positions=target_positions(engine,r)[0];h=engine.capture(r['prompt'],positions)
    u=torch.ones((h.numel(),1));u/=u.norm()
    edit={'kind':'project','positions':positions,'basis':u,'target':h+u[:,0]*5,'beta':1.}
    baseline=engine.score(r['prompt'],r['options'],cache=False)
    zero=engine.score(r['prompt'],r['options'],[{**edit,'beta':0.}],cache=False)
    nonzero=engine.score(r['prompt'],r['options'],[edit],cache=False)
    biaszero=engine.score(r['prompt'],r['options'],biases=[(positions,0.)],cache=False)
    biasnonzero=engine.score(r['prompt'],r['options'],biases=[(positions,1.)],cache=False)
    # Independently reconstruct the live Q/K including per-head normalization and RoPE.
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb,repeat_kv
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    attention=engine.model.model.layers[LAYER].self_attn;captured={};original=ALL_ATTENTION_FUNCTIONS['sdpa']
    def pre(module,args,kwargs):
        hidden=kwargs.get('hidden_states',args[0] if args else None)
        shape=(*hidden.shape[:-1],-1,module.head_dim)
        q=module.q_norm(module.q_proj(hidden).view(shape)).transpose(1,2)
        k=module.k_norm(module.k_proj(hidden).view(shape)).transpose(1,2)
        cos,sin=kwargs['position_embeddings'];captured['q'],captured['k']=apply_rotary_pos_emb(q,k,cos,sin)
    def wrapped(module,q,k,v,mask,*args,**kwargs):
        if module.layer_idx==LAYER:
            captured['q_error']=float((q-captured['q']).abs().max());captured['k_error']=float((k-captured['k']).abs().max())
            group=q.shape[1]//k.shape[1]
            # One real last-query, float32, post-RoPE comparison of native GQA and explicit repeated KV.
            qq=q[:,:,-1:,:].float();kk=k.float();vv=v.float();scale=kwargs['scaling']
            a=torch.nn.functional.scaled_dot_product_attention(qq,kk,vv,enable_gqa=True,scale=scale)
            b=torch.nn.functional.scaled_dot_product_attention(qq,repeat_kv(kk,group),repeat_kv(vv,group),scale=scale)
            captured['gqa_error']=float((a-b).abs().max())
        return original(module,q,k,v,mask,*args,**kwargs)
    handle=attention.register_forward_pre_hook(pre,with_kwargs=True);ALL_ATTENTION_FUNCTIONS.register('sdpa',wrapped)
    try:engine.score(r['prompt'],r['options'],cache=False)
    finally:handle.remove();ALL_ATTENTION_FUNCTIONS.register('sdpa',original)
    delta=lambda a,b:max(abs(a['candidate_logits'][k]-b['candidate_logits'][k]) for k in r['options'])
    checks={'zero_residual':delta(baseline,zero)<1e-5,'zero_bias':delta(baseline,biaszero)<1e-5,
            'residual_actual_change':sum(s['actual_norm'] for s in nonzero['hook']['residual'])>0,
            'non_target_unchanged':all(s['non_target_changed']==0 for s in nonzero['hook']['residual']),
            'score_actual_change':biasnonzero['hook']['attention'][0]['actual_score_rms']>0,
            'rope_q':captured['q_error']<1e-5,'rope_k':captured['k_error']<1e-5,'gqa':captured['gqa_error']<1e-4,
            'bf16':next(engine.model.parameters()).dtype==torch.bfloat16}
    report={'valid':all(checks.values()),'checks':checks,'zero_max_abs':delta(baseline,zero),'target_logit_change':delta(baseline,nonzero),
            'score_logit_change':delta(baseline,biasnonzero),'q_error':captured['q_error'],'k_error':captured['k_error'],'gqa_error':captured['gqa_error'],
            'outputs':[baseline,zero,nonzero,biaszero,biasnonzero],'actual_forward_count':engine.calls,
            'peak_memory_bytes':torch.cuda.max_memory_allocated(),'L0_only':True}
    write_json(Path(out)/'smoke.json',report)
    assert report['valid'],report['checks']
    return report
