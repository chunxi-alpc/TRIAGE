"""Uniform key-score bias using the existing Qwen3 SDPA backend and GQA.

Only queries at/after decision_start are changed. The final prompt query is
included. Zero/empty settings register no hooks and preserve exact logits.
"""
import torch

def biased_mask(mask, positions, key_length, targets, beta, decision_start, dtype, device):
    qlen=len(positions)
    if mask is None:
        allowed=torch.arange(key_length,device=device)[None,:] <= positions[:,None]
        base=torch.zeros((1,1,qlen,key_length),dtype=dtype,device=device)
        base.masked_fill_(~allowed[None,None,:,:],float("-inf"))
    else:
        if mask.ndim!=4: raise ValueError("unsupported_attention_mask_shape")
        base=mask[..., :key_length]
        if base.dtype==torch.bool:
            base=torch.zeros_like(base,dtype=dtype).masked_fill(~base,float("-inf"))
        base=base.expand(base.shape[0],base.shape[1],qlen,key_length).clone()
    selected=torch.zeros(key_length,dtype=torch.bool,device=device)
    if any(t<0 or t>=key_length for t in targets): raise ValueError("target_outside_key_sequence")
    selected[targets]=True
    active=positions>=decision_start
    # Adding a finite value to a masked position never opens the causal mask.
    delta=active[:,None] & selected[None,:]
    return base+delta[None,None,:,:].to(dtype)*beta


def _decision_sdpa(module, query, key, value, attention_mask, **kwargs):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    owner=module._acre_bias_owner
    # Run the original operator for ALL rows, preserving its original kernel/GQA.
    original,_=sdpa_attention_forward(module,query,key,value,attention_mask,**kwargs)
    key_length=key.shape[-2];qlen=query.shape[-2]
    positions=torch.arange(key_length-qlen,key_length,device=query.device)
    active=positions>=owner.decision_start
    if not active.any(): return original,None
    # Only active decision queries are recomputed; no input query receives a bias
    # or a changed numerical kernel. HF Qwen3 handles RoPE/cache before dispatch.
    mask=attention_mask
    if mask is not None:
        mask=mask[..., :key_length]
        mask=mask.expand(mask.shape[0],mask.shape[1],qlen,key_length)[...,active,:]
    modified=biased_mask(mask,positions[active],key_length,owner.targets,owner.beta,owner.decision_start,query.dtype,query.device)
    local=dict(kwargs);local['is_causal']=False
    edited,_=sdpa_attention_forward(module,query[:,:,active,:],key,value,modified,**local)
    output=original.clone();output[:,active,:,:]=edited
    owner.stats['layers'][str(module.layer_idx)]={
        'query_heads':module.config.num_attention_heads,'kv_heads':module.config.num_key_value_heads,
        'active_query_positions':positions[active].tolist(),'target_count':len(owner.targets)}
    return output,None

class AttentionBias:
    def __init__(self, model, targets, beta, decision_start):
        self.model=model;self.targets=sorted(set(targets));self.beta=float(beta)
        self.decision_start=int(decision_start);self.originals=[];self.guards=[]
        self.stats={"beta":self.beta,"targets":self.targets,"decision_start":self.decision_start,"layers":{},"backend":"sdpa; original rows plus decision-only masked SDPA","enabled":bool(self.targets and beta)}
    def __enter__(self):
        if not self.targets or self.beta==0: return self
        from copy import copy
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        if self.model.config.model_type!="qwen3" or self.model.config._attn_implementation!="sdpa":
            raise ValueError("Qwen3 SDPA required; no spectral fallback")
        if self.model.training: raise ValueError("evaluation_mode_required")
        ALL_ATTENTION_FUNCTIONS.register("acre_pilot_sdpa",_decision_sdpa)
        for layer in self.model.model.layers:
            attn=layer.self_attn
            if hasattr(attn,"_acre_bias_owner"): raise ValueError("nested_attention_bias_not_supported")
            self.originals.append((attn,attn.config));attn.config=copy(attn.config)
            attn.config._attn_implementation="acre_pilot_sdpa";attn._acre_bias_owner=self
            def guard(module,args,kwargs):
                cache=kwargs.get("past_key_values")
                if cache is not None and type(cache).__name__!="DynamicCache":
                    raise ValueError("only_uncached_or_dynamic_cache_supported")
                if kwargs.get("output_attentions") or kwargs.get("head_mask") is not None:
                    raise ValueError("unsupported_attention_output_or_head_mask")
                return args,kwargs
            self.guards.append(attn.register_forward_pre_hook(guard,with_kwargs=True))
        return self
    def __exit__(self,*exc):
        for guard in self.guards:guard.remove()
        self.guards=[]
        for attn,config in self.originals:
            attn.config=config;del attn._acre_bias_owner
        self.originals=[]
        return False
