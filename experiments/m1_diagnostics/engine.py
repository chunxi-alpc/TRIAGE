"""BF16 frozen backbones, FP32 label projection, native immutable cache replay."""
import copy,time
from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.cache_utils import DynamicCache
from .common import MODELS
from .data import render

@dataclass
class Context:
    prompt:str
    input_ids:torch.Tensor
    prefix:object
    length:int

class TooLong(ValueError):
    def __init__(self,tokens,limit):
        self.tokens,self.limit=tokens,limit
        super().__init__(f"Original input has {tokens} tokens; model limit is {limit}")

class Engine:
    def __init__(self,model_name,device="cuda:0",model=None,tokenizer=None):
        self.model_name=model_name
        self.tokenizer=tokenizer or AutoTokenizer.from_pretrained(MODELS[model_name],local_files_only=True)
        self.model=model or AutoModelForCausalLM.from_pretrained(MODELS[model_name],local_files_only=True,
            dtype=torch.bfloat16,device_map={"":device},attn_implementation="sdpa")
        self.model.eval().requires_grad_(False)
        self.device=self.model.get_input_embeddings().weight.device
        self.layers=self.model.model.layers;self.n_layers=len(self.layers)
        self.n_heads=int(self.model.config.num_attention_heads);self.head_dim=int(self.layers[0].self_attn.head_dim)
        self.num_heads=self.n_layers*self.n_heads;self._active=False;self.prefill_chunk=4096
        self.label_ids=[self.tokenizer.encode(s,add_special_tokens=False)[0] for s in ["是","否"]]
        assert all(len(self.tokenizer.encode(s,add_special_tokens=False))==1 for s in ["是","否"])
        self._label_weights=self.model.lm_head.weight[self.label_ids].detach().float()
        self.limit=int(self.model.config.max_position_embeddings)
        self.counts=dict(prefills=0,final_forwards=0,backwards=0,prefill_seconds=0.,final_seconds=0.,residual_forwards=0)
    def render(self,prompt,task="binary"):return render(self.tokenizer,prompt,task)
    def ids(self,prompt,task="binary"):
        ids=self.tokenizer(self.render(prompt,task),return_tensors="pt",add_special_tokens=False)["input_ids"].to(self.device)
        if ids.shape[1]>self.limit:raise TooLong(int(ids.shape[1]),self.limit)
        return ids
    def _prefill(self,ids,stop):
        cache=DynamicCache(config=self.model.config)
        with torch.no_grad():
            for start in range(0,stop,self.prefill_chunk):
                end=min(start+self.prefill_chunk,stop)
                output=self.model.model(input_ids=ids[:,start:end],attention_mask=torch.ones_like(ids[:,:end]),
                    past_key_values=cache,use_cache=True,cache_position=torch.arange(start,end,device=self.device),
                    position_ids=torch.arange(start,end,device=self.device).unsqueeze(0))
                cache=output.past_key_values;del output
        for layer in cache.layers:
            if layer.keys is not None:
                layer.keys=layer.keys.detach().contiguous();layer.values=layer.values.detach().contiguous()
        return cache
    def prepare(self,prompt,task="binary"):
        ids=self.ids(prompt,task);before=time.perf_counter();cache=self._prefill(ids,ids.shape[1]-1)
        self.counts["prefills"]+=1;self.counts["prefill_seconds"]+=time.perf_counter()-before
        return Context(prompt,ids,cache,int(ids.shape[1]))
    @staticmethod
    def _copy_cache(prefix):
        cache=copy.copy(prefix);cache.layers=[copy.copy(layer) for layer in prefix.layers]
        return cache
    def _cache(self,context):return self._copy_cache(context.prefix)
    def _score_hidden(self,hidden):
        logits=torch.nn.functional.linear(hidden.float(),self._label_weights)
        return logits[...,0]-logits[...,1]
    def run(self,context,replacements=None,capture=True,grad=False,norm_scales=None,capture_scales=False,label_ids=None):
        if self._active:raise RuntimeError("Overlapping model forwards")
        donors={}
        for h,value in (replacements or {}).items():
            h=int(h)
            if not 0<=h<self.num_heads:raise ValueError("Head outside model")
            donors.setdefault(h//self.n_heads,{})[h%self.n_heads]=value
        raw_heads={};scales={};handles=[];before=time.perf_counter();self._active=True
        def head_hook(i):
            def hook(module,args):
                value=args[0]
                if value.shape!=(1,1,self.n_heads*self.head_dim):raise RuntimeError("Wrong head position")
                raw=value.reshape(1,self.n_heads,self.head_dim)
                if grad:raw.retain_grad()
                if capture or grad:raw_heads[i]=raw
                work=raw
                if i in donors:
                    work=raw.clone();js=sorted(donors[i])
                    work[:,js]=torch.stack([torch.as_tensor(donors[i][j]) for j in js]).to(value.device,value.dtype)
                return (work.reshape_as(value),)+args[1:]
            return hook
        def scale_hook(i):
            def hook(module,args,output):
                x=args[0]
                if capture_scales:scales[i]=torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+module.eps).detach().cpu()
                if norm_scales is not None:
                    s=norm_scales[i].to(x.device)
                    return (x.float()*s*(1.+module.weight.float())).to(x.dtype)
            return hook
        try:
            for i,layer in enumerate(self.layers):
                if capture or grad or i in donors:handles.append(layer.self_attn.o_proj.register_forward_pre_hook(head_hook(i)))
                # Qwen's same-named norm is MLP pre-normalization and must never be changed.
                if self.model_name=="medgemma" and (capture_scales or norm_scales is not None):
                    handles.append(layer.post_attention_layernorm.register_forward_hook(scale_hook(i)))
            with torch.enable_grad() if grad else torch.no_grad():
                embeds=self.model.get_input_embeddings()(context.input_ids[:,-1:]).detach()
                if grad:embeds.requires_grad_(True)
                output=self.model.model(inputs_embeds=embeds,attention_mask=torch.ones_like(context.input_ids),
                    past_key_values=self._cache(context),use_cache=True,
                    cache_position=torch.tensor([context.length-1],device=self.device),
                    position_ids=torch.tensor([[context.length-1]],device=self.device))
                hidden=output.last_hidden_state[0,-1];ell=self._score_hidden(hidden)
                if grad:ell.backward();self.counts["backwards"]+=1
                result=dict(ell=float(ell.detach()),prediction=int(ell.detach()>0),tokens=context.length)
                if label_ids is not None:
                    weight=self.model.lm_head.weight[label_ids].detach().float()
                    result["label_logits"]=torch.nn.functional.linear(hidden.float(),weight).detach().cpu().tolist()
                if capture or grad:
                    result["heads"]=torch.stack([raw_heads[i][0].detach().float().cpu() for i in range(self.n_layers)])
                if grad:
                    result["grads"]=torch.stack([raw_heads[i].grad[0].detach().float().cpu() for i in range(self.n_layers)])
                if capture_scales:result["rms_scales"]=scales
            self.counts["final_forwards"]+=1;self.counts["final_seconds"]+=time.perf_counter()-before
            return result
        finally:
            for h in handles:h.remove()
            self._active=False
    def full_score(self,prompt):
        ids=self.ids(prompt)
        with torch.no_grad():
            output=self.model.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
            return float(self._score_hidden(output.last_hidden_state[0,-1]))

    def residual_run(self,prompt,positions,layers,replacement=None):
        """Patch post-layer residual; recompute every suffix token with native causal caches.
        positions is a mapping of named anchors to token indices.
        """
        ids=self.ids(prompt);captured={};handles=[];offset=[0]
        def hook(i):
            def call(module,args,out):
                value=out[0] if isinstance(out,tuple) else out
                work=value
                for name,pos in positions.items():
                    local=pos-offset[0]
                    if 0<=local<value.shape[1]:
                        captured[(i,name)]=value[0,local].detach().float().cpu()
                        if replacement is not None and replacement[0]==i and replacement[1]==name:
                            work=value.clone();work[0,local]=replacement[2].to(value.device,value.dtype)
                return (work,)+out[1:] if isinstance(out,tuple) else work
            return call
        try:
            for i in layers:handles.append(self.layers[i].register_forward_hook(hook(i)))
            cache=DynamicCache(config=self.model.config)
            with torch.no_grad():
                # The final token is replayed separately to match the head-patching score convention.
                stops=list(range(self.prefill_chunk,ids.shape[1]-1,self.prefill_chunk))+[ids.shape[1]-1,ids.shape[1]]
                start=0
                for end in stops:
                    if end<=start:continue
                    offset[0]=start
                    out=self.model.model(input_ids=ids[:,start:end],attention_mask=torch.ones_like(ids[:,:end]),
                        past_key_values=cache,use_cache=True,cache_position=torch.arange(start,end,device=self.device),
                        position_ids=torch.arange(start,end,device=self.device).unsqueeze(0))
                    cache=out.past_key_values
                    if end==ids.shape[1]:ell=float(self._score_hidden(out.last_hidden_state[0,-1]))
                    del out;start=end
            self.counts["residual_forwards"]+=1
            return dict(ell=ell,residuals=captured)
        finally:
            for h in handles:h.remove()
