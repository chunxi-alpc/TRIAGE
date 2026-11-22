"""Exact suffix replay for single-head Qwen interventions at the final token.

Reuse the unchanged attention output at the intervention layer and the layer's
baseline residual. Recompute o_proj, its MLP, all downstream layers and final norm.
No batching, precision, attention backend or scoring definition is changed.
"""
import time
import torch

class SuffixRunner:
    def __init__(self,engine):
        if engine.text_config.model_type!="qwen3":
            raise ValueError("Suffix replay is validated for this Qwen3 checkpoint")
        self.engine=engine
        self.context=None
        self.inputs={}
        self.attention={}
        self.kwargs={}
        self.suffix_layer_calls=0
        self.queries=0

    def prepare(self,context):
        self.context=context
        self.inputs={};self.attention={};self.kwargs={}
        handles=[]
        def before_layer(i):
            def hook(module,args,kwargs):
                self.inputs[i]=args[0].detach().clone()
                self.kwargs[i]=dict(kwargs)
            return hook
        def before_output(i):
            def hook(module,args):
                self.attention[i]=args[0].detach().clone()
            return hook
        try:
            for i,layer in enumerate(self.engine.layers):
                handles.append(layer.register_forward_pre_hook(before_layer(i),with_kwargs=True))
                handles.append(layer.self_attn.o_proj.register_forward_pre_hook(before_output(i)))
            result=self.engine.run(context,capture=False)
        finally:
            for h in handles:h.remove()
        if len(self.inputs)!=self.engine.n_layers or len(self.attention)!=self.engine.n_layers:
            raise ValueError("Baseline layer coverage incomplete")
        return float(result["ell"])

    def score(self,head,donor):
        engine=self.engine
        if self.context is None:raise RuntimeError("Prepare target before suffix replay")
        layer_id,local=divmod(int(head),engine.n_heads)
        if not 0<=layer_id<engine.n_layers:raise ValueError("Head outside checkpoint")
        started=time.perf_counter()
        with torch.no_grad():
            layer=engine.layers[layer_id]
            work=self.attention[layer_id].clone().reshape(1,engine.n_heads,engine.head_dim)
            work[:,local]=torch.as_tensor(donor).to(work.device,work.dtype)
            hidden=self.inputs[layer_id]+layer.self_attn.o_proj(work.reshape(1,1,-1))
            hidden=hidden+layer.mlp(layer.post_attention_layernorm(hidden))
            cache=engine._cache(self.context)
            for i in range(layer_id+1,engine.n_layers):
                kwargs=dict(self.kwargs[i]);kwargs["past_key_values"]=cache
                hidden=engine.layers[i](hidden,**kwargs)
            hidden=engine.model.model.norm(hidden)
            value=float(engine._score_hidden(hidden[0,-1]).detach())
        self.queries+=1;self.suffix_layer_calls+=engine.n_layers-layer_id
        engine.counts["final_forwards"]+=1
        engine.counts["final_seconds"]+=time.perf_counter()-started
        return value
