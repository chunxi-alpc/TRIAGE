"""Qwen final-position query-head interventions with immutable unedited prefixes."""
from dataclasses import dataclass
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

DEFAULT_MODEL = "/home/gcx/TRIAGE/models/modelscope/Qwen3-8B"
SYSTEM = ("\u6839\u636e\u7ed9\u5b9a\u6307\u5357\u548c\u75c5\u4f8b\uff0c\u5224\u65ad\u5f53\u524d\u8282\u70b9\u7684\u5b8c\u6574\u6761\u4ef6\u662f\u5426\u6210\u7acb\u3002\u4ec5\u8f93\u51fa\u201c\u662f\u201d\u6216\u201c\u5426\u201d\u3002"
          " For English inputs, \u662f means Yes and \u5426 means No.")

@dataclass
class Context:
    prompt: str
    rendered_prompt: str
    input_ids: torch.Tensor
    prefix: tuple
    length: int

class Engine:
    """One frozen model on one GPU; caches never include an edited position.

    heads are pre-edit query outputs, flat IDs are layer * H + j.
    Operators and donor replacements apply only at the final input position.
    clamp_heads makes those outputs independent intervention variables;
    gradients there are joint-clamp derivatives, not natural gradients.
    """
    def __init__(self, model_path=DEFAULT_MODEL, device="cuda:0", *, model=None,
                 tokenizer=None, dtype=None):
        self.device = torch.device(device)
        self.model_path = str(model_path)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        dtype = dtype or (torch.bfloat16 if self.device.type == "cuda" else torch.float32)
        self.model = model or AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, dtype=dtype, device_map={"": str(self.device)},
            attn_implementation="sdpa")
        if model is not None:
            self.model.to(self.device)
        self.model.eval().requires_grad_(False)
        self.layers = self.model.model.layers
        self.n_layers = len(self.layers)
        self.n_heads = int(self.model.config.num_attention_heads)
        self.head_dim = int(self.layers[0].self_attn.head_dim)
        self.num_heads = self.n_layers * self.n_heads
        self._operator_owner = None
        self._compiled_operators = {}
        self._active = False
        ids = [self.tokenizer.encode(s, add_special_tokens=False) for s in ("\u662f", "\u5426")]
        if any(len(x) != 1 for x in ids) or ids[0] == ids[1]:
            raise ValueError("? and ? must be distinct single-token labels")
        self.label_ids = [x[0] for x in ids]
        self.counts = dict(prefills=0, final_forwards=0, backwards=0,
                           prefill_seconds=0., final_seconds=0.)
        if getattr(self.model.model, "has_sliding_layers", False):
            raise ValueError("This engine uses full-attention Qwen prefixes")

    def render(self, prompt):
        return self.tokenizer.apply_chat_template(
            [dict(role="system",content=SYSTEM),dict(role="user",content=prompt)],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def prepare(self, prompt):
        """Compute only the unedited prefix; keep the final token for replay."""
        if self._active:
            raise RuntimeError("Cannot prepare another context during a forward")
        rendered = self.render(prompt)
        batch = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        ids = batch["input_ids"].to(self.device)
        if ids.shape[0] != 1 or ids.shape[1] < 2:
            raise ValueError("Expected one prompt with at least two tokens")
        before = time.perf_counter()
        with torch.no_grad():
            output = self.model.model(
                input_ids=ids[:, :-1], attention_mask=torch.ones_like(ids[:, :-1]),
                use_cache=True)
        cache = output.past_key_values
        prefix = tuple((layer.keys.detach(), layer.values.detach()) for layer in cache.layers)
        self.counts["prefills"] += 1
        self.counts["prefill_seconds"] += time.perf_counter()-before
        return Context(str(prompt), rendered, ids, prefix, int(ids.shape[1]))

    def _cache(self, context):
        """New cache container, referencing prefix tensors without mutation."""
        cache = DynamicCache(config=self.model.config)
        if len(cache.layers) != len(context.prefix):
            raise ValueError("Prefix layer count differs from model")
        for layer, (keys, values) in zip(cache.layers, context.prefix):
            layer.keys, layer.values = keys, values
            layer.dtype, layer.device = keys.dtype, keys.device
            layer.is_initialized = True
        return cache

    def _operators(self, operators):
        if operators is self._operator_owner:
            return self._compiled_operators
        result = {}
        for flat, op in (operators or {}).items():
            flat = int(flat)
            if flat < 0 or flat >= self.num_heads:
                raise ValueError("Query head outside model")
            item = {name: torch.as_tensor(op[name], device=self.device, dtype=torch.float32)
                    for name in ("N","B","mu","S") if name in op}
            d = self.head_dim
            item.setdefault("N", torch.empty(d,0,device=self.device))
            item.setdefault("B", torch.empty(0,d,device=self.device))
            item.setdefault("mu", torch.zeros(d,device=self.device))
            item.setdefault("S", torch.empty(d,0,device=self.device))
            if (item["N"].ndim != 2 or item["N"].shape[0] != d
                    or item["B"].shape != (item["N"].shape[1],d)
                    or item["mu"].shape != (d,)
                    or item["S"].ndim != 2 or item["S"].shape[0] != d):
                raise ValueError("Operator dimensions do not match query-head dimension")
            item["amplification"] = float(op.get("amplification",1.))
            result.setdefault(flat//self.n_heads,{})[flat%self.n_heads] = item
        self._operator_owner, self._compiled_operators = operators, result
        return result

    def run(self, context, operators=None, replacements=None, capture=True,
            grad=False, clamp_heads=None):
        if self._active:
            raise RuntimeError("Engine forwards cannot overlap")
        operators = self._operators(operators)
        donors = {}
        for flat, value in (replacements or {}).items():
            flat = int(flat)
            if flat < 0 or flat >= self.num_heads:
                raise ValueError("Donor query head outside model")
            value = torch.as_tensor(value, device=self.device).reshape(-1)
            if value.numel() != self.head_dim:
                raise ValueError("Donor size differs from query head")
            donors.setdefault(flat//self.n_heads,{})[flat%self.n_heads] = value
        clamps = set(int(h) for h in (clamp_heads or []))
        if any(h<0 or h>=self.num_heads for h in clamps):
            raise ValueError("Clamped query head outside model")
        if any(h//self.n_heads in operators and h%self.n_heads in operators[h//self.n_heads]
               for h in clamps):
            raise ValueError("A clamped head cannot also use an online operator")
        raw_heads, clamp_variables, handles = {}, {}, []
        before=time.perf_counter()
        self._active=True

        def make_hook(layer_index):
            def hook(module, args):
                value=args[0]
                if value.ndim!=3 or value.shape[0]!=1 or value.shape[1]!=1:
                    raise RuntimeError("Interventions must execute only the replayed final token")
                raw=value[:, -1, :].reshape(1,self.n_heads,self.head_dim)
                if grad and raw.requires_grad:
                    raw.retain_grad()
                raw_heads[layer_index]=raw
                work=raw
                if layer_index in operators or layer_index in donors or any(
                        h//self.n_heads==layer_index for h in clamps):
                    work=raw.clone()
                    for j,op in operators.get(layer_index,{}).items():
                        centered=raw[:,j].float()-op["mu"]
                        edited=raw[:,j].float()-(centered@op["B"].T)@op["N"].T
                        if op["amplification"] != 1. and op["S"].shape[1]:
                            edited=edited+(op["amplification"]-1.)*((edited-op["mu"])@op["S"])@op["S"].T
                        work[:,j]=edited.to(value.dtype)
                    for j in range(self.n_heads):
                        flat=layer_index*self.n_heads+j
                        donor=donors.get(layer_index,{}).get(j)
                        if donor is not None or flat in clamps:
                            replacement=(raw[:,j].detach() if donor is None else
                                         donor.to(dtype=value.dtype).reshape(1,-1).detach())
                            if grad and flat in clamps:
                                replacement=replacement.clone().requires_grad_(True)
                                clamp_variables[flat]=replacement
                            work[:,j]=replacement
                # Feed the captured view into o_proj to obtain its gradient.
                return (work.reshape_as(value),)+tuple(args[1:])
            return hook
        try:
            for i,layer in enumerate(self.layers):
                handles.append(layer.self_attn.o_proj.register_forward_pre_hook(make_hook(i)))
            with torch.enable_grad() if grad else torch.no_grad():
                embeds=self.model.get_input_embeddings()(context.input_ids[:, -1:]).detach()
                if grad:
                    embeds.requires_grad_(True)
                output=self.model(
                    inputs_embeds=embeds,
                    attention_mask=torch.ones_like(context.input_ids),
                    past_key_values=self._cache(context),use_cache=True,
                    cache_position=torch.tensor([context.length-1],device=self.device),
                    position_ids=torch.tensor([[context.length-1]],device=self.device),
                    logits_to_keep=1)
                logits=output.logits[0,-1].float()
                ell=logits[self.label_ids[0]]-logits[self.label_ids[1]]
                if grad:
                    ell.backward()
                    self.counts["backwards"]+=1
                result={"ell":float(ell.detach()),"prediction":int(float(ell.detach())>0),
                        "tokens":context.length}
                if capture or grad:
                    result["heads"]=torch.stack([raw_heads[i][0].detach().float().cpu()
                                                for i in range(self.n_layers)])
                if grad:
                    gradients=[]
                    for i in range(self.n_layers):
                        raw=raw_heads[i]
                        g=(torch.zeros_like(raw) if raw.grad is None else raw.grad).detach().float().clone()
                        for j in range(self.n_heads):
                            var=clamp_variables.get(i*self.n_heads+j)
                            if var is not None:
                                g[:,j]=0 if var.grad is None else var.grad.detach().float()
                        gradients.append(g[0].cpu())
                    result["grads"]=torch.stack(gradients)
            self.counts["final_forwards"]+=1
            self.counts["final_seconds"]+=time.perf_counter()-before
            return result
        finally:
            for handle in handles:
                handle.remove()
            self._active=False

    def full_score(self, prompt):
        """Uncached ordinary forward for zero-intervention verification."""
        rendered=self.render(prompt)
        batch=self.tokenizer(rendered,return_tensors="pt",add_special_tokens=False).to(self.device)
        with torch.no_grad():
            logits=self.model(**batch,use_cache=False,logits_to_keep=1).logits[0,-1].float()
        return float(logits[self.label_ids[0]]-logits[self.label_ids[1]])
