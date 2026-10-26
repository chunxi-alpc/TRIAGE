"""Server-only local HF client for the official MEGA Qwen judge functions.

No network access. Load only after the coordinator acquires a GPU for judging.
The same Qwen weights in a judge role do not constitute an independent model.
"""
from types import SimpleNamespace
import os
import time


class LocalResponse(SimpleNamespace):
    def model_dump(self):
        return self.payload


class LocalQwenClient:
    def __init__(self, model_path, served_name="Qwen3_8B", context_length=32768, seed=0):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError("Local judge requires the coordinator-assigned CUDA device")
        if not os.environ.get("CUDA_VISIBLE_DEVICES") or torch.cuda.device_count() != 1:
            raise RuntimeError("Local judge must see exactly one coordinator-assigned CUDA device")
        started = time.monotonic()
        self.torch = torch
        self.served_name, self.context_length, self.seed = served_name, context_length, seed
        self.model_path = str(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype="auto",
            device_map="auto", attn_implementation="sdpa").eval()
        if getattr(self.model.config, "model_type", "") != "qwen3":
            raise ValueError("The local judge must be the declared Qwen3 checkpoint")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True,
                                                       padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.device = next(self.model.parameters()).device
        self.forward_attempts = self.forward_completed = self.requests = 0
        self.prompt_tokens = self.completion_tokens = 0
        def attempt(module, args, kwargs):
            self.forward_attempts += 1
        def complete(module, args, kwargs, output):
            self.forward_completed += 1
        self.hooks = [self.model.register_forward_pre_hook(attempt, with_kwargs=True),
                      self.model.register_forward_hook(complete, with_kwargs=True)]
        # Upstream API did not declare a seed. This local adaptation freezes it once.
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize(self.device)
        self.load_elapsed_seconds = time.monotonic() - started
        self.versions = {"torch": torch.__version__, "transformers": transformers.__version__}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        torch = self.torch
        if request["model"] != self.served_name:
            raise ValueError("Requested judge model differs from the loaded model")
        extra = request.get("extra_body", {})
        if (request.get("temperature") != .7 or request.get("top_p") != .8
                or extra.get("top_k") != 20
                or extra.get("chat_template_kwargs", {}).get("enable_thinking") is not False):
            raise ValueError("Judge request differs from the frozen official sampling settings")
        inputs = self.tokenizer.apply_chat_template(
            request["messages"], tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_tensors="pt", return_dict=True)
        prompt_length = inputs["input_ids"].shape[1]
        maximum = int(request["max_tokens"])
        native_context = int(getattr(self.model.config, "max_position_embeddings", self.context_length))
        if maximum <= 0 or prompt_length + maximum > min(self.context_length, native_context):
            raise ValueError("Judge prompt plus requested generation exceeds the frozen context budget; no truncation")
        inputs = inputs.to(self.device)
        torch.cuda.synchronize(self.device)
        started = time.monotonic()
        before = self.forward_attempts
        self.requests += 1
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, do_sample=True, temperature=.7, top_p=.8, top_k=20,
                max_new_tokens=maximum, pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True)
        torch.cuda.synchronize(self.device)
        elapsed = time.monotonic() - started
        ids = generated.sequences[0, prompt_length:].tolist()
        content = self.tokenizer.decode(ids, skip_special_tokens=True)
        self.prompt_tokens += prompt_length
        self.completion_tokens += len(ids)
        eos_ids = self.model.generation_config.eos_token_id
        eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
        finish = "stop" if ids and ids[-1] in eos_ids else "length"
        payload = {"id": f"local-qwen-{self.requests}", "model": self.served_name,
                   "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                "finish_reason": finish}],
                   "usage": {"prompt_tokens": prompt_length, "completion_tokens": len(ids),
                             "total_tokens": prompt_length + len(ids)},
                   "local_raw": {"generation_token_ids": ids,
                                 "prompt_token_ids": inputs["input_ids"][0].tolist(),
                                 "generation_elapsed_seconds": elapsed,
                                 "forward_attempts_this_request": self.forward_attempts - before,
                                 "process_peak_allocated_gib_so_far": torch.cuda.max_memory_allocated(self.device) / 2**30,
                                 "process_peak_reserved_gib_so_far": torch.cuda.max_memory_reserved(self.device) / 2**30}}
        return LocalResponse(payload=payload, id=payload["id"], model=self.served_name,
                             choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)],
                             usage=SimpleNamespace(**payload["usage"]))

    def cost_summary(self):
        torch = self.torch
        torch.cuda.synchronize(self.device)
        return {"backend": "same_process_local_hf", "versions": self.versions,
                "model_path": self.model_path, "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "loaded_model_num_parameters": sum(parameter.numel() for parameter in self.model.parameters()),
                "load_elapsed_seconds": self.load_elapsed_seconds,
                "forward_attempts": self.forward_attempts, "forward_completed": self.forward_completed,
                "successful_prompt_tokens": self.prompt_tokens, "successful_completion_tokens": self.completion_tokens,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(self.device) / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(self.device) / 2**30,
                "memory_scope": "process allocator high-water marks; includes model loading, never reset per request",
                "judge_rng_seed": self.seed, "rng_policy": "torch CPU/CUDA seeded once after loading; sequential official request order",
                "context_length": self.context_length, "enable_thinking": False,
                "independent_model_claim": False}
