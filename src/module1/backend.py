"""Offline, shared Qwen3 and MedCPT inference; paths come from models.json."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class LocalBackend:
    def __init__(self, config_path=None, device="cuda:0", max_input_tokens=24576):
        config = json.loads(Path(config_path or ROOT / "configs/models.json").read_text())
        models = config["models"]
        self.model_path = models["qwen3_8b"]["path"]
        self.medcpt_path = models["medcpt_medpair"]["path"]
        self.cache_identity = {"model": self.model_path, "revision": models["qwen3_8b"].get("revision"), "parser_version": 1}
        self.device = device
        self.max_input_tokens = max_input_tokens
        self.model = self.tokenizer = self.scorer = self.score_tokenizer = None
        self.calls = []

    def _load_qwen(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa").to(self.device).eval()
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

    def json(self, system, user, max_new_tokens=1536):
        import time
        import torch
        from .trialgpt_matching import parse_json_response
        self._load_qwen()
        started = time.time()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for attempt in range(2):
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            encoded = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            if encoded["input_ids"].shape[1] > self.max_input_tokens:
                raise ValueError("Qwen input exceeds configured token budget; narrow guideline candidates or patient input")
            with torch.inference_mode():
                generated = self.model.generate(**encoded, do_sample=False,
                    max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
            response = self.tokenizer.decode(generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
            self.calls.append({"input_tokens": encoded["input_ids"].shape[1],
                "output_tokens": generated.shape[1] - encoded["input_ids"].shape[1],
                "seconds": round(time.time() - started, 3), "attempt": attempt})
            try:
                return parse_json_response(response)
            except (ValueError, json.JSONDecodeError):
                if attempt:
                    raise ValueError("Qwen did not return valid JSON after one retry")
                messages += [{"role": "assistant", "content": response},
                    {"role": "user", "content": "Return a complete valid JSON object only, matching the requested schema. Be concise."}]

    def score_pairs(self, question, texts):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        if self.scorer is None:
            self.score_tokenizer = AutoTokenizer.from_pretrained(self.medcpt_path, local_files_only=True)
            self.scorer = AutoModelForSequenceClassification.from_pretrained(
                self.medcpt_path, local_files_only=True).to(self.device).eval()
        scores = []
        for offset in range(0, len(texts), 16):
            # Cross-encoder max length is 512; long条款 are covered in overlapping windows.
            pairs, owners = [], []
            for idx, text in enumerate(texts[offset:offset + 16]):
                ids = self.score_tokenizer.encode(text, add_special_tokens=False)
                query_ids = self.score_tokenizer.encode(question, add_special_tokens=False)
                # Cover the query too: appended patient evidence must not silently
                # disappear behind the old first-128-token truncation.
                for qpos in range(0, max(len(query_ids), 1), 96):
                    qids = query_ids[qpos:qpos + 128]
                    width = 512 - len(qids) - 3
                    for pos in range(0, max(len(ids), 1), max(width - 64, 1)):
                        pairs.append([self.score_tokenizer.decode(qids), self.score_tokenizer.decode(ids[pos:pos + width])])
                        owners.append(idx)
                        if pos + width >= len(ids):
                            break
                    if qpos + 128 >= len(query_ids):
                        break
            values = [[] for _ in texts[offset:offset + 16]]
            for start in range(0, len(pairs), 16):
                inputs = self.score_tokenizer(pairs[start:start + 16], padding=True,
                    truncation=True, max_length=512, return_tensors="pt").to(self.device)
                with torch.inference_mode():
                    logits = self.scorer(**inputs).logits
                    probs = torch.sigmoid(logits[:, 0]) if logits.shape[-1] == 1 else logits.softmax(-1)[:, -1]
                for owner, value in zip(owners[start:start + 16], probs.cpu().tolist()):
                    values[owner].append(value)
            scores.extend(max(v) for v in values)
        return scores
