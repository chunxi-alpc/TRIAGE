#!/usr/bin/env python3
"""Single short inference for downloaded Qwen/MedCPT/NLI checkpoints. No training."""
import argparse
import json
from pathlib import Path
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--models", nargs="+", default=["qwen3_1_7b", "medcpt", "deberta_nli"])
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()
config = json.loads((ROOT / "configs/models.json").read_text())["models"]
result = {}
for key in args.models:
    started = time.time()
    checkpoint = config.get(key, {})
    if checkpoint.get("status") != "downloaded":
        result[key] = {"status": "skipped", "reason": "Checkpoint not downloaded"}
        continue
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint["path"], local_files_only=True)
        if checkpoint["task"] == "causal_lm":
            model = AutoModelForCausalLM.from_pretrained(checkpoint["path"], torch_dtype=torch.bfloat16, local_files_only=True).to(args.device).eval()
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": "Reply with only OK."}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=16, do_sample=False)
            result[key] = {"status": "passed", "output": tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)}
        else:
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint["path"], local_files_only=True).to(args.device).eval()
            if checkpoint["task"] == "relevance":
                first, second = "What treats bacterial pneumonia?", "Antibiotics treat bacterial pneumonia."
            else:
                first, second = "The patient has no fever.", "The patient is febrile."
            inputs = tokenizer(first, second, truncation=True, max_length=512, return_tensors="pt").to(args.device)
            with torch.inference_mode():
                logits = model(**inputs).logits.float().cpu()[0]
            result[key] = {"status": "passed", "logits": logits.tolist(), "id2label": model.config.id2label}
            if checkpoint["task"] == "nli":
                result[key]["probabilities"] = logits.softmax(-1).tolist()
        result[key]["seconds"] = round(time.time() - started, 2)
        del model, inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        result[key] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
(ROOT / "results/model_smoke.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(result, ensure_ascii=False))
raise SystemExit(0 if all(v["status"] == "passed" for v in result.values()) else 1)
