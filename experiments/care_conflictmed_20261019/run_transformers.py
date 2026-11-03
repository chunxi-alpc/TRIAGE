#!/usr/bin/env python3
"""A/B exact next-answer log-likelihood baseline. No CARE/steering hooks.

Use a GPU host with pinned model weights and HF Transformers installed. This file
is intentionally separated from the CPU-only data/evaluation tooling.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

from prepare import read_jsonl


def encode_prompt(tokenizer, prompt):
    if getattr(tokenizer, "chat_template", None):
        # Qwen3's thinking channel is unsuitable for first-token A/B scoring.
        try:
            ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                tokenize=True, add_generation_prompt=True,
                                                enable_thinking=False)
            return ids, "chat_thinking_disabled"
        except TypeError:
            raise RuntimeError("Chat template does not support enable_thinking=False; do not score hidden thinking tokens")
    return tokenizer.encode(prompt, add_special_tokens=True), "plain_text"


def option_score(model, tokenizer, prompt_ids, answer):
    import torch
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    if not answer_ids:
        raise ValueError("Empty candidate tokenization")
    # Full continuation score, not a single-token heuristic. Character prefixes
    # are isolated by the chat assistant boundary (or explicit 'Answer:' header).
    inputs = torch.tensor([prompt_ids + answer_ids], device=model.device)
    with torch.inference_mode():
        logits = model(input_ids=inputs).logits[0]
    start = len(prompt_ids)
    log_probs = logits[start - 1:start + len(answer_ids) - 1].log_softmax(dim=-1)
    score = float(log_probs.gather(-1, torch.tensor(answer_ids, device=log_probs.device).unsqueeze(1)).sum())
    return score, len(answer_ids)


def inference(model, tokenizer, item):
    prompt_ids, template = encode_prompt(tokenizer, item["prompt"])
    values = {letter: option_score(model, tokenizer, prompt_ids, letter) for letter in ("A", "B")}
    if values["A"][1] != values["B"][1]:
        raise RuntimeError("A/B candidate token lengths differ; re-evaluate score normalization")
    pred = max(values, key=lambda x: values[x][0])
    return {"id": item["id"], "method": "no_intervention_forced_choice", "prediction": pred,
            "log_likelihood": {k: round(v[0], 8) for k, v in values.items()},
            "prompt_tokens": len(prompt_ids), "template": template,
            "scoring": "candidate_continuation_log_likelihood"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--model", required=True, help="Local verified model dir/revision; no model fetch intended")
    p.add_argument("--split", choices=["train", "dev", "test"], required=True)
    p.add_argument("--track", choices=["advice_choice", "scenario_judgment"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, help="Explicitly exploratory incomplete smoke run")
    p.add_argument("--device-map", default="auto")
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError("Refuse to overwrite predictions")
    if a.limit is not None and a.limit < 1:
        raise ValueError("--limit must be positive")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable. Run on authorized GPU server; do not silently switch to CPU")
    rows = [r for r in read_jsonl(a.inputs) if r["split"] == a.split and r["track"] == a.track]
    if a.limit:
        rows = rows[:a.limit]
    if not rows:
        raise ValueError("No matching rows")
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(a.model, device_map=a.device_map,
                                                torch_dtype="auto", local_files_only=True,
                                                trust_remote_code=False).eval()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    temp = a.out.with_suffix(a.out.suffix + ".incomplete")
    if temp.exists():
        raise FileExistsError("Incomplete output exists; recover explicitly")
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with temp.open("w", encoding="utf-8") as f:
        for n, item in enumerate(rows):
            output = inference(model, tokenizer, item)
            f.write(json.dumps(output) + "\n")
            f.flush()
            if (n + 1) % 50 == 0:
                print(f"{n+1}/{len(rows)}", flush=True)
    temp.replace(a.out)
    blob = a.out.read_bytes()
    contract = {"status": "completed_smoke" if a.limit else "completed_selected_split_track",
                "method": "no_intervention_forced_choice", "n": len(rows), "track": a.track,
                "split": a.split, "model_path": a.model, "pinned_model_hash": None,
                "device_map": a.device_map, "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
                "torch": torch.__version__, "transformers": __import__("transformers").__version__,
                "started_utc": started, "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "predictions_sha256": hashlib.sha256(blob).hexdigest(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    a.out.with_suffix(a.out.suffix + ".manifest.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
