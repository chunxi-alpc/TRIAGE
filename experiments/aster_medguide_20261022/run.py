"""Llama-3.1-8B-Instruct ASTER counterfactual pilot: real inference only.

This uses direct attention-score bias, NOT SEKA or a trained projection.
Oracle rule identities are intervention masks only, except the separately
named explicit-rule-prompt control. All generated answers are model logits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import ALL_ATTENTION_FUNCTIONS, repeat_kv


ARMS = ("baseline", "whole", "changed", "positive", "unsigned", "signed", "reversed", "explicit")
SYSTEM = "Answer the question using any supplied guideline rules. Choose exactly one option letter."
BASE = "Patient:\n{patient}\n\nQuestion:\n{question}\n\nOptions:\n{choices}\n\nAnswer:"
GUIDE_PREFIX = "Clinical guideline rules:\n"


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def atomic_manifest(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_layers(text, total):
    if "-" in text:
        lo, hi = [int(part) for part in text.split("-", 1)]
        layers = list(range(lo, hi + 1))
    else:
        layers = [int(part) for part in text.split(",") if part.strip()]
    if not layers or len(layers) != len(set(layers)) or min(layers) < 0 or max(layers) >= total:
        raise ValueError(f"Invalid selected layers {layers}; model has {total} layers")
    return layers


def char_mask(offsets, span, device):
    if span is None:
        return torch.zeros(len(offsets), dtype=torch.bool, device=device)
    a, b = span
    return torch.tensor([y > a and x < b and y > x for x, y in offsets], dtype=torch.bool, device=device)


class Engine:
    def __init__(self, model_path, device, layers_spec):
        self.device = device
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
        if not self.tokenizer.is_fast:
            raise RuntimeError("A fast tokenizer with offset_mapping is required.")
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, local_files_only=True, attn_implementation="sdpa"
        ).to(device).eval()
        self.layers = parse_layers(layers_spec, self.model.config.num_hidden_layers)
        self.option_ids = {}
        self.option_encoding = None
        self.current = None
        self.diagnostics = []
        self.masks = {}
        # Register precisely the same interface the model's LlamaAttention uses
        # in transformers 4.57.x. The original kernel is retained for all
        # unselected queries and layers.
        self._sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]
        ALL_ATTENTION_FUNCTIONS.register("sdpa", self._attn)
        self.provenance = {
            "model": model_path, "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "dtype": str(self.model.dtype),
            "num_layers": self.model.config.num_hidden_layers,
            "selected_layers": self.layers,
            "device": device,
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "method": "direct_post_rope_attention_score_bias_last_query",
            "source_paper_methods": "not an official SEKA or PASTA implementation",
        }

    def _attn(self, module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        out, weights = self._sdpa(
            module, query, key, value, attention_mask,
            scaling=scaling, dropout=dropout, **kwargs
        )
        if self.current is None or module.layer_idx not in self.layers:
            return out, weights
        if query.shape[0] != 1 or query.shape[-2] != key.shape[-2]:
            raise RuntimeError("Experiment supports batch=1, no KV cache, full forward only.")
        k = repeat_kv(key, module.num_key_value_groups)
        v = repeat_kv(value, module.num_key_value_groups)
        before = torch.matmul(query[:, :, -1:, :].float(), k.float().transpose(-2, -1)) * scaling
        if attention_mask is not None:
            before = before + attention_mask[:, :, -1:, : key.shape[-2]].float()
        # For the final prefill query, all preceding keys are causally visible.
        p0 = torch.softmax(before, dim=-1)
        signed = self.current["signs"]
        assert signed.numel() == before.shape[-1]
        strength = float(self.current["strength"])
        if self.current.get("force_manual") or (strength != 0 and bool(torch.any(signed != 0))):
            after = before + strength * signed[None, None, None, :]
            p1 = torch.softmax(after, dim=-1)
            edited = torch.matmul(p1.to(v.dtype), v).transpose(1, 2).contiguous()
            out = out.clone()
            out[:, -1:, :, :] = edited
        else:
            p1 = p0
        a, b = self.masks["app"], self.masks["sib"]
        def mass(p, mask):
            return float(p[:, :, :, mask].sum(dim=-1).mean().item()) if bool(mask.any()) else 0.0
        self.diagnostics.append({
            "layer": module.layer_idx,
            "app_rule_mass_before": mass(p0, a),
            "sibling_rule_mass_before": mass(p0, b),
            "app_rule_mass_after": mass(p1, a),
            "sibling_rule_mass_after": mass(p1, b),
            "app_rule_tokens": int(a.sum().item()),
            "sibling_rule_tokens": int(b.sum().item()),
            "attention_l1_change": float((p1 - p0).abs().sum(-1).mean().item()),
            "edited_last_query_only": True,
        })
        return out, weights

    def option_token_ids(self, count):
        if count in self.option_ids:
            return self.option_ids[count]
        options = [chr(ord("A") + i) for i in range(count)]
        for prefix in (" ", ""):
            tokens = [self.tokenizer.encode(prefix + option, add_special_tokens=False) for option in options]
            if all(len(item) == 1 for item in tokens) and len({item[0] for item in tokens}) == count:
                self.option_encoding = f"{prefix}<option-letter>"
                ids = [item[0] for item in tokens]
                self.option_ids[count] = ids
                return ids
        raise RuntimeError("Option letters do not have unique single-token continuations.")

    def prompt_and_targets(self, case, variant, explicit=False, closedbook=False):
        version = case["versions"][variant]
        choices = "\n".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(case["options"]))
        user = ""
        if not closedbook:
            user = GUIDE_PREFIX + version["guideline"] + "\n"
        if explicit:
            user += "For this controlled example, R_app applies and R_sib does not apply.\n\n"
        user += BASE.format(patient=case["patient"], question=case["question"], choices=choices)
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = self.tokenizer(
            prompt, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        inp = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        spans = version["spans"]
        if closedbook:
            app = sib = whole = modified = char_mask(offsets, None, self.device)
        else:
            guide_start = prompt.find(version["guideline"])
            if guide_start < 0 or prompt.count(version["guideline"]) != 1:
                raise RuntimeError("Rendered guideline missing or repeated in chat template.")
            def selected(rid):
                s, e = spans[rid]["rule"]
                return char_mask(offsets, [guide_start + s, guide_start + e], self.device)
            app, sib = selected("R_app"), selected("R_sib")
            shared = app & sib
            app, sib = app & ~shared, sib & ~shared
            whole = char_mask(
                offsets, [guide_start, guide_start + len(version["guideline"])], self.device
            )
            changed_rule = version["modified_rule"]
            modified = selected(changed_rule) if changed_rule else char_mask(offsets, None, self.device)
            if not bool(app.any()) or not bool(sib.any()):
                raise RuntimeError("Failed to map both rule spans to input token indices.")
        masks = {"app": app, "sib": sib, "whole": whole, "modified": modified}
        return prompt, inp, masks

    def score(self, case, variant, arm, strength=1.0):
        explicit = arm == "explicit"
        closedbook = arm == "closedbook"
        prompt, inp, masks = self.prompt_and_targets(case, variant, explicit, closedbook)
        self.masks = masks
        zeros = torch.zeros(inp["input_ids"].shape[-1], dtype=torch.float32, device=self.device)
        app, sib = masks["app"], masks["sib"]
        signs = zeros.clone()
        if arm == "whole":
            signs[masks["whole"]] = 1.0
        elif arm == "changed":
            signs[masks["modified"]] = 1.0
        elif arm == "positive":
            signs[app] = 1.0
        elif arm == "unsigned":
            signs[app | sib] = 1.0
        elif arm == "signed":
            signs[app] = 1.0
            signs[sib] = -1.0
        elif arm == "reversed":
            signs[app] = -1.0
            signs[sib] = 1.0
        elif arm not in ("baseline", "manual_zero", "explicit", "closedbook"):
            raise ValueError(arm)
        self.current = {"signs": signs, "strength": strength if bool(signs.any()) else 0.0,
                        "force_manual": arm == "manual_zero"}
        self.diagnostics = []
        now = time.monotonic()
        try:
            with torch.inference_mode():
                full = self.model(**inp, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            logits = full[self.option_token_ids(len(case["options"]))].detach().cpu().tolist()
            best = max(range(len(logits)), key=lambda i: logits[i])
            gold = ord(case["versions"][variant]["gold"]) - ord("A")
            return {
                "status": "completed", "prediction": chr(65 + best),
                "gold": case["versions"][variant]["gold"],
                "correct": best == gold, "candidate_logits": logits,
                "margin": logits[gold] - max(x for i, x in enumerate(logits) if i != gold),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt_tokens": int(inp["input_ids"].shape[-1]),
                "edited_token_count": int((signs != 0).sum().item()),
                "oracle_intervention": arm in ("positive", "unsigned", "signed", "reversed", "explicit"),
                "attention": self.diagnostics,
                "duration_sec": time.monotonic() - now,
            }
        finally:
            self.current = None
            self.masks = {}


def iterate_conditions(strengths):
    conditions = [("baseline", 0.0), ("manual_zero", 0.0), ("whole", 1.0), ("changed", 1.0),
                  ("positive", 1.0), ("unsigned", 1.0), ("explicit", 0.0)]
    conditions.extend(("signed", x) for x in strengths)
    conditions.append(("reversed", 1.0))
    return conditions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default=os.environ.get("ASTER_MODEL", "meta-llama/Llama-3.1-8B-Instruct"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layers", default="16-21")
    ap.add_argument("--strengths", default="0.5,1,2,4")
    ap.add_argument("--max-cases", type=int, default=0, help="0 = all prepared cases")
    args = ap.parse_args()
    strengths = [float(v) for v in args.strengths.split(",") if v.strip()]
    if not strengths or any(v <= 0 for v in strengths):
        ap.error("Supply positive comma-separated --strengths")
    if args.model.startswith("meta-llama/"):
        ap.error("ASTER_MODEL must point to locally downloaded, authorized model files.")
    if not Path(args.model).exists():
        ap.error(f"Model path does not exist: {args.model}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = list(read_jsonl(args.input))
    if args.max_cases:
        cases = cases[:args.max_cases]
    already = {}
    if args.output.exists():
        already = {x["case_id"]: x for x in read_jsonl(args.output)}
    engine = Engine(args.model, args.device, args.layers)
    manifest = {
        **engine.provenance, "status": "running", "cases_requested": len(cases),
        "strengths": strengths, "option_encoding": None, "intervention_conditions": iterate_conditions(strengths),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    atomic_manifest(manifest_path, manifest)
    with args.output.open("a", encoding="utf-8", buffering=1) as sink:
        for case in cases:
            if case["case_id"] in already:
                continue
            record = {
                "case_id": case["case_id"], "family": case["family"],
                "original_answer": case["original_answer"],
                "sibling_answer": case["sibling_answer"],
                "replacement_answer": case["replacement_answer"],
                "variants": {},
            }
            try:
                record["closedbook"] = engine.score(case, "A", "closedbook", 0.0)
                for variant in "ABC":
                    arms = {}
                    for arm, strength in iterate_conditions(strengths):
                        name = arm if arm != "signed" else f"signed@{strength:g}"
                        try:
                            arms[name] = engine.score(case, variant, arm, strength)
                        except Exception as exc:
                            arms[name] = {"status": "failed", "error": repr(exc)}
                            traceback.print_exc()
                            if args.device.startswith("cuda"):
                                torch.cuda.empty_cache()
                    record["variants"][variant] = arms
            except Exception as exc:
                record["error"] = repr(exc)
                traceback.print_exc()
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest["status"] = "completed"
    manifest["cases_recorded"] = len(list(read_jsonl(args.output)))
    manifest["option_encoding"] = engine.option_encoding
    if args.device.startswith("cuda"):
        manifest["cuda_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        manifest["cuda_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    atomic_manifest(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "cases": manifest["cases_recorded"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
