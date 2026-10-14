"""Necessary projection and intervention checks on existing non-test data."""
import argparse
import copy
import json
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from src.module3 import build_projections, GatedAttentionIntervention


def graph(action_gate=0.8, status="applicable", conflict="disagreement"):
    return {"schema_version": "module1.v1", "applicability": [{"rule_id": "r1", "status": status, "decisive_condition_ids": ["c1"]}],
            "condition_evidence_relations": [{"rule_id":"r1", "condition_id":"c1",
                "status":"satisfied", "confidence": .9, "evidence":[{"text":"synthetic control"}]}],
            "action_gate": {"r1": action_gate}, "binding_gate": {"r1": {"c1": 0.8}},
            "conflict_scores": [{"rule_id": "r1", "status": conflict, "score": 1.,
                                 "prior_support": "oppose" if conflict == "disagreement" else "unknown",
                                 "guided_support": "support" if conflict == "disagreement" else "unknown"}],
            "relation_weights": [{"rule_id": "r1", "condition_id": "c1", "weight": .8,
                                  "binding_gate": .8, "action_gate": action_gate}]}


def mapping(length):
    return {"schema_version": "pipeline_tokens.v1", "prompt_length": length,
            "rules": {"r1": {"action_tokens": [3, 4]}},
            "relations": [{"rule_id": "r1", "condition_id": "c1", "condition_tokens": [1, 2],
                           "evidence_tokens": [5, 6], "action_tokens": [3, 4]}]}


def checks(model, encoded, components, projection):
    model.eval()
    length = encoded["input_ids"].shape[-1]
    tm = mapping(length)
    with torch.no_grad():
        base = model(**encoded, use_cache=False).logits.float()
        with GatedAttentionIntervention(model, components, projection, graph(), tm) as active:
            edited = model(**encoded, use_cache=False).logits.float()
        with GatedAttentionIntervention(model, components, projection, graph(), tm, context_strength=0., prior_strength=0.) as neutral:
            adapted = model(**encoded, use_cache=False).logits.float()
        tests = {"zero_strength_adapter_max_logit_delta":float((adapted-base).abs().max()),
                 "zero_strength_adapter_exact":bool(torch.equal(adapted,base)),
                 "active_max_logit_delta": float((edited - base).abs().max()), "active_stats": active.stats,
                 "control_gate_semantics": "module1.v1 synthetic controls; no artificial clinical conflict labels"}
        assert tests["active_max_logit_delta"] > tests["zero_strength_adapter_max_logit_delta"]
        assert active.stats["context_changed_elements"] > 0
        assert active.stats["prior_changed_elements"] > 0
        for name, value in [("zero_gate", graph(0.)), ("inapplicable", graph(.8, "inapplicable")),
                            ("unknown_applicability", graph(.8, "unknown"))]:
            with GatedAttentionIntervention(model, components, projection, value, tm) as inactive:
                actual = model(**encoded, use_cache=False).logits.float()
            exact = bool(torch.equal(base, actual))
            tests[name] = {"exact_logits_equal": exact, "stats": inactive.stats}
            assert exact and inactive.stats["context_changed_elements"] == 0 and inactive.stats["prior_changed_elements"] == 0
        with GatedAttentionIntervention(model, components, projection, graph(conflict="unknown"), tm) as uncertain:
            model(**encoded, use_cache=False)
        tests["unknown_conflict"] = uncertain.stats
        assert uncertain.stats["prior_changed_elements"] == 0
        low = graph()
        low["condition_evidence_relations"][0]["confidence"] = .4
        with GatedAttentionIntervention(model, components, projection, low, tm) as low_confidence:
            model(**encoded, use_cache=False)
        tests["low_confidence_conflict"] = low_confidence.stats
        assert low_confidence.stats["prior_changed_elements"] == 0
        # Direct tensor check verifies expanded-key edit leaves every unselected query head exact.
        probe = torch.randn(1, model.config.num_attention_heads, length, model.config.head_dim,
                            device=model.device, dtype=next(model.parameters()).dtype)
        l, h = components["selected_heads"][0]["layer"], components["selected_heads"][0]["head"]
        operation = GatedAttentionIntervention(model, components, projection, graph(), tm)
        result = operation._edit_keys(probe, l, [h], None)
        unselected = [x for x in range(model.config.num_attention_heads) if x != h]
        tests["gqa_unselected_query_keys_exact"] = bool(torch.equal(probe[:, unselected], result[:, unselected]))
        assert tests["gqa_unselected_query_keys_exact"]
        with GatedAttentionIntervention(model, components, projection, graph(), tm) as generation:
            output = model.generate(**encoded, max_new_tokens=12, do_sample=False, use_cache=True)
        tests["cached_generation_tokens"] = output[0, length:].tolist()
        tests["cached_generation_stats"] = generation.stats
        restored = model(**encoded, use_cache=False).logits.float()
        tests["restored_exact_logits_equal"] = bool(torch.equal(base, restored))
        assert tests["restored_exact_logits_equal"]
    return tests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--components", default="results/module2/components.json")
    args = parser.parse_args()
    out = Path("results/module3")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if args.cpu:
        torch.manual_seed(5)
        config = Qwen3Config(vocab_size=80, hidden_size=64, intermediate_size=128,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                            head_dim=16, max_position_embeddings=128, pad_token_id=0,
                            bos_token_id=1, eos_token_id=2, attention_dropout=0.)
        model = Qwen3ForCausalLM(config).eval()
        components = {"selected_heads": [{"layer": 1, "head": 2}]}
        basis = torch.eye(16)[:, :4]
        projection = {"model_config": {}, "components": {"1:2": {"layer": 1, "head": 2,
                       "key_basis": basis, "prior_basis": basis}}}
        encoded = {"input_ids": torch.tensor([[1, 4, 7, 9, 3, 2, 6, 8]]),
                   "attention_mask": torch.ones(1, 8, dtype=torch.long)}
        report = checks(model, encoded, components, projection)
        report["scope"] = "tiny random Qwen3 CPU implementation checks, not clinical results"
        report["elapsed_seconds"] = time.time() - started
        (out / "cpu_checks.json").write_text(json.dumps(report, indent=2))
        print("CPU checks passed")
        return
    from src.module2.data import build_calibration_pairs
    components = json.loads(Path(args.components).read_text())
    if not components["selected_heads"]:
        raise ValueError("No causally beneficial components selected; report module2 outcome instead of fallback")
    path = "/home/gcx/TRIAGE/models/modelscope/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
             torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0").eval()
    from src.pipeline.tokens import build_prompt_and_mapping
    samples = json.loads(Path("results/module1_smoke.json").read_text())["samples"][:1]
    records = []
    for sample in samples:
        prompt, _, token_map = build_prompt_and_mapping(tokenizer, sample["patient"], sample["question"], sample["result"])
        records.append({"text": prompt, "source": sample["input_source"], "split": "train_val",
                        "sample_id": sample["input_source"]["guideline_id"],
                        "token_indices":sorted({i for r in token_map["relations"] for key in ["condition_tokens","evidence_tokens","action_tokens"] for i in r[key]})})
    projection_path = out / "projections.pt"
    build_projections(model, tokenizer, records, components, projection_path, rank=4)
    prompt = records[0]["text"]
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384)
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    report = checks(model, encoded, components, projection_path)
    report.update({"model": path, "calibration_sources": [r["source"] for r in records],
                   "scope": "real Qwen3 model computation checks with synthetic gates on train_val text",
                   "elapsed_seconds": time.time() - started,
                   "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30})
    sample = samples[0]
    prompt, actual_encoded, token_mapping = build_prompt_and_mapping(tokenizer, sample["patient"], sample["question"], sample["result"])
    actual_encoded = {k:v.to(model.device) for k,v in actual_encoded.items()}
    with torch.inference_mode():
        original = model(**actual_encoded, use_cache=False).logits[0,-1].float().cpu()
        with GatedAttentionIntervention(model, components, projection_path, sample["result"], token_mapping) as live:
            changed = model(**actual_encoded, use_cache=False).logits[0,-1].float().cpu()
    report["actual_module1_control"] = {"module1_source": "results/module1_smoke.json",
           "input_source":sample["input_source"], "stats":live.stats,
           "max_abs_next_token_logit_delta":float((changed-original).abs().max()),
           "mapped_relation_count":len(token_mapping["relations"])}
    assert live.stats["context_changed_elements"] > 0
    assert live.stats["prior_changed_elements"] == 0
    (out / "gpu_checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("GPU checks passed")

if __name__ == "__main__":
    main()
