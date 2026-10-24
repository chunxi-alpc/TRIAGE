"""Real Qwen3 joint attention backend; PyTorch is required only at construction.

The attention adapters have not been validated by a real model merely because
the CPU data/evaluator tests pass. The run journal records that distinction.
"""
from __future__ import annotations

import contextlib
import gzip
import hashlib
import importlib
import importlib.metadata
import io
import json
import math
import os
import sys
from pathlib import Path


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_zero_outputs(original, zero):
    """Exact task-output comparison; a logit shift must not hide behind softmax."""
    audit = {"comparison_policy": "exact equality; no floating-point tolerance",
             "candidate_count": 4, "does_not_validate_nonzero_or_learned_hooks": True}
    passed = True
    for field, label in (("candidate_logits", "candidate_logit"),
                         ("candidate_probabilities", "candidate_probability")):
        left, right = original[field], zero[field]
        if len(left) != 4 or len(right) != 4 or not all(math.isfinite(x) for x in left + right):
            raise ValueError("Invalid candidate outputs in actual-model zero control")
        difference = max(abs(a - b) for a, b in zip(left, right))
        audit[label + "_max_difference"] = difference
        audit[label + "_exactly_equal"] = left == right
        passed = passed and left == right
    audit["status"] = "passed" if passed else "failed_output_mismatch"
    return audit


class ModelCompatibilityError(ValueError):
    def __init__(self, message, audit):
        super().__init__(message)
        self.compatibility_audit = audit


class LearnedAssets:
    """Verified, explicitly learned operators; no random or identity fallback."""
    def __init__(self, path, confirmation_families):
        self.path = Path(path)
        self.manifest = json.loads(self.path.read_text())
        data = self.manifest
        if data.get("artifact_type") != "learned_applicability_subspace" or data.get("method") != "B4":
            raise ValueError("Only explicitly identified learned B4 assets are accepted")
        if not data.get("training_family_ids") or not data.get("source_repository_commit"):
            raise ValueError("Learned assets lack training provenance")
        used = set(data["training_family_ids"]) | set(data.get("selection_family_ids", []))
        if used & set(confirmation_families) or data.get("confirmation_family_ids_used"):
            raise ValueError("Learned assets or their selection used confirmation families")
        if data.get("model_revision") != "906bfd4b4dc7f14ee4320094d8b41684abff8539":
            raise ValueError("Learned operator model revision differs from the frozen Qwen3 experiment")
        for fold in ("0", "1", "2"):
            entry = data["folds"][fold]
            operator_file = self.path.parent / entry["operator_path"]
            if hashlib.sha256(operator_file.read_bytes()).hexdigest() != entry["operator_sha256"]:
                raise ValueError(f"Operator hash mismatch in fold {fold}")
        self.loaded = {}

    def get(self, fold, torch, expected_location):
        fold = str(fold)
        if fold in self.loaded:
            return self.loaded[fold]
        entry = self.manifest["folds"][fold]
        if tuple(entry["location"]) != tuple(expected_location):
            raise ValueError("Learned operator location differs from frozen direct-control location")
        path = self.path.parent / entry["operator_path"]
        content = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
        loaded = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
        operator = loaded[entry["operator_key"]] if entry.get("operator_key") else loaded
        operator = operator.float()
        if tuple(operator.shape) != (128, 128) or not torch.isfinite(operator).all():
            raise ValueError("Invalid learned operator dimensions or values")
        if not torch.allclose(operator, operator.T, atol=1e-5, rtol=1e-5):
            raise ValueError("B4 operator is not symmetric")
        if not torch.allclose(operator @ operator, operator, atol=2e-4, rtol=2e-4):
            raise ValueError("B4 operator is not an orthogonal projector")
        self.loaded[fold] = operator.numpy()
        return self.loaded[fold]


class QwenJointBackend:
    def __init__(self, model_path, device, protocol, learned_assets=None):
        model_path = Path(model_path).resolve()
        if not model_path.is_dir():
            raise ValueError("An explicit existing local model directory is required")
        from module2.compatibility import load_frozen_manifest, config_audit, environment_audit
        reference_path = Path(__file__).resolve().parents[2] / "care_qwen3_4b_base_20261016/results/manifest.json"
        original = load_frozen_manifest(reference_path, protocol)
        actual_weights = {path.name: file_digest(path) for path in sorted(model_path.glob("*.safetensors"))}
        expected_weights = {key: value["sha256"] for key, value in original["weights"].items()}
        if not actual_weights or actual_weights != expected_weights:
            raise ValueError("Local model shards do not match the frozen Qwen3-4B-Base weight SHA256 values")
        observed_versions = {}
        for name in original["versions"]:
            try:
                observed_versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                observed_versions[name] = None
        self.compatibility_audit = {
            "environment": environment_audit(observed_versions, original, protocol),
            "weight_shards_match_frozen_reference": True,
        }
        if not self.compatibility_audit["environment"]["compatible"]:
            raise ModelCompatibilityError("Runtime versions differ from the frozen model contract", self.compatibility_audit)
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        resolved_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        self.compatibility_audit["before_load"] = config_audit(resolved_config.to_dict(), original, protocol)
        if not self.compatibility_audit["before_load"]["compatible"]:
            raise ModelCompatibilityError("Resolved model configuration differs from the frozen model", self.compatibility_audit)
        self.torch, self.protocol = torch, protocol
        self.learned_assets = learned_assets
        self.forward_calls = 0
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation=protocol["attention"]).to(device).eval()
        config = self.model.config
        self.compatibility_audit["after_load"] = config_audit(config.to_dict(), original, protocol)
        if not self.compatibility_audit["after_load"]["compatible"]:
            raise ModelCompatibilityError("Loaded model configuration differs from the frozen model", self.compatibility_audit)
        attention = getattr(config, "_attn_implementation", None)
        dtype = str(self.model.dtype)
        self.compatibility_audit["attention_and_dtype"] = {
            "compatible": attention == "sdpa" and dtype == "torch.bfloat16",
            "expected_attention": "sdpa", "observed_attention": attention,
            "expected_dtype": "torch.bfloat16", "observed_dtype": dtype,
        }
        if not self.compatibility_audit["attention_and_dtype"]["compatible"]:
            raise ModelCompatibilityError("Loaded attention implementation or dtype differs from the frozen model", self.compatibility_audit)
        self.provenance = {
            "model_id": protocol["model_id"], "model_revision": protocol["model_revision"],
            "model_path": str(model_path), "weight_sha256": actual_weights,
            "weight_shards_match_frozen_reference": True,
            "compatibility_audit": self.compatibility_audit,
            "reference_manifest_sha256": file_digest(reference_path),
            "config_sha256": file_digest(model_path / "config.json"),
            "tokenizer_file_sha256": {path.name: file_digest(path) for path in sorted(model_path.glob("*token*")) if path.is_file()},
            "tokenizer_configuration": {"class": type(self.tokenizer).__name__, "is_fast": self.tokenizer.is_fast,
                                        "padding_side": self.tokenizer.padding_side, "truncation_side": self.tokenizer.truncation_side,
                                        "model_max_length": self.tokenizer.model_max_length,
                                        "bos_token_id": self.tokenizer.bos_token_id, "eos_token_id": self.tokenizer.eos_token_id},
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers")},
            "device": str(self.model.device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_version": torch.version.cuda, "model_instance_shared_across_all_arms": True,
        }
        if self.model.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.model.device)
            self.provenance["gpu"] = {"name": props.name, "total_memory_bytes": props.total_memory,
                                       "compute_capability": [props.major, props.minor]}
        self.option_ids = []
        for letter in "ABCD":
            encoded = self.tokenizer.encode(" " + letter, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError("A/B/C/D candidates must be single tokens")
            self.option_ids.append(encoded[0])
        self.geometry_cls = None

    def _geometry(self):
        if self.geometry_cls is None:
            module2 = Path(__file__).resolve().parents[1] / "module2"
            sys.path.insert(0, str(module2))
            self.geometry_cls = importlib.import_module("backend").InterventionGeometry
        return self.geometry_cls

    def encode(self, prompt, targets):
        torch = self.torch
        encoded = self.tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoded.pop("offset_mapping")[0].tolist()
        n = len(offsets)
        if n > 2048:
            raise ValueError("Prompt exceeds the prospectively fixed 2048-token limit; truncation is prohibited")
        mapped, seen = [], set()
        for target in targets:
            start, end = target["span"]
            indices = [i for i, (left, right) in enumerate(offsets) if right > start and left < end and right > left]
            if not indices:
                raise ValueError("Action span has no tokenizer support")
            # A token can have a leading space. It cannot extend into another
            # word, candidate option, or another action's intervention support.
            if prompt[offsets[indices[0]][0]:start].strip() or prompt[end:offsets[indices[-1]][1]].strip():
                raise ValueError("Action span crosses a non-whitespace token boundary")
            if seen & set(indices):
                raise ValueError("Two action spans share intervention tokens")
            seen.update(indices)
            mapped.append(dict(target, token_indices=indices))
        device = self.model.device
        inputs = {"input_ids": encoded["input_ids"].to(device)}
        minimum = torch.finfo(self.model.dtype).min
        mask = torch.full((n, n), minimum, dtype=self.model.dtype, device=device).triu(1)
        inputs["attention_mask"] = mask[None, None]
        return inputs, mapped

    def _forward(self, inputs):
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1].float()
        self.forward_calls += 1
        probabilities = logits[self.option_ids].softmax(-1).cpu().tolist()
        vocabulary_mass = logits.softmax(-1)[self.option_ids].sum().item()
        return {"prediction": "ABCD"[max(range(4), key=lambda i: probabilities[i])],
                "candidate_logits": logits[self.option_ids].cpu().tolist(),
                "candidate_probabilities": probabilities, "full_vocabulary_candidate_mass": vocabulary_mass}

    def signed_zero_control(self, prompt, outer_fold):
        original = self.score(prompt, [], "structured_prompt", outer_fold)
        try:
            zero = self.score(prompt, [], "signed_bias", outer_fold)
            audit = compare_zero_outputs(original, zero) | {"zero_forward_attempted": True}
        except Exception as error:
            audit = {"status": "model_error", "zero_forward_attempted": True,
                     "error": type(error).__name__ + ": " + str(error)}
        return original, audit

    @contextlib.contextmanager
    def _direct(self, location, targets, n, rms):
        torch = self.torch
        layer, head = location
        additions = torch.zeros((32, n, n), device=self.model.device, dtype=torch.float32)
        squared_sum, actual_squared_sum, support = 0.0, 0.0, 0
        for target in targets:
            value = target["sign"] * rms * target["magnitude"]
            actual = float(torch.tensor(value, dtype=self.model.dtype))
            for token in target["token_indices"]:
                additions[4*head:4*head+4, token:, token] = value
                count = 4 * (n - token)
                squared_sum += count * value ** 2
                actual_squared_sum += count * actual ** 2
                support += count
        def hook(module, args, kwargs):
            mask = kwargs.get("attention_mask")
            if mask is None or mask.ndim != 4:
                raise ValueError("Expected explicit 4-D causal attention mask")
            kwargs["attention_mask"] = (mask.expand(1, 32, n, n).float() + additions[None]).to(self.model.dtype)
            return args, kwargs
        handle = self.model.model.layers[layer].self_attn.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            yield {"edited_rules": len(targets), "full_causal_support_entries": support,
                   "requested_combined_score_rms": (squared_sum / support) ** .5 if support else 0.0,
                   "actual_combined_score_rms": (actual_squared_sum / support) ** .5 if support else 0.0,
                   "zero_edit": not bool(support)}
        finally:
            handle.remove()

    def score(self, prompt, targets, mode, outer_fold):
        if mode not in {"baseline", "structured_prompt", "signed_bias", "learned_subspace"}:
            raise ValueError(mode)
        frozen = self.protocol["frozen_folds"][str(outer_fold)]
        location, rms = (frozen["layer"], frozen["kv_head"]), frozen["score_rms"]
        inputs, mapped = self.encode(prompt, targets)
        n = inputs["input_ids"].shape[1]
        if mode in {"baseline", "structured_prompt"}:
            return self._forward(inputs) | {"intervention": {"mode": mode, "zero_edit": True}, "prompt_tokens": n}
        if mode == "signed_bias":
            with self._direct(location, mapped, n, rms) as audit:
                result = self._forward(inputs)
            return result | {"intervention": audit | {"mode": mode, "location": location}, "prompt_tokens": n}
        if self.learned_assets is None:
            raise FileNotFoundError("Learned subspace assets were not supplied")
        operator = self.learned_assets.get(outer_fold, self.torch, location)
        # Queries and keys come from this same complete prompt, including this
        # source's structure. This single-layer capture is never a donor case.
        capture, handles = {location[0]: {}}, []
        module = self.model.model.layers[location[0]].self_attn
        for kind in ("q", "k"):
            def save(module, args, out, kind=kind):
                capture[location[0]][kind] = out[0].detach().clone()
            handles.append(getattr(module, kind + "_norm").register_forward_hook(save))
        try:
            self._forward(inputs)
        finally:
            for handle in handles:
                handle.remove()
        edits, audits, combined_sum, combined_support = [], [], 0.0, 0
        for target in mapped:
            geometry = self._geometry()(self.model, capture, target["token_indices"], location, self.protocol)
            changed, audit = geometry.key_intervention("b4", "calibrated", {"b4": operator},
                                                       target["sign"], rms * target["magnitude"])
            edits.append((target["token_indices"], changed))
            audits.append(audit | {"rule_id": target["rule_id"]})
            count = audit["score_support_count"]
            combined_sum += count * audit["actual_rms"] ** 2
            combined_support += count
        def replace(module, args, out):
            result = out.clone()
            for indices, values in edits:
                result[0, indices, location[1]] = values
            return result
        handle = module.k_norm.register_forward_hook(replace)
        try:
            result = self._forward(inputs)
        finally:
            handle.remove()
        result.update(prompt_tokens=n, intervention={
            "mode": mode, "location": location, "edited_rules": len(edits), "per_rule": audits,
            "combined_actual_score_rms": (combined_sum / combined_support) ** .5 if combined_support else 0.0,
            "full_causal_support_entries": combined_support,
            "certificate_scope": "fixed-layer last-prompt query; all four GQA query heads",
            "evidence_budget_implemented": False,
            "future_generation_direction_guaranteed": False,
        })
        return result
