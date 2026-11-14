"""Official CAA difference-of-means and intervention, adapted to Qwen3/E1.

GPU work is explicit through smoke/train/select/evaluate/run. describe only
prepares configuration. No A/P input-difference direction is used.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import json
import math
import os
from pathlib import Path
import time

from .data import REPO, load_bundle, get_prompt

METHOD = "CAA-E1-task-adapted"
OFFICIAL = Path("/home/gcx/aster_codex03_external_assets_20261112/official/CAA-official-snapshot")
DEFAULT_OUT = REPO / "reports/aster_codex03_external_20261112/caa"
DEFAULT_MULTIPLIERS = [-1., -.5, 0., .5, 1.]
MODEL_PATH = "/home/gcx/TRIAGE/models/modelscope/Qwen3-8B"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def fit_examples():
    unique = {}
    for view in load_bundle("fit")["views"]:
        uid = view["uid"]
        label = int(view["label"])
        if uid in unique and unique[uid]["label"] != label:
            raise ValueError("One fit prompt has conflicting behavior labels: " + uid)
        unique.setdefault(uid, dict(uid=uid, label=label, source_id=view["source_id"]))
    return list(unique.values())


def load_official_functions(filename, names, namespace):
    """Compile the named upstream function nodes unchanged; avoid legacy imports."""
    tree = ast.parse((OFFICIAL / filename).read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != set(names):
        raise ValueError("Required official functions are absent: " + filename)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *functions], type_ignores=[]))
    exec(compile(module, str(OFFICIAL / filename), "exec"), namespace)
    return {name: namespace[name] for name in names}


def describe(output=DEFAULT_OUT, layer=18):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / ".gitignore").write_text("*.pt\n*.npy\n*.jsonl\n", encoding="utf-8")
    examples = fit_examples()
    revision = (OFFICIAL / "SOURCE_COMMIT").read_text().strip()
    config = dict(
        method=METHOD, official_repository="https://github.com/nrimsky/CAA",
        official_revision=revision, official_snapshot=str(OFFICIAL),
        official_training_core="generate_vectors.py::generate_save_vectors_for_behavior (unmodified function AST)",
        official_intervention_core="utils/helpers.py::add_vector_from_position (unmodified function AST)",
        implementation_status="Qwen3 architecture and E1 behavioral data adaptation",
        model=MODEL_PATH, model_dtype="bfloat16", vector_dtype="float32",
        dataset="E1 / Dataset A", training_pairs=2540, unique_training_prompts=len(examples),
        behavior="Correct versus incorrect single-token answer to the same unchanged E1 prompt",
        positive="Gold answer token: yes if label=1, no if label=0",
        negative="Opposite answer token to the identical prompt",
        original_pair_type_used_for_direction=False, layer=int(layer), layer_indexing="zero-based",
        layer_selection="fixed before development and test", multipliers=DEFAULT_MULTIPLIERS,
        vector_normalization="None: the official cross-behavior equal-norm transform is identity for one behavior",
        extraction="Canonical prompt token IDs + answer token + EOS; official [-2] selects the answer token",
        extraction_note="Upstream tokenizer does not explicitly append EOS. This port explicitly appends one terminal EOS so [-2] has the required answer-token meaning under Qwen3.",
        prefix_reuse="Unedited prompt prefix and last prompt token are computed once, then immutable cache copies process correct/incorrect answer+EOS suffixes",
        intervention="Official inclusive position mask from the last user <|im_end|> through assistant template and subsequent answer positions",
        scoring="Unchanged canonical system/chat prompt, enable_thinking=False, FP32 yes-minus-no label projection",
        selection="Development macro paired correctness over A/P/C/F; tie: natural original-node correctness, then smaller absolute multiplier",
        test_policy="No test access until multiplier is selected and frozen",
        stochastic_training=False, seeds="Not applicable: deterministic paired activation averaging",
        raw_residual_export="Unedited last prompt token at final decoder output before final norm; all fit UIDs, never answer activations",
        expected_peak_memory_gib=[20,30], estimated_hours=[1,3],
    )
    config_path = output / "configuration.json"
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError("Existing CAA configuration differs; use a separate output directory")
    write_json(config_path, config)
    write_json(output / "implementation.json", dict(
        official_revision=revision,
        reused_functions=[config["official_training_core"], config["official_intervention_core"]],
        differences=[
            "Replace Llama-specific model and tokenizer wrappers with the existing canonical Qwen3 scorer.",
            "Define behavior pairs from each unique E1 prompt's correct and incorrect answer, and label the result task-adapted.",
            "Append a terminal EOS and select the answer activation at official index -2.",
            "Reuse unedited prefix computation across the two answers; preserve the causal answer representations.",
            "Accumulate the original official mean formula in FP32, retaining the BF16 model.",
            "Map the official instruction-end boundary to Qwen3's closing user im_end token, inclusive.",
            "Use the existing binary score evaluator; avoid unused full-vocabulary logits during activation extraction.",
        ],
        original_objective="Mean positive-minus-negative post-block answer representations, followed by additive steering",
        added_ap_cf_loss=False,
    ))
    return config


def make_engine():
    import torch
    from experiments.aster_cr.engine import Engine
    torch.set_num_threads(4)
    return Engine("qwen")


class BehaviorDataset:
    """Architecture adapter for the upstream ComparisonDataset interface."""
    engine = None
    examples = None

    def __init__(self, data_path, token, model_name_path, use_chat):
        self.data = type(self).examples
        self.engine = type(self).engine

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        import torch
        item = self.data[index]
        ids = self.engine.ids(get_prompt(item["uid"]))
        eos = self.engine.tokenizer.eos_token_id
        if eos is None or ids.shape[1] + 2 > self.engine.limit:
            raise ValueError("CAA answer+EOS sequence is unavailable or exceeds context length")
        answer_ids = self.engine.label_ids
        correct = answer_ids[0 if item["label"] else 1]
        wrong = answer_ids[1 if item["label"] else 0]
        suffix = lambda token: torch.tensor([[token, eos]], device=ids.device, dtype=ids.dtype)
        return torch.cat((ids, suffix(correct)), dim=1), torch.cat((ids, suffix(wrong)), dim=1)


class QwenExtractionAdapter:
    """Expose the upstream wrapper interface, computing only needed hidden states."""
    def __init__(self, engine, layer, output, examples, export_raw):
        import torch
        from experiments.aster_cr.engine import SYSTEM
        self.engine, self.layer, self.output = engine, layer, Path(output)
        self.device, self.model_name_path, self.use_chat = engine.device, MODEL_PATH, True
        self.examples, self.export_raw = examples, export_raw
        self.active_capture = False
        self.active_raw_capture = False
        self.activation = None
        self.raw_residual = None
        self.cached_ids = None
        self.cached_prefix = None
        self.prompt_index = -1
        self.counts = dict(prompt_prefills=0, prompt_final_forwards=0, answer_suffix_forwards=0)
        self.started = time.perf_counter()
        self.raw_dir = self.output / "raw_prompt_residuals"
        if export_raw:
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            self.raw_index = self.raw_dir / "index.jsonl"
            self.saved_raw = set()
            if self.raw_index.exists():
                self.saved_raw = {json.loads(s)["uid"] for s in self.raw_index.read_text().splitlines() if s.strip()}
            write_json(self.raw_dir / "metadata.json", dict(
                model=MODEL_PATH, dtype="float32_saved_from_bfloat16", hidden_size=int(engine.model.config.hidden_size),
                system_prompt=SYSTEM, tokenizer=MODEL_PATH, enable_thinking=False,
                location="last_decoder_layer_output_pre_final_norm", position="last_original_prompt_token",
                protocol="Canonical BF16 prefix, separate final-prompt-token forward, FP32 yes/no projection",
                uid_count=len(examples), official_caa_revision=(OFFICIAL / "SOURCE_COMMIT").read_text().strip(),
            ))
        def capture(module, args, output):
            if self.active_capture:
                value = output[0] if isinstance(output, tuple) else output
                self.activation = value.float()
        def capture_raw(module, args, output):
            if self.active_raw_capture:
                value = output[0] if isinstance(output, tuple) else output
                self.raw_residual = value[0, -1].detach().float().cpu()
        self.handles = [engine.layers[layer].register_forward_hook(capture),
                        engine.layers[-1].register_forward_hook(capture_raw)]

    def set_save_internal_decodings(self, value):
        pass

    def reset_all(self):
        self.activation = None

    def get_last_activations(self, layer):
        if layer != self.layer or self.activation is None:
            raise ValueError("No requested answer activation")
        return self.activation

    def get_logits(self, tokens):
        import numpy as np
        import torch
        eng = self.engine
        prompt_ids = tokens[:, :-2]
        with torch.no_grad():
            if self.cached_ids is None or not torch.equal(prompt_ids, self.cached_ids):
                self.prompt_index += 1
                item = self.examples[self.prompt_index]
                before = time.perf_counter()
                self.active_capture = False
                self.active_raw_capture = False
                self.cached_prefix = None
                prefix = eng._prefill(prompt_ids, prompt_ids.shape[1] - 1)
                self.counts["prompt_prefills"] += 1
                pos = prompt_ids.shape[1] - 1
                self.active_raw_capture = self.export_raw
                result = eng.model.model(
                    input_ids=prompt_ids[:, -1:], attention_mask=torch.ones_like(prompt_ids),
                    past_key_values=prefix, use_cache=True,
                    cache_position=torch.tensor([pos], device=eng.device),
                    position_ids=torch.tensor([[pos]], device=eng.device),
                )
                self.active_raw_capture = False
                self.counts["prompt_final_forwards"] += 1
                self.cached_prefix = result.past_key_values
                self.cached_ids = prompt_ids.clone()
                raw_ell = float(eng._score_hidden(result.last_hidden_state[0, -1]))
                if self.export_raw and item["uid"] not in self.saved_raw:
                    path = self.raw_dir / (item["uid"] + ".npy")
                    np.save(path, self.raw_residual.numpy(), allow_pickle=False)
                    torch.cuda.synchronize(eng.device)
                    append_jsonl(self.raw_index, dict(
                        uid=item["uid"], path=str(path), ell=raw_ell, tokens=int(prompt_ids.shape[1]),
                        source_id=item["source_id"], label=item["label"],
                        position="last_original_prompt_token", location="last_decoder_layer_output_pre_final_norm",
                        model=MODEL_PATH, prompt_protocol="aster_cr.engine.SYSTEM; thinking disabled",
                        feature_seconds=time.perf_counter()-before,
                    ))
                    self.saved_raw.add(item["uid"])
                if self.prompt_index % 50 == 0:
                    write_json(self.output / "status.json", dict(
                        state="extracting", completed_prompts=self.prompt_index+1, total_prompts=len(self.examples),
                        seconds=time.perf_counter()-self.started, counts=self.counts,
                        physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    ))
                del result, prefix
            self.active_capture = True
            start = prompt_ids.shape[1]
            result = eng.model.model(
                input_ids=tokens[:, -2:], attention_mask=torch.ones_like(tokens),
                past_key_values=eng._copy_cache(self.cached_prefix), use_cache=True,
                cache_position=torch.arange(start, start+2, device=eng.device),
                position_ids=torch.arange(start, start+2, device=eng.device).unsqueeze(0),
            )
            self.active_capture = False
            self.counts["answer_suffix_forwards"] += 1
            del result
        # The official caller only consumes stored activations, not returned logits.
        return None

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.cached_prefix = None
        self.activation = None
        self.cached_ids = None


def vector_path(output, layer):
    return Path(output) / f"vector_layer{layer}.pt"


def train(output=DEFAULT_OUT, layer=18, engine=None, limit=None):
    import torch
    from tqdm import tqdm
    output = Path(output)
    config = describe(output, layer)
    if limit is None and (output / "training.json").exists():
        old = read_json(output / "training.json")
        if old.get("complete") and vector_path(output, layer).exists():
            return old
    examples = fit_examples()
    if limit is not None:
        examples = examples[:limit]
    examples_file = output / "behavior_examples.jsonl"
    with examples_file.open("w", encoding="utf-8") as handle:
        for item in examples:
            handle.write(json.dumps(item) + "\n")
    eng = engine or make_engine()
    if not 0 <= layer < len(eng.layers):
        raise ValueError("CAA layer outside model")
    torch.cuda.reset_peak_memory_stats(eng.device)
    started = time.perf_counter()
    wrapper = QwenExtractionAdapter(eng, layer, output, examples, export_raw=True)
    BehaviorDataset.engine, BehaviorDataset.examples = eng, examples
    namespace = dict(
        t=torch, os=os, tqdm=tqdm, ComparisonDataset=BehaviorDataset, HUGGINGFACE_TOKEN=None,
        get_ab_data_path=lambda behavior: str(examples_file),
        get_vector_dir=lambda behavior: str(output),
        get_activations_dir=lambda behavior: str(output / "answer_activations"),
        get_vector_path=lambda behavior, current_layer, model: str(vector_path(output, current_layer)),
        get_activations_path=lambda behavior, current_layer, model, polarity:
            str(output / "answer_activations" / f"layer{current_layer}_{polarity}.pt"),
    )
    function = load_official_functions(
        "generate_vectors.py", ["generate_save_vectors_for_behavior"], namespace
    )["generate_save_vectors_for_behavior"]
    try:
        function([layer], True, "E1_correct_answer", wrapper)
        torch.cuda.synchronize(eng.device)
        elapsed = time.perf_counter() - started
        vector = torch.load(vector_path(output, layer), map_location="cpu", weights_only=True)
        if not torch.isfinite(vector).all() or vector.norm() == 0:
            raise ValueError("CAA produced a nonfinite or zero direction")
        result = dict(
            complete=True, method=METHOD, smoke_only=limit is not None,
            actual_experiment_results=False, stage="direction_estimated; evaluation pending",
            unique_training_prompts=len(examples), e1_training_pairs=2540 if limit is None else None,
            layer=layer, vector_file=str(vector_path(output, layer)), vector_norm=float(vector.norm()),
            parameter_count=vector.numel(), trainable_parameter_count=0, stored_parameter_count=vector.numel(),
            model_parameter_count=sum(p.numel() for p in eng.model.parameters()),
            training_wall_seconds=elapsed, training_gpu_hours=elapsed / 3600,
            training_peak_memory_gib=torch.cuda.max_memory_allocated(eng.device)/2**30,
            counts=wrapper.counts, official_revision=config["official_revision"],
            physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
            original_objective_preserved=True, task_adapted=True,
            raw_prompt_residuals=str(wrapper.raw_dir),
        )
        write_json(output / "training.json", result)
        write_json(output / "status.json", dict(state="trained", **result))
        return result
    finally:
        wrapper.close()
        BehaviorDataset.engine, BehaviorDataset.examples = None, None


class CAAScorer:
    def __init__(self, engine, vector, layer, multiplier):
        import torch
        self.engine, self.layer, self.multiplier = engine, int(layer), float(multiplier)
        self.vector = vector.to(engine.device, dtype=torch.float32) * self.multiplier
        namespace = {"t": torch}
        funcs = load_official_functions("utils/helpers.py", [
            "add_vector_from_position", "find_last_subtensor_position", "find_instruction_end_postion"
        ], namespace)
        self.add_vector = funcs["add_vector_from_position"]
        self.find_end = funcs["find_instruction_end_postion"]
        self.end_tokens = torch.tensor(
            engine.tokenizer.encode("<|im_end|>", add_special_tokens=False), device=engine.device)

    def __call__(self, prompt, uid=None):
        import torch
        eng = self.engine
        ids = eng.ids(prompt)
        boundary = self.find_end(ids[0], self.end_tokens)
        if boundary < 0:
            raise ValueError("Qwen user instruction boundary is missing")
        def hook(module, args, kwargs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            positions = kwargs.get("position_ids")
            if positions is None:
                raise ValueError("Qwen layer does not expose position IDs")
            changed = self.add_vector(hidden.clone(), self.vector, positions, from_pos=boundary)
            return (changed,) + output[1:] if isinstance(output, tuple) else changed
        handle = eng.layers[self.layer].register_forward_hook(hook, with_kwargs=True)
        try:
            context = eng.prepare(prompt)
            result = eng.run(context, capture=False)
            torch.cuda.synchronize(eng.device)
            return dict(ell=result["ell"], tokens=result["tokens"],
                        forward_count=math.ceil((result["tokens"]-1)/eng.prefill_chunk)+1,
                        peak_memory_gib=torch.cuda.max_memory_allocated(eng.device)/2**30)
        finally:
            handle.remove()


def smoke(output=DEFAULT_OUT, layer=18, engine=None):
    import torch
    eng = engine or make_engine()
    smoke_dir = Path(output) / "smoke"
    train(smoke_dir, layer, eng, limit=2)
    vector = torch.load(vector_path(smoke_dir, layer), map_location="cpu", weights_only=True)
    prompts = load_bundle("dev")["views"][:2]
    records = []
    for view in prompts:
        text = get_prompt(view["uid"])
        base = CAAScorer(eng, vector, layer, 0.)(text)["ell"]
        first = CAAScorer(eng, vector, layer, 1.)(text)["ell"]
        loaded = torch.load(vector_path(smoke_dir, layer), map_location="cpu", weights_only=True)
        second = CAAScorer(eng, loaded, layer, 1.)(text)["ell"]
        records.append(dict(uid=view["uid"], raw=base, steered=first, reloaded=second,
                            output_change=first-base, reload_difference=abs(first-second)))
    import numpy as np
    residual_checks = []
    for example in fit_examples()[:2]:
        saved = np.load(smoke_dir / "raw_prompt_residuals" / (example["uid"] + ".npy"))
        raw = eng.score(get_prompt(example["uid"]), capture=True)
        difference = float(np.max(np.abs(saved - raw["residual"].numpy())))
        residual_checks.append(dict(uid=example["uid"], max_abs_difference=difference,
                                    shape=list(saved.shape), location="last_original_prompt_token_pre_final_norm"))
    residual_match = all(row["max_abs_difference"] <= 1e-6 for row in residual_checks)
    changed = sum(abs(row["output_change"]) > 1e-6 for row in records)
    consistent = all(row["reload_difference"] <= 1e-6 for row in records)
    result = dict(complete=bool(changed and consistent and residual_match), actual_experiment_results=False,
                  purpose="Real-weight intervention and reload smoke only", changed_count=changed,
                  reload_consistent=consistent, raw_prompt_residual_match=residual_match,
                  residual_checks=residual_checks, records=records, layer=layer)
    write_json(Path(output) / "smoke.json", result)
    if not result["complete"]:
        raise RuntimeError("CAA real-weight smoke did not show output change and reload consistency")
    return result


def select(output=DEFAULT_OUT, layer=18, engine=None):
    import torch
    output = Path(output)
    if (output / "frozen.json").exists():
        return read_json(output / "frozen.json")
    config = describe(output, layer)
    training = read_json(output / "training.json")
    if training.get("smoke_only") or training["unique_training_prompts"] != 4637:
        raise ValueError("Development selection requires the full E1 direction")
    eng = engine or make_engine()
    vector = torch.load(vector_path(output, layer), map_location="cpu", weights_only=True)
    bundle = load_bundle("dev")
    views = {v["uid"]: v for v in bundle["views"]}
    candidates = []
    start = time.perf_counter()
    for multiplier in config["multipliers"]:
        scorer = CAAScorer(eng, vector, layer, multiplier)
        name = f"layer{layer}_alpha{multiplier:g}"
        target = output / "development" / (name + ".jsonl")
        target.parent.mkdir(parents=True, exist_ok=True)
        records = {}
        if target.exists():
            records = {r["uid"]: r for r in (json.loads(s) for s in target.read_text().splitlines() if s.strip())}
        for index, (uid, view) in enumerate(views.items()):
            if uid not in records:
                before = time.perf_counter()
                result = scorer(get_prompt(uid), uid)
                record = dict(uid=uid, method=METHOD, split="dev", multiplier=multiplier,
                              execution_kind="model_inference", status="ok", **result,
                              inference_seconds=time.perf_counter()-before)
                append_jsonl(target, record)
                records[uid] = record
            if index % 100 == 0:
                write_json(output / "status.json", dict(state="selecting_on_dev", multiplier=multiplier,
                           completed=index+1, total=len(views), seconds=time.perf_counter()-start))
        per_kind = {}
        for kind in ("A", "P", "C", "F"):
            pairs = [p for p in bundle["pairs"] if p["kind"] == kind]
            correct = sum((records[p["left"]]["ell"] > 0) == bool(p["left_label"]) and
                          (records[p["right"]]["ell"] > 0) == bool(p["right_label"]) for p in pairs)
            per_kind[kind] = correct / len(pairs)
        natural = [v for v in views.values() if v.get("original")]
        natural_accuracy = sum((records[v["uid"]]["ell"] > 0) == bool(v["label"]) for v in natural) / len(natural)
        candidates.append(dict(layer=layer, multiplier=multiplier,
                               macro_pair_accuracy=sum(per_kind.values())/4, pair_accuracy=per_kind,
                               natural_node_accuracy=natural_accuracy,
                               inference_wall_seconds=sum(r["inference_seconds"] for r in records.values()),
                               scored_inputs=len(records), predictions=str(target)))
    chosen = max(candidates, key=lambda c: (c["macro_pair_accuracy"], c["natural_node_accuracy"], -abs(c["multiplier"])))
    result = dict(frozen=True, method=METHOD, split_used="dev", layer=layer,
                  multiplier=chosen["multiplier"], selection_rule=config["selection"],
                  candidates=candidates, selected=chosen,
                  development_wall_seconds=sum(c["inference_wall_seconds"] for c in candidates),
                  test_accessed_for_selection=False, training=training)
    write_json(output / "frozen.json", result)
    return result


def evaluate_split(split, output=DEFAULT_OUT, engine=None):
    import torch
    from .inference import score_split
    from .evaluate import evaluate
    output = Path(output)
    frozen = read_json(output / "frozen.json")
    if not frozen.get("frozen"):
        raise ValueError("CAA must be frozen before path evaluation")
    eng = engine or make_engine()
    layer = int(frozen["layer"])
    vector = torch.load(vector_path(output, layer), map_location="cpu", weights_only=True)
    scorer = CAAScorer(eng, vector, layer, frozen["multiplier"])
    scores = output / "predictions" / (split + ".jsonl")
    metadata = dict(frozen["training"], method=METHOD, task_adapted=True,
                    layer=layer, multiplier=frozen["multiplier"],
                    development_selection_gpu_hours=frozen["development_wall_seconds"]/3600,
                    official_repository="https://github.com/nrimsky/CAA",
                    frozen_configuration=str(output / "frozen.json"))
    collected = score_split(METHOD, split, scorer, scores, metadata)
    collected["inference_gpu_hours"] = collected["inference_wall_seconds"] / 3600
    collected["inference_seconds_per_input"] = collected["inference_wall_seconds"] / collected["inference_input_count"]
    write_json(scores.with_suffix(".metadata.json"), collected)
    result = evaluate(METHOD, scores, split, DEFAULT_OUT.parent, metadata=collected)
    write_json(output / "status.json", dict(state=split+"_evaluated", actual_experiment_results=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["describe", "smoke", "train", "select", "evaluate", "run"], required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    args = parser.parse_args()
    if args.mode == "describe":
        config = describe(args.output, args.layer)
        print(json.dumps(dict(status="prepared_only", method=METHOD, layer=args.layer,
                              training_prompts=config["unique_training_prompts"])))
        return
    engine = make_engine()
    if args.mode == "smoke":
        result = smoke(args.output, args.layer, engine)
    elif args.mode == "train":
        result = train(args.output, args.layer, engine)
    elif args.mode == "select":
        result = select(args.output, args.layer, engine)
    elif args.mode == "evaluate":
        result = evaluate_split(args.split, args.output, engine)
    else:
        smoke(args.output, args.layer, engine)
        train(args.output, args.layer, engine)
        select(args.output, args.layer, engine)
        evaluate_split("dev", args.output, engine)
        result = evaluate_split("test", args.output, engine)
        write_json(args.output / "status.json", dict(state="complete", actual_experiment_results=True,
                                                    method=METHOD, frozen=str(args.output / "frozen.json")))
    print(json.dumps(dict(status="finished", mode=args.mode, output=str(args.output))))


if __name__ == "__main__":
    main()
