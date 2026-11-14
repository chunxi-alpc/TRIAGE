"""Official RePS components adapted to E1 one-token preference responses.

The official AxBench preference_loss function and intervention classes are
executed from the pinned source, without rewriting their mathematics. RePS uses
scaled_simpo, orig_add/orig_sub and null_it_out; BiPO is a different baseline.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import get_prompt, load_bundle
from experiments.aster_cr.engine import SYSTEM, Engine
from experiments.m1_diagnostics.engine import Context
from experiments.m1_diagnostics.common import MODELS

REPO = Path(__file__).resolve().parents[2]
ASSETS = Path("/home/gcx/aster_codex03_external_assets_20261112")
UPSTREAM = ASSETS / "official/axbench"
UPSTREAM_COMMIT = "41c8332543e5a631f9a8c0a9df38799893ace758"
DEFAULT_OUTPUT = REPO / "reports/aster_codex03_external_20261112/reps"
METHOD = "RePS-E1-task-adapted"
OFFICIAL_CONFIG = "axbench/sweep/wuzhengx/reps/experiments/p_vector_dps_g2-9b_nfs_axbench.yaml"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def official_components():
    """Load exact upstream source without importing unrelated AxBench benchmarks."""
    dependency_path = str(ASSETS / "python")
    if dependency_path not in sys.path:
        sys.path.insert(0, dependency_path)
    source = UPSTREAM / "axbench/models/interventions.py"
    spec = importlib.util.spec_from_file_location("aster_official_axbench_interventions", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    loss_source = UPSTREAM / "axbench/models/preference_model.py"
    tree = ast.parse(loss_source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "preference_loss")
    namespace = dict(torch=torch, F=F, Tuple=Tuple)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(loss_source), "exec"), namespace)
    return module, namespace["preference_loss"]


def unique_views(split):
    views = {}
    for row in load_bundle(split)["views"]:
        key = row["uid"]
        if key in views and int(views[key]["label"]) != int(row["label"]):
            raise ValueError("Conflicting existing labels for " + key)
        views[key] = row
    return [views[key] for key in sorted(views)]


def training_configuration(args):
    return dict(
        method=METHOD, seed=args.seed, model="Qwen3-8B", model_path=str(MODELS["qwen"]),
        train_data="E1 / Dataset A", train_pair_count=2540, layer=args.layer,
        layer_numbering="zero_based", rank=1, learning_rate=args.lr, epochs=args.epochs,
        batch_size=1, gradient_accumulation_steps=1, weight_decay=0.0,
        optimizer="AdamW", schedule="linear_decay", clipping_norm=1.0,
        loss_type="scaled_simpo", beta=1.0, gemma=0.0, simpo_scaler=1.0,
        reference_free=True, label_smoothing=0.0,
        preference_pairs=["orig_add", "orig_sub"], substraction_type="null_it_out",
        steering_factors=[1.0], positions="all", exclude_bos=True, prefix_length=1,
        dropout=0.0, intervention_positions_dropout=0.0,
        official_repository="https://github.com/stanfordnlp/axbench",
        official_commit=UPSTREAM_COMMIT, official_reference_config=OFFICIAL_CONFIG,
        isolated_dependency=dict(pyvene="0.1.8", location=str(ASSETS/"python")),
        torch_version=torch.__version__,
        official_components=["PreferenceVectorIntervention", "AdditionIntervention", "preference_loss"],
        supervision="Each unique fit prompt uses its correct yes/no label as positive response and opposite label as negative response. orig_sub reverses these preferences.",
        implementation_differences=[
            "Qwen3-8B and E1 task responses replace Gemma concept-generation data; this is explicitly task-adapted.",
            "Exact upstream loss is loaded from its AST node to avoid imports of unrelated AxBench benchmark tooling.",
            "Exact upstream intervention classes are connected by a PyTorch output hook supporting Qwen3 Tensor output.",
            "One-token responses share the same causal prompt hidden state. Compute only the final vocabulary logits, rather than logits for every prompt and appended response token.",
            "Reference prompt forward is shared between positive and nulling branches. Branch losses are backpropagated separately and averaged, as in upstream minibatching.",
            "Frozen BF16 backbone with FP32 intervention parameters; exact source operations receive FP32 residuals and return to BF16.",
            "Use fixed one-epoch small-batch training budget and one preset layer; choose only inference strength on development data.",
            "Inference calls official AdditionIntervention, which uses the trained direction and a multiplier and does not add its saved training bias.",
            "Retain upstream attention-mask length scaling: prompt length plus one response token.",
        ],
        system_prompt=SYSTEM, answer_labels=["\u662f", "\u5426"], decision_rule="ell > 0",

        evaluation_projection="Canonical Engine chunked prefill and final-token replay, FP32 yes/no output-head rows",
        training_projection="Native BF16 full-vocabulary final-token logits, then FP32 log-softmax",
    )


class RePSRunner:
    def __init__(self, config, training=False):
        self.config = config
        self.official, self.preference_loss = official_components()
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model_path"], local_files_only=True, dtype=torch.bfloat16,
            device_map={"": "cuda:0"}, attn_implementation="sdpa")
        self.model.requires_grad_(False)
        self.model.config.use_cache = False
        self.device = self.model.get_input_embeddings().weight.device
        self.labels = [self.tokenizer.encode(x, add_special_tokens=False) for x in ["\u662f", "\u5426"]]
        if any(len(x) != 1 for x in self.labels):
            raise ValueError("Expected the common single-token yes/no labels")
        self.label_ids = [x[0] for x in self.labels]
        if len(set(self.label_ids)) != 2:
            raise ValueError("Yes/no labels must have distinct token ids")
        self.label_weights = self.model.lm_head.weight[self.label_ids].detach().float()
        self.ax = self.official.PreferenceVectorIntervention(
            embed_dim=self.model.config.hidden_size, low_rank_dimension=1,
            dropout=0.0, intervention_positions_dropout=0.0).to(self.device)
        self.addition = self.official.AdditionIntervention(
            embed_dim=self.model.config.hidden_size, low_rank_dimension=1).to(self.device)
        self.addition.requires_grad_(False)
        self.factor = None
        self.mode = "training" if training else "inference"
        self.handle = self.model.model.layers[config["layer"]].register_forward_hook(self._hook, with_kwargs=True)
        self.engine = Engine("qwen", model=self.model, tokenizer=self.tokenizer)
        self.forward_count = 0
        if training:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.train()
        else:
            self.model.eval()
        self.ax.train(training)

    def _hook(self, module, inputs, kwargs, output):
        if self.factor is None:
            return output
        value = output[0] if isinstance(output, tuple) else output
        # exclude the initial template token, matching upstream prefix_length=1.
        active = value.float()
        positions = kwargs.get("position_ids")
        if positions is None:
            positions = torch.arange(value.shape[1], device=value.device).unsqueeze(0)
        if self.mode == "training":
            changed = self.ax(active, subspaces={"steering_factor": torch.tensor([self.factor], device=self.device)}).output
        else:
            changed = self.addition(active, subspaces={
                "idx": torch.zeros(1, dtype=torch.long, device=self.device),
                "max_act": torch.ones(1, device=self.device),
                "mag": torch.tensor([self.factor], device=self.device)})
        result = torch.where((positions >= 1).unsqueeze(-1), changed.to(value.dtype), value)
        return (result,) + output[1:] if isinstance(output, tuple) else result

    def token_ids(self, prompt):
        rendered = self.tokenizer.apply_chat_template(
            [dict(role="system", content=SYSTEM), dict(role="user", content=prompt)],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        if ids.shape[1] + 1 > self.model.config.max_position_embeddings:
            raise ValueError("Complete prompt exceeds Qwen3 context; no truncation is allowed")
        return ids

    def hidden(self, ids, factor):
        self.factor = factor
        output = self.model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        self.forward_count += 1
        return output.last_hidden_state[:, -1]

    def response_logps(self, ids, factor):
        hidden = self.hidden(ids, factor)
        logits = self.model.lm_head(hidden)
        return F.log_softmax(logits.float(), dim=-1)[:, self.label_ids]

    def sync_inference(self):
        self.addition.proj.load_state_dict(self.ax.proj.state_dict())

    def score_ids(self, ids, strength, raw=False):
        self.mode = "inference"
        self.factor = None if raw else strength
        prefix = self.engine._prefill(ids, ids.shape[1]-1)
        context = Context("", ids, prefix, int(ids.shape[1]))
        result = self.engine.run(context, capture=False)
        self.forward_count += math.ceil((ids.shape[1]-1)/self.engine.prefill_chunk) + 1
        torch.cuda.synchronize()
        return float(result["ell"])

    def scorer(self, strength):
        self.model.eval()
        self.ax.eval()
        self.sync_inference()
        def score(prompt, uid):
            ids = self.token_ids(prompt)
            ell = self.score_ids(ids, strength)
            return dict(ell=ell, tokens=int(ids.shape[1]), forward_count=math.ceil((ids.shape[1]-1)/self.engine.prefill_chunk)+1,
                        peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30)
        return score

    def close(self):
        self.handle.remove()


def checkpoint(path, runner, optimizer, scheduler, step, elapsed, config):
    torch.save(dict(
        vector={k: v.detach().cpu() for k, v in runner.ax.state_dict().items()},
        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), step=step,
        training_wall_seconds=elapsed, config=config,
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
        training_forward_count=runner.forward_count,
        peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,
    ), path)


def train(args, smoke=False):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    config = training_configuration(args)
    if smoke:
        config.update(smoke_only=True, train_pair_count=None)
    config_path = output / "training_config.json"

    if config_path.exists() and read_json(config_path) != config:
        raise ValueError("Output has another training configuration")
    write_json(config_path, config)
    if (output / "training.json").exists() and (output / "model.pt").exists():
        return read_json(output / "training.json")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    fit = unique_views("fit")
    random.Random(args.seed).shuffle(fit)
    if smoke:
        fit = fit[:8]
    with (output / "fit_inputs.jsonl").open("w", encoding="utf-8") as f:
        for row in fit:
            f.write(json.dumps({k: row[k] for k in ("uid", "source_id", "label")}, ensure_ascii=False) + "\n")
    runner = RePSRunner(config, training=True)
    initial = runner.ax.proj.weight.detach().clone()
    optimizer = torch.optim.AdamW(runner.ax.parameters(), lr=args.lr, weight_decay=0.0)
    total_steps = len(fit) * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.0, 1 - step / max(1, total_steps)))
    step_start, previous_seconds, previous_peak = 0, 0.0, 0.0
    last_checkpoint = output / "checkpoint.pt"
    if last_checkpoint.exists():
        saved = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        if saved["config"] != config:
            raise ValueError("Checkpoint belongs to another configuration")
        runner.ax.load_state_dict(saved["vector"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        step_start, previous_seconds = saved["step"], saved["training_wall_seconds"]
        runner.forward_count = saved.get("training_forward_count", 3*step_start)
        previous_peak = saved.get("peak_memory_gib", 0.0)
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    loss_log = output / "training_steps.jsonl"
    with loss_log.open("a", encoding="utf-8", buffering=1) as log:
        for step in range(step_start, total_steps):
            row = fit[step % len(fit)]
            ids = runner.token_ids(get_prompt(row["uid"]))
            label = int(row["label"])
            chosen, rejected = (0, 1) if label else (1, 0)
            runner.mode = "training"
            with torch.no_grad():
                reference = runner.response_logps(ids, None)
            length = torch.tensor([ids.shape[1] + 1], dtype=torch.float32, device=runner.device)
            optimizer.zero_grad(set_to_none=True)
            branch_losses = []
            for factor, win, lose in [(1.0, chosen, rejected), (0.0, rejected, chosen)]:
                policy = runner.response_logps(ids, factor)
                losses, _, _ = runner.preference_loss(
                    policy[:, win], policy[:, lose],
                    reference[:, win], reference[:, lose],
                    beta=1.0, gemma=0.0, simpo_scaler=1.0,
                    winning_lens=length, losing_lens=length,
                    label_smoothing=0.0, loss_type="scaled_simpo", reference_free=True)
                loss = losses.mean() / 2
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite RePS loss")
                loss.backward()
                branch_losses.append(float(loss.detach()) * 2)
            grad_norm = torch.nn.utils.clip_grad_norm_(runner.ax.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize()
            elapsed = previous_seconds + time.perf_counter() - started
            log.write(json.dumps(dict(step=step + 1, uid=row["uid"], tokens=int(ids.shape[1]),
                                      positive_loss=branch_losses[0], nulling_loss=branch_losses[1],
                                      mean_loss=sum(branch_losses)/2, gradient_norm=float(grad_norm),
                                      elapsed_seconds=elapsed)) + "\n")
            if (step + 1) % 100 == 0 or step + 1 == total_steps:
                checkpoint(last_checkpoint, runner, optimizer, scheduler, step + 1, elapsed, config)
                write_json(output / "status.json", dict(stage="training", step=step+1, total_steps=total_steps,
                                                        elapsed_seconds=elapsed))
    elapsed = previous_seconds + time.perf_counter() - started
    vector = {k: v.detach().cpu() for k, v in runner.ax.state_dict().items()}
    torch.save(dict(vector=vector, config=config), output / "model.pt")
    params = sum(p.numel() for p in runner.ax.parameters())
    metadata = dict(
        config, actual_experiment_results=not smoke, execution_kind="smoke_validation" if smoke else "training",
        train_unique_inputs=len(fit), training_steps=total_steps, parameter_count=params,
        trainable_parameter_count=params, stored_parameter_count=params,
        backbone_parameter_count=sum(p.numel() for p in runner.model.parameters()),
        training_wall_seconds=elapsed, training_gpu_hours=elapsed/3600,
        training_peak_memory_gib=max(previous_peak,torch.cuda.max_memory_allocated()/2**30),
        training_forward_count=runner.forward_count, training_backward_count=2*total_steps, model_file=str(output/"model.pt"),
        parameter_change_l2=float(torch.linalg.vector_norm(runner.ax.proj.weight.detach()-initial)),
        gpu_visibility=os.environ.get("CUDA_VISIBLE_DEVICES"), selected_checkpoint="final_epoch",
    )
    if smoke:
        runner.model.eval()
        runner.sync_inference()
        probes = unique_views("dev")[:4]
        before = []

        for row in probes:
            ids = runner.token_ids(get_prompt(row["uid"]))
            before.append(dict(uid=row["uid"], raw_ell=runner.score_ids(ids, 1.0, raw=True),
                               adapted_ell=runner.score_ids(ids, 1.0)))
        # Reload actual saved weights into the same backbone; no new GPU model is needed.
        restored = torch.load(output/"model.pt", map_location="cpu", weights_only=False)
        with torch.no_grad():
            runner.ax.proj.weight.zero_()
            runner.ax.proj.bias.zero_()
        runner.ax.load_state_dict(restored["vector"])
        runner.sync_inference()
        for row in before:
            ids = runner.token_ids(get_prompt(row["uid"]))
            row["reloaded_ell"] = runner.score_ids(ids, 1.0)
            row["absolute_reload_difference"] = abs(row["reloaded_ell"]-row["adapted_ell"])
        max_reload = max(x["absolute_reload_difference"] for x in before)
        max_change = max(abs(x["adapted_ell"]-x["raw_ell"]) for x in before)
        metadata.update(probes=before, max_reload_difference=max_reload, max_output_change=max_change,
                        smoke_passed=bool(metadata["parameter_change_l2"] > 0 and max_change > 1e-6 and max_reload <= 1e-6))
        write_json(output/"smoke.json", metadata)
        if not metadata["smoke_passed"]:
            raise RuntimeError("RePS smoke failed: inspect smoke.json")
    write_json(output/"training.json", metadata)
    write_json(output/"status.json", dict(stage="smoke_complete" if smoke else "trained", complete=True))
    runner.close()
    return metadata


def load_runner(output):
    saved = torch.load(output/"model.pt", map_location="cpu", weights_only=False)
    runner = RePSRunner(saved["config"])
    runner.ax.load_state_dict(saved["vector"])
    runner.sync_inference()
    return runner, saved["config"]


def select(args):
    output = args.output
    frozen_path = output/"frozen.json"
    if frozen_path.exists():
        return read_json(frozen_path)
    runner, config = load_runner(output)
    views = unique_views("dev")
    candidates = [float(x) for x in args.strengths.split(",")]
    if not candidates or len(candidates) > 12 or any(x <= 0 for x in candidates):
        raise ValueError("Expected one to twelve positive development strengths")
    output_scores = output/"dev_selection_scores.jsonl"
    existing = {}
    if output_scores.exists():
        for line in output_scores.read_text().splitlines():
            row = json.loads(line)
            existing[(row["uid"], row["strength"])] = row
    stats = {x: [] for x in candidates}
    started = time.perf_counter()
    with output_scores.open("a", encoding="utf-8", buffering=1) as handle:
        for i, row in enumerate(views):
            ids = None
            for strength in candidates:
                key = (row["uid"], strength)
                if key not in existing:
                    if ids is None:
                        ids = runner.token_ids(get_prompt(row["uid"]))
                    ell = runner.score_ids(ids, strength)
                    record = dict(uid=row["uid"], strength=strength, ell=ell, label=int(row["label"]))
                    existing[key] = record
                    handle.write(json.dumps(record)+"\n")
                record = existing[key]
                stats[strength].append(float(np.logaddexp(0, -(2*record["label"]-1)*record["ell"])))
            if (i+1)%100 == 0:
                write_json(output/"status.json", dict(stage="development_selection", inputs=i+1,total_inputs=len(views)))
    summary = [dict(strength=s, mean_binary_nll=float(np.mean(stats[s])), unique_dev_inputs=len(stats[s])) for s in candidates]
    chosen = min(summary, key=lambda x:(x["mean_binary_nll"],x["strength"]))
    frozen = dict(config, selected_strength=chosen["strength"], selection_metric="mean dev binary negative log likelihood",
                  selection_candidates=summary, selection_wall_seconds=time.perf_counter()-started,
                  selection_split="dev", decision_threshold=0.0, frozen_before_test=True)
    write_json(frozen_path, frozen)
    runner.close()
    return frozen


def evaluate(args):
    from .inference import score_split
    from .evaluate import evaluate as summarize
    output = args.output
    frozen = read_json(output/"frozen.json")
    metadata = read_json(output/"training.json")
    if metadata.get("execution_kind") == "smoke_validation":
        raise ValueError("Smoke weights cannot be evaluated as a full E1 result")
    runner, config = load_runner(output)
    method = METHOD+"-seed"+str(config["seed"])
    metadata.update(frozen_configuration=frozen, method=method, execution_kind="model_inference")
    scores = output/args.split/"scores.jsonl"
    collected = score_split(method, args.split, runner.scorer(frozen["selected_strength"]), scores, metadata)
    metadata["inference_gpu_hours"] = collected["inference_wall_seconds"] / 3600
    result = summarize(method, scores, args.split, DEFAULT_OUTPUT.parent, metadata)
    runner.close()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["smoke","train","select","evaluate","all","cpu-check"])
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT/"seed11")
    p.add_argument("--seed", type=int, choices=[11,22,33], default=11)
    p.add_argument("--layer", type=int, default=18)
    p.add_argument("--lr", type=float, default=0.04)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--strengths", default="0.5,1.0,2.0")

    p.add_argument("--split", choices=["dev","test"], default="test")
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.command=="cpu-check":
        module, loss = official_components()
        ax=module.PreferenceVectorIntervention(embed_dim=8,low_rank_dimension=1)
        values=ax(torch.zeros(1,2,8),subspaces={"steering_factor":torch.ones(1)})
        result=dict(official_component_shape=list(values.output.shape),official_loss_loaded=callable(loss),
                    gpu_used=False,source_commit=UPSTREAM_COMMIT)
        write_json(args.output/"cpu_import.json",result)
    elif args.command=="smoke":result=train(args,True)
    elif args.command=="train":result=train(args)
    elif args.command=="select":result=select(args)
    elif args.command=="evaluate":result=evaluate(args)
    else:
        train(args)
        select(args)
        result=evaluate(args)
    print(json.dumps(dict(command=args.command,output=str(args.output),complete=True),ensure_ascii=False))


if __name__=="__main__":
    main()
