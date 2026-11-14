"""PSR S-PSR_MSE task adaptation using exact Nokia implementation components.

Fit-only teacher prompts produce binary responses. Successful teacher examples
are distilled with the official aligned-activation MSE and Focused module.
No task-pair consistency penalty replaces the original teacher objective.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import get_prompt
from .reps import unique_views, write_json, read_json
from experiments.aster_cr.engine import SYSTEM, Engine
from experiments.m1_diagnostics.engine import Context
from experiments.m1_diagnostics.common import MODELS

REPO=Path(__file__).resolve().parents[2]
ASSETS=Path("/home/gcx/aster_codex03_external_assets_20261112")
UPSTREAM=ASSETS/"official/steer-like-the-llm"
COMMIT="3d916c618d146c5d657f055e432a432b0fa493c6"
REPORT=REPO/"reports/aster_codex03_external_20261112"
METHOD="PSR-MSE-E1-task-adapted"
TEACHER_INSTRUCTION=(
    "Evaluate only the current condition using the patient's decisive facts and the literal guideline condition. "
    "Do not let downstream actions, irrelevant background, or stylistic differences change the decision. "
    "Preserve sensitivity to changes in decisive facts or in the condition. "
    "Answer with the requested single Chinese yes/no label.")
LABELS=["\u662f","\u5426"]


def official_components():
    source=str(UPSTREAM/"src")
    if source not in sys.path:sys.path.insert(0,source)
    from ase.steering.focused_steering import FocusedSteeringModule,FocusedSteeredModelConfig
    from ase.steering.steering_base import mse_on_aligned_activations
    return FocusedSteeringModule,FocusedSteeredModelConfig,mse_on_aligned_activations


def configuration(args,smoke=False):
    return dict(method=METHOD,variant="S-PSR_MSE",seed=args.seed,model="Qwen3-8B",
        model_path=str(MODELS["qwen"]),train_data="E1 / Dataset A",train_pair_count=None if smoke else 2540,
        layer=args.layer,layer_numbering="zero_based",rank=1,batch_size=1,epochs=args.epochs,
        learning_rate=args.lr,weight_decay=1e-6,optimizer="AdamW",schedule="linear_decay",
        native_objective="prompt_steering_imitation",layers_to_imitate="all",normalize_psi_loss=False,
        steering_location="answer_only",location_activation_fn="relu",use_location_bias=True,
        use_steering_coeff_bias=False,regularization_coefficient=0.1,
        training_user_steering_coefficient=1.0,
        official_repository="https://github.com/Nokia-Bell-Labs/steer-like-the-llm",official_commit=COMMIT,
        official_components=["FocusedSteeringModule","mse_on_aligned_activations"],
        official_reference_config="src/ase/experiment_utils/default_model_configs.py",
        teacher_system_prompt=SYSTEM+"\n"+TEACHER_INSTRUCTION,
        teacher_response="Argmax of common FP32 yes/no scores; one-token constrained response.",
        teacher_success_filter="Fit-only ground truth checks teacher response; retain correct teacher examples.",
        system_prompt=SYSTEM,answer_labels=LABELS,decision_rule="ell > 0",
        position_alignment="Official answer_only includes the final prompt token (position >= last_input_token_positions); training aligns final prompt plus teacher-response token.",
        implementation_differences=[
            "Qwen3-8B and E1 correctness teacher instruction replace original trait/concept prompts; explicitly task-adapted.",
            "Teacher responses are task-constrained single-token yes/no decisions; fit truth is used only for successful-teacher filtering.",
            "Exact upstream Focused module and aligned MSE are imported directly without modifying official files.",
            "For answer_only steering, unchanged causal prefixes are cached with the common Engine; only final prompt and response positions require a differentiable suffix.",
            "Cache the frozen teacher's aligned two-position residuals at all layers, and reuse across random seeds.",
            "Use the official unnormalized MSE option, the official default 0.1 regularization, and a fixed one-epoch budget.",
            "Canonical Engine prefill/final-token inference and FP32 yes/no projection are retained.",
        ],
        smoke_only=smoke,torch_version=torch.__version__)


class PSRRunner:
    def __init__(self,config):
        self.config=config
        module,config_cls,self.mse=official_components()
        self.tokenizer=AutoTokenizer.from_pretrained(config["model_path"],local_files_only=True)
        self.model=AutoModelForCausalLM.from_pretrained(config["model_path"],local_files_only=True,
            dtype=torch.bfloat16,device_map={"":"cuda:0"},attn_implementation="sdpa")
        self.engine=Engine("qwen",model=self.model,tokenizer=self.tokenizer)
        self.model.eval().requires_grad_(False)
        self.device=self.model.get_input_embeddings().weight.device
        token_lists=[self.tokenizer.encode(x,add_special_tokens=False) for x in LABELS]
        if any(len(t)!=1 for t in token_lists) or len({t[0] for t in token_lists})!=2:
            raise ValueError("Common yes/no labels must be distinct single tokens")
        self.label_ids=[t[0] for t in token_lists]
        native_config=config_cls(layers=[config["layer"]],device=str(self.device),

            steering_location="answer_only",use_location_bias=True,use_steering_coeff_bias=False,
            regularization_coefficient=config["regularization_coefficient"],dtype=torch.float32)
        self.ax=module(native_config,self.model.config.hidden_size).to(self.device)
        self.factor=None
        self.capture=False
        self.residuals={}
        self.regularization=None
        self.forward_count=0
        self.handles=[self.model.model.layers[config["layer"]].register_forward_hook(self._intervene)]
        for index,layer in enumerate(self.model.model.layers):
            self.handles.append(layer.register_forward_hook(self._capture_hook(index)))

    def _intervene(self,module,inputs,output):
        if self.factor is None:return output
        value=output[0] if isinstance(output,tuple) else output
        # This hook is active only on final prompt / answer suffix positions.
        # The unchanged prefix is computed with factor=None.
        result=self.ax(value,last_system_token_positions=torch.zeros(1,dtype=torch.long,device=self.device),
            last_input_token_positions=torch.zeros(1,dtype=torch.long,device=self.device),
            user_steering_coeffs=torch.tensor([self.factor],device=self.device))
        self.regularization=result.regularization_loss
        return (result.steered_output,)+output[1:] if isinstance(output,tuple) else result.steered_output

    def _capture_hook(self,index):
        def hook(module,inputs,output):
            if self.capture:self.residuals[index]=output[0] if isinstance(output,tuple) else output
        return hook

    def token_ids(self,prompt,teacher=False):
        system=self.config["teacher_system_prompt"] if teacher else SYSTEM
        rendered=self.tokenizer.apply_chat_template([dict(role="system",content=system),dict(role="user",content=prompt)],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=self.tokenizer(rendered,return_tensors="pt",add_special_tokens=False)["input_ids"].to(self.device)
        if ids.shape[1]+1>self.model.config.max_position_embeddings:
            raise ValueError("Complete input exceeds Qwen3 context; no truncation")
        return ids

    def prefix(self,ids):
        self.factor=None;self.capture=False
        result=self.engine._prefill(ids,ids.shape[1]-1)
        self.forward_count+=math.ceil((ids.shape[1]-1)/self.engine.prefill_chunk)
        return result

    def score_ids(self,ids,strength=1.0,raw=False,prefix=None):
        prefix=self.prefix(ids) if prefix is None else prefix
        self.factor=None if raw else strength
        self.capture=False
        context=Context("",ids,prefix,int(ids.shape[1]))
        result=self.engine.run(context,capture=False)
        self.forward_count+=1
        torch.cuda.synchronize()
        return float(result["ell"])

    def suffix_residuals(self,ids,response_id,strength=None,prefix=None):
        prefix=self.prefix(ids) if prefix is None else prefix
        suffix=torch.cat([ids[:,-1:],torch.tensor([[response_id]],device=self.device)],dim=1)
        start=ids.shape[1]-1;end=ids.shape[1]+1
        self.factor=strength;self.capture=True;self.residuals={};self.regularization=None
        self.model.model(input_ids=suffix,attention_mask=torch.ones((1,end),dtype=torch.long,device=self.device),
            past_key_values=self.engine._copy_cache(prefix),use_cache=True,
            cache_position=torch.arange(start,end,device=self.device),
            position_ids=torch.arange(start,end,device=self.device).unsqueeze(0))
        self.forward_count+=1
        result=[self.residuals[i] for i in range(len(self.model.model.layers))]
        self.capture=False;self.residuals={}
        return result,self.regularization

    def loss(self,student,teacher):
        z=torch.zeros(1,dtype=torch.long,device=self.device)
        e=torch.ones(1,dtype=torch.long,device=self.device)
        return self.mse(student,z,e,teacher,z,e)[0]

    def scorer(self,strength):
        def score(prompt,uid):
            ids=self.token_ids(prompt)
            before=self.forward_count
            value=self.score_ids(ids,strength)
            return dict(ell=value,tokens=int(ids.shape[1]),forward_count=self.forward_count-before,
                peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
        return score

    def close(self):
        for handle in self.handles:handle.remove()
        self.handles=[]


def fit_candidates(seed,smoke):
    values=unique_views("fit")
    if smoke:
        # Same two fit examples for every implementation check; no test use.
        values=values[:2]
    return values


def build_teacher(args,smoke=False):
    folder=args.output/"teacher" if smoke else args.teacher_dir
    folder.mkdir(parents=True,exist_ok=True)
    config=configuration(args,smoke)
    fixed={k:config[k] for k in ("model_path","teacher_system_prompt","teacher_response","teacher_success_filter","answer_labels")}
    fixed["smoke_only"]=smoke
    spec=folder/"teacher_config.json"
    if spec.exists() and read_json(spec)!=fixed:raise ValueError("Teacher cache belongs to another configuration")
    write_json(spec,fixed)
    if (folder/"summary.json").exists():return folder,read_json(folder/"summary.json")
    existing={}
    records_file=folder/"records.jsonl"
    if records_file.exists():
        for line in records_file.read_text().splitlines():
            row=json.loads(line);existing[row["uid"]]=row
    runner=PSRRunner(config)

    torch.cuda.reset_peak_memory_stats()
    candidates=fit_candidates(args.seed,smoke)
    with records_file.open("a",encoding="utf-8",buffering=1) as f:
        for i,row in enumerate(candidates):
            uid=row["uid"]
            if uid in existing:continue
            began=time.perf_counter();forward_before=runner.forward_count
            ids=runner.token_ids(get_prompt(uid),teacher=True)
            with torch.no_grad():
                prefix=runner.prefix(ids)
                ell=runner.score_ids(ids,raw=True,prefix=prefix)
                prediction=int(ell>0)
                accepted=prediction==int(row["label"])
                if accepted:
                    response_id=runner.label_ids[0 if prediction else 1]
                    residuals,_=runner.suffix_residuals(ids,response_id,prefix=prefix)
                    torch.save(dict(uid=uid,response_id=response_id,
                        residuals=[x.detach().cpu() for x in residuals]),folder/(uid+".pt"))
            torch.cuda.synchronize()
            record=dict(uid=uid,source_id=row["source_id"],label=int(row["label"]),teacher_prediction=prediction,
                teacher_ell=ell,accepted=accepted,tokens=int(ids.shape[1]),wall_seconds=time.perf_counter()-began,
                forward_count=runner.forward_count-forward_before,peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
            existing[uid]=record;f.write(json.dumps(record)+"\n")
            if (i+1)%100==0:write_json(folder/"status.json",dict(stage="teacher",inputs=i+1,total_inputs=len(candidates)))
    records=[existing[r["uid"]] for r in candidates]
    result=dict(candidate_unique_inputs=len(records),accepted_unique_inputs=sum(r["accepted"] for r in records),
        rejected_unique_inputs=sum(not r["accepted"] for r in records),split="fit",
        teacher_wall_seconds=sum(r["wall_seconds"] for r in records),
        teacher_peak_memory_gib=max(r.get("peak_memory_gib",0.0) for r in records),
        teacher_forward_count=sum(r["forward_count"] for r in records) if all("forward_count" in r for r in records) else None,teacher_config=fixed,
        source_pair_count=None if smoke else 2540,actual_experiment_results=False)
    write_json(folder/"summary.json",result)
    runner.close()
    return folder,result


def train(args,smoke=False):
    args.output.mkdir(parents=True,exist_ok=True)
    config=configuration(args,smoke)
    config_file=args.output/"training_config.json"
    if config_file.exists() and read_json(config_file)!=config:raise ValueError("Different output configuration")
    write_json(config_file,config)
    if (args.output/"training.json").exists() and (args.output/"model.pt").exists():return read_json(args.output/"training.json")
    teacher_folder,teacher_summary=build_teacher(args,smoke)
    records=[json.loads(s) for s in (teacher_folder/"records.jsonl").read_text().splitlines()]
    fit=[r for r in records if r["accepted"]]
    if not fit:raise RuntimeError("No successful fit-only teacher responses; do not substitute gold responses")
    random.Random(args.seed).shuffle(fit)
    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);np.random.seed(args.seed)
    runner=PSRRunner(config)
    initial=torch.cat([x.detach().flatten() for x in runner.ax.parameters()]).clone()
    optimizer=torch.optim.AdamW(runner.ax.parameters(),lr=args.lr,weight_decay=1e-6)
    steps=len(fit)*args.epochs
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda i:max(0.0,1-i/max(1,steps)))
    start_step,previous_seconds,previous_peak=0,0.0,0.0
    if (args.output/"checkpoint.pt").exists():
        saved=torch.load(args.output/"checkpoint.pt",map_location="cpu",weights_only=False)
        if saved["config"]!=config:raise ValueError("Checkpoint configuration differs")
        runner.ax.load_state_dict(saved["vector"]);optimizer.load_state_dict(saved["optimizer"]);scheduler.load_state_dict(saved["scheduler"])
        start_step,previous_seconds=saved["step"],saved["fit_seconds"]
        runner.forward_count=saved.get("training_forward_count",0)
        previous_peak=saved.get("peak_memory_gib",0.0)
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with (args.output/"training_steps.jsonl").open("a",encoding="utf-8",buffering=1) as f:
        for step in range(start_step,steps):
            row=fit[step%len(fit)]
            cached=torch.load(teacher_folder/(row["uid"]+".pt"),map_location="cpu",weights_only=True)
            teacher=[x.to(runner.device) for x in cached["residuals"]]
            ids=runner.token_ids(get_prompt(row["uid"]))
            optimizer.zero_grad(set_to_none=True)
            student,regularization=runner.suffix_residuals(ids,cached["response_id"],strength=1.0)
            mse=runner.loss(student,teacher)
            loss=mse if regularization is None else mse+regularization
            if not torch.isfinite(loss):raise ValueError("Nonfinite official PSR loss")
            loss.backward();optimizer.step();scheduler.step()
            torch.cuda.synchronize()
            elapsed=previous_seconds+time.perf_counter()-started
            f.write(json.dumps(dict(step=step+1,uid=row["uid"],loss=float(loss.detach()),mse=float(mse.detach()),
                regularization=0.0 if regularization is None else float(regularization.detach()),elapsed_seconds=elapsed))+"\n")
            if (step+1)%100==0 or step+1==steps:
                torch.save(dict(vector={k:v.detach().cpu() for k,v in runner.ax.state_dict().items()},
                    optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),step=step+1,fit_seconds=elapsed,config=config,training_forward_count=runner.forward_count,
                    peak_memory_gib=max(previous_peak,torch.cuda.max_memory_allocated()/2**30)),args.output/"checkpoint.pt")
                write_json(args.output/"status.json",dict(stage="training",step=step+1,total_steps=steps))
    fit_seconds=previous_seconds+time.perf_counter()-started

    vector={k:v.detach().cpu() for k,v in runner.ax.state_dict().items()}
    torch.save(dict(vector=vector,config=config),args.output/"model.pt")
    params=sum(x.numel() for x in runner.ax.parameters())
    change=float(torch.linalg.vector_norm(torch.cat([x.detach().flatten() for x in runner.ax.parameters()])-initial))
    metadata=dict(config,execution_kind="smoke_validation" if smoke else "training",
        actual_experiment_results=not smoke,parameter_count=params,trainable_parameter_count=params,stored_parameter_count=params,
        backbone_parameter_count=sum(x.numel() for x in runner.model.parameters()),training_steps=steps,
        train_unique_inputs=len(fit),teacher_summary=teacher_summary,teacher_directory=str(teacher_folder),
        parameter_fit_wall_seconds=fit_seconds,training_wall_seconds=fit_seconds+teacher_summary["teacher_wall_seconds"],
        training_gpu_hours=(fit_seconds+teacher_summary["teacher_wall_seconds"])/3600,
        training_peak_memory_gib=max(previous_peak,torch.cuda.max_memory_allocated()/2**30,teacher_summary["teacher_peak_memory_gib"]),
        training_forward_count=runner.forward_count,training_backward_count=steps,parameter_change_l2=change,gpu_visibility=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cost_accounting="Per-seed standalone cost includes reusable teacher preparation. Actual multi-seed execution prepares the teacher only once.")
    if smoke:
        probes=[]
        for row in unique_views("dev")[:2]:
            ids=runner.token_ids(get_prompt(row["uid"]))
            probes.append(dict(uid=row["uid"],raw_ell=runner.score_ids(ids,raw=True),adapted_ell=runner.score_ids(ids)))
        saved=torch.load(args.output/"model.pt",map_location="cpu",weights_only=False)
        with torch.no_grad():
            for parameter in runner.ax.parameters():parameter.zero_()
        runner.ax.load_state_dict(saved["vector"])
        for row in probes:
            row["reloaded_ell"]=runner.score_ids(runner.token_ids(get_prompt(row["uid"])))
            row["absolute_reload_difference"]=abs(row["adapted_ell"]-row["reloaded_ell"])
        metadata.update(probes=probes,max_reload_difference=max(r["absolute_reload_difference"] for r in probes),
            max_output_change=max(abs(r["adapted_ell"]-r["raw_ell"]) for r in probes))
        metadata["smoke_passed"]=change>0 and metadata["max_reload_difference"]<=1e-6 and metadata["max_output_change"]>1e-6
        write_json(args.output/"smoke.json",metadata)
        if not metadata["smoke_passed"]:raise RuntimeError("PSR smoke failed; do not launch a full run")
    write_json(args.output/"training.json",metadata)
    write_json(args.output/"status.json",dict(stage="smoke_complete" if smoke else "trained",complete=True))
    runner.close()
    return metadata


def load_runner(output):
    saved=torch.load(output/"model.pt",map_location="cpu",weights_only=False)
    runner=PSRRunner(saved["config"]);runner.ax.load_state_dict(saved["vector"])
    return runner,saved["config"]


def select(args):
    frozen=args.output/"frozen.json"
    if frozen.exists():return read_json(frozen)
    runner,config=load_runner(args.output)
    candidates=[float(x) for x in args.strengths.split(",")]
    if not candidates or len(candidates)>12 or any(x<=0 for x in candidates):raise ValueError("Invalid strength candidates")
    views=unique_views("dev")
    scores_file=args.output/"dev_selection_scores.jsonl";existing={}
    if scores_file.exists():
        for line in scores_file.read_text().splitlines():
            r=json.loads(line);existing[(r["uid"],r["strength"])]=r
    values={x:[] for x in candidates};began=time.perf_counter()
    with scores_file.open("a",encoding="utf-8",buffering=1) as f:
        for i,row in enumerate(views):
            ids=None
            for strength in candidates:
                key=(row["uid"],strength)
                if key not in existing:
                    if ids is None:ids=runner.token_ids(get_prompt(row["uid"]))
                    record=dict(uid=row["uid"],strength=strength,label=int(row["label"]),ell=runner.score_ids(ids,strength))
                    existing[key]=record;f.write(json.dumps(record)+"\n")
                r=existing[key]
                values[strength].append(float(np.logaddexp(0,-(2*r["label"]-1)*r["ell"])))
            if (i+1)%100==0:write_json(args.output/"status.json",dict(stage="development_selection",inputs=i+1,total_inputs=len(views)))
    summary=[dict(strength=s,mean_binary_nll=float(np.mean(v)),unique_dev_inputs=len(v)) for s,v in values.items()]
    selected=min(summary,key=lambda r:(r["mean_binary_nll"],r["strength"]))
    result=dict(config,selected_strength=selected["strength"],selection_metric="mean dev binary negative log likelihood",
        selection_split="dev",selection_candidates=summary,selection_wall_seconds=time.perf_counter()-began,
        decision_threshold=0.0,frozen_before_test=True)
    write_json(frozen,result);runner.close();return result


def evaluate(args):
    from .inference import score_split
    from .evaluate import evaluate as summarize
    frozen=read_json(args.output/"frozen.json");metadata=read_json(args.output/"training.json")
    if metadata["execution_kind"]=="smoke_validation":raise ValueError("Smoke weights are not full E1 results")
    runner,config=load_runner(args.output)
    method=METHOD+"-seed"+str(config["seed"])
    metadata.update(frozen_configuration=frozen,method=method,execution_kind="model_inference")

    scores=args.output/args.split/"scores.jsonl"
    collected=score_split(method,args.split,runner.scorer(frozen["selected_strength"]),scores,metadata)
    metadata["inference_gpu_hours"]=collected["inference_wall_seconds"]/3600
    result=summarize(method,scores,args.split,REPORT,metadata)
    runner.close();return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=["cpu-check","smoke","teacher","train","select","evaluate","all"])
    p.add_argument("--output",type=Path,default=REPORT/"psr/seed11")
    p.add_argument("--teacher-dir",type=Path,default=REPORT/"psr/teacher_fit")
    p.add_argument("--seed",type=int,choices=[11,22,33],default=11)
    p.add_argument("--layer",type=int,default=18)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--epochs",type=int,default=1)
    p.add_argument("--strengths",default="0.5,1.0,2.0")
    p.add_argument("--split",choices=["dev","test"],default="test")
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.command=="cpu-check":
        module,config_cls,mse=official_components()
        ax=module(config_cls(layers=[18],use_steering_coeff_bias=False,regularization_coefficient=.1),8)
        x=torch.ones(1,2,8)
        y=ax(x,torch.zeros(1,dtype=torch.long),torch.zeros(1,dtype=torch.long),torch.ones(1))
        write_json(args.output/"cpu_import.json",dict(official_module_shape=list(y.steered_output.shape),official_mse_loaded=callable(mse),gpu_used=False,official_commit=COMMIT))
    elif args.command=="smoke":train(args,True)
    elif args.command=="teacher":build_teacher(args)
    elif args.command=="train":train(args)
    elif args.command=="select":select(args)
    elif args.command=="evaluate":evaluate(args)
    else:
        train(args);select(args);evaluate(args)
    print(json.dumps(dict(command=args.command,output=str(args.output),complete=True)))


if __name__=="__main__":main()
