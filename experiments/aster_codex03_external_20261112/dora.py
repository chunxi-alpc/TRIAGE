"""Qwen3 DoRA via official PEFT; one-answer-token SFT on E1 only."""
from __future__ import annotations
import argparse, gc, json, math, random, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model
from .data import load_bundle, get_prompt
from .inference import score_split
from .evaluate import evaluate
from experiments.aster_cr.engine import Engine, SYSTEM

MODEL="/home/gcx/TRIAGE/models/modelscope/Qwen3-8B"
ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/"reports/aster_codex03_external_20261112"
METHOD="DoRA-E1-SFT"
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    tmp.replace(path)
def append(path,obj):
    with Path(path).open("a",encoding="utf-8") as f:
        f.write(json.dumps(obj,ensure_ascii=False,allow_nan=False)+"\n")
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
def load_base():
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,
        device_map={"":"cuda:0"},attn_implementation="sdpa")
    model.requires_grad_(False)
    return model,tok
def encode(tok,prompt):
    rendered=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],
        tokenize=False,add_generation_prompt=True,enable_thinking=False)
    return tok(rendered,add_special_tokens=False,return_tensors="pt")["input_ids"].to("cuda:0")
def fit_rows():
    bundle=load_bundle("fit");pairs=bundle["pairs"]
    n=Counter((p["kind"],p["source_id"]) for p in pairs)
    sources=defaultdict(set)
    for p in pairs:sources[p["kind"]].add(p["source_id"])
    w=defaultdict(float);labels={}
    for p in pairs:
        pw=1/(4*len(sources[p["kind"]])*n[p["kind"],p["source_id"]])
        for side in ("left","right"):
            uid=p[side];label=int(p[side+"_label"])
            if uid in labels and labels[uid]!=label:raise ValueError("Conflicting E1 label "+uid)
            labels[uid]=label;w[uid]+=pw/2
    N=len(labels)
    return [dict(uid=u,label=labels[u],weight=w[u]*N) for u in sorted(labels)]
def configuration(seed,lr,epochs,rank):
    return dict(method=METHOD,seed=seed,model_path=MODEL,model_revision=None,
        model_revision_note="Installed local snapshot does not declare an immutable upstream revision; no revision inferred.",
        backbone="Qwen3-8B",train_data="E1 / Dataset A",train_pairs=2540,
        objective="Full-vocabulary causal cross-entropy for the single correct answer token",
        supervision="E1 endpoint gold answers; no A/P consistency or C/F preservation loss",
        sample_weighting="Equal A/P/C/F mass, equal source mass within class, aggregate both endpoints onto unique inputs",
        implementation="Official PEFT DoRA LoraConfig(use_dora=True)",
        upstream_audit_revision="f8796e8",data_version="20261111-sol-agents-v2",peft_version="0.21.2",official_reference="https://github.com/NVlabs/DoRA",
        official_revision="7e2f10abbe8efe212c8fca1d983ae1d04ef13a18",
        implementation_differences=["Qwen3 q_proj/v_proj across all 36 layers",
          "One-token clinical answer SFT, original DoRA magnitude/direction parameterization unchanged"],
        rank=rank,lora_alpha=2*rank,lora_dropout=0.0,target_modules=["q_proj","v_proj"],
        learning_rate=lr,epochs=epochs,micro_batch=1,gradient_accumulation=8,
        optimizer="AdamW",weight_decay=0.0,warmup_ratio=.03,gradient_clip=1.0,
        dtype="bfloat16",gradient_checkpointing=True,answer_labels=["是","否"],
        threshold=0.0,hyperparameter_selection="Single configuration fixed before development/test; no test selection")
def inject(base,rank):
    return get_peft_model(base,LoraConfig(task_type="CAUSAL_LM",r=rank,lora_alpha=2*rank,
        target_modules=["q_proj","v_proj"],lora_dropout=0.0,use_dora=True,bias="none"))
def train(model,tok,rows,config,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    params=[p for p in model.parameters() if p.requires_grad]
    nparams=sum(p.numel() for p in params)
    base=model.get_base_model()
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    base.enable_input_require_grads();base.config.use_cache=False
    model.train()
    optim=torch.optim.AdamW(params,lr=config["learning_rate"],weight_decay=0.0)
    accumulation=config["gradient_accumulation"]
    updates_per_epoch=math.ceil(len(rows)/accumulation);total=updates_per_epoch*config["epochs"]
    warmup=max(1,round(total*config["warmup_ratio"]))
    step=0;started=time.perf_counter();forward_count=0;loss_total=0.0
    target_ids=[tok.encode(s,add_special_tokens=False) for s in ["是","否"]]
    if any(len(v)!=1 for v in target_ids):raise ValueError("Answer labels are not single tokens")
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(config["epochs"]):
        order=list(range(len(rows)));random.Random(config["seed"]+epoch).shuffle(order)
        optim.zero_grad(set_to_none=True)
        for i,j in enumerate(order):
            r=rows[j];ids=encode(tok,get_prompt(r["uid"]))
            hidden=base.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).last_hidden_state[:,-1]
            logits=base.lm_head(hidden).float()
            target=torch.tensor([target_ids[0 if r["label"] else 1][0]],device=ids.device)
            loss=F.cross_entropy(logits,target)*r["weight"]
            window=min(accumulation,len(rows)-(i//accumulation)*accumulation)
            (loss/window).backward()
            value=float(loss.detach());loss_total+=value;forward_count+=1
            if (i+1)%accumulation==0 or i+1==len(order):
                step+=1
                rate=step/warmup if step<=warmup else max(0.,(total-step+1)/max(1,total-warmup))
                for group in optim.param_groups:group["lr"]=config["learning_rate"]*rate
                torch.nn.utils.clip_grad_norm_(params,config["gradient_clip"])
                optim.step();optim.zero_grad(set_to_none=True)
                if step==1 or step%20==0 or step==total:
                    status=dict(stage="training",seed=config["seed"],epoch=epoch+1,micro_step=i+1,
                        optimizer_steps=step,total_optimizer_steps=total,loss=value,
                        seconds=time.perf_counter()-started,peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
                    append(out/"training.jsonl",status);write(out/"status.json",status)
            del hidden,logits,loss,ids
    torch.cuda.synchronize()
    seconds=time.perf_counter()-started
    costs=dict(parameter_count=nparams,trainable_parameter_count=nparams,stored_parameter_count=nparams,
        backbone_parameter_count=sum(p.numel() for p in base.parameters())-nparams,
        training_wall_seconds=seconds,training_gpu_hours=seconds/3600,
        training_peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,
        training_forward_count=forward_count,training_optimizer_steps=step,
        training_unique_inputs=len(rows),training_mean_weighted_loss=loss_total/max(1,forward_count),
        training_cost_scope="Training loop including tokenization; excludes model loading and evaluation")
    model.save_pretrained(out/"adapter",safe_serialization=True)
    config=dict(config,**costs);write(out/"training_and_implementation.json",config)
    base.gradient_checkpointing_disable();base.config.use_cache=True
    if hasattr(base,"disable_input_require_grads"):base.disable_input_require_grads()
    model.eval();del optim,params
    gc.collect();torch.cuda.empty_cache()
    return config
def scorer(model,tok):
    base=model.get_base_model() if hasattr(model,"get_base_model") else model
    engine=Engine("qwen",model=base,tokenizer=tok,device="cuda:0")
    def call(prompt,uid):
        before=engine.counts["prefills"]+engine.counts["final_forwards"]
        r=engine.score(prompt,capture=False)
        # Prefix chunks are counted separately below when a prompt spans chunks.
        return dict(ell=r["ell"],tokens=r["tokens"],
            forward_count=math.ceil(max(0,r["tokens"]-1)/engine.prefill_chunk)+1,
            peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
    return call
def run_seed(args):
    seed_all(args.seed);out=Path(args.output)/f"seed{args.seed}";out.mkdir(parents=True,exist_ok=True)
    conf=configuration(args.seed,args.lr,args.epochs,args.rank)
    write(out/"configuration.json",conf);write(out/"status.json",dict(stage="loading",seed=args.seed))
    base,tok=load_base()
    saved=out/"training_and_implementation.json"
    if (out/"adapter/adapter_config.json").exists() and saved.exists():
        previous=json.loads(saved.read_text(encoding="utf-8"))
        for key in ("seed","learning_rate","rank","epochs"):
            if previous[key]!=conf[key]:raise ValueError("Saved training configuration differs: "+key)
        conf=previous
        model=PeftModel.from_pretrained(base,out/"adapter",is_trainable=False)
    else:
        model=inject(base,args.rank)
        conf=train(model,tok,fit_rows(),conf,out)
    write(out/"frozen_configuration.json",dict(conf,frozen_before_test=True))
    call=scorer(model,tok)
    for split in ("dev","test"):
        write(out/"status.json",dict(stage="scoring",split=split,seed=args.seed))
        torch.cuda.reset_peak_memory_stats()
        scores=out/split/"scores.jsonl"
        m=score_split(METHOD+f"-seed{args.seed}",split,call,scores,metadata=conf)
        m["inference_gpu_hours"]=m["inference_wall_seconds"]/3600
        m["inference_cost_scope"]="Unique-input evaluation, including pair inputs and visited dynamic-path inputs"
        write(scores.with_suffix(".metadata.json"),m)
        evaluate(METHOD+f"-seed{args.seed}",scores,split,REPORT,metadata=m)
    write(out/"completion.json",dict(complete=True,seed=args.seed,adapter=str(out/"adapter")))
    write(out/"status.json",dict(stage="complete",seed=args.seed))
def smoke(args):
    seed_all(args.seed);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    base,tok=load_base()
    dev=load_bundle("dev")["views"][:2]
    raw_call=scorer(base,tok)
    before=[raw_call(get_prompt(r["uid"]),r["uid"])["ell"] for r in dev]
    del raw_call
    model=inject(base,args.rank)
    conf=configuration(args.seed,args.lr,1,args.rank);conf["gradient_accumulation"]=1
    subset=fit_rows()[:4]
    for r in subset:r["weight"]=1.0
    conf=train(model,tok,subset,conf,out)
    call=scorer(model,tok)
    after=[call(get_prompt(r["uid"]),r["uid"])["ell"] for r in dev]
    model.load_adapter(out/"adapter",adapter_name="restored",is_trainable=False)
    model.set_adapter("restored")
    restored=[call(get_prompt(r["uid"]),r["uid"])["ell"] for r in dev]
    change=max(abs(a-b) for a,b in zip(before,after))
    reload_error=max(abs(a-b) for a,b in zip(after,restored))
    result=dict(passed=change>0 and reload_error<1e-4,fit_inputs=4,dev_inputs=2,
        max_logit_change=change,reload_max_abs_error=reload_error,
        parameter_count=conf["parameter_count"],peak_memory_gib=conf["training_peak_memory_gib"],
        training_seconds=conf["training_wall_seconds"],physical_gpu=5,
        actual_formal_result=False,before=before,after=after,restored=restored)
    write(out/"smoke_result.json",result)
    if not result["passed"]:raise RuntimeError("DoRA effect/reload smoke failed")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("command",choices=["smoke","run-seed"])
    p.add_argument("--seed",type=int,default=11);p.add_argument("--rank",type=int,default=8)
    p.add_argument("--lr",type=float,default=1e-4);p.add_argument("--epochs",type=int,default=1)
    p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    if args.command=="smoke":smoke(args)
    else:run_seed(args)
