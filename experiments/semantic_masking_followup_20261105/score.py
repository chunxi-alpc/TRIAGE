"""Identical binary scoring with reuse keyed by complete rendered input and model settings."""
import argparse
import importlib.util
import json
import os
import time
import traceback
from pathlib import Path
from common import CONFIG, DATA, LOGS, RESULTS, append_row, read_rows, write_json

SYSTEM = """判断当前节点的条件是否成立。只判断该条件，不选择后续治疗方案。
祖先条件给定为到达当前节点的已知条件；不能把后续处置建议当作患者事实。
根据病例事实、当前条件及给定的指南说明作答。只输出“是”或“否”，不输出解释。
For English inputs, 是 means Yes and 否 means No. Judge the current condition only."""

def make_prompt(row):
    ancestors="\n".join(f"{i+1}. {a['condition']} -> {'是' if a['answer']=='yes' else '否'}"
                         for i,a in enumerate(row["ancestors"])) or "(无)"
    return (f"指南相关原文：\n{row['context']}\n\n"
            f"到达当前节点的祖先条件：\n{ancestors}\n\n"
            f"病例：\n{row['patient']}\n\n"
            f"当前判断条件：\n{row['condition']}\n\n答案：")

def settings(token_ids):
    return dict(model=CONFIG["qwen"],dtype="bfloat16",attention="sdpa",device="single_cuda_gpu",
        enable_thinking=False,add_generation_prompt=True,add_special_tokens=False,
        use_cache=False,logits_to_keep=1,labels=["是","否"],label_token_ids=token_ids,
        scoring="next_token_binary_log_softmax",system=SYSTEM,seed=CONFIG["seed"])

def reusable(record, rendered_prompt, model_settings):
    return (record.get("status")=="complete" and record.get("rendered_prompt")==rendered_prompt
            and record.get("model_settings")==model_settings)

def rebind_score(cached, row, origin):
    result=dict(cached)
    result.update({k:row[k] for k in ("score_id","example_id","node_id","source_id","population","view","gold","split")})
    result.update(correct=result["prediction"]==row["gold"],reused_from=origin,user_prompt=make_prompt(row),
                  reused_score_id=cached.get("score_id"))
    return result

def legacy_cache(model_settings):
    old=Path(CONFIG["old_run"])
    config=json.loads((old/"config.json").read_text())
    source=(old/"score.py").read_text()
    required=['dtype=torch.bfloat16','attn_implementation="sdpa"','enable_thinking=False',
              'add_generation_prompt=True','add_special_tokens=False','use_cache=False','logits_to_keep=1']
    if config["qwen"]!=CONFIG["qwen"] or config["seed"]!=CONFIG["seed"] or not all(s in source for s in required):
        return {}
    # Compare the legacy system message as well as the actual saved rendered input.
    import ast
    module=ast.parse(source)
    old_system=next(ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign) and
                    any(isinstance(t,ast.Name) and t.id=="SYSTEM" for t in n.targets))
    if old_system!=SYSTEM:
        return {}
    cache={}
    for stage in ("pilot","full"):
        for shard in (0,1):
            prompts={r["score_id"]:r["prompt"] for r in read_rows(old/f"logs/scoring_prompts_{stage}_{shard}.jsonl")}
            for row in read_rows(old/f"results/scores_{stage}_{shard}.jsonl"):
                prompt=prompts.get(row["score_id"])
                if row.get("status")=="complete" and prompt is not None and row.get("label_token_ids")==model_settings["label_token_ids"]:
                    record=dict(row,rendered_prompt=prompt,model_settings=model_settings,
                        legacy_settings_evidence="Legacy config, literal score.py settings, system text, label IDs and saved rendered prompt")
                    cache[prompt]=record
    return cache

def worker(stage, shard):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4); torch.manual_seed(CONFIG["seed"])
    rows=read_rows(DATA/f"inputs_{stage}.jsonl")[shard::2]
    started=time.time()
    status_path=RESULTS/f"scoring_{stage}_{shard}_status.json"
    write_json(status_path,dict(status="loading_tokenizer",total=len(rows),pid=os.getpid(),
        physical_gpu=os.environ["CUDA_VISIBLE_DEVICES"]))
    tokenizer=AutoTokenizer.from_pretrained(CONFIG["qwen"],local_files_only=True)
    labels=[tokenizer.encode(s,add_special_tokens=False) for s in ("是","否")]
    if any(len(ids)!=1 for ids in labels):
        raise RuntimeError("Binary labels must be single tokens")
    token_ids=[ids[0] for ids in labels]
    signature=settings(token_ids)
    cache=legacy_cache(signature)
    for parent in CONFIG.get("prior_score_runs", []):
        for prior_stage in ("pilot", "full"):
            for prior_shard in (0, 1):
                for record in read_rows(Path(parent)/f"results/scores_{prior_stage}_{prior_shard}.jsonl"):
                    if record.get("status")=="complete" and record.get("model_settings")==signature:
                        cache[record["rendered_prompt"]]=record
    for prior in ("pilot","full"):
        for part in (0,1):
            for record in read_rows(RESULTS/f"scores_{prior}_{part}.jsonl"):
                if record.get("status")=="complete" and record.get("model_settings")==signature:
                    cache[record["rendered_prompt"]]=record
    output=RESULTS/f"scores_{stage}_{shard}.jsonl"
    existing={r["score_id"]:r for r in read_rows(output) if r.get("status")=="complete"}
    model=None
    successful=reused=inferred=0
    with output.open("a") as handle:
        for i,row in enumerate(rows):
            messages=[dict(role="system",content=SYSTEM),dict(role="user",content=make_prompt(row))]
            text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            current=existing.get(row["score_id"])
            if current and reusable(current,text,signature) and current.get("gold")==row["gold"]:
                successful+=1; reused+=1
                continue
            cached=cache.get(text)
            if cached and reusable(cached,text,signature):
                record=rebind_score(cached,row,"identical_rendered_input_and_model_settings")
                append_row(handle,record)
                successful+=1; reused+=1
            else:
                record={k:row[k] for k in ("score_id","example_id","node_id","source_id","population","view","gold","split")}
                record.update(model_settings=signature,rendered_prompt=text,user_prompt=make_prompt(row))
                row_started=time.time()
                try:
                    if model is None:
                        write_json(status_path,dict(status="loading_model",total=len(rows),pid=os.getpid()))
                        model=AutoModelForCausalLM.from_pretrained(CONFIG["qwen"],local_files_only=True,
                            dtype=torch.bfloat16,device_map={"":"cuda:0"},attn_implementation="sdpa")
                        model.eval(); torch.cuda.reset_peak_memory_stats()
                    inputs=tokenizer(text,add_special_tokens=False,return_tensors="pt").to("cuda")
                    count=int(inputs["input_ids"].shape[1])
                    if count>CONFIG["scoring_max_tokens"]:
                        raise ValueError(f"{count} input tokens; source was not truncated")
                    with torch.inference_mode():
                        logits=model(**inputs,use_cache=False,logits_to_keep=1).logits[0,-1].float()
                        raw_logp=torch.log_softmax(logits,dim=-1)[token_ids]
                        binary_logp=torch.log_softmax(logits[token_ids],dim=-1)
                    p_yes=float(binary_logp[0].exp())
                    record.update(status="complete",label_token_ids=token_ids,
                        raw_log_probabilities=raw_logp.cpu().tolist(),log_probabilities=binary_logp.cpu().tolist(),
                        p_yes=p_yes,margin=float(binary_logp[0]-binary_logp[1]),prediction=int(p_yes>=0.5),
                        correct=int(p_yes>=0.5)==row["gold"],input_tokens=count,seconds=time.time()-row_started)
                    successful+=1; inferred+=1
                    cache[text]=record
                except Exception as exc:
                    traceback.print_exc()
                    record.update(status="failed",error=str(exc),seconds=time.time()-row_started)
                    if "out of memory" in str(exc).lower():
                        torch.cuda.empty_cache()
                append_row(handle,record)
            if (i+1)%10==0 or i+1==len(rows):
                write_json(status_path,dict(status="running",done=i+1,total=len(rows),successful=successful,
                    reused=reused,inferred=inferred,elapsed_seconds=time.time()-started,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if model is not None else 0,pid=os.getpid()))
    write_json(status_path,dict(status="complete",done=len(rows),total=len(rows),successful=successful,
        reused=reused,inferred=inferred,elapsed_seconds=time.time()-started,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if model is not None else 0,pid=os.getpid()))
    if rows and not successful:
        raise RuntimeError("No binary scores completed")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["pilot","full"],required=True)
    parser.add_argument("--shard",type=int,choices=[0,1],required=True)
    args=parser.parse_args()
    worker(args.stage,args.shard)
