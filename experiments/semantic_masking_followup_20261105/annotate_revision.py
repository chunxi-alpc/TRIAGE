"""Separate management extraction from semantic-dependency annotation."""
import argparse
import json
import os
import time
import traceback
from common import CONFIG, DATA, LOGS, RESULTS, append_row, read_rows, write_json
from revision import all_annotations, normalize_annotation
from source_annotations import INSTRUCTION, rule_spans
from annotate import parse_json
from annotation_queue import queue_path, claim, finish, remaining, nodes_for

EXTRACT = """
Extract independent management action clauses from the supplied ORIGINAL guideline text.
This is a text-editing task, not a dependency judgment. Extract actions even if the current condition needs no definition.
Only conservative_cuts inside designated deleted blocks are eligible. Leave the introductory condition, subject eligibility,
exceptions, definitions and numeric diagnostic cutoffs in place. Select the smallest complete action clause.
For example, from "If the tumor is resectable, surgical resection is recommended.", select "surgical resection is recommended."
Do not remove "If the tumor is resectable". Preserve other definitions and conditions regardless of whether they matter here.
Use exact verbatim characters and real block IDs. Never quote a reconstructed clinical question.
Return one JSON object containing:
conservative_cuts: a list of block_id and quote objects for independent action clauses in designated deleted blocks.
action_swap_safe: boolean; true only when two DIFFERENT independent action clauses from the supplied blocks can exchange text
without moving or removing any condition, definition or diagnostic cutoff.
action_swap_spans: the two block_id and quote objects if safe, otherwise an empty list.
extraction_reason: briefly identify extracted actions, or explain why no independent clause can be separated.
The existence of a self-contained condition is NOT a reason to leave conservative_cuts empty.
"""
def worker(stage, shard, shared_queue=False):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.set_num_threads(4)
    nodes=[n for n in read_rows(DATA/"nodes.jsonl") if stage=="full" or n["pilot"]]
    previous=all_annotations()
    todo=[n for n in nodes[shard::2] if not previous.get(n["node_id"],{}).get("management_extraction_completed") or previous.get(n["node_id"],{}).get("semantic_schema")!=2]
    if shared_queue:
        first=claim(queue_path(stage),dict(pid=os.getpid(),physical_gpu=os.environ["CUDA_VISIBLE_DEVICES"]))
        work_count=1+remaining(queue_path(stage)) if first is not None else 0
        work=nodes_for(stage,nodes,first) if first is not None else iter(())
    else:
        work_count=len(todo)
        work=iter(todo)
    specs={s["node_id"]:s for s in read_rows(DATA/"construction_specs.jsonl")}
    status_path=RESULTS/f"annotation_{stage}_{shard}_status.json";started=time.time()
    write_json(status_path,dict(status="loading_model",total=work_count,pid=os.getpid(),physical_gpu=os.environ["CUDA_VISIBLE_DEVICES"],shared_queue=shared_queue))
    if not work_count:
        write_json(status_path,dict(status="complete",done=0,total=0,successful=0,elapsed_seconds=0))
        return
    tokenizer=AutoTokenizer.from_pretrained(CONFIG["medgemma"],local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(CONFIG["medgemma"],local_files_only=True,
        dtype=torch.bfloat16,device_map={"":"cuda:0"},attn_implementation="sdpa")
    model.eval();torch.cuda.reset_peak_memory_stats()
    successful=0
    with (DATA/f"annotations_{stage}_{shard}.jsonl").open("a") as output, \
         (LOGS/f"annotation_requests_{stage}_{shard}.jsonl").open("a") as requestlog, \
         (LOGS/f"annotation_raw_{stage}_{shard}.jsonl").open("a") as rawlog:
        def request(node,prompt,task,max_tokens=2000):
            error=None
            for attempt in range(2):
                text=tokenizer.apply_chat_template([dict(role="user",content=prompt+
                    ("\nCorrect this JSON error: "+error if error else ""))],tokenize=False,add_generation_prompt=True)
                inputs=tokenizer(text,return_tensors="pt",add_special_tokens=False).to("cuda")
                count=int(inputs["input_ids"].shape[1])
                append_row(requestlog,dict(node_id=node["node_id"],task=task,attempt=attempt,prompt=text,input_tokens=count))
                if count>CONFIG["annotation_max_input_tokens"]:
                    raise ValueError(f"{count} annotation tokens; source retained without truncation")
                with torch.inference_mode():
                    generated=model.generate(**inputs,max_new_tokens=max_tokens,do_sample=False,pad_token_id=tokenizer.pad_token_id)
                raw=tokenizer.decode(generated[0,count:],skip_special_tokens=True)
                append_row(rawlog,dict(node_id=node["node_id"],task=task,attempt=attempt,text=raw))
                try:
                    value=parse_json(raw)
                    if task=="semantics":
                        for field in ("source_condition_action_mixed","source_semantic_dependency","dataset_definition_completed"):
                            if value.get(field) not in ("yes","no","unknown"):
                                raise ValueError(field+" must be exactly one of yes, no, unknown")
                    return value,count,int(generated.shape[1]-count)
                except ValueError as exc:
                    error=str(exc)
                    if attempt:
                        raise
        processed=0
        for i,node in enumerate(work):
            before=time.time();record=dict(node_id=node["node_id"],source_id=node["source_id"])
            try:
                old=previous.get(node["node_id"])
                if old and old.get("status")=="complete" and old.get("semantic_schema")==2:
                    value=dict(old["annotation"]);n_input=old.get("input_tokens",0);n_output=0
                else:
                    payload={k:node[k] for k in ("node_id","condition","ancestors","node_mapping",
                        "designated_management","designated_delete_block_ids","blocks")}
                    payload["original_rule_evidence"]=rule_spans(node)
                    payload["construction_specification"]=specs.get(node["node_id"])
                    value,n_input,n_output=request(node,INSTRUCTION+"\nDATA:\n"+json.dumps(payload,ensure_ascii=False),"semantics")
                if node["designated_delete_block_ids"]:
                    management_started=time.time()
                    targets=[b for b in node["blocks"] if b["block_id"] in node["designated_delete_block_ids"]]
                    payload=dict(condition=node["condition"],original_rule_evidence=rule_spans(node),
                        designated_management=node["designated_management"],
                        designated_delete_block_ids=node["designated_delete_block_ids"],blocks=targets)
                    if old and old.get("management_extraction_completed"):
                        extraction=dict(old["annotation"],extraction_reason=old["annotation"].get("management_extraction_reason"))
                    else:
                        extraction,_,_=request(node,EXTRACT+"\nDATA:\n"+json.dumps(payload,ensure_ascii=False),"management",1200)
                else:
                    management_started=time.time()
                    extraction=dict(conservative_cuts=[],action_swap_safe=False,action_swap_spans=[],
                        extraction_reason="No designated management located")
                value.update({k:extraction.get(k) for k in ("conservative_cuts","action_swap_safe","action_swap_spans")})
                value["management_extraction_reason"]=extraction.get("extraction_reason")
                record=normalize_annotation(node,value)
                record.update(management_extraction_completed=True,semantic_schema=2,input_tokens=n_input,output_tokens=n_output,
                    semantic_annotation_reused=bool(old and old.get("semantic_schema")==2),
                    annotation_seconds_for_projection=max(time.time()-before,old.get("annotation_seconds_for_projection",0) if old else 0))
                successful+=1
            except Exception as exc:
                traceback.print_exc();record.update(status="failed",error=str(exc))
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
            record["seconds"]=time.time()-before;append_row(output,record)
            processed=i+1
            if shared_queue:
                finish(queue_path(stage),node["node_id"],record["status"])
            write_json(status_path,dict(status="running",done=i+1,total=(i+1+remaining(queue_path(stage))) if shared_queue else len(todo),successful=successful,shared_queue=shared_queue,
                elapsed_seconds=time.time()-started,mean_seconds_per_node=(time.time()-started)/(i+1),
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,pid=os.getpid()))
    write_json(status_path,dict(status="complete",done=processed,total=processed,successful=successful,shared_queue=shared_queue,
        elapsed_seconds=time.time()-started,mean_seconds_per_node=(time.time()-started)/processed,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,pid=os.getpid()))
    if not successful:
        raise RuntimeError("No annotations completed")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["pilot","full"],required=True)
    parser.add_argument("--shard",type=int,choices=[0,1],required=True)
    parser.add_argument("--shared-queue",action="store_true")
    args=parser.parse_args();worker(args.stage,args.shard,args.shared_queue)
