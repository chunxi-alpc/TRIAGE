"""Automatic semantic labels; source deletion targets are supplied, never inferred here."""
import argparse
import json
import os
import time
import traceback
from common import CONFIG, DATA, LOGS, RESULTS, append_row, read_rows, write_json
from revision import all_annotations, normalize_annotation

INSTRUCTION = """
Annotate an automatic research pilot using ONLY the supplied original guideline blocks.
The downstream management passage and the blocks deleted by the experiment are FIXED in DATA.
They WILL be deleted even if the condition is semantically independent. Never change the deletion list.
There are three different questions:
1. Does the ORIGINAL condition passage mix a condition with management?
2. Does interpretation of that ORIGINAL condition require a definition/explanation inside the designated deleted blocks?
3. Does the dataset question already supply that required definition?
Use null if the source does not establish the answer. Mere co-location or treatment relevance is not semantic dependence.
A dependency positive needs a verbatim definition from the deleted passage and an explanation of the required term.
Quote literal characters from blocks, including original punctuation and math. No paraphrased source quotations.
For conservative_cuts, identify independent management words/sentences inside designated deleted blocks, while retaining every
condition, definition, numeric threshold, exception and footnote needed to interpret the rule.
For necessary_restores, quote ONLY necessary definitions from the deleted blocks. Never put a patient answer or recommendation
into a restoration. Leave empty when no necessary definition is removed. This is permitted; no positive quota exists.
action_swap_spans may contain two independent management-only source spans if both are located in these supplied blocks and
at least one is among the conservative cuts. If not available, action_swap_safe=false. Do not invent alternative management.
Ancestors are supplied as known fixed conditions. A constructed case's remaining clinical facts are explicitly listed.
If a construction specification is supplied, assess whether its complete predicate and fixed facts preserve the WHOLE original
question. Numeric ranges retain both ends; OR branches require fixed facts for the branches not varied. No population response
percentage may be treated as an individual clinical threshold. Original factual yes/no wording is kept, with no added variant.
A rejected construction may be reported invalid with a concrete reason; use unknown for uncertainty.
Return only one JSON object, using this schema:
{
 "source_condition_action_mixed": null,
 "source_semantic_dependency": null,
 "dataset_definition_completed": null,
 "dependency_reason": "brief explanation tied to source",
 "rule_evidence": [{"block_id":0,"quote":"literal original condition"}],
 "conservative_cuts": [{"block_id":0,"quote":"independent management only"}],
 "necessary_restores": [{"block_id":0,"quote":"necessary definition only"}],
 "direct_information_sufficient": null,
 "ancestors_information_sufficient": null,
 "full_information_sufficient": null,
 "action_swap_safe": false,
 "action_swap_spans": [],
 "missing_reference_queries": [],
 "construction_review": {"status":"not_applicable/valid/invalid/unknown","reason":"brief reason"}
}
"""
def parse_json(text):
    start=text.find("{")
    if start<0:
        raise ValueError("No JSON object")
    value,_=json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(value,dict):
        raise ValueError("Expected object")
    return value

from source_annotations import INSTRUCTION, rule_spans

def worker(stage, shard):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    nodes=[n for n in read_rows(DATA/"nodes.jsonl") if stage=="full" or n["pilot"]]
    done=all_annotations()
    todo=[n for n in nodes[shard::2] if n["node_id"] not in done]
    specs={s["node_id"]:s for s in read_rows(DATA/"construction_specs.jsonl")}
    status_path=RESULTS/f"annotation_{stage}_{shard}_status.json"
    started=time.time()
    write_json(status_path,dict(status="loading_model",total=len(todo),pid=os.getpid(),physical_gpu=os.environ["CUDA_VISIBLE_DEVICES"]))
    if not todo:
        write_json(status_path,dict(status="complete",done=0,total=0,elapsed_seconds=0))
        return
    tokenizer=AutoTokenizer.from_pretrained(CONFIG["medgemma"],local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(CONFIG["medgemma"],local_files_only=True,
        dtype=torch.bfloat16,device_map={"":"cuda:0"},attn_implementation="sdpa")
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    successful=0
    with (DATA/f"annotations_{stage}_{shard}.jsonl").open("a") as output, \
         (LOGS/f"annotation_requests_{stage}_{shard}.jsonl").open("a") as requestlog, \
         (LOGS/f"annotation_raw_{stage}_{shard}.jsonl").open("a") as rawlog:
        for i,node in enumerate(todo):
            before=time.time()
            record=dict(node_id=node["node_id"],source_id=node["source_id"])
            try:
                payload={k:node[k] for k in ("node_id","condition","ancestors","node_mapping",
                    "designated_management","designated_delete_block_ids","blocks")}
                payload["construction_specification"]=specs.get(node["node_id"])
                payload["original_rule_evidence"]=rule_spans(node)
                prompt=INSTRUCTION+"\nDATA:\n"+json.dumps(payload,ensure_ascii=False)
                error=None
                for attempt in range(2):
                    current=prompt+("\nPrevious JSON error: "+error if error else "")
                    text=tokenizer.apply_chat_template([dict(role="user",content=current)],tokenize=False,add_generation_prompt=True)
                    inputs=tokenizer(text,return_tensors="pt",add_special_tokens=False).to("cuda")
                    count=int(inputs["input_ids"].shape[1])
                    append_row(requestlog,dict(node_id=node["node_id"],attempt=attempt,prompt=text,input_tokens=count))
                    if count>CONFIG["annotation_max_input_tokens"]:
                        raise ValueError(f"{count} input tokens exceed annotation capacity; source retained without truncation")
                    with torch.inference_mode():
                        generated=model.generate(**inputs,max_new_tokens=CONFIG["annotation_max_new_tokens"],
                            do_sample=False,pad_token_id=tokenizer.pad_token_id)
                    raw=tokenizer.decode(generated[0,count:],skip_special_tokens=True)
                    append_row(rawlog,dict(node_id=node["node_id"],attempt=attempt,text=raw))
                    try:
                        value=parse_json(raw)
                        for field in ("source_condition_action_mixed","source_semantic_dependency","dataset_definition_completed"):
                            if value.get(field) not in ("yes","no","unknown"):
                                raise ValueError(field+" must be exactly one of yes, no, unknown")
                        break
                    except ValueError as exc:
                        error=str(exc)
                        if attempt:
                            raise
                record=normalize_annotation(node,value)
                record.update(input_tokens=count,output_tokens=int(generated.shape[1]-count))
                successful+=1
            except Exception as exc:
                traceback.print_exc()
                record.update(status="failed",error=str(exc))
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
            record["seconds"]=time.time()-before
            append_row(output,record)
            write_json(status_path,dict(status="running",done=i+1,total=len(todo),successful=successful,
                elapsed_seconds=time.time()-started,mean_seconds_per_node=(time.time()-started)/(i+1),
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,pid=os.getpid()))
    write_json(status_path,dict(status="complete",done=len(todo),total=len(todo),successful=successful,
        elapsed_seconds=time.time()-started,mean_seconds_per_node=(time.time()-started)/len(todo),
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,pid=os.getpid()))
    if not successful:
        raise RuntimeError("No automatic annotations completed")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["pilot","full"],required=True)
    parser.add_argument("--shard",type=int,choices=[0,1],required=True)
    args=parser.parse_args()
    worker(args.stage,args.shard)
