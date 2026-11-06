"""MedGemma-assisted additions and label-independent semantic masking."""
import argparse,copy,json,os,time,traceback
from pathlib import Path
from .common import *
from .data import condition,canonical_guide,replace_all,set_condition
MASK_INSTRUCTION="""Extract a semantic reading view of this entire decision tree for the CURRENT condition.
No patient, truth labels, expected answers or construction records are supplied. Keep every decision condition and transition unchanged.
The application masks all terminal action payloads and all supplied definition passages except the IDs you retain.
Select a terminal action ONLY if its name/description is necessary to interpret the current condition or an ancestor.
Select original definition passages necessary to interpret any retained condition; omit independent treatment recommendations.
Return only one compact valid JSON object: {"keep_action_ids":["L3"],"keep_definition_indices":[0,2],"reason":"One short sentence."}.
Output IDs and indices first. Do not reproduce the tree, quotations, or explanations for every retained item.
No Markdown, commentary, or text after the closing brace. Keep reason to one short sentence.
Use only actual IDs/indices from INPUT. Empty lists are allowed."""
from .construction_prompts import MAIN_INSTRUCTION,CONDITION_INSTRUCTION,DEF_INSTRUCTION
from .construction_support import repair_condition_state,source_edit_text
REVISION=3
KNOWN_RECONSTRUCTION={"expanded-development-732-G0","expanded-development-428-G0"}

def validate(g,v):
    if g["cohort"]=="definition":
        if "definition_valid" not in v or type(v["definition_valid"]) is not bool:
            raise ValueError("Definition schema requires explicit boolean definition_valid")
        if v["definition_valid"] is False:
            return dict(available=False,reason=v.get("reason","Explicitly unavailable definition example"))
        required=("short_condition","shared_record","fact_sentences","facts","predicate",
                  "ancestor_predicates","decisive_sentences","protected_action_ids",
                  "condition_change","clinical_background","treatment_preference","reason")
        missing=[key for key in required if key not in v]
        if missing:raise ValueError("Definition schema missing required fields: "+", ".join(missing))
        if not isinstance(v["short_condition"],str) or not v["short_condition"].strip():
            raise ValueError("Definition short_condition must be nonempty text")
        if not isinstance(v["shared_record"],str):
            raise ValueError("Definition shared_record must be text")
        if (not isinstance(v["fact_sentences"],list) or len(v["fact_sentences"])!=2 or
                any(not isinstance(x,str) or not x.strip() for x in v["fact_sentences"])):
            raise ValueError("Definition fact_sentences must contain two nonempty observations")
    fs=v["facts"];expr=v["predicate"]
    if len(fs)!=2 or [int(evaluate(expr,x)) for x in fs]!=[0,1]:raise ValueError("Complete predicate expected state labels [0,1]; got "+str([int(evaluate(expr,x)) for x in fs]))
    anc={a["gid"]:a["predicate"] for a in v["ancestor_predicates"]}
    for a in g["ancestors"]:
        if a["gid"] not in anc:raise ValueError("Missing complete ancestor predicate "+a["gid"])
        if any(int(evaluate(anc[a["gid"]],f))!=a["label"] for f in fs):raise ValueError("An ancestor "+a["gid"]+" expected "+str(a["label"])+" in BOTH patients; got "+str([int(evaluate(anc[a["gid"]],f)) for f in fs]))
    used=fields(expr)
    for ex in anc.values():used|=fields(ex)
    for key in ("clinical_background","treatment_preference"):
        p=v.get(key,{})
        if p.get("valid"):
            if len(p.get("texts",[]))!=2 or p["texts"][0]==p["texts"][1]:
                p.update(valid=False,reason="Two distinct backgrounds not supplied")
            elif not p.get("changed_fields") or used&set(p["changed_fields"]):
                p.update(valid=False,reason="Changed field belongs to target/ancestor condition")
    repair_condition_state(g,v)
    c=v.get("condition_change",{})
    if c.get("valid"):
        try:
            s=int(c["base_state"]); ed=c["edits"]
            if s not in (0,1):raise ValueError("C base_state must be 0 or 1")
            original=source_edit_text(g)
            before_target=condition(g)
            after_target=condition(dict(g,guideline=replace_all(g["guideline"],ed)))
            before_defs=[d["quote"] for d in g["definitions"]]
            after_defs=[d["quote"] for d in replace_all(g["definitions"],ed)]
            if before_target==after_target and before_defs==after_defs:raise ValueError("C does not change the current condition or a necessary definition")
            if not ed or any(not e["old"] or e["old"] not in original or e["old"]==e["new"] for e in ed):
                raise ValueError("C source edit not found or unchanged")
            if int(evaluate(c["predicate"],fs[s]))==s:raise ValueError("Complete condition does not flip")
            if g["cohort"]=="definition" and before_defs==after_defs:
                raise ValueError("Definition pair does not modify a supplied definition")
            # Edits to ancestor text must include an executable updated predicate.
            updated=c.get("ancestor_predicates",[])
            updated={a["gid"]:a["predicate"] for a in updated}
            for a in g["ancestors"]:
                touched=any(e["old"] in a["condition"] for e in ed)
                ae=updated.get(a["gid"],anc[a["gid"]])
                if touched and a["gid"] not in updated:raise ValueError("C changes an ancestor without its updated predicate")
                if int(evaluate(ae,fs[s]))!=a["label"]:raise ValueError("C changes ancestor truth")
        except (ValueError,KeyError,IndexError,TypeError) as exc:c.update(valid=False,reason=str(exc))
    if g["cohort"]=="definition":
        if len(v["fact_sentences"])!=2 or not v["short_condition"]:raise ValueError("Missing definition patients/short condition")
        terms=[d.get("target_term","") for d in g.get("definition_links",[])]
        if not any(t and t.lower() in v["short_condition"].lower() for t in terms):
            raise ValueError("Short condition omits original abbreviated term")
    return dict(available=True,original_truths=[0,1],ancestors_preserved=len(g["ancestors"]),C_valid=c.get("valid",False))
class Generator:
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        self.torch=torch;torch.set_num_threads(4)
        self.tokenizer=AutoTokenizer.from_pretrained(MEDGEMMA,local_files_only=True)
        self.model=AutoModelForCausalLM.from_pretrained(MEDGEMMA,local_files_only=True,dtype=torch.bfloat16,
                        device_map={"":"cuda:0"},attn_implementation="sdpa").eval()
        self.model.requires_grad_(False)
    def generate(self,instruction,payload,max_new=2600):
        self.last_response=None
        tok=self.tokenizer
        text=tok.apply_chat_template([dict(role="user",content=instruction+"\nINPUT:\n"+json.dumps(payload,ensure_ascii=False))],
                                    tokenize=False,add_generation_prompt=True)
        inputs=tok(text,return_tensors="pt",add_special_tokens=False).to("cuda")
        with self.torch.inference_mode():
            out=self.model.generate(**inputs,max_new_tokens=max_new,do_sample=False,pad_token_id=tok.pad_token_id)
        result=tok.decode(out[0,inputs.input_ids.shape[1]:],skip_special_tokens=True)
        self.last_response=dict(text=result,input_tokens=int(inputs.input_ids.shape[1]),
                                output_tokens=int(out.shape[1]-inputs.input_ids.shape[1]))
        start=result.find("{")
        if start<0:raise ValueError("No JSON in generated response: "+result[:120])
        value,_=json.JSONDecoder().raw_decode(result[start:])
        return value,result,int(inputs.input_ids.shape[1]),int(out.shape[1]-inputs.input_ids.shape[1])
def ordered_groups(root,limit=None):
    """Keep serial ordering fixed before applying a worker's shard stride."""
    root=Path(root)
    groups=rows(root/"data/groups_main.jsonl")+rows(root/"data/groups_definition.jsonl")
    first=[];seen=set()
    for g in groups:
        if g["cohort"]=="main" and g["split"]!="test" and g["source_id"] not in seen:
            first.append(g);seen.add(g["source_id"])
    first_ids={g["group_id"] for g in first}
    groups=first+[g for g in groups if g["group_id"] not in first_ids]
    if limit is not None:
        if limit<0:raise ValueError("limit must be nonnegative")
        groups=groups[:limit]
    return groups

def _combine_records(paths,key):
    """Preserve a saved success; otherwise retain the latest available record."""
    combined={}
    for path in paths:
        for record in rows(path):
            ident=record[key];previous=combined.get(ident)
            rank=lambda r:(r.get("construction_revision",0),r.get("status")=="complete",bool(r.get("value",{}).get("condition_change",{}).get("valid")))
            if previous is None or rank(record)>rank(previous):
                combined[ident]=record
    return combined

def _required_masks(group,record):
    if record is None:return None
    value=record.get("value",{})
    if group["cohort"]=="definition" and not value.get("definition_valid"):
        # A completed unavailable construction, or a recorded failed attempt,
        # has no semantic view to extract and is still a processed node.
        return []
    names=["base","swap"]
    if value.get("condition_change",{}).get("valid"):names.append("condition")
    return [group["group_id"]+":"+name for name in names]

def _definition_schema_incomplete(record):
    value=record.get("value",{})
    if type(value.get("definition_valid")) is not bool:return True
    if value["definition_valid"] is False:return False
    return any(key not in value for key in
               ("short_condition","shared_record","fact_sentences","facts","predicate",
                "ancestor_predicates","decisive_sentences","protected_action_ids",
                "condition_change","clinical_background","treatment_preference","reason"))

def _needs_reconstruction(group,record):
    if record is None:return True
    if record.get("construction_revision",0)>=REVISION:return False
    if record.get("status")!="complete":return True
    return group["cohort"]=="definition" and _definition_schema_incomplete(record)

def _needs_condition_repair(group,record):
    # Existing successful main constructions are retained in revision 3.
    # Newly reconstructed nodes still receive the focused C attempt below.
    return bool(group["cohort"]=="definition" and record and record.get("status")=="complete"
                and record.get("validation",{}).get("available",True)
                and record.get("value",{}).get("definition_valid") is True
                and not record.get("value",{}).get("condition_change",{}).get("valid")
                and record.get("construction_revision",0)<REVISION)

def _finished(group,record,masks):
    if _needs_reconstruction(group,record) or _needs_condition_repair(group,record):return False
    required=_required_masks(group,record)
    return required is not None and all(key in masks for key in required)

def merge_construction_shards(root=DEFAULT_ROOT,shards=2):
    """Called by the parent only after workers exit; no worker writes masters."""
    if shards<1:raise ValueError("shards must be positive")
    root=Path(root)
    constructions=_combine_records(
        [root/"data/constructions.jsonl"]+
        [root/f"data/constructions_{shard}.jsonl" for shard in range(shards)],
        "group_id")
    masks=_combine_records(
        [root/"data/semantic_masks.jsonl"]+
        [root/f"data/semantic_masks_{shard}.jsonl" for shard in range(shards)],
        "key")
    write_rows(root/"data/constructions.jsonl",constructions.values())
    write_rows(root/"data/semantic_masks.jsonl",masks.values())
    return dict(constructions=len(constructions),semantic_masks=len(masks),shards=shards)

def construct(root=DEFAULT_ROOT,limit=None,shard=0,shards=1):
    if shards<1 or not 0<=shard<shards:
        raise ValueError("Require shards >= 1 and 0 <= shard < shards")
    root=Path(root)
    groups=ordered_groups(root,limit)[shard::shards]
    suffix=f"_{shard}" if shards>1 else ""
    stage="construction"+suffix
    construction_path=root/f"data/constructions{suffix}.jsonl"
    mask_path=root/f"data/semantic_masks{suffix}.jsonl"
    construction_paths=[root/"data/constructions.jsonl"]
    mask_paths=[root/"data/semantic_masks.jsonl"]
    if shards>1:
        construction_paths.append(construction_path)
        mask_paths.append(mask_path)
    done=_combine_records(construction_paths,"group_id")
    masks=_combine_records(mask_paths,"key")
    todo=[g for g in groups if not _finished(g,done.get(g["group_id"]),masks)]
    initial_done=len(groups)-len(todo)
    common=dict(shard=shard,shards=shards,total=len(groups),pending_at_start=len(todo),
                resumed_nodes=initial_done,physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
    if not todo:
        status(root,stage,state="complete",done=len(groups),remaining=0,
               elapsed_seconds=0.,calls=0,**common)
        return dict(done=len(groups),total=len(groups),calls=0,shard=shard,shards=shards)
    status(root,stage,state="loading",done=initial_done,remaining=len(todo),**common)
    start=time.time();gen=Generator();n_calls=0;unavailable=0
    for ix,g in enumerate(todo):
        gid=g["group_id"];rec=done.get(gid)
        reconstruct=_needs_reconstruction(g,rec)
        if reconstruct:
            # Keep the original rows for provenance; replacement has a new revision.
            for name in ("base","swap","condition"):masks.pop(gid+":"+name,None)
            payload=dict(current_condition=condition(g),ancestors=g["ancestors"],tree=canonical_guide(g["guideline"]),
                         definitions=g["definitions"],shared_record=g["shared_record"],fact_sentences=g["facts"])
            if g["cohort"]=="definition":
                payload.update(source_rule_evidence=g["source_rule_evidence"],definition_links=g["definition_links"],
                               existing_patient=g["existing_patient"])
            if g["cohort"]=="main":
                payload["patients"]=[dict(state=j,label=j,text=g["shared_record"]+"\n"+g["facts"][j]) for j in (0,1)]
            instruction=DEF_INSTRUCTION if g["cohort"]=="definition" else MAIN_INSTRUCTION
            rec=dict(group_id=gid,source_id=g["source_id"],split=g["split"],cohort=g["cohort"],construction_revision=REVISION)
            construction_max_new=4096 if g["cohort"]=="definition" else 2600
            for attempt in range(2):
                logged=False
                try:
                    n_calls+=1
                    value,raw,ni,no=gen.generate(instruction,payload,max_new=construction_max_new)
                    append(root/f"logs/construction_raw{suffix}.jsonl",
                           dict(group_id=gid,attempt=attempt,construction_revision=REVISION,text=raw,input_tokens=ni,output_tokens=no))
                    logged=True
                    check=validate(g,value)
                    rec.update(status="complete",value=value,validation=check)
                    rec.pop("error",None)
                    break
                except Exception as exc:
                    rec.update(status="failed",error=str(exc))
                    previous=getattr(gen,"last_response",None)
                    if previous and not logged:
                        append(root/f"logs/construction_raw{suffix}.jsonl",
                               dict(group_id=gid,attempt=attempt,construction_revision=REVISION,parse_error=str(exc),**previous))
                    if previous:
                        payload["previous_response"]=previous["text"]
                        if g["cohort"]=="definition" and previous.get("output_tokens",0)>=construction_max_new:
                            construction_max_new=6144
                    payload["validation_error"]=str(exc)
                    instruction+="\nCorrect the previous response using the error AND the original patients; never change their facts."
            # A focused C request follows a valid factual construction when needed.
        condition_attempt=(rec.get("status")=="complete" and rec.get("validation",{}).get("available",True)
            and not rec.get("value",{}).get("condition_change",{}).get("valid")
            and (reconstruct or _needs_condition_repair(g,rec)))
        if condition_attempt:
            rec=copy.deepcopy(rec);v=rec["value"]
            payload=dict(current_node=g["gid"],current_condition=condition(g),
                tree=canonical_guide(g["guideline"]),definitions=g["definitions"],ancestors=g["ancestors"],
                facts=v["facts"],predicate=v["predicate"],ancestor_predicates=v["ancestor_predicates"],
                previous_condition_change=v.get("condition_change",{}),cohort=g["cohort"])
            if g["cohort"]=="main":
                payload["patients"]=[dict(state=j,label=j,text=g["shared_record"]+"\n"+g["facts"][j]) for j in (0,1)]
            else:
                payload["patients"]=[dict(state=j,label=j,text=v["shared_record"]+"\n"+v["fact_sentences"][j]) for j in (0,1)]
                payload["short_condition"]=v["short_condition"]
            logged=False
            try:
                from .numeric_conditions import numeric_condition_candidate
                numeric=numeric_condition_candidate(g,v)
                if numeric is not None:
                    candidate=dict(condition_change=numeric);raw=json.dumps(candidate,ensure_ascii=False)
                    ni=no=0;origin="numeric_threshold_from_verified_facts"
                else:
                    n_calls+=1
                    candidate,raw,ni,no=gen.generate(CONDITION_INSTRUCTION,payload,max_new=2200 if g["cohort"]=="definition" else 1400)
                    origin="MedGemma"
                append(root/f"logs/condition_repair_raw{suffix}.jsonl",
                       dict(group_id=gid,text=raw,input_tokens=ni,output_tokens=no,construction_revision=REVISION,origin=origin))
                logged=True
                proposed=copy.deepcopy(v)
                proposed["condition_change"]=candidate.get("condition_change",candidate)
                check=validate(g,proposed)
                if proposed["condition_change"].get("valid"):
                    rec.update(value=proposed,validation=check)
                else:rec["condition_repair_unavailable"]=proposed["condition_change"].get("reason")
            except Exception as exc:
                rec["condition_repair_error"]=str(exc)
                previous=getattr(gen,"last_response",None)
                if previous and not logged:
                    append(root/f"logs/condition_repair_raw{suffix}.jsonl",
                           dict(group_id=gid,parse_error=str(exc),construction_revision=REVISION,**previous))
            rec["construction_revision"]=REVISION
        if reconstruct or condition_attempt:
            append(construction_path,rec);done[gid]=rec
        value=rec.get("value",{})
        definition_unavailable=g["cohort"]=="definition" and not value.get("definition_valid")
        if definition_unavailable:
            unavailable+=1
        else:
            guide=g["guideline"]
            if g["cohort"]=="definition":guide=set_condition(guide,g["gid"],value["short_condition"])
            from .data import protected_swap
            swapped,_=protected_swap(dict(g,guideline=guide),value.get("protected_action_ids",[]))
            contexts=[("base",guide,g["definitions"]),("swap",swapped,g["definitions"])]
            c=value.get("condition_change",{})
            if c.get("valid"):
                contexts.append(("condition",replace_all(guide,c["edits"]),replace_all(g["definitions"],c["edits"])))
            for key,gg,dd in contexts:
                mk=gid+":"+key
                if mk in masks:continue
                payload=dict(current_node=g["gid"],current_condition=condition(dict(g,guideline=gg)),
                             tree=canonical_guide(gg),definitions=[dict(index=j,text=d["quote"]) for j,d in enumerate(dd)])
                entry=dict(key=mk,group_id=gid,construction_revision=rec.get("construction_revision",0))
                mask_logged=False
                try:
                    n_calls+=1
                    v,raw,ni,no=gen.generate(MASK_INSTRUCTION,payload,max_new=1200)
                    leaves=canonical_guide(gg)["leaves"]
                    v["keep_action_ids"]=[x for x in v.get("keep_action_ids",[]) if x in leaves]
                    v["keep_definition_indices"]=[int(x) for x in v.get("keep_definition_indices",[]) if 0<=int(x)<len(dd)]
                    entry.update(status="complete",value=v,input_tokens=ni,output_tokens=no)
                    append(root/f"logs/masking_raw{suffix}.jsonl",dict(key=mk,text=raw))
                    mask_logged=True
                except Exception as exc:
                    previous=getattr(gen,"last_response",None)
                    if previous and not mask_logged:
                        append(root/f"logs/masking_raw{suffix}.jsonl",dict(key=mk,parse_error=str(exc),**previous))
                    entry.update(status="failed",error=str(exc),value=dict(
                        keep_action_ids=list(canonical_guide(gg)["leaves"]),
                        keep_definition_indices=list(range(len(dd))),
                        reason="Extraction failed; preserve all supplied text"))
                append(mask_path,entry);masks[mk]=entry
        elapsed=time.time()-start
        processed=initial_done+ix+1
        status(root,stage,state="running",done=processed,remaining=len(groups)-processed,
               elapsed_seconds=elapsed,mean_seconds_per_node=elapsed/(ix+1),calls=n_calls,
               unavailable_definitions=unavailable,
               peak_reserved_gib=gen.torch.cuda.max_memory_reserved()/2**30,**common)
        if ix==19:
            write_json(root/f"pilot20_timing{suffix}.json",dict(
                stage="MedGemma construction + semantic extraction",shard=shard,shards=shards,
                nodes=20,seconds=elapsed,projected_remaining_hours=(len(todo)-20)*elapsed/20/3600))
    elapsed=time.time()-start
    status(root,stage,state="complete",done=len(groups),remaining=0,elapsed_seconds=elapsed,
           calls=n_calls,unavailable_definitions=unavailable,**common)
    return dict(done=len(groups),total=len(groups),calls=n_calls,shard=shard,shards=shards)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    parser.add_argument("--limit",type=int)
    parser.add_argument("--shard",type=int,default=0)
    parser.add_argument("--shards",type=int,default=1)
    parser.add_argument("--merge-shards",action="store_true")
    args=parser.parse_args()
    if args.merge_shards:
        merge_construction_shards(args.root,args.shards)
    else:
        construct(args.root,args.limit,args.shard,args.shards)
