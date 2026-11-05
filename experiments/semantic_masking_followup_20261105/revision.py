"""Preserve sampling, locate mandatory management spans, and build repaired inputs."""
import argparse
import collections
import json
import re
from functools import lru_cache
from pathlib import Path
from common import CONFIG, DATA, RESULTS, ROOT, blocks_for, compact, locate_quote, read_rows, render, write_json, write_rows
from predicates import make_specs, patient_for, evaluate

OLD = Path(CONFIG["old_run"])
ACTION = re.compile(r"(?i)therap|treat|resect|surveill|follow.up|recommend|surgery|surgical|radiat|chemotherap|transplant|administer|manage|monitor|biopsy|剂量|用药|治疗|切除|手术|建议|推荐|随访|监测|复查|检查|处置|干预")
def normalized_with_positions(text):
    chars, positions = [], []
    for i, char in enumerate(text):
        if not re.match(r"[\s$\\{}_*]", char):
            for c in char.lower():
                chars.append(c)
                positions.append(i)
    return "".join(chars), positions

@lru_cache(maxsize=48)
def source_content(path):
    text = Path(path).read_text()
    norm, positions = normalized_with_positions(text)
    return text, blocks_for(text), norm, positions

def locate_mapping(node):
    text, blocks, norm, positions = source_content(node["source_file"])
    spans, missing = [], []
    rule_quotes = {compact(q) for q in node["node_mapping"]}
    for query in dict.fromkeys(node["action_mapping"]):
        if compact(query) in rule_quotes or not ACTION.search(query):
            continue
        q = compact(query)
        start = norm.find(q) if q else -1
        if start < 0:
            missing.append(query)
            continue
        a, b = positions[start], positions[start+len(q)-1]+1
        touched = [block for block in blocks if block["start"] < b and block["end"] > a]
        for block in touched:
            left, right = max(a,block["start"]), min(b,block["end"])
            quote = text[left:right]
            if quote.strip() and ACTION.search(quote):
                spans.append(dict(block_id=block["block_id"],quote=quote,
                    source_start=left,source_end=right,
                    line_start=text.count("\n",0,left)+1,line_end=text.count("\n",0,right)+1,
                    mapping_quote=query,location_method="normalized_exact_to_verbatim"))
        # One designated downstream passage per node, fixed without model outcomes.
        if spans:
            break
    return spans, missing

def prepare_revision():
    old_nodes = read_rows(OLD/"data/nodes.jsonl")
    old_inputs = read_rows(OLD/"data/inputs_full.jsonl")
    specs = make_specs(old_nodes, old_inputs)
    spec_map = {s["node_id"]:s for s in specs}
    pilot_ids = [s["node_id"] for s in specs if s["split"] == "development"]
    # Include a known action-present-but-not-removed case, plus original development pilot nodes.
    required = "afd5d03b38791ef112c4c90e963feed0:15"
    pilot_ids += [required]
    pilot_ids += [n["node_id"] for n in old_nodes if n["split"]=="development" and n["pilot"]]
    pilot_ids += [n["node_id"] for n in old_nodes if n["split"]=="development"]
    pilot_ids = list(dict.fromkeys(pilot_ids))[:20]
    nodes = []
    for original in old_nodes:
        node = dict(original)
        spans, missing = locate_mapping(node)
        all_blocks = {b["block_id"]:b for b in source_content(node["source_file"])[1]}
        selected = {b["block_id"]:b for b in node["blocks"]}
        for span in spans:
            selected[span["block_id"]] = all_blocks[span["block_id"]]
        node.update(blocks=sorted(selected.values(),key=lambda b:b["start"]),
            designated_management=spans,
            designated_delete_block_ids=sorted({s["block_id"] for s in spans}),
            management_mapping_unlocated=missing,
            management_location_status="located" if spans else "unlocated_or_no_action_mapping",
            try_construction=node["node_id"] in spec_map,
            pilot=node["node_id"] in pilot_ids)
        nodes.append(node)
    if len(read_rows(DATA/"sources.jsonl"))!=40 or sum(n["role"]=="statistical" for n in nodes)!=379:
        raise ValueError("Original sampling changed")
    write_rows(DATA/"nodes.jsonl",nodes)
    write_rows(DATA/"construction_specs.jsonl",specs)
    write_json(RESULTS/"revision_preparation.json",dict(
        original_run=str(OLD),sources=40,statistical_nodes=379,total_nodes=len(nodes),
        original_construction_pool=len(specs),repaired_constructible=sum(s["eligible"] for s in specs),
        excluded_constructions=[dict(node_id=s["node_id"],reason=s["reason"]) for s in specs if not s["eligible"]],
        management_located=sum(bool(n["designated_management"]) for n in nodes),
        pilot_nodes=pilot_ids,split_policy="Original source split unchanged; Q sample unchanged",
        score_budget=CONFIG["score_budget"]))

def clean_spans(items, blocks, allowed=None):
    valid, errors = [], []
    for item in items or []:
        try:
            bid, quote = item["block_id"],item["quote"]
            if allowed is not None and bid not in allowed:
                raise ValueError("span outside designated deleted block")
            a,b = locate_quote(blocks[bid],quote)
            if any(x["block_id"]==bid and max(a,x["start_in_block"])<min(b,x["end_in_block"]) for x in valid):
                raise ValueError("overlapping spans")
            valid.append(dict(block_id=bid,quote=quote,start_in_block=a,end_in_block=b,
                source_start=blocks[bid]["start"]+a,source_end=blocks[bid]["start"]+b))
        except (KeyError,TypeError,ValueError) as exc:
            errors.append(str(exc))
    return valid, errors

def normalize_annotation(node, value):
    from source_annotations import ground_annotation
    value=ground_annotation(node,value)
    if value.get("semantic_review_version")==3:
        from dependency_rules import reconcile_review
        value=reconcile_review(node,value)
    blocks = {b["block_id"]:b for b in node["blocks"]}
    deleted = node["designated_delete_block_ids"]
    evidence, e1 = clean_spans(value.get("rule_evidence",[]),blocks)
    definitions, e2 = clean_spans(value.get("necessary_restores",[]),blocks,deleted)
    all_definitions, definition_errors = clean_spans(value.get("missing_definitions",[]),blocks)
    e2+=definition_errors
    all_definitions=all_definitions or definitions
    cuts, e3 = clean_spans(value.get("conservative_cuts",[]),blocks,deleted)
    from conservative import preserve_notes
    fragments,retained=preserve_notes(cuts,blocks)
    cuts,fragment_errors=clean_spans(fragments,blocks,deleted)
    e3+=fragment_errors
    value["conservative_retained_notes"]=retained
    # An independent action span must not overlap a restored definition.
    cuts = [cut for cut in cuts if not any(cut["block_id"]==definition["block_id"] and
        max(cut["start_in_block"],definition["start_in_block"]) <
        min(cut["end_in_block"],definition["end_in_block"]) for definition in all_definitions+evidence)]
    swaps, e4 = clean_spans(value.get("action_swap_spans",[]),blocks)
    if not value.get("action_swap_safe") or len(swaps)!=2:
        swaps = []
    if swaps and (swaps[0]["quote"]==swaps[1]["quote"] or any(
        s["block_id"]==p["block_id"] and max(s["start_in_block"],p["start_in_block"]) <
        min(s["end_in_block"],p["end_in_block"]) for s in swaps for p in all_definitions+evidence)):
        swaps=[]
    if swaps and not any(any(s["block_id"]==c["block_id"] and s["quote"] in c["quote"]
                            for c in cuts) for s in swaps):
        swaps=[]
    for name in ("source_semantic_dependency","source_condition_action_mixed","dataset_definition_completed"):
        if type(value.get(name)) is not bool:
            value[name]=None
    if value["source_semantic_dependency"] is True and not all_definitions:
        value["source_semantic_dependency"]=None
    value.update(rule_evidence=evidence,necessary_restores=definitions,
                 missing_definitions=all_definitions,definition_removed=bool(definitions),
                 conservative_cuts=cuts,action_swap_spans=swaps,action_swap_safe=bool(swaps),
                 direct_delete_block_ids=deleted,
                 source_traceable=bool(evidence),
                 span_errors=e1+e2+e3+e4)
    return dict(status="complete",node_id=node["node_id"],source_id=node["source_id"],
        blocks=node["blocks"],annotation=value,
        unresolved_references=value.get("missing_reference_queries",[]),
        validation=dict(source_traceable=bool(evidence),mask_valid=True,
            dependency_evidence_valid=type(value["source_semantic_dependency"]) is bool,
            errors=value["span_errors"]))

def all_annotations():
    output = {}
    for name in ("annotations_pilot_0.jsonl","annotations_pilot_1.jsonl",
                 "annotations_full_0.jsonl","annotations_full_1.jsonl"):
        for row in read_rows(DATA/name):
            if row["status"]=="complete":
                output[row["node_id"]]=row
    return output

def irrelevant_control(node, length):
    pattern=r"(?i)copyright|all rights reserved|版权所有|版权|通信作者|通讯作者|利益冲突|correspondence|conflicts? of interest"
    pool=[]
    for block in source_content(node["source_file"])[1]:
        for match in re.finditer(r"[^\n]+",block["text"]):
            if re.search(pattern,match.group()):
                pool.append(dict(block_id=block["block_id"],text=match.group(),start=block["start"]+match.start()))
    if not pool:
        return None
    block = min(pool,key=lambda b:abs(len(b["text"])-length))
    # Contiguous original excerpt, never generated clinical advice.
    quote = block["text"][:length]
    return dict(block_id=block["block_id"],quote=quote,source_start=block["start"],
                source_end=block["start"]+len(quote),requested_characters=length,
                actual_characters=len(quote),control_type="publication_metadata")

def views_for(node, annotation):
    blocks=node["blocks"]; value=annotation["annotation"]
    deleted=node["designated_delete_block_ids"]
    necessary=collections.defaultdict(list)
    for span in value.get("necessary_restores",[]):
        necessary[span["block_id"]].append(span["quote"])
    necessary={k:"\n".join(v) for k,v in necessary.items()}
    controls={bid:irrelevant_control(node,len(text)) for bid,text in necessary.items()}
    irrelevant={bid:control["quote"] for bid,control in controls.items() if control}
    views=dict(full=render(blocks),direct=render(blocks,deleted=deleted),
        conservative=render(blocks,cuts=value.get("conservative_cuts",[])),
        necessary_restore=render(blocks,deleted=deleted,insertions=necessary),
        irrelevant_restore=render(blocks,deleted=deleted,insertions=irrelevant),
        condition_ancestors="")
    swaps=value.get("action_swap_spans",[])
    if value.get("action_swap_safe") and len(swaps)==2:
        views["action_swap"]=render(blocks,replacements=[
            dict(block_id=swaps[0]["block_id"],quote=swaps[0]["quote"],replacement=swaps[1]["quote"]),
            dict(block_id=swaps[1]["block_id"],quote=swaps[1]["quote"],replacement=swaps[0]["quote"])])
    changes=dict(designated_management=node["designated_management"],deleted_block_ids=deleted,
        deleted_blocks=[b for b in blocks if b["block_id"] in deleted],
        conservative_cuts=value.get("conservative_cuts",[]),
        necessary_restores=value.get("necessary_restores",[]),irrelevant_controls=controls,
        definition_lost=value.get("definition_lost",bool(necessary)),
        removed_definition_spans=value.get("removed_definition_spans",[]),
        restoration_available=bool(necessary),matched_irrelevant_available=bool(necessary) and len(irrelevant)==len(necessary),
        direct_changed=views["direct"]!=views["full"],conservative_changed=views["conservative"]!=views["full"],
        action_swap_spans=swaps,location_status=node["management_location_status"])
    return views,changes

def build(stage):
    nodes=read_rows(DATA/"nodes.jsonl")
    annotations=all_annotations()
    for node in nodes:
        prior=annotations.get(node["node_id"])
        if prior:
            annotations[node["node_id"]]=dict(prior,**normalize_annotation(node,prior["annotation"]))
    write_rows(DATA/"annotations.jsonl",annotations.values())
    specs={s["node_id"]:s for s in read_rows(DATA/"construction_specs.jsonl")}
    inputs,metadata,excluded=[],[],[]
    for node in nodes:
        if stage=="pilot" and not node["pilot"]:
            continue
        annotation=annotations.get(node["node_id"])
        if annotation is None:
            # Missing semantic labels do not turn deterministic source edits into missing nodes.
            annotation=normalize_annotation(node,{})
            excluded.append(dict(node_id=node["node_id"],reason="semantic_annotation_unavailable; deterministic direct edit retained"))
        views,changes=views_for(node,annotation)
        value=annotation["annotation"]
        common=dict(node_id=node["node_id"],source_id=node["source_id"],publisher=node["publisher"],
            split=node["split"],ancestors=node["ancestors"],role=node["role"],
            source_semantic_dependency=value.get("source_semantic_dependency"),
            dataset_definition_completed=value.get("dataset_definition_completed"),
            source_traceable=annotation["validation"]["source_traceable"],
            dependency_evidence_valid=annotation["validation"]["dependency_evidence_valid"],
            unresolved_references=annotation.get("unresolved_references",[]),
            direct_information_sufficient=value.get("direct_information_sufficient"),
            ancestors_information_sufficient=value.get("ancestors_information_sufficient"),
            full_information_sufficient=value.get("full_information_sufficient"),
            changes=changes)
        metadata.append(dict(**common,condition=node["condition"],blocks=node["blocks"],
            annotation=value,source_file=node["source_file"],original_case_id=node["original_case_id"]))
        if node["original_case"] and node["original_gold"] is not None and (node["role"]=="statistical" or stage=="pilot"):
            example_id=node["node_id"]+":original"
            for view in ("full","direct","conservative","necessary_restore","irrelevant_restore","condition_ancestors"):
                inputs.append(dict(**common,example_id=example_id,score_id=example_id+":"+view,
                    population="original",view=view,condition=node["condition"],patient=node["original_case"],
                    context=views[view],gold=node["original_gold"],
                    gold_origin="Unchanged MEGA-CDP case_paths decision_outcome",teacher_answer=None,
                    origin="dataset-generated original vignette"))
        spec=specs.get(node["node_id"])
        if not spec:
            continue
        review=value.get("construction_review") or {}
        if not spec["eligible"] or review.get("status")=="invalid":
            excluded.append(dict(node_id=node["node_id"],reason=spec.get("reason") or review.get("reason"),
                                 population="constructed",review=review))
            continue
        for level in ("below","above","same_side"):
            for background in ("neutral","cue"):
                patient,facts=patient_for(spec,level,background)
                # The original is already a factual binary question; avoid duplicate neutral wording.
                wording="original"; condition=node["condition"]
                example_id=f"{node['node_id']}:constructed:{level}:{background}:{wording}"
                row=dict(**common,example_id=example_id,population="constructed",condition=condition,
                    patient=patient,gold=int(evaluate(spec["predicate"],facts)),facts=facts,
                    metric_level=level,metric_value=spec["values"][level],background=background,
                    wording=wording,wording_changed=False,wording_noop_reason="Original factual proposition retained",
                    unit=spec["unit"],gold_origin="Complete executable predicate",construction=spec,
                    origin="基于真实指南构造")
                # Numerical predicates are explicit in the question; unrelated ancestry stays supplied.
                row["ancestors_information_sufficient"]=True
                row["direct_information_sufficient"]=True
                for view in ("full","direct","condition_ancestors","action_swap"):
                    if view in views:
                        inputs.append(dict(**row,score_id=example_id+":"+view,view=view,context=views[view]))
    if len(inputs)>CONFIG["score_budget"]:
        raise RuntimeError(f"{len(inputs)} inputs exceed budget {CONFIG['score_budget']}")
    write_rows(DATA/f"inputs_{stage}.jsonl",inputs)
    write_rows(DATA/f"edits_{stage}.jsonl",metadata)
    write_rows(DATA/f"input_exclusions_{stage}.jsonl",excluded)
    summary=dict(stage=stage,scores=len(inputs),nodes=len(metadata),
        constructed_nodes=len({r["node_id"] for r in inputs if r["population"]=="constructed"}),
        direct_changed_nodes=sum(m["changes"]["direct_changed"] for m in metadata),
        direct_unchanged_nodes=sum(not m["changes"]["direct_changed"] for m in metadata),
        by_population_view=dict(collections.Counter(r["population"]+"/"+r["view"] for r in inputs)),
        exclusions=len(excluded))
    write_json(RESULTS/f"input_summary_{stage}.json",summary)
    return summary

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["prepare","build"])
    parser.add_argument("--stage",choices=["pilot","full"],default="pilot")
    args=parser.parse_args()
    if args.command=="prepare":
        prepare_revision()
    else:
        print(json.dumps(build(args.stage)))
