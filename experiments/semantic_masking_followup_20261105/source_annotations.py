"""Deterministic source anchors and explicit semantic-label choices for MedGemma."""
from common import compact

def rule_spans(node):
    from revision import source_content
    text,blocks,norm,positions=source_content(node["source_file"])
    present={b["block_id"] for b in node["blocks"]}
    output=[]
    for query in dict.fromkeys(node.get("node_mapping",[])):
        q=compact(query)
        start=norm.find(q) if q else -1
        if start<0:
            continue
        a,b=positions[start],positions[start+len(q)-1]+1
        for block in blocks:
            if block["block_id"] not in present or block["end"]<=a or block["start"]>=b:
                continue
            left,right=max(a,block["start"]),min(b,block["end"])
            quote=text[left:right]
            if quote.strip():
                output.append(dict(block_id=block["block_id"],quote=quote,
                    source_start=left,source_end=right,mapping_quote=query))
    return output

def ground_annotation(node,value):
    value=dict(value)
    for field in ("source_condition_action_mixed","source_semantic_dependency","dataset_definition_completed",
                  "direct_information_sufficient","ancestors_information_sufficient","full_information_sufficient",
                  "action_swap_safe"):
        item=value.get(field)
        if isinstance(item,str):
            value[field]={"yes":True,"no":False,"unknown":None,"true":True,"false":False}.get(item.lower())
    anchored=rule_spans(node)
    if anchored:
        value["rule_evidence"]=anchored
    value["rule_anchor_method"]="normalized exact node mapping resolved to literal source" if anchored else "model source quote or missing"
    return value

INSTRUCTION = """
Task: annotate the ORIGINAL guideline condition and a SPECIFIC, ALREADY DESIGNATED downstream management passage.
DATA includes original_rule_evidence with real source block IDs and literal text, designated_management, and source blocks.
The experiment always deletes designated_delete_block_ids. Independence of the condition DOES NOT cancel deletion.
Assess source meaning, not this patient's answer. Return a JSON object with the fields described below.
There is no default answer: select yes, no, or unknown for EACH semantic question after reading the actual evidence.
Use unknown only when the evidence is insufficient. A numerical threshold already stated in a rule is usually self-contained;
a shorthand such as a risk class may require a definition. Co-location alone is not dependence.

Required fields:
source_condition_action_mixed: string yes/no/unknown. Does the mapped ORIGINAL condition passage contain management advice too?
source_semantic_dependency: string yes/no/unknown. Does interpreting the mapped ORIGINAL condition require an additional
definition or explanation beyond that condition's own text? If yes, quote it in missing_definitions, even when its block is
retained. Whether this particular deletion removes it is a SEPARATE question. Do not count an unrelated recommendation as a definition.
dataset_definition_completed: string yes/no/unknown. Has the dataset's rewritten condition already supplied that definition?
dependency_reason: a brief explanation citing the relevant real block IDs.
conservative_cuts: list of objects with block_id and quote. Extract the minimal INDEPENDENT MANAGEMENT CLAUSE from designated
deleted blocks, preserving the condition, all exceptions and all definitions. For "If X, give treatment A", cut only "give
treatment A", retaining "If X". Use an empty list only if no independent action clause can be separated.
missing_definitions: list of block_id/quote objects for additional definitions required by the original condition, anywhere in supplied source.
necessary_restores: the subset of those definitions inside DESIGNATED DELETED blocks. ONLY verbatim definitions needed by the original
condition, from designated deleted blocks. Do not return recommendations, invented definitions, case answers, or retained text.
direct_information_sufficient, ancestors_information_sufficient, full_information_sufficient: each string yes/no/unknown.
action_swap_safe: boolean. True only for two independent action clauses located in the supplied original blocks.
action_swap_spans: list of two block_id/quote objects if safe, otherwise empty. At least one must be among conservative cuts.
missing_reference_queries: list of unresolved explicitly referenced definitions, otherwise empty.
construction_review: object with status and reason. Choose exactly one status from valid, invalid, unknown, not_applicable.
For an eligible supplied construction, valid means its FULL predicate and fixed facts preserve the complete current condition.
Ancestors are already supplied as fixed known facts, not missing clinical facts. Only the selected metric changes.
All alternatives in OR conditions are included, all interval endpoints retained, and only nonclinical backgrounds vary.
Already ineligible constructions need no further review: choose not_applicable.

Use actual JSON booleans only for action_swap_safe; semantic labels use one chosen string.
Source quotes must be verbatim, never the rewritten question unless it literally appears in the original source.
If a source excerpt is already supplied in original_rule_evidence, do not re-quote it in a rule_evidence field.
Do not copy field descriptions as values. Do not supply an answer for any patient.
""";
