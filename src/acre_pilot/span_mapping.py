"""Exact source coordinates, with explicit disambiguation and fail-closed mapping."""
from src.pipeline.tokens import char_tokens
SYSTEM = ("Use explicit patient findings and the complete guideline to answer the question. "
          "Follow only rules whose required conditions hold. Do not invent evidence. "
          "Select exactly one supplied option; select insufficient evidence if needed. "
          "Return only its single capital letter without explanation or punctuation.")
FIELDS = ("action_spans", "supporting_evidence", "refuting_evidence", "exception_spans")

def locate(source, span):
    text = span if isinstance(span, str) else span.get("text", "")
    if not isinstance(text, str) or not text:
        raise ValueError("empty_or_invalid_span")
    if isinstance(span, dict) and "start" in span:
        a, b = span["start"], span.get("end")
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(source) or source[a:b] != text:
            raise ValueError("invalid_explicit_coordinates")
        return a, b
    positions = []; start = 0
    while True:
        a = source.find(text, start)
        if a < 0: break
        positions.append((a, a + len(text))); start = a + 1
    if isinstance(span, dict):
        before, after = span.get("before", ""), span.get("after", "")
        positions = [(a,b) for a,b in positions if source[:a].endswith(before) and source[b:].startswith(after)]
    if len(positions) != 1:
        raise ValueError("span_not_found" if not positions else "ambiguous_duplicate_span")
    return positions[0]

def options_for(item):
    actions = list(dict.fromkeys(r["action"] for r in item["guideline"]["rules"]))
    actions.append("Insufficient explicit patient evidence to select a guideline recommendation.")
    labels="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if item.get("decision_protocol")=="single_token_label.v2": labels+="0123456789abcdefghijklmnopqrstuvwxyz"
    if len(actions)>len(labels): raise ValueError("option_count_exceeds_single_token_protocol")
    return [{"label": labels[i], "action": a} for i,a in enumerate(actions)]

def build_prompt(tokenizer, item, structured=None):
    if not tokenizer.is_fast: raise ValueError("fast_tokenizer_required")
    parts=[]; fields={}; cursor=0
    def append(text, key=None):
        nonlocal cursor
        if key is not None: fields[key]=(cursor,cursor+len(text))
        parts.append(text); cursor+=len(text)
    append("Complete guideline rules:\n")
    for rule in item["guideline"]["rules"]:
        append("\nRule "+rule["rule_id"]+":\n")
        append(rule["text"], ("rule",rule["rule_id"]))
        append("\n")
    append("\nPatient record:\n"); append(item["patient"],("patient",))
    append("\nQuestion:\n"); append(item["question"],("question",))
    if structured is not None:
        import json
        append("\nIndependent annotations (may contain errors):\n"+json.dumps(structured,ensure_ascii=False))
    append("\nDecision options:\n")
    for row in options_for(item): append(row["label"]+". "+row["action"]+"\n")
    append("Answer with one supplied label." if item.get("decision_protocol")=="single_token_label.v2" else "Answer with one letter.")
    user="".join(parts)
    system=SYSTEM.replace("single capital letter", "single supplied label") if item.get("decision_protocol")=="single_token_label.v2" else SYSTEM
    prompt=tokenizer.apply_chat_template([{"role":"system","content":system},{"role":"user","content":user}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    # Full content must occur exactly once, including after template processing.
    a,b=locate(prompt,user)
    encoded=tokenizer(prompt,add_special_tokens=False,return_offsets_mapping=True,return_tensors="pt")
    offsets=encoded.pop("offset_mapping")[0].tolist()
    mapping={"fields":{k:(a+x,a+y) for k,(x,y) in fields.items()},"offsets":offsets,"prompt_tokens":len(offsets)}
    for k,(x,y) in mapping["fields"].items():
        if prompt[x:y] != user[fields[k][0]:fields[k][1]]: raise ValueError("chat_template_alignment_failure")
    return prompt,encoded,mapping

def map_annotations(item, annotations, mapping):
    rules={r["rule_id"]:r for r in item["guideline"]["rules"]}; out=[]
    for ann in annotations:
        rid=ann["rule_id"]; row={**ann,"mapped":{},"mapping_errors":[]}
        try:
            ax,ay=locate(rules[rid]["text"],rules[rid]["action"])
            base_action=mapping["fields"][("rule",rid)][0]
            row["original_action_tokens"]=char_tokens(mapping["offsets"],base_action+ax,base_action+ay)
        except ValueError:
            row["original_action_tokens"]=[]
            row["mapping_errors"].append({"field":"original_action","reason":"canonical_action_not_uniquely_located"})
        for field in FIELDS:
            source=item["patient"] if "evidence" in field else rules[rid]["text"]
            key=("patient",) if "evidence" in field else ("rule",rid)
            base=mapping["fields"][key][0]; spans=[]
            for span in ann.get(field,[]):
                try:
                    x,y=locate(source,span); tokens=char_tokens(mapping["offsets"],base+x,base+y)
                    if not tokens: raise ValueError("no_mapped_tokens")
                    covered=[mapping["offsets"][t] for t in tokens]
                    if min(a for a,b in covered)>base+x or max(b for a,b in covered)<base+y: raise ValueError("incomplete_token_coverage")
                    spans.append({"text":source[x:y],"source_start":x,"source_end":y,"prompt_start":base+x,"prompt_end":base+y,"tokens":tokens})
                except (ValueError,TypeError,KeyError) as e:
                    row["mapping_errors"].append({"field":field,"span":span,"reason":str(e)})
            row["mapped"][field]=spans
        out.append(row)
    return out

def select_targets(mapped, mapping, method):
    targets=set(); invalid=[]
    for row in mapped:
        if not row["relevant"]: continue
        if method=="B1":
            targets.update(char_tokens(mapping["offsets"],*mapping["fields"][("rule",row["rule_id"])])); continue
        fields=("action_spans","supporting_evidence") if method=="B2" or row["applicability"]=="applicable" else ("refuting_evidence","exception_spans") if row["applicability"]=="inapplicable" else ()
        for field in fields:
            for span in row["mapped"][field]:
                if method=="B3" and row["applicability"]=="inapplicable" and field=="exception_spans":
                    if not row["original_action_tokens"] or set(span["tokens"]) & set(row["original_action_tokens"]):
                        invalid.append({"rule_id":row["rule_id"],"field":field,"reason":"exception_overlaps_action_or_action_unmapped"});continue
                targets.update(span["tokens"])
        invalid.extend(row["mapping_errors"])
    return sorted(targets),invalid
