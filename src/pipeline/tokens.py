"""Exact character-to-token mapping after the model chat template is applied."""
import json
SYSTEM = ("You are a clinical decision model. Use the supplied patient evidence, question, "
          "and current guideline. Activate only applicable rules, and preserve uncertainty. "
          "Return only a concise JSON object with decision, actions (list), status "
          "(decided or insufficient_evidence), and rationale. Do not invent patient evidence. Only applicable rule actions may be recommended. "
          "A not_applicable rule is blocked, and an unknown rule cannot justify a decision. "
          "Check each action against explicit patient evidence, including contraindications. "
          "If no rule is applicable, return insufficient_evidence and no actions.")

def char_tokens(offsets,start,end):
    if start < 0 or end < start:
        raise ValueError("Invalid character span")
    if start == end:
        return []
    return [i for i,(a,b) in enumerate(offsets) if b>a and a<end and b>start]

def build_prompt_and_mapping(tokenizer,patient,question,module1_output):
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer with exact offsets is required")
    parts,cursor,fields=[],0,{}
    def append(text,key=None):
        nonlocal cursor
        start=cursor
        parts.append(text)
        cursor+=len(text)
        if key is not None: fields[key]=(start,cursor)
    append("Question:\n")
    append(question,("question",))
    append("\n\nPatient note:\n")
    append(patient,("patient",))
    append("\n\nCurrent guideline rules:\n")
    apps={r["rule_id"]:r["status"] for r in module1_output["applicability"]}
    relations={(r["rule_id"],r["condition_id"]):r for r in module1_output["condition_evidence_relations"]}
    for rule in module1_output["candidate_rules"]:
        rid=rule["rule_id"]
        start=cursor
        append("\nRule "+rid+"; applicability="+apps.get(rid,"unknown")+"\n")
        append("Original applicable guideline clause:\n" if apps.get(rid)=="applicable" else "Blocked or uncertain branch; its action cannot justify a recommendation.\n")
        append(rule.get("text","") if apps.get(rid)=="applicable" else "",("rule_text",rid))
        append("\n")
        append("Logic: "+json.dumps(rule.get("logic","AND"),ensure_ascii=False)+"\n")
        for role,key in (("condition","conditions"),("exception","exceptions")):
            for item in rule.get(key,[]):
                cid=item["condition_id"]
                row=relations.get((rid,cid),{})
                text=row.get("condition") or (("NOT ("+item["text"]+")") if item.get("negated") else item["text"])
                append(role.title()+" "+cid+"; status="+row.get("status","unknown")+": ")
                append(text,("condition",rid,cid))
                append("\n")
        append("Exception logic: "+json.dumps(rule.get("exception_logic","OR"),ensure_ascii=False)+"\n")
        append("Recommendation: ")
        append(rule.get("action","") if apps.get(rid)=="applicable" else "",("action",rid))
        append("\n")
        fields[("rule",rid)]=(start,cursor)
    append("\nAnswer the question using the patient note and these rules.")
    user="".join(parts)
    prompt=tokenizer.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":user}],
        tokenize=False,add_generation_prompt=True,enable_thinking=False)
    user_start=prompt.find(user)
    if user_start<0 or prompt.find(user,user_start+1)>=0:
        raise ValueError("Chat template did not preserve unique user content")
    encoded=tokenizer(prompt,add_special_tokens=False,return_offsets_mapping=True,return_tensors="pt")
    offsets=encoded.pop("offset_mapping")[0].tolist()
    chars={k:(user_start+a,user_start+b) for k,(a,b) in fields.items()}
    toks={k:char_tokens(offsets,*s) for k,s in chars.items()}
    mapping={"schema_version":"pipeline_tokens.v1","prompt_length":len(offsets),
             "patient_tokens":toks[("patient",)],"question_tokens":toks[("question",)],
             "rules":{},"relations":[],"character_spans":[]}
    for rule in module1_output["candidate_rules"]:
        rid=rule["rule_id"]
        mapping["rules"][rid]={"rule_tokens":toks[("rule",rid)],
            "action_tokens":toks[("action",rid)],"condition_tokens":{},
            "original_text_tokens":toks[("rule_text",rid)],
            "prompt_char_span":{"start":chars[("rule_text",rid)][0],"end":chars[("rule_text",rid)][1]},
            "source_span":rule.get("source_span"),
            "action_exposed":apps.get(rid)=="applicable",
            "source_coordinate_note":"Source span is copied from module1; for list/structured rules start/end are local clause characters with clause index, not full guideline offsets."}
        source=rule.get("source_span",{})
        if source.get("unit")=="character" and "start" in source and "end" in source:
            if int(source["end"])-int(source["start"])!=len(rule.get("text","")):
                raise ValueError("Module1 source span length differs from original rule text")
        for key in ("conditions","exceptions"):
            for c in rule.get(key,[]):
                mapping["rules"][rid]["condition_tokens"][c["condition_id"]]=toks[("condition",rid,c["condition_id"])]
    patient_start=chars[("patient",)][0]
    for row in module1_output["relation_weights"]:
        rid,cid=row["rule_id"],row["condition_id"]
        ev=[]
        for e in row.get("evidence",[]):
            a,b=int(e["start"]),int(e["end"])
            if a<0 or b>len(patient) or patient[a:b]!=e["text"]:
                raise ValueError("Module1 evidence is not an exact patient character span")
            ev+=char_tokens(offsets,patient_start+a,patient_start+b)
        rr=mapping["rules"].get(rid,{})
        mapping["relations"].append({"rule_id":rid,"condition_id":cid,
            "condition_tokens":rr.get("condition_tokens",{}).get(cid,[]),
            "evidence_tokens":sorted(set(ev)),"action_tokens":rr.get("action_tokens",[]),
            "weight":row.get("weight",0),"binding_gate":row.get("binding_gate",0),
            "action_gate":row.get("action_gate",0),"prior_competition_weight":row.get("prior_competition_weight",0)})
    for key,(a,b) in chars.items():
        mapping["character_spans"].append({"field":list(key),"start":a,"end":b,"tokens":toks[key]})
    return prompt,encoded,mapping
