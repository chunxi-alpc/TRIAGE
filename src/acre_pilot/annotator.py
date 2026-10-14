"""One independent model annotation per case. No references or final decision."""
import hashlib,json,time
from pathlib import Path
import torch
from .span_mapping import FIELDS
STATES=("applicable","inapplicable","unknown")
SYSTEM="You annotate clinical rule applicability and exact text spans. Never produce a final clinical decision. Output JSON only."
INSTRUCTION="""For EVERY supplied rule, independently determine relevance to the question and applicability from explicit patient evidence. Return {"rules":[{"rule_id":"original ID","relevant":true,"applicability":"applicable|inapplicable|unknown","action_spans":["exact recommendation text"],"supporting_evidence":["exact patient substring"],"refuting_evidence":["exact patient substring"],"exception_spans":["exact triggered guideline exception"]}]}. Actions and exceptions must be exact substrings of that rule's text; evidence must be exact substrings of the patient record. No paraphrases. Unknown when necessary conditions cannot be decided. For inapplicable rules include explicit refuting evidence, not their action as evidence. Supporting evidence may still be recorded for their satisfied conditions. Give only the shortest sufficient excerpts. For a repeated excerpt use {"text":"...","before":"immediately preceding text","after":"immediately following text"} to disambiguate. Do not output any final decision, reference path, or answer choice. Do not omit rules.\n"""

def identity(item,backend,config):
    payload={"rules":[{"rule_id":r["rule_id"],"text":r["text"]} for r in item["guideline"]["rules"]],"patient":item["patient"],"question":item["question"],"model":backend.model_path,"prompt":SYSTEM+INSTRUCTION,"max_new_tokens":config["annotation_max_new_tokens"]}
    return payload,hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def parse(raw,rules):
    raw=raw.strip()
    if raw.startswith("```"): raw="\n".join(raw.splitlines()[1:-1])
    value=json.loads(raw); rows=value["rules"]
    if set(value)!={"rules"}: raise ValueError("unexpected_top_level_fields")
    expected={r["rule_id"] for r in rules};seen=set()
    for row in rows:
        if set(row)!={"rule_id","relevant","applicability",*FIELDS}: raise ValueError("annotation_schema_mismatch")
        rid=row["rule_id"]
        if rid not in expected or rid in seen: raise ValueError("invalid_or_duplicate_rule_id")
        seen.add(rid)
        if type(row["relevant"]) is not bool or row["applicability"] not in STATES: raise ValueError("invalid_state_or_relevance")
        for f in FIELDS:
            if not isinstance(row[f],list) or any(not isinstance(s,(str,dict)) for s in row[f]): raise ValueError("invalid_span_list")
            for span in row[f]:
                if isinstance(span,dict) and ("text" not in span or set(span)-{"text","start","end","before","after"}):
                    raise ValueError("invalid_span_fields")
    if seen!=expected: raise ValueError("omitted_rules")
    return rows

def annotate(backend,item,config,cache_dir):
    payload,key=identity(item,backend,config);path=Path(cache_dir)/(key+".json")
    if path.exists():
        saved=json.loads(path.read_text())
        if saved["cache_key"]!=key: raise ValueError("cache_identity_mismatch")
        return saved
    user=INSTRUCTION+json.dumps({k:payload[k] for k in ("rules","patient","question")},ensure_ascii=False)
    prompt=backend.tokenizer.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":user}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    inputs=backend.tokenizer(prompt,return_tensors="pt",add_special_tokens=False).to(backend.model.device)
    start=time.time()
    if inputs["input_ids"].shape[1]>config["max_input_tokens"]: raise ValueError("annotation_input_budget_exceeded")
    with torch.inference_mode():
        generated=backend.model.generate(**inputs,do_sample=False,max_new_tokens=config["annotation_max_new_tokens"],pad_token_id=backend.tokenizer.eos_token_id)
    raw=backend.tokenizer.decode(generated[0,inputs["input_ids"].shape[1]:],skip_special_tokens=True)
    errors=[]
    try: rows=parse(raw,item["guideline"]["rules"])
    except (ValueError,KeyError,TypeError) as e:
        errors.append(str(e));rows=[{"rule_id":r["rule_id"],"relevant":False,"applicability":"unknown",**{f:[] for f in FIELDS}} for r in item["guideline"]["rules"]]
    saved={"cache_key":key,"input":payload,"raw_output":raw,"rules":rows,"errors":errors,"input_tokens":inputs["input_ids"].shape[1],"output_tokens":generated.shape[1]-inputs["input_ids"].shape[1],"seconds":time.time()-start,"reference_used":False}
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(saved,ensure_ascii=False,indent=2)+"\n")
    return saved

def direct_decision(mapped, options, rules):
    # Rule IDs and actions are supplied guideline metadata, not reference labels.
    # Use the complete original action rather than penalizing an annotator that
    # correctly quotes only part of a recommendation. Abstain on conflicting rules.
    original={r['rule_id']:r['action'] for r in rules};actions=set()
    for row in mapped:
        if row['relevant'] and row['applicability']=='applicable':
            actions.add(original[row['rule_id']])
    if len(actions)!=1:return None
    action=next(iter(actions));matches=[o['label'] for o in options[:-1] if o['action']==action]
    return matches[0] if len(matches)==1 else None
