#!/usr/bin/env python3
"""Local Qwen adapter for the official TrialGPT-Matching prompts.
No Azure client, training, annotations, regenerated answers, or synthetic patient evidence.
"""
import argparse,ast,hashlib,json,re,sys,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT/"third_party/TrialGPT"

def official_prompt_functions():
    source=REPO/"trialgpt_matching/TrialGPT.py"
    tree=ast.parse(source.read_text())
    wanted={"parse_criteria","print_trial","get_matching_prompt"}
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in wanted]
    if len(nodes)!=3:raise RuntimeError("Official TrialGPT prompt functions missing")
    namespace={}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),"exec"),namespace)
    return namespace,hashlib.sha256(source.read_bytes()).hexdigest()

def criteria_list(text,funcs):
    numbered=funcs["parse_criteria"](text)
    return [m.group(2).strip() for m in re.finditer(r"(?ms)^(\d+)\.\s*(.*?)(?=^\d+\.\s|\Z)",numbered)]

def sentences_from_patient(patient):
    from nltk.tokenize import sent_tokenize,PunktSentenceTokenizer
    try:
        sentences=sent_tokenize(patient)
        method="official_nltk_sent_tokenize"
    except LookupError:
        sentences=PunktSentenceTokenizer().tokenize(patient)
        method="nltk_PunktSentenceTokenizer_without_resource_download"
    if not sentences:raise ValueError("Empty patient note")
    for sentence in sentences:
        if sentence not in patient:raise ValueError("Sentence is not an exact patient-note quote")
    return sentences,method

def smoke_input(funcs):
    # Select existing criteria with explicit demographic patient evidence, without new facts or labels.
    sources=sorted((REPO/"dataset").glob("*/retrieved_trials.json"))
    for source in sources:
        dataset=json.loads(source.read_text())
        for instance in dataset:
            if not re.search(r"\b\d+[- ]year[- ]old\b",instance["patient"],re.I):
                continue
            for relevance in ["2","1","0"]:
                for trial in instance.get(relevance,[]):
                    inclusion=criteria_list(trial.get("inclusion_criteria",""),funcs)
                    age_ids=[i for i,c in enumerate(inclusion)
                             if len(c)<120 and re.search(r"\bage\b|\baged\b",c,re.I)
                             and re.search(r"\d",c)]
                    if not age_ids:
                        continue
                    first=age_ids[0]
                    selected_ids=[first]
                    if first+1<len(inclusion):
                        selected_ids.append(first+1)
                    selected=dict(trial)
                    selected["inclusion_criteria"]="\n\n".join(inclusion[i] for i in selected_ids)
                    selected["exclusion_criteria"]=""
                    return {"patient":instance["patient"],"patient_id":instance.get("patient_id"),
                            "trial":selected,"source":str(source),
                            "selection":"one existing official patient/trial; one original age criterion and at most one adjacent original inclusion criterion; no new reference labels",
                            "original_criterion_ids":{"inclusion":selected_ids,"exclusion":[]}}
    raise RuntimeError("Existing official patient with age evidence and original age criterion not found")

def parse_json_response(response):
    text=re.sub(r"<think>.*?</think>","",response,flags=re.S).strip()
    fence=chr(96)*3
    fenced=re.search(re.escape(fence)+r"(?:json)?\s*(.*?)"+re.escape(fence),text,re.S)
    if fenced:text=fenced.group(1).strip()
    start=text.find("{")
    if start<0:raise ValueError("Model response has no JSON object")
    obj,_=json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(obj,dict):raise ValueError("Model JSON must be an object")
    return obj

def normalize(raw,criterion_type,criteria,sentences):
    labels={"inclusion":{"included":("满足","满足"),"not included":("不满足","不满足")},
            "exclusion":{"excluded":("满足","不满足"),"not excluded":("不满足","满足")}}
    rows=[]
    for i,criterion in enumerate(criteria):
        item=raw.get(str(i))
        valid=isinstance(item,list) and len(item)==3 and isinstance(item[0],str) and isinstance(item[1],list) and isinstance(item[2],str)
        original_label=item[2].strip().lower() if valid else None
        ids=item[1] if valid else []
        valid_ids=sorted(set(j for j in ids if isinstance(j,int) and not isinstance(j,bool) and 0<=j<len(sentences)))
        bad_ids=[j for j in ids if not isinstance(j,int) or isinstance(j,bool) or not 0<=j<len(sentences)]
        evidence=[{"sentence_id":j,"text":sentences[j]} for j in valid_ids]
        label_supported=original_label in labels[criterion_type] or original_label in {"not enough information","not applicable"}
        classified=labels[criterion_type].get(original_label,("证据不足","证据不足"))
        if not evidence or bad_ids or not valid or not label_supported:
            classified=("证据不足","证据不足")
        rows.append({
            "criterion_id":i,"criterion_type":criterion_type,"criterion":criterion,
            "original_label":original_label,"condition_status":classified[0],
            "eligibility_status":classified[1],"evidence":evidence,
            "evidence_quotes_verified":bool(evidence) and not bad_ids,
            "not_applicable":original_label=="not applicable",
            "invalid_sentence_ids":bad_ids,"response_schema_valid":valid and label_supported,
            "mapping_note":"exclusion条件满足表示排除，eligibility_status为不满足" if criterion_type=="exclusion" else "inclusion条件满足表示本项入组要求满足"
        })
    return rows

def main():
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input",type=Path,help="JSON: patient string plus official TrialGPT trial dict")
    group.add_argument("--smoke",action="store_true",help="Use one existing official patient/trial and 1-2 existing inclusion criteria with demographic evidence")
    parser.add_argument("--model",type=Path)
    parser.add_argument("--output",type=Path,default=ROOT/"results/trialgpt_smoke.json")
    parser.add_argument("--max-new-tokens",type=int,default=768)
    args=parser.parse_args()
    start=time.time()
    initial_smoke=None
    if args.smoke and args.output.exists():
        previous=json.loads(args.output.read_text())
        initial_smoke=previous.get("initial_smoke") or {
            k:previous[k] for k in ["successful_condition_outputs","elapsed_seconds","gpu_peak_memory_gib","results","all_evidence_quotes_are_patient_substrings"] if k in previous}
    funcs,source_hash=official_prompt_functions()
    sample=smoke_input(funcs) if args.smoke else json.loads(args.input.read_text())
    patient=sample["patient"];trial=sample["trial"]
    sentences,method=sentences_from_patient(patient)
    numbered_patient="\n".join(f"{i}. {sentence}" for i,sentence in enumerate(sentences))
    model_path=args.model
    if model_path is None:
        cfg=json.loads((ROOT/"configs/models.json").read_text())
        model_path=Path(cfg["models"]["qwen3_8b"]["path"])
    tokenizer=AutoTokenizer.from_pretrained(str(model_path),local_files_only=True)
    tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(str(model_path),local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation="sdpa").to("cuda:0").eval()
    model.generation_config.temperature=None;model.generation_config.top_p=None;model.generation_config.top_k=None
    output={"model":str(model_path),"official_repository":"https://github.com/ncbi-nlp/TrialGPT",
            "official_prompt_file":str(REPO/"trialgpt_matching/TrialGPT.py"),
            "official_prompt_sha256":source_hash,"patient_id":sample.get("patient_id"),
            "trial_id":trial.get("NCTID"),"source":sample.get("source",str(args.input) if args.input else None),
            "selection":sample.get("selection"),"original_criterion_ids":sample.get("original_criterion_ids"),
            "patient_note":patient,"patient_sentences":[{"sentence_id":i,"text":s} for i,s in enumerate(sentences)],
            "sentence_tokenization":method,"synthetic_consent_sentence_added":False,"results":[],
            "raw_model_labels":{},"errors":[],"test_kind":"inference only; no reference answers created"}
    for criterion_type in ["inclusion","exclusion"]:
        criteria=criteria_list(trial.get(f"{criterion_type}_criteria",""),funcs)
        if not criteria:continue
        system,user=funcs["get_matching_prompt"](trial,criterion_type,numbered_patient)
        prompt=tokenizer.apply_chat_template([{"role":"system","content":system},{"role":"user","content":user}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        encoded=tokenizer(prompt,return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated=model.generate(**encoded,max_new_tokens=args.max_new_tokens,do_sample=False,pad_token_id=tokenizer.eos_token_id)
        response=tokenizer.decode(generated[0,encoded["input_ids"].shape[1]:],skip_special_tokens=True)
        try:
            raw=parse_json_response(response)
            output["raw_model_labels"][criterion_type]=raw
            output["results"].extend(normalize(raw,criterion_type,criteria,sentences))
        except (ValueError,json.JSONDecodeError) as error:
            output["errors"].append({"criterion_type":criterion_type,"error":str(error),"response":response})
    if initial_smoke is not None:output["initial_smoke"]=initial_smoke
    output["elapsed_seconds"]=round(time.time()-start,2)
    output["gpu_peak_memory_gib"]=round(torch.cuda.max_memory_allocated()/1024**3,3)
    output["successful_condition_outputs"]=len(output["results"])
    output["all_evidence_quotes_are_patient_substrings"]=all(e["text"] in patient for row in output["results"] for e in row["evidence"])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"output":str(args.output),"conditions":len(output["results"]),"errors":len(output["errors"]),"gpu_peak_memory_gib":output["gpu_peak_memory_gib"]},ensure_ascii=False))
    return 1 if not output["results"] or output["errors"] or any(not x["response_schema_valid"] for x in output["results"]) else 0

if __name__=="__main__":sys.exit(main())
