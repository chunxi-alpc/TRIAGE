"""Deterministic JSON grammar with exact-source patient evidence IDs; no gold inputs."""
import json,re,hashlib,time
from pathlib import Path
import torch
from .annotator import STATES
VERSION='source_ids_json_grammar.v3'
SYSTEM='Annotate each original guideline rule independently. Never give a final clinical decision. Select exact patient evidence IDs; do not invent evidence.'
INSTRUCTION='Determine relevance and applicability (applicable, inapplicable, unknown) for EVERY listed rule from explicit patient findings. Required conditions all must hold; absent information means unknown, not false. support_ids identify clauses supporting necessary conditions, refute_ids clauses contradicting a necessary condition. A refuting clause is NOT support. Select null for unused slots. Use evidence sufficient for the full rule, including preceding conditions. Original guideline actions will be preserved exactly. The grammar fixes JSON syntax and rule IDs; choose only relevance, states and evidence IDs. No final decision, option letter, reference answer or path.\n'

def evidence_catalog(patient):
    boundaries=[0]+[m.end() for m in re.finditer(r'[。！？;；\n]|(?<!\d)\.(?=\s+[A-Z]|\s*$)',patient)]+[len(patient)]
    bounds=sorted(set(boundaries));rows=[]
    for a,b in zip(bounds,bounds[1:]):
        while a<b and patient[a].isspace():a+=1
        while b>a and patient[b-1].isspace():b-=1
        if a<b:rows.append({'id':len(rows),'text':patient[a:b],'start':a,'end':b})
    return rows

class JSONTokenGrammar:
    """Token FSM: fixed JSON fields and finite prefix-free alternatives."""
    def __init__(self,tokenizer,rules,catalog,slots=6,legacy_quoted_states=False):
        self.segments=[];self.prompt_length=None
        def fixed(s):self.segments.append([tokenizer.encode(s,add_special_tokens=False)])
        def choice(values):self.segments.append([tokenizer.encode(v,add_special_tokens=False) for v in values])
        fixed('{"rules":[')
        for i,rule in enumerate(rules):
            if i:fixed(',')
            fixed('{"rule_id":'+json.dumps(rule['rule_id'])+',"relevant":');choice(['true','false'])
            if legacy_quoted_states:
                fixed(',"applicability":');choice([json.dumps(s) for s in STATES])
            else:
                # A shared quote prefix is fixed for ALL states. Otherwise Qwen
                # may encode quote+in as one token but quote separately for the
                # other labels, suppressing inapplicable before it can be chosen.
                fixed(',"applicability":"');choice(list(STATES));fixed('"')
            for key in ('support_ids','refute_ids'):
                fixed(',"'+key+'":[')
                for j in range(slots):
                    if j:fixed(',')
                    choice(['null']+[str(row['id']) for row in catalog])
                fixed(']')
            fixed('}')
        fixed(']}')
        self.maximum_tokens=sum(max(len(s) for s in segment) for segment in self.segments)
        self.last_prefix=();self.last_state=(0,())
        for segment in self.segments:
            if any(a!=b and len(a)<=len(b) and b[:len(a)]==a for a in segment for b in segment):raise ValueError('non_prefix_free_grammar_choice')
    def allowed(self,batch_id,input_ids):
        ids=tuple(input_ids.tolist()[self.prompt_length:]);segment=0;partial=()
        if ids[:len(self.last_prefix)]==self.last_prefix:
            segment,partial=self.last_state;remaining=ids[len(self.last_prefix):]
        else:remaining=ids
        for token in remaining:
            partial=partial+(token,);alternatives=self.segments[segment]
            matching=[a for a in alternatives if tuple(a[:len(partial)])==partial]
            if not matching:raise ValueError('token_outside_annotation_grammar')
            if any(len(a)==len(partial) for a in matching):segment+=1;partial=()
        self.last_prefix=ids;self.last_state=(segment,partial)
        if segment==len(self.segments):return [self.eos]
        return sorted({a[len(partial)] for a in self.segments[segment] if tuple(a[:len(partial)])==partial})

def payload_for(item,backend,config):
    rules=[{'rule_id':r['rule_id'],'text':r['text'],'action':r['action']} for r in item['guideline']['rules']]
    catalog=evidence_catalog(item['patient'])
    payload={'version':config.get('annotation_protocol',VERSION),'rules':rules,'patient':item['patient'],'question':item['question'],'catalog':catalog,'model':backend.model_path,'evidence_slots':config.get('annotation_evidence_slots',6),'prompt':SYSTEM+INSTRUCTION}
    return payload,hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def normalize(raw,rules,catalog):
    data=json.loads(raw)
    if set(data)!={'rules'} or len(data['rules'])!=len(rules):raise ValueError('compact_schema_mismatch')
    out=[];byid={r['id']:r for r in catalog}
    for row,rule in zip(data['rules'],rules):
        if set(row)!={'rule_id','relevant','applicability','support_ids','refute_ids'} or row['rule_id']!=rule['rule_id']:raise ValueError('compact_rule_identity_failure')
        if type(row['relevant']) is not bool or row['applicability'] not in STATES:raise ValueError('invalid_state')
        converted={'rule_id':rule['rule_id'],'relevant':row['relevant'],'applicability':row['applicability'],'action_spans':[rule['action']], 'supporting_evidence':[],'refuting_evidence':[],'exception_spans':[]}
        for key,field in [('support_ids','supporting_evidence'),('refute_ids','refuting_evidence')]:
            ids=[]
            for value in row[key]:
                if value is None:continue
                if type(value) is not int or value not in byid:raise ValueError('evidence_id_out_of_range')
                if value not in ids:ids.append(value)
            converted[field]=[{'text':byid[v]['text'],'start':byid[v]['start'],'end':byid[v]['end']} for v in ids]
        out.append(converted)
    return out

def annotate_constrained(backend,item,config,cache_dir):
    payload,key=payload_for(item,backend,config);path=Path(cache_dir)/(key+'.json')
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['cache_key']!=key:raise ValueError('annotation_cache_mismatch')
        return saved
    prompt=backend.tokenizer.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':INSTRUCTION+json.dumps({k:payload[k] for k in ('rules','patient','question','catalog')},ensure_ascii=False)}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    inputs=backend.tokenizer(prompt,add_special_tokens=False,return_tensors='pt').to(backend.model.device);n=inputs['input_ids'].shape[1]
    if n>config['max_input_tokens']:raise ValueError('annotation_input_budget_exceeded')
    grammar=JSONTokenGrammar(backend.tokenizer,payload['rules'],payload['catalog'],payload['evidence_slots'],legacy_quoted_states=payload['version']=='source_ids_json_grammar.v2');grammar.prompt_length=n;grammar.eos=backend.tokenizer.eos_token_id
    start=time.time()
    with torch.inference_mode():
        generated=backend.model.generate(**inputs,do_sample=False,max_new_tokens=grammar.maximum_tokens+1,prefix_allowed_tokens_fn=grammar.allowed,pad_token_id=grammar.eos)
    raw=backend.tokenizer.decode(generated[0,n:],skip_special_tokens=True);rows=normalize(raw,payload['rules'],payload['catalog'])
    saved={'cache_key':key,'input':payload,'raw_output':raw,'rules':rows,'errors':[],'input_tokens':n,'output_tokens':generated.shape[1]-n,'seconds':time.time()-start,'reference_used':False,'annotation_protocol':payload['version'],'action_source':'original supplied guideline rule; not predicted or gold-selected','exact_source_catalog':True}
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(saved,ensure_ascii=False,indent=2)+'\n');return saved

def compact_for_prompt(annotation):
    """Lossless normalization: source text stored once, rule evidence by catalog ID."""
    catalog=annotation['input']['catalog'];index={(r['start'],r['end'],r['text']):r['id'] for r in catalog};rows=[]
    for row in annotation['rules']:
        compact={k:row[k] for k in ('rule_id','relevant','applicability','action_spans','exception_spans')}
        for field in ('supporting_evidence','refuting_evidence'):
            compact[field]=[index[(e['start'],e['end'],e['text'])] for e in row[field]]
        rows.append(compact)
    # Prove exact semantic/coordinate equality rather than silently dropping spans.
    byid={r['id']:r for r in catalog}
    restored=[]
    for row in rows:
        r=dict(row)
        for field in ('supporting_evidence','refuting_evidence'):
            r[field]=[{k:byid[i][k] for k in ('text','start','end')} for i in row[field]]
        restored.append(r)
    if restored!=annotation['rules']:raise ValueError('compact_annotation_not_lossless')
    return {'evidence_reference_format':'Each supporting_evidence/refuting_evidence integer identifies exactly one original patient span in patient_evidence_catalog. Actions and applicability are unchanged.','patient_evidence_catalog':catalog,'rules':rows}
