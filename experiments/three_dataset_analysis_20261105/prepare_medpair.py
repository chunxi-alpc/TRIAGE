import sys
from dataclasses import replace
from transformers import AutoTokenizer
from common import *
sys.path.insert(0,str(REPO/'src'))
from represent_faithful.data.schema import ClinicalCase
from represent_faithful.data.prompts import build_prompt
from represent_faithful.runtime import answer_choice_token_ids

def main():
 tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
 source=REPO/'data/private/medpair/processed/physician_clean.jsonl'
 inputs=[];manifest=[];unique={};pairs=[]
 cases=[x for x in rows(source) if x['metadata']['split'] in ('dev','test')]
 pilot_ids=[x['id'] for x in cases if x['metadata']['split']=='dev'][:20]
 for raw in cases:
  c=ClinicalCase.from_dict(raw);split=raw['metadata']['split'];labels=[x.label for x in c.answer_choices];tids=answer_choice_token_ids(tok,labels)
  def add(condition,ev,extra=None):
   text=build_prompt(replace(c,evidence=tuple(ev))).text;key=(text,tuple(labels))
   if key not in unique:
    sid=f'medpair_input_{len(unique):04d}';unique[key]=sid
    inputs.append(dict(score_id=sid,prompt=text,labels=labels,option_token_ids=tids,gold=c.gold_option,id=c.id,split=split,condition=condition,pilot=c.id in pilot_ids,input_tokens=len(tok.encode(text,add_special_tokens=True))))
   manifest.append(dict(id=c.id,split=split,condition=condition,score_id=unique[key],gold=c.gold_option,labels=labels,
    evidence_sids=[e.sid for e in ev],source_path=str(source),source_dataset=raw['source_dataset'],**(extra or {})))
  add('Full',c.evidence);add('High',[e for e in c.evidence if e.priority_level==2])
  hi=[i for i,e in enumerate(c.evidence) if e.priority_level==2];un=[i for i,e in enumerate(c.evidence) if e.priority_level==0]
  if not hi or not un:continue
  lengths=[len(tok.encode(e.text,add_special_tokens=False)) for e in c.evidence]
  h,u=min(((h,u) for h in hi for u in un),key=lambda ij:(abs(lengths[ij[0]]-lengths[ij[1]]),ij[0],ij[1]))
  base=[e for i,e in enumerate(c.evidence) if i not in (h,u)];pos=min(h,u)
  high=list(base);high.insert(pos,c.evidence[h]);unrelated=list(base);unrelated.insert(pos,c.evidence[u])
  meta=dict(h_sid=c.evidence[h].sid,u_sid=c.evidence[u].sid,h_tokens=lengths[h],u_tokens=lengths[u],signed_token_gap=lengths[h]-lengths[u],absolute_token_gap=abs(lengths[h]-lengths[u]),insertion_position=pos,selection='minimum absolute token length gap; ties by high then irrelevant original sentence order')
  pairs.append(dict(id=c.id,split=split,**meta))
  add('C',base,meta);add('C+h',high,meta);add('C+u',unrelated,meta)
 write_rows(ROOT/'data/medpair_inputs.jsonl',inputs);write_rows(ROOT/'data/medpair_views.jsonl',manifest);write_rows(ROOT/'data/medpair_addback_pairs.jsonl',pairs)
 write_json(ROOT/'data/medpair_preparation.json',dict(cases=len(cases),views=len(manifest),unique_inputs=len(inputs),reused_identical_inputs=len(manifest)-len(inputs),pilot_case_ids=pilot_ids,pilot_inputs=sum(r['pilot'] for r in inputs),max_tokens=max(r['input_tokens'] for r in inputs),addback_dev=sum(p['split']=='dev' for p in pairs),addback_test=sum(p['split']=='test' for p in pairs),model=str(MODEL),dtype='bfloat16',prompt='original represent_faithful.data.prompts.build_prompt; plain text; space-prefixed option tokens'))
if __name__=='__main__':main()
