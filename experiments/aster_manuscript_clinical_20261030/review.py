"""Independent on-server Llama review; deterministic rule labels remain authoritative."""
import argparse,json,random,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from .data import read,write,jsonl,clinical_validator
def predicate(spec):
 op,a,*b=spec
 return f'Current patient age is between {a} and {b[0]} years, inclusive.' if op=='range' else f'Current patient age is {dict(ge="at least",gt="greater than",le="at most",lt="less than")[op]} {a} years.'
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=False)
 torch.set_num_threads(4);torch.manual_seed(20261030);start=time.time()
 path='/home/gcx/TRIAGE/models/modelscope/Meta-Llama-3.1-8B-Instruct'
 tok=AutoTokenizer.from_pretrained(path,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda:0').eval()
 rows=read(Path(a.data)/'candidate_pairs.jsonl');seen={};accepted=[];reviews=[];rejected=[]
 for i,row in enumerate(rows):
  anchor=row['clinical_verification']['original_case_id']
  if anchor not in seen:
   patients=list(row['states'].values());random.Random(20261030+i).shuffle(patients);names=['A','B','C']
   text='You are an independent clinical-data reviewer. Use only documented facts. Do not infer a missing age from a diagnosis. Review the following age-only counterfactual clinical records. Predicate: '+predicate(row['clinical_verification']['age_predicate'])+'\nOriginal source question: '+row['clinical_verification']['source_condition']
   text+='\nClassify each record as true, false or unknown for this predicate. Flag only explicit conflicts (for example a treatment duration exceeding age). Return ONLY JSON: {"labels":{"A":"true/false/unknown","B":"...","C":"..."},"explicit_conflict":false,"reason":"brief reason"}. No expected labels are supplied.\n'
   text+='\n'.join(name+':\n'+patient for name,patient in zip(names,patients))
   prompt=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True)
   ids=tok(prompt,return_tensors='pt',add_special_tokens=False).to('cuda:0')
   with torch.inference_mode():result=model.generate(**ids,max_new_tokens=160,do_sample=False,pad_token_id=tok.eos_token_id)
   answer=tok.decode(result[0,ids['input_ids'].shape[1]:],skip_special_tokens=True)
   try:
    parsed=json.JSONDecoder().raw_decode(answer[answer.index('{'):])[0]
    table={patient:parsed['labels'][name].lower().strip() for name,patient in zip(names,patients)}
    seen[anchor]={'labels':table,'explicit_conflict':bool(parsed.get('explicit_conflict',False)),'reason':parsed.get('reason',''),'raw':answer}
   except Exception as e:seen[anchor]={'parse_failure':str(e),'raw':answer}
  review=seen[anchor];truths=clinical_validator(row);expected={s:'unknown' if t is None else str(t).lower() for s,t in truths.items()}
  ok='labels' in review and not review['explicit_conflict'] and all(review['labels'][row['states'][s]]==v for s,v in expected.items())
  record={'pair_id':row['pair_id'],'source_id':row['source_id'],'split':row['split'],'accepted':ok,'review':review}
  reviews.append(record)
  if ok:
   row['clinical_verification']['independent_review']={'model':'Meta-Llama-3.1-8B-Instruct','local_snapshot_revision':'not recorded by prior acquisition','clinical_rule_labels_agree':True,'explicit_conflict':False,'clinician_review':False}
   row['verification']['verifier']='age-predicate-v1 + independent Llama consistency review'
   accepted.append(row)
  else:rejected.append({'pair_id':row['pair_id'],'split':row['split'],'source_id':row['source_id'],'reason':review.get('reason',review.get('parse_failure','label disagreement'))})
  write(out/'status.json',{'status':'running','reviewed_triplets':len(reviews),'accepted':len(accepted),'rejected':len(rejected)})
 jsonl(out/'accepted_pairs.jsonl',accepted);jsonl(out/'reviews_private.jsonl',reviews)
 write(out/'audit.json',{'submitted':len(rows),'accepted':len(accepted),'rejected':rejected,'validator':'independent deterministic age predicate plus auxiliary Llama review','human_clinician_review':False,'reviewer_model_path':path,'reviewer_upstream_revision':None,'backbone_labels_used':False,'elapsed_seconds':time.time()-start})
 write(out/'status.json',{'status':'complete','reviewed_triplets':len(reviews),'accepted':len(accepted),'rejected':len(rejected),'elapsed_seconds':time.time()-start})
if __name__=='__main__':main()

