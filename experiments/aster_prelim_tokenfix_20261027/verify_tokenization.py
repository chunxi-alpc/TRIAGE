"""Compare actual model input encoding with template encoding; no GPU inference."""
from pathlib import Path
import hashlib,json
from transformers import AutoTokenizer
from model_input import encode_prompts
D=Path(__file__).resolve().parent;OLD=D.parent/'aster_prelim_repaired_20261027'
rows=[json.loads(s) for s in (D/'private/cases.jsonl').read_text().splitlines() if '"variant": "NC"' in s]
reports={}
for name,folder in [('qwen','Qwen3-8B'),('llama','Meta-Llama-3.1-8B-Instruct')]:
 tok=AutoTokenizer.from_pretrained('/home/gcx/TRIAGE/models/modelscope/'+folder,local_files_only=True,padding_side='left')
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 texts=[]
 for fn,field in [(OLD/'private'/name/'prompts.jsonl','prompt'),(OLD/'private'/('p2_'+name)/'prompts.jsonl','text')]:
  texts.extend(json.loads(s)[field] for s in fn.read_text().splitlines())
 # Reconstruct the exact P2 closed-book prompts, including original options.
 for row in rows:
  text='Patient:\n'+row['patient']+'\n\n'+row['source_question']+'\nOptions:\n'+'\n'.join(chr(65+i)+': '+v for i,v in enumerate(row['options']))+'\nReturn only the option letter.'
  texts.append(tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False))
 oldeq=0;dup=0;records=[];bos=tok.bos_token_id
 for text in texts:
  old=tok(text)['input_ids'];new=encode_prompts(tok,text)['input_ids'];expected=tok.encode(text,add_special_tokens=False)
  assert new==expected
  same=old==new;oldeq+=same;duplicate=bos is not None and old[:2]==[bos,bos];dup+=duplicate
  if bos is not None:assert new[0]==bos and new[1]!=bos
  records.append({'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),'old_tokens':len(old),'new_tokens':len(new),'identical':same,'old_duplicate_BOS':duplicate,'new_tokens_sha256':hashlib.sha256(json.dumps(new).encode()).hexdigest()})
 # Test the exact batched encode helper used by both runners, with variable lengths.
 for i in range(0,len(texts),4):
  subset=texts[i:i+4];enc=encode_prompts(tok,subset,padding=True,return_tensors='pt')
  for j,text in enumerate(subset):assert enc['input_ids'][j][enc['attention_mask'][j].bool()].tolist()==tok.encode(text,add_special_tokens=False)
 for content in ['Which branch condition applies? Return only 1 or 2.','Patient facts remain unchanged.']:
  msgs=[{'role':'user','content':content}];text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
  assert encode_prompts(tok,text)['input_ids']==tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,enable_thinking=False)
 reports[name]={'status':'PASS','prompts_checked':len(texts),'old_and_fixed_identical':oldeq,'old_duplicate_BOS':dup,'fixed_duplicate_BOS':0,'canonical_template_and_batched_helper_agree':True,'safe_to_reuse_old_outputs':oldeq==len(texts)}
 (D/'results'/(name+'_tokenization_audit.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records))
(D/'results/tokenization_audit.json').write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(reports))
