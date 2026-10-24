"""Content-addressed reuse of PR23 and interrupted-run inference, without model calls."""
import hashlib,json,re
from pathlib import Path
PROTOCOL='Qwen3-8B|BF16|eager|greedy8|gain0.5|CADalpha1|model_eos'
def key(prompt,labels,intervention=None,closed=None,beta=None,cap=8):
 bias={}
 if intervention:
  for name,gain in [('positive',.5),('other',intervention['other_gain'])]:
   for i in intervention[name]:
    span=tuple(intervention['spans'][i]);bias[span]=bias.get(span,0)+gain
 return hashlib.sha256(json.dumps([PROTOCOL,prompt,sorted(labels),sorted((a,b,g) for (a,b),g in bias.items() if g),closed,beta,cap],ensure_ascii=False).encode()).hexdigest()
def request(c,v,m,r,old):
 labels=c['options'];prompt=old.final_prompt(c,v);iv=None;closed=None;beta=None;cap=8
 if m.startswith('branch_'):return key(old.branch_prompt(c,v,m[7:]),[chr(65+i) for i in range(len(c['branches']))])
 if m in ['existing_prune','gold_branch_prune']:
  single=c.get('gold_rule',{}).get(v,c['sib_rule'] if v=='CS' else c['app_rule'])
  prompt=old.prune_prompt(c,v,single=single) if m=='existing_prune' else old.prune_prompt(c,v,bs=[c['gold_branch'][v]])
 elif m in ['visible_prune','hidden_prune','hidden_top2']:
  first=r.get('predicted_branch');second=r.get('runner_up')
  if first is None:return None
  prompt=old.prune_prompt(c,v,bs=[first]+([second] if m=='hidden_top2' and second is not None else []))
 elif m in ['existing_app','existing_signed','hidden_signed','hidden_unsigned']:
  end=prompt.index('Patient:\n');starts=[x.start() for x in re.finditer('IF the clinical sequence is: ',prompt[:end])];spans=[[x,(starts[j+1] if j+1<len(starts) else end)-2] for j,x in enumerate(starts)]
  if m.startswith('existing_'):
   pos=[c.get('gold_rule',{}).get(v,c['sib_rule'] if v=='CS' else c['app_rule'])];other=c['app_rule'] if v=='CS' else c['sib_rule'];other=[] if other is None else [other];og=-.5 if m=='existing_signed' else 0.
  else:
   first=r.get('predicted_branch');second=r.get('runner_up')
   if first is None:return None
   pos=c['branches'][first]['rule_indices'];other=c['branches'][second]['rule_indices'] if second is not None else [];og=-.5 if m=='hidden_signed' else .5
  iv={'spans':spans,'positive':pos,'other':other,'other_gain':og}
 elif m.startswith('CAD'):closed=c['closed_book_prompt'];cap=8 if m=='CAD8' else 128;beta=.1 if m=='CAD_filtered128' else None
 elif m!='existing_baseline':return None
 return key(prompt,labels,iv,closed,beta,cap)
def rows(p):
 if Path(p).exists():
  with Path(p).open() as f:
   for s in f:yield json.loads(s)
def build():
 import run as runner
 old=runner.old;private=Path('/home/gcx/qwen800_methods_20261023/private');root=Path('/home/gcx/aster_validation_20261022/private/run500');cases={r['id']:r for r in rows(private/'cases.jsonl')};cache={};sources=[]
 def add(k,r,path):
  if k is None or 'completion' not in r or r.get('upstream_branch_parse_failure'):return
  val={x:r[x] for x in ['completion','option_logits','generated_tokens','max_new_tokens','stop_reason','input_tokens'] if x in r};val['reused_from']=str(path);cache.setdefault(k,val)
 def source(path):
  if path.exists():sources.append({'path':str(path),'sha256':runner.sha(path)})
 # Historical ABC: exact old prompts and intervention spans, both correct/wrong assignments.
 data={(r['id'],r['variant'][0]):r for r in rows(root/'all.jsonl') if r['id'] in cases}
 for name in ['qwen_full_8token_original.jsonl','qwen_full.jsonl','cad200/original_filtered.jsonl','cad_fix/qwen_original_filtered.jsonl']:
  path=root/name;source(path)
  for r in rows(path):
   c=data.get((r['id'],r['variant'][0]));m=r['method']
   if c is None or m not in ['baseline','prune','app','signed','cad','prune_cad']:continue
   prompt=c['prompt_text'];iv=None;closed=None;beta=None;cap=8
   wrong=r.get('assigned_wrong',False);a,b=(c['sibling_char_span'],c['applicable_char_span']) if wrong else (c['applicable_char_span'],c['sibling_char_span'])
   if m in ['prune','prune_cad']:prompt='Follow this supplied rule even if it conflicts with medical memory:\n'+c['sibling_rule_text' if wrong else 'applicable_rule_text']+'\n\n'+c['closed_book_prompt']
   if m in ['app','signed']:
    if r.get('gain')!=.5:continue
    iv={'spans':[a,b],'positive':[0],'other':[1],'other_gain':-.5 if m=='signed' else 0.}
   if m in ['cad','prune_cad']:closed=c['closed_book_prompt'];cap=r.get('max_new_tokens',8);beta=r.get('plausibility_beta')
   add(key(prompt,c['options'],iv,closed,beta,cap),r,path)
 # Historical branch methods and current smoke/interrupted outputs.
 for inp,outs in [(root/'mechanism/priority200/test.jsonl',[root/'mechanism/priority200/qwen_complete_parser_v2.jsonl']),(private/'cases.jsonl',[private/'predictions0.jsonl',private/'predictions1.jsonl',private/'smoke_output.jsonl',private/'smoke_cs_output.jsonl'])]:
  cs={c['id']:c for c in rows(inp) if c['id'] in cases};source(inp)
  for path in outs:
   source(path)
   for r in rows(path):
    c=cs.get(r['id'])
    if c is not None:add(request(c,r['variant'],r['method'],r,old),r,path)
 dest=private/'reuse_cache.jsonl';dest.write_text(''.join(json.dumps({'key':k,'result':v},ensure_ascii=False)+'\n' for k,v in cache.items()))
 # Count reusable target rows where upstream branches are already available.
 counts={};matched_ids=set();eligible=0
 for c in cases.values():
  for v in c['variant_rules']:
   for m in runner.METHODS+['branch_hidden','branch_visible','branch_hide_inapp']:
    r={}
    if m.startswith(('hidden_','visible_')):
     mode=m.split('_')[0];labs=[chr(65+i) for i in range(len(c['branches']))];br=cache.get(key(old.branch_prompt(c,v,mode),labs))
     if br is None:continue
     pred=old.parse(br['completion'],labs)
     if pred not in labs:continue
     ranking=sorted(br.get('option_logits',{}),key=br.get('option_logits',{}).get,reverse=True);r={'predicted_branch':labs.index(pred),'runner_up':next((labs.index(x) for x in ranking if x!=pred),None)}
    k=request(c,v,m,r,old)
    if k in cache:
     src=cache[k]['reused_from'];counts[src]=counts.get(src,0)+1;matched_ids.add(c['id']);eligible+=1
 report={'cache_sha256':runner.sha(dest),'unique_cached_requests':len(cache),'target_rows_with_matching_cached_inference':eligible,'target_cases_with_matching_cached_inference':len(matched_ids),'by_source':counts,'sources':sources,'historical500_overlap':len({i for i,v in data}),'policy':'Exact prompt, labels, attention spans/weights and decoding settings; reparse completion and rescore against PR24 frozen gold/KU. Original overlap is reuse, not independent replication.'}
 (Path(__file__).parent/'results/reuse_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='sources'},indent=2))
if __name__=='__main__':build()
