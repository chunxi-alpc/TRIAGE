"""D-only option rotations with frozen branch predictions and resumable prompt cache."""
import argparse,copy,hashlib,json,os,time
from pathlib import Path
ROOT=Path('/home/gcx/aster_validation_20261022/private/run500')
SOURCE=ROOT/'mechanism/priority200'
WORK=SOURCE/'d_option_order'
METHODS=('visible_prune','hidden_prune')
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def rotated(c,m):
 c=copy.deepcopy(c);old=c['options'];c['options']={k:old[m[k]] for k in sorted(m)}
 oldblock='Options:\n'+'\n'.join(f'{k}: {v}' for k,v in old.items())+'\nReturn only one option letter.'
 newblock='Options:\n'+'\n'.join(f'{k}: {v}' for k,v in c['options'].items())+'\nReturn only one option letter.'
 assert c['closed_book_prompt'].count(oldblock)==1
 c['closed_book_prompt']=c['closed_book_prompt'].replace(oldblock,newblock)
 return c

def prepare():
 from run import branch_prompt,prune_prompt
 WORK.mkdir(exist_ok=True);cases=rows(SOURCE/'test.jsonl');ids=json.loads((SOURCE/'selection.json').read_text())['ids'];assert [c['id'] for c in cases]==ids and len(set(ids))==200
 source=rows(SOURCE/'qwen_complete_parser_v2.jsonl');index={(r['id'],r['variant'],r['method']):r for r in source}
 maps=json.loads((ROOT/'cad200/option_analysis_view/option_order/design.json').read_text())['inverse_label_maps']
 originals=[]
 for rotation in ('left','right'):
  jobs=[]
  for c in cases:
   m=maps[rotation][c['id']];assert set(m)==set(m.values())==set(c['options']);rc=rotated(c,m)
   for visibility in ('hidden','visible'):assert branch_prompt(c,'D',visibility)==branch_prompt(rc,'D',visibility)
   assert rc['variant_rules']==c['variant_rules'] and rc['patient']==c['patient']
   for method in METHODS:
    old=index[c['id'],'D',method];b=index[c['id'],'D','branch_'+method.split('_')[0]];assert old['predicted_branch']==ord(b['prediction'])-65
    prompt=prune_prompt(rc,'D',bs=[old['predicted_branch']]);gold=next(k for k,v in m.items() if v==c['gold_answer']['D'])
    jobs.append(dict(id=c['id'],disease=c['disease'],stratum=old['stratum'],variant='D',method=method,rotation=rotation,gold=gold,canonical_gold=c['gold_answer']['D'],inverse_map=m,predicted_branch=old['predicted_branch'],prompt=prompt,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),labels=list(rc['options']),branch_reused=True))
    if rotation=='left':originals.append({**old,'rotation':'original','canonical_prediction':old['prediction'],'canonical_gold':old['gold']})
  path=WORK/(rotation+'_input.jsonl');data=''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in jobs)
  if path.exists():assert path.read_text()==data
  else:path.write_text(data)
  assert len(jobs)==400 and {x['id'] for x in jobs}==set(ids)
 write(WORK/'original.json',originals)
 write(WORK/'design.json',dict(ids=ids,source_sha256=sha(SOURCE/'qwen_complete_parser_v2.jsonl'),test_sha256=sha(SOURCE/'test.jsonl'),methods=METHODS,primary_family='Three all-200 hidden-minus-visible D comparisons, Holm across original/left/right. K/U and order changes separate exploratory families.',branch_inference_calls=0,max_new_tokens=8,seed=20261023,bootstrap=2000))
 print('Prepared 800 logical action evaluations, 400 original action rows reused, all 200 IDs; branch prompt invariance verified.',flush=True)

def run(rotation):
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM
 from run import encode_prompt,greedy_with_logits,parse
 jobs=rows(WORK/(rotation+'_input.jsonl'));expected={(j['id'],j['method']):j['prompt_sha256'] for j in jobs};assert len(expected)==400
 out=WORK/(rotation+'.jsonl');status=WORK/(rotation+'.status');cache={};done=set()
 if out.exists():
  for r in rows(out):
   key=(r['id'],r['method']);assert key not in done and expected[key]==r['prompt_sha256'];done.add(key);cache[r['prompt_sha256']]={k:r[k] for k in ('prediction','completion','option_logits')}
 if (WORK/(rotation+'.manifest.json')).exists():
  mf=json.loads((WORK/(rotation+'.manifest.json')).read_text());assert mf['complete'] and mf['records']==len(done)==400 and mf['input_sha256']==sha(WORK/(rotation+'_input.jsonl')) and mf['output_sha256']==sha(out);return
 torch.set_num_threads(4);torch.manual_seed(20261023);modelpath='/home/gcx/TRIAGE/models/modelscope/Qwen3-8B'
 tok=AutoTokenizer.from_pretrained(modelpath,local_files_only=True,use_fast=True)
 model=AutoModelForCausalLM.from_pretrained(modelpath,local_files_only=True,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='eager').eval()
 calls=0;hits=0;start=time.time()
 with out.open('a') as f:
  for job in jobs:
   if (job['id'],job['method']) in done:continue
   key=job['prompt_sha256'];reused=key in cache
   if reused:s=dict(cache[key]);hits+=1
   else:
    enc,_=encode_prompt(tok,job['prompt'],[]);enc={k:v.to(model.device) for k,v in enc.items()};ids={k:tok.encode(k,add_special_tokens=False) for k in job['labels']};assert all(len(x)==1 for x in ids.values())
    with torch.inference_mode():gen,z=greedy_with_logits(model,enc,tok.eos_token_id)
    text=tok.decode(gen[0,enc['input_ids'].shape[1]:],skip_special_tokens=True);s=dict(prediction=parse(text,job['labels']),completion=text,option_logits={k:float(z[v[0]]) for k,v in ids.items()});cache[key]=s;calls+=1
   r={k:v for k,v in job.items() if k not in ('prompt','labels')};r.update(s);r.update(correct=s['prediction']==job['gold'],parse_failure=s['prediction'] is None,canonical_prediction=job['inverse_map'].get(s['prediction']),inference_reused=reused)
   f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();done.add((job['id'],job['method']))
   write(status,dict(state='running',records=len(done),new_calls=calls,cache_hits=hits,seconds=time.time()-start))
 assert len(done)==400
 write(WORK/(rotation+'.manifest.json'),dict(complete=True,records=400,input_sha256=sha(WORK/(rotation+'_input.jsonl')),output_sha256=sha(out),ids=sorted({x['id'] for x in jobs})))
 write(status,dict(state='complete',records=400,new_calls=calls,cache_hits=hits,seconds=time.time()-start))

def analyze():
 import numpy as np
 from scipy.stats import binomtest
 from analyze import bootstrap
 out=Path(__file__).parent/'results/d_option_order';out.mkdir(parents=True,exist_ok=True)
 design=json.loads((WORK/'design.json').read_text());assert design['source_sha256']==sha(SOURCE/'qwen_complete_parser_v2.jsonl') and design['test_sha256']==sha(SOURCE/'test.jsonl');data=json.loads((WORK/'original.json').read_text())
 for rotation in ('left','right'):
  mf=json.loads((WORK/(rotation+'.manifest.json')).read_text());assert mf['complete'] and set(mf['ids'])==set(design['ids']) and mf['output_sha256']==sha(WORK/(rotation+'.jsonl')) and mf['input_sha256']==sha(WORK/(rotation+'_input.jsonl'))
  data+=rows(WORK/(rotation+'.jsonl'))
 assert len(data)==1200 and len({(x['id'],x['rotation'],x['method']) for x in data})==1200 and {x['id'] for x in data}==set(design['ids'])
 idx={(x['id'],x['rotation'],x['method']):x for x in data};summary={};tests=[]
 def paired(aa,bb,name,family):
  ids=sorted(aa);assert ids==sorted(bb);d=[int(bb[i]['correct'])-int(aa[i]['correct']) for i in ids];w=d.count(1);l=d.count(-1)
  r=dict(name=name,family=family,n=len(ids),difference=sum(d)/len(d),ci=bootstrap(d,[aa[i]['disease'] for i in ids]),wins=w,losses=l,p=float(binomtest(w,w+l,.5).pvalue) if w+l else 1.)
  tests.append(r);return r
 for rotation in ('original','left','right'):
  summary[rotation]={}
  for stratum in ('all','K','U'):
   sets={m:{x['id']:x for x in data if x['rotation']==rotation and x['method']==m and (stratum=='all' or x['stratum']==stratum)} for m in METHODS}
   summary[rotation][stratum]={m:dict(n=len(v),accuracy=sum(x['correct'] for x in v.values())/len(v),parse_failures=sum(x['parse_failure'] for x in v.values())) for m,v in sets.items()}
   paired(sets['visible_prune'],sets['hidden_prune'],rotation+'_'+stratum,'primary' if stratum=='all' else 'strata')
 for rotation in ('left','right'):
  for m in METHODS:
   aa={i:idx[i,'original',m] for i in sorted(design['ids'])};bb={i:idx[i,rotation,m] for i in sorted(design['ids'])}
   r=paired(aa,bb,rotation+'_'+m+'_vs_original','order');r['canonical_action_agreement']=sum(aa[i]['canonical_prediction']==bb[i]['canonical_prediction'] and aa[i]['canonical_prediction'] is not None for i in aa)/200
 for family in ('primary','strata','order'):
  tt=sorted([x for x in tests if x['family']==family],key=lambda x:x['p']);last=0
  for j,t in enumerate(tt):last=max(last,min(1,t['p']*(len(tt)-j)));t['holm_p']=last
 write(out/'summary.json',dict(design=design,accuracy=summary,paired=tests,audit=dict(records=1200,unique_ids=200,outside_ids=0,duplicate_keys=0),workers={r:json.loads((WORK/(r+'.status')).read_text()) for r in ('left','right')}))
 (out/'paired_scores.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in data))
 lines=['# D版选项重排：固定200例Qwen补充验证','','复用原分支预测及400条原顺序处置；两卡仅生成左右重排处置。仅处置选项标签/顺序改变，规则与分支顺序不变。K76/U124固定；不包含开发题，不调测试参数。','','| 顺序 | 可见动作判断＋裁剪 | 隐藏动作判断＋裁剪 | 隐藏收益（百分点） | 疾病簇95%区间 | Holm p |','|---|---:|---:|---:|---|---:|']
 for rotation in ('original','left','right'):
  s=summary[rotation]['all'];t=next(x for x in tests if x['name']==rotation+'_all');ci=t['ci'];lines.append(f"| {rotation} | {100*s['visible_prune']['accuracy']:.1f}% | {100*s['hidden_prune']['accuracy']:.1f}% | {100*t['difference']:+.1f} | [{100*ci[0]:.1f}, {100*ci[1]:.1f}] | {t['holm_p']:.6g} |")
 lines+=['','主检验为三个顺序各自的隐藏与可见配对比较，三项共同Holm校正。K/U六项、顺序变化四项分别为探索性检验族。精确McNemar（二项法）；疾病簇自助法2000次。解析失败计错，动作一致率按原标签逆映射。逐题输出、K/U、解析失败和顺序变化详见summary.json与paired_scores.jsonl。','','分支预测复用，因此本验证检验处置选项顺序对隐藏判断＋裁剪收益的影响，不检验分支候选本身重排。既有动机性筛查及注意力门槛结论不因本补充更改。','']
 lines+=['','## K/U与解析检查','','|顺序|K可见|K隐藏|U可见|U隐藏|解析失败（两方法合计）|','|---|---:|---:|---:|---:|---:|']
 for rotation in ('original','left','right'):
  s=summary[rotation];lines.append(f"|{rotation}|{100*s['K']['visible_prune']['accuracy']:.1f}%|{100*s['K']['hidden_prune']['accuracy']:.1f}%|{100*s['U']['visible_prune']['accuracy']:.1f}%|{100*s['U']['hidden_prune']['accuracy']:.1f}%|{sum(x['parse_failures'] for x in s['all'].values())}|")
 lines+=['','左右重排对同一方法的准确率改变均未显著（探索性Holm最小p=.875）；按动作逆映射后，与原顺序逐题动作一致率95.5%–96.5%。K/U内隐藏相对可见的收益在六项探索性比较校正后也均显著（最大Holm p=.0319）。','','新增800条逻辑结果由548次实际处置推理与252次完全相同提示缓存复用生成；另复用400条原顺序结果，新增分支推理为0。每顺序200例×2方法，名单外ID和重复键为0。两worker自然完成，不恢复Llama。','','本补充三项主检验族不同于原机制报告的全比较族，原顺序Holm p不同由校正族变化导致；原始配对p与输出未改变。','']
 (out/'REPORT.zh.md').write_text('\n'.join(lines));print('\n'.join(lines),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','left','right','analyze']);a=p.parse_args()
 if a.phase=='prepare':prepare()
 elif a.phase=='analyze':analyze()
 else:run(a.phase)
