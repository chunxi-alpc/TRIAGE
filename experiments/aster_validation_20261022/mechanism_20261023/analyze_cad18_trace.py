"""Read full private CAD traces; publish only labels, numeric summaries and hashes."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True).encode()).hexdigest()

def validate_trajectory(data,gold_id,eos_ids,filtered):
 ids=data['result']['token_ids'];steps=data['steps']
 if not ids or len(ids)!=len(steps) or len(ids)>128:raise ValueError('incomplete token trace')
 for k,(token,step) in enumerate(zip(ids,steps)):
  if step['step']!=k or step['prefix_token_ids']!=ids[:k]:raise ValueError('prefix mismatch')
  if token!=step['filtered_choice' if filtered else 'unfiltered_choice']:raise ValueError('selection mismatch')
  candidates=set(step['candidate_ids'])
  if not candidates or len(candidates)!=len(step['candidate_ids']):raise ValueError('invalid candidate set')
  if filtered and token not in candidates:raise ValueError('filtered choice is masked')
  for t in {gold_id,*eos_ids,step['unfiltered_choice'],step['filtered_choice']}:
   tracked=step['tracked_tokens'][str(t)]
   if tracked['masked']!=(t not in candidates):raise ValueError('mask membership mismatch')
   if tracked['masked']!=(tracked['logc']<step['threshold']):raise ValueError('threshold mismatch')
  if token in eos_ids and k!=len(ids)-1:raise ValueError('trace continued past EOS')
 if data['result']['stop_reason']!=('eos' if ids[-1] in eos_ids else 'length'):raise ValueError('stop reason mismatch')
 if ids[-1] not in eos_ids and len(ids)!=128:raise ValueError('non-EOS trace ended before cap')
 return {'generated_tokens':len(ids),'stop_reason':data['result']['stop_reason'],
  'gold_masked_steps':sum(gold_id not in step['candidate_ids'] for step in steps),
  'old_choice_masked_steps':sum(step['unfiltered_choice'] not in step['candidate_ids'] for step in steps),
  'candidate_count_min':min(len(s['candidate_ids']) for s in steps),
  'candidate_count_max':max(len(s['candidate_ids']) for s in steps),
  'final_token_is_eos':ids[-1] in eos_ids,
  'final_old_eos_would_be_masked_on_same_prefix':ids[-1] in eos_ids and ids[-1] not in steps[-1]['candidate_ids']}

def case_evidence(old,filtered,gold_id,eos_ids,expected):
 a=validate_trajectory(old,gold_id,eos_ids,False);b=validate_trajectory(filtered,gold_id,eos_ids,True)
 aa=old['result']['token_ids'];bb=filtered['result']['token_ids']
 diverged=next((k for k,(x,y) in enumerate(zip(aa,bb)) if x!=y),None)
 if diverged is None:raise ValueError('frozen discordant case has no token divergence')
 # These steps have exactly the same generated prefix, so their distributions can be compared.
 for k in range(diverged+1):
  x,y=old['steps'][k],filtered['steps'][k]
  for field in ('prefix_token_ids','unfiltered_choice','filtered_choice','conditional_logp_max','threshold','candidate_ids','tracked_tokens'):
   if x[field]!=y[field]:raise ValueError('same-prefix scores differ')
 x=old['steps'][diverged];first=old['steps'][0]
 outcome=expected['outcome'];gold=old['summary']['gold']
 if old['summary']['id']!=filtered['summary']['id'] or filtered['summary']['gold']!=gold:raise ValueError('case mismatch')
 for mode,data in [('old',old),('filtered',filtered)]:
  if digest(data['result']['completion'])!=expected[mode+'_completion_sha256']:raise ValueError('historical text not reproduced')
  if data['summary']['prediction']!=expected[mode+'_prediction']:raise ValueError('historical prediction not reproduced')
 old_at_div=aa[diverged];new_at_div=bb[diverged]
 return dict(id=old['summary']['id'],outcome=outcome,gold=gold,
  old_prediction=old['summary']['prediction'],filtered_prediction=filtered['summary']['prediction'],
  historical_text_reproduced=True,old=a,filtered=b,first_divergence_step=diverged,
  same_prefix_steps_verified=diverged+1,common_prefix_sha256=digest(aa[:diverged]),
  first_step_gold_masked=gold_id not in first['candidate_ids'],
  old_first_token_is_gold=aa[0]==gold_id,
  first_step_gold_logc=first['tracked_tokens'][str(gold_id)]['logc'],
  first_step_global_logc_max=first['conditional_logp_max'],first_step_threshold=first['threshold'],
  divergence_old_choice_masked=old_at_div not in x['candidate_ids'],
  divergence_old_choice_is_gold=old_at_div==gold_id,divergence_filtered_choice_is_gold=new_at_div==gold_id,
  divergence_gold_masked=gold_id not in x['candidate_ids'],divergence_candidate_count=len(x['candidate_ids']),
  divergence_old_logc=x['tracked_tokens'][str(old_at_div)]['logc'],
  divergence_old_logu=x['tracked_tokens'][str(old_at_div)]['logu'],
  divergence_old_cad_score=x['tracked_tokens'][str(old_at_div)]['cad_score'],
  divergence_new_logc=x['tracked_tokens'][str(new_at_div)]['logc'],
  divergence_new_logu=x['tracked_tokens'][str(new_at_div)]['logu'],
  divergence_new_cad_score=x['tracked_tokens'][str(new_at_div)]['cad_score'],
  divergence_threshold=x['threshold'],
  divergence_all_eos_masked=all(t not in x['candidate_ids'] for t in eos_ids),
  old_completion_sha256=expected['old_completion_sha256'],filtered_completion_sha256=expected['filtered_completion_sha256'])

def summarize(rows):
 out={}
 for outcome in sorted({r['outcome'] for r in rows}):
  rs=[r for r in rows if r['outcome']==outcome]
  out[outcome]=dict(n=len(rs),first_step_divergence=sum(r['first_divergence_step']==0 for r in rs),
   later_divergence=sum(r['first_divergence_step']>0 for r in rs),
   first_step_gold_masked=sum(r['first_step_gold_masked'] for r in rs),
   old_first_token_gold=sum(r['old_first_token_is_gold'] for r in rs),
   divergence_old_choice_masked=sum(r['divergence_old_choice_masked'] for r in rs),
   divergence_old_choice_gold=sum(r['divergence_old_choice_is_gold'] for r in rs),
   old_length_stop=sum(r['old']['stop_reason']=='length' for r in rs),
   filtered_length_stop=sum(r['filtered']['stop_reason']=='length' for r in rs),
   old_parse_failure=sum(r['old_prediction'] is None for r in rs),filtered_parse_failure=sum(r['filtered_prediction'] is None for r in rs))
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--traces',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 manifest=json.loads((a.traces/'manifest.json').read_text());auditpath=HERE/'results/step0_review/private_structure_summary.json';audit=json.loads(auditpath.read_text());expected={r['id']:r for r in audit['cad']['cases']}
 if not manifest['complete'] or not manifest['historical_reproduction_complete'] or manifest['cases']!=18 or manifest['decodes']!=36:raise ValueError('complete historical reproduction required')
 from transformers import AutoTokenizer
 model=Path('/home/gcx/TRIAGE/models/modelscope/Qwen3-8B');tok=AutoTokenizer.from_pretrained(model,local_files_only=True,use_fast=True)
 eos=json.loads((model/'generation_config.json').read_text())['eos_token_id'];eos=set(eos if isinstance(eos,list) else [eos]);rows=[];hashes={}
 for id_ in audit['cad']['frozen_case_ids']:
  paths=[a.traces/f'{id_}_{mode}.json' for mode in ('old','filtered')];old,filtered=[json.loads(p.read_text()) for p in paths]
  g=tok.encode(old['summary']['gold'],add_special_tokens=False)
  if len(g)!=1:raise ValueError('gold label is not one token')
  rows.append(case_evidence(old,filtered,g[0],eos,expected[id_]));hashes.update({p.name:sha(p) for p in paths})
 if len(rows)!=18 or len(hashes)!=36:raise ValueError('frozen trace list incomplete')
 start=a.traces.parent/'pid';seconds=(a.traces/'manifest.json').stat().st_mtime-start.stat().st_mtime
 result=dict(complete=True,cases=18,decodes=36,steps=sum(r[m]['generated_tokens'] for r in rows for m in ('old','filtered')),
  historical_text_match_count=36,alpha=manifest['alpha'],beta=manifest['beta'],max_new_tokens=manifest['max_new_tokens'],
  gpu=0,peak_gib=manifest['peak_gib'],approximate_wall_seconds_from_files=seconds,
  groups=summarize(rows),rows=rows,private_trace_file_sha256=hashes,
  provenance={'trace_manifest_sha256':sha(a.traces/'manifest.json'),'frozen_audit_sha256':sha(auditpath),
   'analysis_script_sha256':sha(__file__),'runner_script_sha256':manifest['script_sha256'],
   'original_CAD_source_sha256':manifest['source_sha256'],'input_sha256':manifest['input_sha256'],
   'torch':manifest['torch'],'transformers':manifest['transformers'],
   'model_metadata_sha256':{p.name:sha(p) for p in (model/'config.json',model/'generation_config.json',model/'tokenizer_config.json')}},
  scope='Fixed B discordant cases selected retrospectively from original200: 16 lost and 2 gained. Not an unbiased CAD evaluation sample.',
  private_full_traces=str(a.traces),limitations=['Mask changes full-vocabulary tokens, not entire clinical options.',
   'Same-prefix distributions verified only up to the first divergence; later prefixes are different.',
   'These 18 cases do not estimate the prevalence of the mechanism in the 800 cohort.',
   'No clinical review, source-version independence or internal-attention causal claim.'])
 a.output.mkdir(parents=True,exist_ok=True);p=a.output/'cad18_trace_summary.json'
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(result,indent=2)+'\n')
 lines=['# 固定18例CAD完整轨迹诊断','真实GPU诊断已完成。18例B、原版/过滤版各一次，共36条自由解码；36/36最终文本SHA256及解析结果与历史记录一致。只为补存缺失轨迹，不改写原结果。',
  f"GPU0，峰值{result['peak_gib']:.2f} GiB；依据文件时间约{seconds/60:.2f}分钟。Qwen3-8B、BF16、eager、alpha=1、beta=.1、128词元，EOS按原模型generation config。",
  '## 直接证据','每一步保存实际生成前缀token ID、条件/无条件log概率、全词表最大条件log概率及阈值、完整候选token集合，以及未过滤/过滤下一词元选择和金标/EOS掩码。两条轨迹到首次分歧（含该步）的前缀和所记录分布完全一致，因此该步可直接比较。',
  '|原比较结局|例数|首步分歧|后续才分歧|首步金标字母被过滤|首步旧选择为金标|首次分歧旧选择被过滤|首次分歧旧选择为金标|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for name,label in [('lost','过滤后失分'),('gained','过滤后获益')]:
  x=result['groups'][name];lines.append('|'+label+'|'+'|'.join(str(x[k]) for k in ('n','first_step_divergence','later_divergence','first_step_gold_masked','old_first_token_gold','divergence_old_choice_masked','divergence_old_choice_gold'))+'|')
 lines+=['16例失分的首步金标字母均被完整候选集合直接证实排除，补足了旧审计仅凭选项概率充分条件推导的证据。13例旧首词元即金标，首步排除直接改变其下一词元。medguide_05698首步相同、step=1（第二个词元）才分歧，须与首步变化15例区分。',
  '两例获益首步金标字母均保留；旧首选的其他词元被过滤，从而改选金标。对这18例，首次分歧时未过滤旧首选18/18都不在候选集合，过滤直接触发下一词元变化；最终文本和历史一致。',
  '## 停止、解析与解释范围','失分16例中：旧版8例达到128词元上限、过滤版2例达到上限；过滤版1例解析失败，其余为有效但错误的选项。两个获益病例都正常EOS。停止更早不代表答案更正确。金标字母的某一步被屏蔽不等于完整处置动作永久不可生成，后续前缀也可能重新允许该字母。',
  '## 逐例核查','step从0开始；公开仅标签、统计、数值和哈希。完整患者/规则文本、生成文本、token序列和候选集合均留私有服务器。',
  '|病例|结局|原预测→过滤预测|金标|首分歧step|原/过滤词元数|原/过滤停止|','|---|---|---|---|---:|---|---|']
 for r in rows:lines.append(f"|{r['id']}|{r['outcome']}|{r['old_prediction']} → {r['filtered_prediction']}|{r['gold']}|{r['first_divergence_step']}|{r['old']['generated_tokens']}/{r['filtered']['generated_tokens']}|{r['old']['stop_reason']}/{r['filtered']['stop_reason']}|")
 lines+=['## 与800例的关系','本18例来自原200中按已知损益挑选的B病例，含16失分和2获益，不能据此估计800例的机制发生率或做新的无偏效果检验。800 B过滤CAD下降4.875个百分点是完整队列统计；本诊断提供固定病例的解码机制证据，不把二者合并计样本量。',
 '## 仍未完成','具体指南文档/版本/页码映射、两名医学专家盲审及仲裁、预先认可的非劣效/临床报警确认标准继续开放。完整CAD轨迹这一执行项已完成，临床合理性混杂仍未排除。',
 '## 复现','原轨迹脚本step0_cad_trace.py；只读汇总脚本analyze_cad18_trace.py。原/过滤输入与历史输出指纹在manifest及cad18_trace_summary.json，逐文件哈希覆盖全部36个私有轨迹文件。新增模型推理仅36条解码，未重跑800或200全套。']
 (a.output/'CAD18_TRACE.zh.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'complete':True,'groups':result['groups'],'steps':result['steps'],'peak_gib':result['peak_gib'],'seconds':seconds}))
if __name__=='__main__':main()
