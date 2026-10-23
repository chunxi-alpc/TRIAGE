"""Read-only CAD/branch-action comparison on the exact selected 200 test IDs."""
import argparse,json,collections,datetime,hashlib
from pathlib import Path
from analyze import bootstrap,paired

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert not a.out.exists();r=a.root;selection=json.loads((r/'mechanism/priority200/selection.json').read_text());ids=set(selection['ids']);assert len(ids)==200
 mech=Path(__file__).resolve().parent/'results/priority200/qwen/full_parser_v2/summary.json';report={'time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'cases':200,'selection_sha256':hashlib.sha256((r/'mechanism/priority200/selection.json').read_bytes()).hexdigest(),'CAD_cells':{},'paired_filtered_vs_unfiltered':{},'pending':[],'mechanism':json.loads(mech.read_text()),'read_only':True};buckets={};safe=[]
 sources=[('CAD8',r/'qwen_full_8token_original.jsonl'),('CAD128',r/'qwen_full.jsonl')]+[(f'CAD_filtered_{order}',r/'cad_fix'/f'qwen_{order}_filtered.jsonl') for order in ['original','left','right']]
 for revision,source in sources:
  mf=source.with_suffix('.manifest.json')
  if not source.exists() or not mf.exists() or not json.loads(mf.read_text()).get('complete',True):report['pending'].append(revision);continue
  grouped=collections.defaultdict(list)
  for line in source.read_text().splitlines():
   x=json.loads(line)
   if x['id'] not in ids or x['method'] not in ['cad','prune_cad']:continue
   k=(x['method'],x['variant'][0],x.get('assigned_wrong',False));grouped[k].append(x)
  assert len(grouped)==9
  for (method,variant,wrong),rows in grouped.items():
   assert len(rows)==200 and {x['id'] for x in rows}==ids
   key='/'.join(map(str,[revision,method,variant,wrong]));bad=sum(x['prediction'] is None for x in rows);correct=sum(x['correct'] for x in rows);report['CAD_cells'][key]={'n':200,'correct':correct,'accuracy':correct/200,'parse_failures':bad,'valid_choice_errors':200-bad-correct,'parse_failure_rate':bad/200,'accuracy_disease_ci':bootstrap([int(x['correct']) for x in rows],[x['disease'] for x in rows]),'original_action_selection_rate':sum(x['prediction']==x['original_answer'] for x in rows)/200,'sibling_misuse_rate':sum(x['prediction']==x['sibling_action_label'] for x in rows)/200};buckets[revision,method,variant,wrong]=rows
   for x in rows:safe.append({k:x[k] for k in ['id','disease','variant','method','prediction','gold_answer','original_answer','sibling_action_label','correct','stratum']}|{'revision':revision,'assigned_wrong':wrong,'parse_failure':x['prediction'] is None})
 for key,aa in buckets.items():
  revision,method,v,wrong=key
  if revision=='CAD128' and ('CAD_filtered_original',method,v,wrong) in buckets:report['paired_filtered_vs_unfiltered']['/'.join(map(str,[method,v,wrong]))]=paired(aa,buckets['CAD_filtered_original',method,v,wrong])
 tests=sorted(report['paired_filtered_vs_unfiltered'].values(),key=lambda x:x['p']);prev=0
 for j,x in enumerate(tests):prev=max(prev,min(1,x['p']*(len(tests)-j)));x['p_holm']=prev
 report['CAD_complete']=not report['pending'];a.out.mkdir(parents=True);(a.out/'summary.json').write_text(json.dumps(report,indent=2));(a.out/'cad_scores.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in safe))
 lines=['# Qwen同批200题：CAD与机制实验联合汇总',f"更新时间：{report['time']}。固定200题，K76/U124，40开发题不计入。",'CAD候选过滤版及三个顺序均已完整结束。' if report['CAD_complete'] else 'CAD尚未全部结束，以下为已完成版本；缺失版本不计为错误，也不冒充修复结果。待完成：'+', '.join(report['pending']),'|方法|A|B|C|D|','|---|---:|---:|---:|---:|']
 for method in ['existing_baseline','existing_signed','visible_prune','hidden_prune','hidden_top2','existing_prune']:
  lines.append('|'+method+'|'+'|'.join(f"{100*report['mechanism']['cells'][v+'/'+method+'/all']['accuracy']:.1f}%" for v in 'ABCD')+'|')
 for revision in ['CAD8','CAD128','CAD_filtered_original','CAD_filtered_left','CAD_filtered_right']:
  if (revision,'cad','A',False) not in buckets:continue
  lines.append('|'+revision+'|'+'|'.join(f"{100*report['CAD_cells']['/'.join(map(str,[revision,'cad',v,False]))]['accuracy']:.1f}%" for v in 'ABC')+'|未运行|')
 lines+=['\n## CAD解析失败与真实选择错误','|版本/条件|正确|解析失败|有效但选错|','|---|---:|---:|---:|']
 for key,x in report['CAD_cells'].items():lines.append(f"|{key}|{x['correct']}|{x['parse_failures']}|{x['valid_choice_errors']}|")
 lines+=['\n每行分母均为200；正确、解析失败、有效但选错互斥且合计200。CAD无D/CS输出，不能填入其分数或计算相同BCD/BCDCS调和平均。左右重排与原顺序分开列出，不混合报告。','机制结论：分支与动作错误并存；K组动作诱导有害翻转尚未达到校正后的预设注意力门槛，未执行阻断。隐藏动作裁剪在D上优于可见预测裁剪，但A/C非全面优势，仍低于金标。CS2例仅作描述。','完整机制结果与逐题统计见../priority200/qwen/full_parser_v2。CAD逐题统计见cad_scores.jsonl；同批过滤前后配对检验和疾病簇区间见summary.json。Llama保持保存暂停。'];(a.out/'REPORT.zh.md').write_text('\n\n'.join(lines[:3])+'\n\n'+'\n'.join(lines[3:])+'\n');print(json.dumps({'complete':report['CAD_complete'],'pending':report['pending'],'path':str(a.out)}))
if __name__=='__main__':main()
