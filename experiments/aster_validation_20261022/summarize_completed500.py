"""Read-only summary of completed 500-case original-order evaluations; CPU only."""
import argparse,json,datetime,concurrent.futures
from pathlib import Path

def extract(item):
 model,root=item;path=root/model/'full/summary.json';d=json.loads(path.read_text());assert d['complete'] and d['n_cases']==500
 variants=['A_original','B_applicable_rule_changed','C_sibling_rule_changed'];methods=['baseline','annotate','prune','app','signed','unsigned','cad','prune_cad'];result={'n_cases':500,'K':d['metrics']['all/A_original/closed_book/all']['correct'],'cells':{},'paired_vs_baseline':{}}
 for variant in variants:
  for method in methods:
   key=f'all/{variant}/{method}/all';x=dict(d['metrics'][key]);n=x['n'];bad=round(x['parse_failure_rate']*n);x.update(parse_failures=bad,valid_choice_errors=n-bad-x['correct']);result['cells'][variant[0]+'/'+method]=x
   for stratum in ['all','K','U']:
    comparison=d['comparisons'].get(f'all/{variant}/{method}/{stratum}')
    if comparison:result['paired_vs_baseline'][variant[0]+'/'+method+'/'+stratum]=comparison
 return model,result

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent/'results/run500');p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:models=dict(pool.map(extract,[(m,a.root) for m in ['llama','qwen']]))
 report={'updated_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'scope':'500 held-out cases, original option order, free generation. CAD here is the completed unfiltered 128-token version; candidate-filtered CAD pending. No partial option-order or development performance included.','models':models};(a.out/'summary.json').write_text(json.dumps(report,indent=2))
 lines=['# 已完成500题结果（原选项顺序）','仅汇总已完整完成的测试；40开发题与尚未完成的重排结果不混入。所有准确率为自由生成后解析。K为原闭卷原动作偏好：Llama 212，Qwen 183。','|模型|方法|A准确率|B准确率|C准确率|','|---|---|---:|---:|---:|']
 for model,d in models.items():
  for method in ['baseline','annotate','prune','app','signed','unsigned']:
   cells=[d['cells'][v+'/'+method] for v in 'ABC'];lines.append('|'+model+'|'+method+'|'+'|'.join(f"{100*x['accuracy']:.1f}%" for x in cells)+'|')
 lines+=['\n## 配对比较：带符号引导对基线','|模型/版本|差值（百分点）|疾病簇95%区间|Holm p|','|---|---:|---|---:|']
 for model,d in models.items():
  for v in 'ABC':
   x=d['paired_vs_baseline'][v+'/signed/all'];lines.append(f"|{model}/{v}|{100*x['accuracy_delta']:.1f}|{[round(100*z,2) for z in x['disease_bootstrap_ci']]}|{x['holm_p']:.4g}|")
 lines+=['\n## 已完成CAD128只读统计','这是未过滤128词元版本，不是排队中的候选过滤修改版；后者尚未完成，不能提前评价修复效果。','|模型/版本|正确/500|解析失败/500|有效选择错误/500|','|---|---:|---:|---:|']
 for model,d in models.items():
  for v in 'ABC':
   x=d['cells'][v+'/cad'];lines.append(f"|{model}/{v}|{x['correct']}|{x['parse_failures']}|{x['valid_choice_errors']}|")
 lines+=['\n## 目前可得出的结论','给出金标单规则后准确率大幅提高；这表明规则筛选或动作隔离有帮助，尚不能区分分支判断失败和分支内动作竞争。带符号引导的作用因模型和版本而异，不能替代规则隔离。新机制实验尚未开始正式推理，动作诱导分支偏移与隐藏动作裁剪的独立收益目前无正式结果。','原动作选择率、误用率、解析失败率、K/U配对统计及疾病簇区间见summary.json。原统计来源与逐题数值仍保留在各模型full目录。']
 (a.out/'REPORT.zh.md').write_text('\n\n'.join(lines[:2])+'\n\n'+'\n'.join(lines[2:])+'\n');print(json.dumps({'report':str(a.out/'REPORT.zh.md'),'models':{m:d['n_cases'] for m,d in models.items()}}))
if __name__=='__main__':main()
