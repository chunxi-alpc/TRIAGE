"""Public report for the seed-0 development improvement; exports no clinical text."""
import argparse,csv,gzip,json,time,collections
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from continuation_report import paired
from improve_branch import read,key,argmax,BASE,CORE

def writecsv(path,rows):
 if not rows:return
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 cases={key(c):c for c in read(BASE/'cases.jsonl') if c['split']=='dev' and c['seed']==0 and c['variant'] in CORE}
 meta={};summary=[];branches=[];conditional=[];intervals=[];alltables={};costs=[]
 for model in ('qwen','llama'):
  root=a.root/model;meta[model]={}
  for name in ('manifest','status','fusion_choices','FAILED'):
   p=root/(name+'.json')
   if p.exists():meta[model][name]=json.loads(p.read_text())
  tables={};branch={}
  for stage in ('answers','branches'):
   for p in sorted((root/stage).glob('*.jsonl')):
    rows=read(p);table={r['key']:r for r in rows};(tables if stage=='answers' else branch)[p.stem]=table
    dest=a.out/'records'/model/stage/(p.stem+'.jsonl.gz');dest.parent.mkdir(parents=True,exist_ok=True)
    allowed=('key','id','tree','seed','variant','split','p','assessments','prediction','scores','correct','gold_option','oracle','seconds','reused','baseline_source')
    with dest.open('wb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as f:
     for r in rows:f.write((json.dumps({k:r[k] for k in allowed if k in r})+'\n').encode())
  if tables:
   common=set.intersection(*(set(t) for t in tables.values()));alltables[model]=tables
   for name,table in tables.items():
    for variant in (*CORE,'all'):
     ks=[k for k in common if variant=='all' or cases[k]['variant']==variant]
     if ks:summary.append({'model':model,'method':name,'variant':variant,'rows':len(ks),'cases':len({cases[k]['id'] for k in ks}),'accuracy':sum(table[k]['correct'] for k in ks)/len(ks),'oracle':table[ks[0]]['oracle']})
    times=[r['seconds'] for r in table.values() if not r['reused']]
    costs.append({'model':model,'method':name,'measured_calls':len(times),'incremental_seconds':sum(times),'reused_rows':sum(r['reused'] for r in table.values())})
   for src in ('condition','weighted_oof','equal_external'):
    for method in ('direct','crop','soft_all','hard_all'):
     name=src+'_'+method
     if name not in tables:continue
     for baseline in ('blind_'+method,src+'_direct',src+'_crop'):
      if baseline==name or baseline not in tables:continue
      ci=paired(list(tables[name].values()),list(tables[baseline].values()))
      if ci['n']:intervals.append({'model':model,'method':name,'baseline':baseline,**ci})
  branch_common=set.intersection(*(set(t) for t in branch.values())) if branch else set()
  for name,table in branch.items():
   ks=[k for k in branch_common if k in cases]
   if not ks:continue
   correct=[k for k in ks if argmax(table[k]['p'])==cases[k]['gold_rule']]
   branches.append({'model':model,'source':name,'rows':len(ks),'cases':len({cases[k]['id'] for k in ks}),'branch_accuracy':len(correct)/len(ks),'mean_max_probability':sum(max(table[k]['p']) for k in ks)/len(ks),'high_confidence_count':sum(max(table[k]['p'])>=.9 for k in ks)})
   for method in ('direct','crop','soft_all','hard_all','soft_late'):
    ans=tables.get(name+'_'+method,{})
    for correct_flag in (True,False):
     common=[k for k in ks if k in ans and (argmax(table[k]['p'])==cases[k]['gold_rule'])==correct_flag]
     if common:conditional.append({'model':model,'source':name,'method':method,'branch_correct':correct_flag,'rows':len(common),'answer_accuracy':sum(ans[k]['correct'] for k in common)/len(common)})
 writecsv(a.out/'accuracy.csv',summary);writecsv(a.out/'branches.csv',branches);writecsv(a.out/'conditional.csv',conditional);writecsv(a.out/'costs.csv',costs)
 (a.out/'status.json').write_text(json.dumps(meta,indent=2));(a.out/'paired_intervals.json').write_text(json.dumps(intervals,indent=2))
 if summary:
  fig,axes=plt.subplots(1,2,figsize=(15,8))
  for ax,model in zip(axes,('qwen','llama')):
   vals=[r for r in summary if r['model']==model and r['variant']=='all'];ax.barh([r['method'] for r in vals],[r['accuracy'] for r in vals]);ax.set_xlim(0,1);ax.set_xlabel('Development accuracy (seed 0)');ax.set_title(model)
  fig.tight_layout();fig.savefig(a.out/'comparison.png',dpi=150);plt.close(fig)
 lines=['# ASTER 单种子开发集改进实验','',f'导出时间：{time.strftime("%Y-%m-%d %H:%M:%S UTC",time.gmtime())}。运行中数据仅为中间结果。','','只使用 seed=0、200例开发集的 NC/AC/IC/SW 四卷，800条，实际4棵树；不读取测试集结果选参。GPU6顺序运行Qwen3-8B和Llama3.1-8B。原始锁定实验继续独立运行。','','## 实际状态','']
 for model,m in meta.items():
  s=m.get('status',{});lines.append(f"- {model}: {s.get('stage','尚未启动')}，{s.get('completed_rows',0)}/{s.get('total_rows',800)}；峰值显存 {s.get('peak_allocated_GiB',0):.2f} GiB；GPU占用 {s.get('gpu_hours_this_process',0):.3f}小时。")
  if 'FAILED' in m:lines.append('  错误：'+m['FAILED']['error'])
 lines+=['','## 分支判断','','|模型|来源|行数|分支准确率|','|---|---|---:|---:|']
 for r in branches:lines.append(f"|{r['model']}|{r['source']}|{r['rows']}|{r['branch_accuracy']:.2%}|")
 lines+=['','## 相同已完成样本的答案准确率','','|模型|方法|共同完成行数|准确率|','|---|---|---:|---:|']
 for r in summary:
  if r['variant']=='all':lines.append(f"|{r['model']}|{r['method']}|{r['rows']}|{r['accuracy']:.2%}|")
 lines+=['','## 方法与解释','','- condition：每条规则单独判断满足、不满足或信息不足。只输入病人和该规则条件，输出三个候选标签的概率；各规则满足概率归一化为分支分布。相同病人和条件跨卷复用判断。归一化概率不声称经过校准。','- blind：复用原模型盲判；equal_external：原盲判与1.7B隐藏处置判断等权平均。','- weighted_oof：来源顺序为新条件判断、原盲判、1.7B辅助判断。预先固定7种权重，包括完全不使用辅助模型。每次用其余3棵树的宏平均分支准确率选择权重，再应用到留出的树；权重和训练树记录在status.json。属于探索性开发比较，不是独立测试结论。','- 每种来源均比较直接映射、整条裁剪、全层软门控（λ=2）、全层硬门控；指南处置是唯一门控目标，选项不干预，回答阶段不盲判。另比较新/旧判断的后半层软门控。','- agreement_gate/crop：新条件判断和旧盲判选择相同分支时才干预，否则回退无干预。没有把原始高概率直接当作可靠性。','- gold_crop仅作为明确的金标上限。预测函数只接收不含金标的输入；gold_rule只用于开发集选权重和评分。','- baseline_source标记复用的实际旧预测；新的逐例耗时是增量耗时。相同配置在同一病例内复用答案。','- 配对区间按树重采样10000次，4棵树带来的不确定性必须保留。条件正确/错误的分组结果在conditional.csv。','- 不因为本轮结果改写原已锁定测试协议；是否改进以全部800条完成后比较。','','复现：在115服务器运行 `bash experiments/aster_e0_e8_20261024/launch_improvement.sh`。私有输入和完整日志保留于 `/home/gcx/aster_e0_e8_20261024/private/improvement_seed0_v1`。公开records只含数值和病例编号，不含病人、规则或选项原文。']
 (a.out/'报告.zh.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'models':{m:v.get('status',{}).get('stage') for m,v in meta.items()},'summary_rows':len(summary)}))
if __name__=='__main__':main()
