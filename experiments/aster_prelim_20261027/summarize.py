"""Aggregate only this run, retain failures, and enforce the P1 stopping gate."""
from pathlib import Path
import collections,csv,hashlib,json,random,shutil,statistics
BASE=Path(__file__).resolve().parent
VOLS=['NC','IC','AC','SW','CS']
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 res=BASE/'results';res.mkdir(exist_ok=True);audit=json.loads((res/'data_audit.json').read_text());allrows=[];states={};summary=[];paired=[]
 for name in ['qwen','llama']:
  src=BASE/'private'/name
  if not (src/'DONE.json').exists():states[name]='not_completed';continue
  states[name]='completed';rows=[json.loads(s) for s in (src/'predictions.jsonl').read_text().splitlines()]
  assert len(rows)==2*sum(audit['counts'].values())
  for r in rows:r['model']=name
  allrows+=rows
  shutil.copyfile(src/'manifest.json',res/f'{name}_manifest.json')
  shutil.copyfile(BASE/'private'/f'{name}.log',res/f'{name}_run.log')
  for vol in VOLS:
   rs=[r for r in rows if r['variant']==vol];byid=collections.defaultdict(dict)
   for r in rs:byid[r['id']][r['condition']]=r
   n=len(byid);assert len(rs)==2*n and all(set(x)=={'blind','visible'} for x in byid.values())
   b=sum(x['blind']['correct'] for x in byid.values());v=sum(x['visible']['correct'] for x in byid.values())
   diffs=collections.defaultdict(list)
   for cid,x in byid.items():
    d=int(x['visible']['correct'])-int(x['blind']['correct']);diffs[x['blind']['tree']].append(d)
    paired.append({'model':name,'variant':vol,'id':cid,'tree':x['blind']['tree'],'gold':x['blind']['gold'],'blind_prediction':x['blind']['prediction'],'visible_prediction':x['visible']['prediction'],'blind_correct':x['blind']['correct'],'visible_correct':x['visible']['correct'],'visible_minus_blind':d})
   rng=random.Random(20261027);trees=sorted(diffs);boot=[]
   for _ in range(5000):
    ds=[d for t in rng.choices(trees,k=len(trees)) for d in diffs[t]];boot.append(100*sum(ds)/len(ds))
   boot.sort()
   summary.append({'model':name,'volume':vol,'n':n,'trees':len(trees),'blind_correct':b,'visible_correct':v,'blind_accuracy_pct':100*b/n,'visible_accuracy_pct':100*v/n,'visible_minus_blind_pp':100*(v-b)/n,'cluster_ci_low_pp':boot[124],'cluster_ci_high_pp':boot[4874],'blind_parse_failures':sum(x['blind']['parse_failure'] for x in byid.values()),'visible_parse_failures':sum(x['visible']['parse_failure'] for x in byid.values())})
 if summary:
  with (res/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 (res/'raw_predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in allrows))
 (res/'paired_results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in paired))
 complete=all(s=='completed' for s in states.values())
 sw=[r for r in summary if r['volume']=='SW'];drop=-statistics.mean(r['visible_minus_blind_pp'] for r in sw) if complete else None
 p1='STOP' if complete and drop<3 else 'PASS_SW_GATE' if complete else 'INCOMPLETE'
 status={'P1':p1,'models':states,'SW_model_macro_blind_minus_visible_pp':drop,'threshold_pp':3,'P1_scope':'two-model paired sibling-branch diagnostic; CS below requested size','CS_actual':audit['CS_actual'],'CS_requested':200,'P2':'NOT_RUN_P1_STOP' if p1=='STOP' else 'NOT_RUN','P3':'NOT_RUN','P4':'NOT_RUN','P5':'NOT_RUN','P6':'NOT_RUN','test_inference_count':0,'later_stages_automatically_launched':False}
 dump(res/'gate_status.json',status)
 lines=['# ASTER P1–P6 预实验执行记录（2026-10-27）','','基线 main：`781ee863784c60f1f6710ba5eede22bc8ee6b93c`。全部推理在 115 服务器执行；本轮测试集推理数为 0。','',f'P1 状态：**{p1}**。这是 P1 完成时的记录；后续状态见 [最新执行状态](RUN_STATUS.zh.md)。','', '| 模型 | 卷 | 例数 | 盲判 % | 明判 % | 明判−盲判（百分点） |','|---|---|---:|---:|---:|---:|']
 for r in summary:lines.append(f"| {r['model']} | {r['volume']} | {r['n']} | {r['blind_accuracy_pct']:.2f} | {r['visible_accuracy_pct']:.2f} | {r['visible_minus_blind_pp']:+.2f} |")
 lines+=['',f'预先固定的 P1 判断：两个模型 SW 的“盲判−明判”百分点差取等权平均，达到 3 才继续。本轮为 {drop if drop is not None else "待完成"}。']
 if p1=='STOP':lines+=['','**按 P1 止损，不进入 P2。当前开发集实现未达到现象门槛；不能把这次结果写成 TLBS 或 ASTER 有效的证据。**']
 lines+=['','数据及解释范围：','',f'- NC/IC/AC/SW 各 200 例，来自 4 个已有开发树条目；CS 仅 {audit["CS_actual"]} 例，来自其中 1 个树条目，未达到约 200 例的目标。没有复制病例补数。','- 沿用此前冻结的 10 个开发树／45 个测试树清单；本轮只读取独立 dev.jsonl 和划分元数据。没有读取测试病例或旧测试预测；额外只检查过 pancreatic_11 的开发来源记录。','- 原始全路径中，同条件可有多处置。本轮比较条件不同的同父兄弟分支，以固定随机顺序赋予 1/2 编号，避免把处置编号误当条件分支。属于两分支诊断，不能外推成完整树生成结果。','- 盲判移除终端处置、选项和病历列内嵌问题；保留条件和病史中的既往治疗事实。明判保留原问题、选项和两分支处置。','- NC/AC/IC/SW 沿用来源参考路径标签；未宣称独立人工核验。CS 用病历中明确的疗程结束时间和互补阈值验证翻转，排除原因见 exclusions.json。','- 不按模型表现筛选病例；所有解析失败计错。相同文本提示只计算一次，逐例保留复用键。','- 置信区间按树聚类配对 bootstrap；只有 4 个树，CS 仅 1 个树，区间信息有限，不作为正式测试推断。','','可复现产物：prepare.py、run_p1.py、launch_p1.sh、summarize.py、protocol.json；results/ 下为哈希、划分、逐例原始输出、汇总、门槛状态和运行日志；private/ 保存完整提示及病例，未提交。','','P6 已收到完整协议，但 P1 门槛之前不构造其数据、不运行生成实验；未进行人工核验或评分器验收。']
 (BASE/'REPORT.zh.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(status))
if __name__=='__main__':main()
