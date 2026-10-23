"""Collect only safe, explicit experiment artifacts; status never implies unrun results."""
import datetime,hashlib,json,shutil,subprocess
from pathlib import Path
from zoneinfo import ZoneInfo
exp=Path(__file__).resolve().parent;private=Path('/home/gcx/aster_validation_20261022/private');out=exp/'results';out.mkdir(exist_ok=True)
for name in ('dataset_manifest.json','model_fingerprints.json','positive_preparation.json','condition_labels.json'):
 p=private/name
 if p.exists():shutil.copyfile(p,out/name)
status={'updated_at':datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'models':{},'all_experiments_complete':False,'protocol':{'source_cases':7747,'paired_cases':3645,'heldout_cases':3605,'main_records_per_model':189540,'unsigned_records_per_model':21870,'final_records_per_model':211410},'blocked_or_unrun':{'model_predicted_applicability':'not run; noise study uses controlled random corruption','context_dpo_canoe':'not run; trained checkpoints/configuration absent','model_selected_key_sentences':'not run; unknown-applicability comparison covers baseline/whole/edited/CAD'},'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in exp.iterdir() if p.suffix in ('.py','.sh','.json')},'seka_source_revision':subprocess.check_output(['git','-C','/home/gcx/TRIAGE/third_party/SEKA','rev-parse','HEAD'],text=True).strip()}
for name in ('llama','qwen'):
 dst=out/name;dst.mkdir(exist_ok=True)
 for suffix in ('_diag.manifest.json','_full.manifest.json','_unsigned.manifest.json','_complete.manifest.json'):
  p=private/(name+suffix)
  if p.exists():shutil.copyfile(p,dst/p.name)
 state={k:(private/(name+k)).read_text().strip() if (private/(name+k)).exists() else 'not_started' for k in ('.status','.finish_status')}
 summary=dst/'full/summary.json';state['main_results_exist']=summary.exists()
 final_manifest=private/(name+'_complete.manifest.json')
 required_metrics=[private/'positive'/name/arm/'efficacy_metrics.json' for arm in ('baseline','seka')]
 state['required_artifacts_present']=summary.exists() and final_manifest.exists() and all(p.exists() for p in required_metrics)
 state['all_model_stages_complete']=state['.finish_status']=='complete' and state['required_artifacts_present'] and json.loads(final_manifest.read_text()).get('complete',False)
 pos=private/'positive'/name
 for arm in ('baseline','seka'):
  p=pos/arm/'efficacy_metrics.json'
  if p.exists():
   (dst/'positive').mkdir(exist_ok=True);shutil.copyfile(p,dst/'positive'/(arm+'_efficacy_metrics.json'))
 # Numeric per-example official benchmark results, excluding prompt/generation strings.
 for arm in ('baseline','seka'):
  p=pos/arm/'efficacy.json'
  if p.exists():
   d=json.loads(p.read_text());samples=d.get('samples',[]) if isinstance(d,dict) else d if isinstance(d,list) else []
   safe=[{k:r[k] for k in ('id','target_score','comparator_score') if k in r} for r in samples if isinstance(r,dict)]
   (dst/'positive').mkdir(exist_ok=True);(dst/'positive'/(arm+'_scores.json')).write_text(json.dumps(safe))
 status['models'][name]=state
status['all_model_runs_complete']=all(v['all_model_stages_complete'] for v in status['models'].values())
status['all_experiments_complete']=status['all_model_runs_complete'] and not status['blocked_or_unrun']
(out/'STATUS.json').write_text(json.dumps(status,indent=2))
lines=['# 本轮实际状态','',f"更新时间：{status['updated_at']}",'','全量范围：7,747 条来源病例中可构造的 3,645 例；40 例开发、3,605 例正式评价。','']
for name,state in status['models'].items():
 lines.extend([f"- {name}: 主任务 {state['.status']}；后续阶段 {state['.finish_status']}。"])
 p=out/name/'diagnostic/summary.json'
 if p.exists():
  d=json.loads(p.read_text());lines.append(f"  开发集增益={d['selected_gain']}；零增益分数最大误差={d['zero_bias_max_error']}。")
 p=out/name/'full/REPORT.txt'
 if p.exists():lines.extend(['','```text',p.read_text().strip(),'```'])
lines.extend(['','原始病例、指南、生成文字和详细日志保留在服务器 private 目录。安全数值输出、配置、统计和来源指纹保存在本目录。','SEKA 正对照使用原始 CounterFact 经 PASTA 原函数整理，独立构建模型专用 last10 投影；模型版本与原论文不同，不能称为论文表格的逐项复现。','专家条件标注、预测适用关系、Context-DPO/CANOE 和模型自选关键句的未运行状态见 STATUS.json；不会以随机噪声实验或直接分数偏置替代这些结果。'])
(out/'REPORT.zh.md').write_text('\n'.join(lines)+'\n')
