"""Write a truthful stage snapshot and optionally publish only this experiment directory."""
from pathlib import Path
import argparse,datetime,hashlib,json,subprocess
BASE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--publish',action='store_true');a=p.parse_args()
 states={};records={}
 for stage,sub in [('P1',''),('P2','p2'),('P3','p3'),('P4','p4'),('P5','p5')]:
  path=BASE/'results'/sub/'gate_status.json'
  if path.exists():
   r=json.loads(path.read_text());records[stage]=r;states[stage]=r[stage]
  else:
   names=[BASE/'private'/(stage.lower()+'_'+n)/'manifest.json' for n in ['qwen','llama']]
   states[stage]='RUNNING' if any(x.exists() for x in names) else 'NOT_RUN'
 states['P6']='NOT_RUN'
 first_stop=next((s for s in ['P1','P2','P3','P4','P5'] if any(k in states[s] for k in ['STOP','FAIL','INCOMPLETE'])),None)
 for s in ['P2','P3','P4','P5','P6']:
  if first_stop and int(s[1:])>int(first_stop[1:]) and states[s]=='NOT_RUN':states[s]='NOT_RUN_PREDECESSOR_STOP'
 diagnostics=[]
 for s in records.get('P4',{}).get('summary',[]):
  if not s['passes']['U4']:
   diagnostics.append({'model':s['model'],'failed':'U4','ordinary_prune_gap_pp':s['U4_gap_pp'],'position_preserved_prediction_agreement':s['U4_position_preserved_prediction_agreement'],'inference':'If position-preserved comparison matches while ordinary pruning differs, encoding-position changes are a supported explanation; this does not waive the user text-pruning threshold.'})
 result={'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'base_commit':'781ee863784c60f1f6710ba5eede22bc8ee6b93c','branch':'exp/aster-p1-p6-20261027','stages':states,'first_stop':first_stop,'test_inference_count':0,'CS_actual':17,'CS_requested':200,'P4_failure_diagnostics':diagnostics,'P6_manual_verified_set_available':False}
 (BASE/'results/overall_status.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 lines=['# ASTER P1–P6 实际执行状态','','基于 main `781ee863784c60f1f6710ba5eede22bc8ee6b93c`，在 115 服务器 GPU 0、1 执行。仅使用已冻结开发树。本轮测试集推理为 **0**。','', '| 阶段 | 实际状态 | 记录 |','|---|---|---|']
 links={'P1':'REPORT.zh.md','P2':'P2_REPORT.zh.md','P3':'P3_REPORT.zh.md','P4':'P4_REPORT.zh.md','P5':'P5_REPORT.zh.md','P6':'results/p6_readiness.json'}
 for s,status in states.items():lines.append(f"| {s} | {status} | [{s} 记录]({links[s]}) |" if (BASE/links[s]).exists() else f'| {s} | {status} | 尚无完成报告 |')
 lines+=['','已确认的观测：','','- P1 SW：Qwen 盲判 80.5%、明判 69.5%，差距 11.0 个百分点；Llama 66.0% 与 50.5%，差距 15.5。事前固定的模型等权平均差距 13.25，超过 3 的继续门槛。','- P2 中性代号：Qwen 明判比盲判高 13.0 个百分点，Llama 低 1.5；未触发本轮中性代号止损。去选项与顺序对效应影响明显，闭卷置信度没有稳定单调关系，不能宣称全部替代解释已被排除。','- P1/P2 熟悉标准条件同提示的 400 条对应记录／模型，预测 100% 一致。']
 if 'P3' in records:
  for name,z in records['P3'].get('window_choices',{}).items():lines.append(f"- P3 {name}：最佳四层窗口 {z['best_four_layer_window']}，恢复比例 {z['best_recovery_fraction']}；后续层选择 {z['selected_layers']}。这是开发选择，不是独立测试。")
 if 'P4' in records:
  for z in records['P4'].get('summary',[]):lines.append(f"- P4 {z['model']}：U1 最大键差 {z['U1_max_abs_difference']:.9g}，U2 一致率 {100*z['U2_all_four_argmax_agreement']:.2f}%，U3 差 {z['U3_gap_pp']:.2f} 分，U4 差 {z['U4_gap_pp']:.2f} 分；通过情况 {z['passes']}。")
 if first_stop:lines+=['',f'**在 {first_stop} 停止，后续关口未运行。未用未完成项目或预期值补齐结果。**']
 elif any(v=='RUNNING' for v in states.values()):lines+=['','仍有实验运行中，不能把计划启动、脚本已写或进程稳定当作实验完成。后续必须读取对应完成标记与门槛结果。']
 lines+=['','范围与未完成事项：','','- 四卷各 200 例，来自 4 个已划入开发集的树条目；CS 仅 17 例，未达到约 200 例目标。当前为同父兄弟两分支诊断，不等于完整树或路径生成验证。','- 处置来源和“陌生处置”只是本轮操作性定义，未证明模型参数记忆中的熟悉度。所有模型差异与负结果保留。','- P3 首次启动的词元对齐断言失败已保留；修正后重新开始，失败启动未计为有效实验。','- P6 找到的既有材料仅 16 例，相关报告明确注明 AI 审阅、尚无临床专家人工审核。不能算作当前要求的 ≥120 配对、≥20 指南且 ≥40 多节点／跨片段病例。','- 完整病例、提示与内部激活仅保存在服务器 private/；公开目录保存可复现脚本、协议、模型／数据指纹、逐例输出、汇总、运行日志和审计。','', '服务器工作目录：`/home/gcx/TRIAGE-worktrees/aster-prelim-20261027/experiments/aster_prelim_20261027`。','',f'状态更新时间（UTC）：{result["updated_at"]}。']
 (BASE/'RUN_STATUS.zh.md').write_text('\n'.join(lines)+'\n')
 # File inventory excludes itself and source text. Hashes bind published artifacts.
 inventory={str(f.relative_to(BASE)):hashlib.sha256(f.read_bytes()).hexdigest() for f in BASE.rglob('*') if f.is_file() and not any(x in f.relative_to(BASE).parts for x in ['private','__pycache__']) and f.name!='artifact_sha256.json'}
 (BASE/'results/artifact_sha256.json').write_text(json.dumps(inventory,indent=2)+'\n')
 if a.publish:
  repo=BASE.parents[1];branch=subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip();assert branch=='exp/aster-p1-p6-20261027'
  subprocess.run(['git','add','experiments/aster_prelim_20261027'],cwd=repo,check=True)
  if subprocess.run(['git','diff','--cached','--quiet'],cwd=repo).returncode:
   subprocess.run(['git','commit','-m','Record actual ASTER preliminary experiment outcomes and stopping status'],cwd=repo,check=True)
  subprocess.run(['git','push','origin',branch],cwd=repo,check=True)
 print(json.dumps(result))
if __name__=='__main__':main()
