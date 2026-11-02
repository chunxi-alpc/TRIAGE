"""One completion wait and aggregate publication; no resource/status polling."""
import argparse,json,os,select,subprocess,traceback
from pathlib import Path
from .data import BASE,write
from .publish import publish

def main():
 p=argparse.ArgumentParser();p.add_argument('--pid',required=True,type=int);a=p.parse_args();run=BASE
 status=run/'completion_publication.json'
 write(status,{'status':'waiting_for_launcher_exit','launcher_pid':a.pid,'polls_for_progress':False})
 try:
  try:fd=os.pidfd_open(a.pid)
  except ProcessLookupError:fd=None
  if fd is not None:
   event=select.poll();event.register(fd,select.POLLIN);event.poll();os.close(fd)
  launch=json.loads((run/'official_test_launch.json').read_text())
  if launch['status']!='complete':
   write(status,{'status':'not_published_as_complete','launcher_state':launch['status'],'reason':'interrupted or incomplete run; existing units preserved'});return
  out=Path('experiments/aster_manuscript_clinical_20261030/public/clinical01');publish(run,out)
  rows=json.loads((run/'summary_official_test/comparisons.json').read_text());sem=json.loads((run/'summary_official_test/official_Qwen_scores.json').read_text())
  lines=['# ASTER 官方测试完成汇总','',f"已完成 {launch['cases']} 例、{launch['expected_units']} 个预测单元；技术失败及评分失败保留在分母中。",'',
   '| 方法 | 终点正确 | 总分母 | 路径F1 | 技术失败 |','|---|---:|---:|---:|---:|']
  for r in rows:lines.append(f"| {r['method']} | {r.get('leaf_correct','不适用')} | {r['denominator']} | {r.get('path_rule_f1','不适用')} | {r['technical_failures']} |")
  lines+=['','Qwen 适配官方评分（不是 GPT 主表；PCC 为官方DTW成本，单独保留，不混同结论准确率）：','',
   '| 方法 | Qwen结论分（失败计0） | 成功评分 | 评分失败 |','|---|---:|---:|---:|']
  for r in sem:lines.append(f"| {r['method']} | {r['Qwen_judged_official_accuracy_failures_as_zero']} | {r['scoring_successes']} | {r['scoring_failures']} |")
  lines+=['','指南簇区间、覆盖率、纠正/退化、成本和配对算子差值见公开 JSON。完整方法、临床年龄标注范围、模型/代码修订、原失败与修复证据见[临床执行报告](clinical-results-20261030.zh.md)。',
   '','这只表示 MEGA-CDP 官方批次完成；MedGUIDE 端到端、医生裁定和更丰富临床混杂控制尚未完成，不将整个研究任务标记为完成。','']
  doc=Path('docs/ASTER/clinical-official-results-20261030.zh.md');doc.write_text('\n'.join(lines))
  branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip()
  if branch!='codex/aster-manuscript-implementation-20261030':raise RuntimeError('branch changed; aggregate files saved, publication push skipped')
  subprocess.run(['git','add',str(out),str(doc)],check=True)
  changed=subprocess.run(['git','diff','--cached','--quiet','--',str(out),str(doc)]).returncode
  if changed:subprocess.run(['git','commit','--only','-m','results(aster): preserve completed frozen official clinical evaluation',str(out),str(doc)],check=True)
  subprocess.run(['git','push','origin',branch],check=True,timeout=180)
  write(status,{'status':'aggregate_results_pushed','branch':branch,'clinical_raw_data_published':False,'full_research_complete':False})
 except Exception as e:
  write(status,{'status':'completion_publication_failed','error_type':type(e).__name__,'reason':str(e),'clinical_outputs_preserved':True});traceback.print_exc();raise
if __name__=='__main__':main()
