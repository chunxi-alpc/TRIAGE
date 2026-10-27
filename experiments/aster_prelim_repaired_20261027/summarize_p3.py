from pathlib import Path
import collections,csv,json,shutil
from run_p1 import dump
BASE=Path(__file__).resolve().parent

def main():
 res=BASE/'results/p3';res.mkdir(parents=True,exist_ok=True);summary=[];probes=[];states={};choices={};agreement={};raw=[]
 for name in ['qwen','llama']:
  src=BASE/'private'/('p3_'+name)
  for suffix in ['failed_alignment.log','.log']:
   fn=BASE/'private'/('p3_'+name+('_' if suffix[0]!='.' else '')+suffix)
   if fn.exists():shutil.copyfile(fn,res/fn.name)
  if not (src/'DONE.json').exists():states[name]='not_completed';continue
  assert json.loads((src/'manifest.json').read_text())['knockout_mask_verified'];states[name]='completed';rs=[json.loads(s) for s in (src/'predictions.jsonl').read_text().splitlines()];raw+=rs
  bymethod=collections.defaultdict(list)
  for r in rs:bymethod[r['method']].append(r)
  acc={k:100*sum(r['correct'] for r in v)/len(v) for k,v in bymethod.items()};gap=acc['blind']-acc['visible']
  for method,rows in bymethod.items():summary.append({'model':name,'method':method,'n':len(rows),'accuracy_pct':acc[method],'improvement_over_visible_pp':acc[method]-acc['visible'],'recovery_fraction':(acc[method]-acc['visible'])/gap if gap>0 else None})
  windows=[k for k in acc if k.startswith('knock_') and k!='knock_all'];best=max(windows,key=lambda k:(acc[k],-int(k.split('_')[1])))
  recovery=(acc[best]-acc['visible'])/gap if gap>0 else None
  choices[name]={'best_four_layer_window':best,'best_accuracy_pct':acc[best],'baseline_gap_pp':gap,'best_recovery_fraction':recovery,'selected_layers':best if recovery is not None and recovery>=.5 and acc[best]>acc['visible'] else 'all','selection_scope':'exploratory development selection; threshold heuristic is not proof of localization'}
  p1=[json.loads(s) for s in (BASE/'private'/name/'predictions.jsonl').read_text().splitlines()];old={(r['id'],r['condition']):r for r in p1 if r['variant']=='SW'}
  agreement[name]={c:sum(r['prediction']==old[r['id'],c]['prediction'] for r in bymethod[c])/len(bymethod[c]) for c in ['blind','visible']}
  probes+=json.loads((src/'probes.json').read_text());shutil.copyfile(src/'manifest.json',res/(name+'_manifest.json'))
 if summary:
  with (res/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 (res/'raw_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in raw));dump(res/'probes.json',probes)
 gate={'P3':'COMPLETED' if all(v=='completed' for v in states.values()) else 'INCOMPLETE','models':states,'window_choices':choices,'P1_free_generation_prediction_agreement':agreement,'P4':'NOT_RUN','P5':'NOT_RUN','P6':'NOT_RUN','test_inference_count':0,'prior_run_failures':'see ../aster_prelim_20261027; no old results reused'};dump(res/'gate_status.json',gate)
 lines=['# P3 分层干预实测','',f'状态：**{gate["P3"]}**。只用 169 个 SW 开发病例。','', '| 模型 | 方法 | 准确率 % | 相对明判变化（百分点） | 差距恢复比例 |','|---|---|---:|---:|---:|']
 for r in summary:
  ratio='未定义' if r['recovery_fraction'] is None else f"{r['recovery_fraction']:.3f}"
  lines.append(f"| {r['model']} | {r['method']} | {r['accuracy_pct']:.2f} | {r['improvement_over_visible_pp']:+.2f} | {ratio} |")
 lines+=['','窗口选择：`'+json.dumps(choices,ensure_ascii=False)+'`。','', '解释范围：','', '- 敲除仅切断最后一个判断位置对两条终端处置的注意力；选项与其他位置仍可传递信息。它检验直接读取路径，不声称消除所有间接路径。','- K/V 从中性代号运行修补到熟悉处置运行；修补诱饵段及必要的分词边界词元，段外 token ID 完全一致。修补发生在 k_proj/v_proj 输出，Qwen 的 K 归一化与 RoPE 继续正常执行。','- P3 使用 1/2 的下一词元概率读出，另行报告与 P1 自由生成的一致率。恢复比例用 P3 自己的盲判／明判差距作分母，不混用分母。','- 线性探针逐层在 3 个开发树训练、剩余开发树评估；全程未读取测试树。探针可解码不等于因果证据。','- 旧版的词元对齐和布尔掩码失败记录保留于原实验目录；本轮从头运行，不复用旧结果。','- 不能找到足够恢复的窗口时，按用户规则改用全层；不把预期的第 13–22 层或 K/V 恢复比例填入结果。']
 (BASE/'P3_REPORT.zh.md').write_text('\n'.join(lines)+'\n');print(json.dumps(gate))
if __name__=='__main__':main()
