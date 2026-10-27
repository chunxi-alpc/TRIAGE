from pathlib import Path
import json,shutil
from run_p1 import dump
BASE=Path(__file__).resolve().parent

def main():
 res=BASE/'results/p4';res.mkdir(parents=True,exist_ok=True);summ=[];states={}
 for name in ['qwen','llama']:
  src=BASE/'private'/('p4_'+name)
  log=BASE/'private'/('p4_'+name+'.log')
  if log.exists():shutil.copyfile(log,res/log.name)
  if not (src/'DONE.json').exists():states[name]='not_completed';continue
  states[name]='completed';s=json.loads((src/'summary.json').read_text());summ.append(s)
  for fn in ['summary.json','records.jsonl','U1_keys.json','manifest.json']:shutil.copyfile(src/fn,res/(name+'_'+fn))
 complete=all(v=='completed' for v in states.values());passed=complete and all(s['all_pass'] for s in summ)
 gate={'P4':'PASS' if passed else 'FAIL_STOP' if complete else 'INCOMPLETE','models':states,'summary':summ,'P5':'NOT_RUN','P6':'NOT_RUN','test_inference_count':0};dump(res/'gate_status.json',gate)
 lines=['# P4 实现与泄漏检查','',f'状态：**{gate["P4"]}**。P5/P6 尚未运行。','','| 模型 | U1 最大键差 | U2 一致率 | U3 准确率差（百分点） | U4 准确率差（百分点） | 全部通过 |','|---|---:|---:|---:|---:|---|']
 for s in summ:lines.append(f"| {s['model']} | {s['U1_max_abs_difference']:.9g} | {100*s['U2_all_four_argmax_agreement']:.2f}% | {s['U3_gap_pp']:.2f} | {s['U4_gap_pp']:.2f} | {s['all_pass']} |")
 lines+=['','测试定义（本次使用 FP32、关闭 TF32；此前 BF16 结果另行归档）：','','- U1 在 4 个开发树各 5 例中只改病历年龄事实，逐层比较全部指南词元的实际 RoPE 后 K；要求最大绝对差严格为 0。未把它扩写成完整缓存性能验证。','- U2 对 200 个底稿四卷逐一改变处置，按用户最新指示取消 99% 一致率门槛；保留实际一致率，验证可见词元相同、固定长度隐藏处置替换不改变分支分数，并保存完整概率。','- 条件与处置按片段分别分词，盲判使用非处置累计位置，避免不可见处置长度改变可见位置。非处置在全部盲判层不可读处置，处置片段彼此隔离。','- U3 对相同片段序列分别使用遮罩隐藏和直接移除处置词元，比较 200 例准确率。它验证本轮接口，不等于 P1 原提示格式的同等性能。','- U4 在 200 个 SW 病例给定金标分支，比较同步压缩位置编号的硬门控与普通文本裁剪，预测处置选项；选项保留，未使用额外选项—规则映射来门控。','- 另记录不压缩位置的硬门控与保留原位置的裁剪结果，区分位置变化与掩码错误。硬门控位置压缩是新增的实现步骤，不可据此声称只加对数偏置就与普通文本裁剪严格等价。它只是诊断对照，不能替代 U4 主门槛。','- 当前只验证分支／选项读出，未实现自由文本生成的多步缓存与阶段切换；P6 仍未验证。']
 for s in summ:
  lines+=['',f"{s['model']}：强门控准确率 {s['U4_hard_gate_accuracy_pct']:.2f}%，普通裁剪 {s['U4_plain_prune_accuracy_pct']:.2f}%；普通裁剪预测一致率 {100*s['U4_prediction_agreement']:.2f}%，保留位置的裁剪预测一致率 {100*s['U4_position_preserved_prediction_agreement']:.2f}%，后者最大概率差 {s['U4_position_preserved_max_probability_difference']:.9g}。"]
 if complete and not passed:lines+=['','**任何一项未通过即停止，不进入 P5/P6。失败不能用更换参考运行、平均掉模型差异或删除病例来消除。需据逐例结果定位问题后再重新验收。**']
 (BASE/'P4_REPORT.zh.md').write_text('\n'.join(lines)+'\n');print(json.dumps(gate))
if __name__=='__main__':main()
