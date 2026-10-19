"""Summarize all real stage-1/2 saved predictions, without rerunning inference."""
import csv,json
from pathlib import Path
from engine import read_cases
from run_attention import summarize
HERE=Path(__file__).resolve().parent
def main():
 results=HERE/"results";metrics={};flat=[]
 for directory in sorted(results.iterdir()):
  if not directory.is_dir() or not directory.name.startswith(("group1_","group2_")):continue
  prov=json.loads((directory/"provenance.json").read_text())
  split=prov["split"];cases=read_cases(split)
  basepath=results/("group1_gpu0" if split=="dev" else "group1_confirmation_main")/"baseline.jsonl"
  baseline=list(map(json.loads,basepath.read_text().splitlines()))
  metrics[directory.name]={}
  for config in prov["configs"]:
   name=config["name"];path=directory/(name+".jsonl")
   if not path.exists():continue
   rows=list(map(json.loads,path.read_text().splitlines()))
   summary=summarize(cases,rows,baseline)
   if config.get("with_structure"):
    structurepath=results/"group1_confirmation_main/structure_only.jsonl"
    if structurepath.exists():
     s=summarize(cases,rows,list(map(json.loads,structurepath.read_text().splitlines())))
     summary["versus_structure_only"]={k:s[k] for k in ["corrected","damaged","net_corrected","prediction_changes","mean_margin_delta"]}
   metrics[directory.name][name]=summary
   flat.append(dict(directory=directory.name,split=split,arm=name,**{k:v for k,v in summary.items() if not isinstance(v,dict)}))
 (results/"group12_metrics.json").write_text(json.dumps(metrics,indent=2))
 keys=list(dict.fromkeys(k for row in flat for k in row))
 with (results/"group12_metrics.csv").open("w") as f:
  writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(flat)
 main=metrics["group1_confirmation_main"];rep=metrics["group1_confirmation_representative"];g2=metrics["group2_confirmation_effect"];ctrl=metrics["group2_controls_confirmation"]
 lines=["# 第一、二组真实 GPU 实验结果","",
 "沿用 PR #9 的 96 例 dev 和 192 例 confirmation。这批数据已使用过，结果仅作探索性机制实验。Qwen3-4B-Base 在 GPU 0–3 实际推理；所有规则和动作保留在原始提示中，无结构提示臂只将状态交给注意力程序。","",
 "第一组完成 24 个层段/强度配置，每臂 96 例。dev 基线 35/96，24 个配置主要配对均 0/12。按主要配对、病例正确数、平均 margin 顺序选定第18–23层 λ=1 和代表第12–17层 λ=0.5。第二组完成 A–E × 最后1/16行十臂，每臂96例。E最后16行39/96，E最后1行38/96，主要配对均0/12。配置在 confirmation 前固定。","",
 "| confirmation 条件 | 正确病例/192 | 主要配对/24 | 相对原始基线纠错/破坏 |","|---|---:|---:|---:|"]
 for group in [main,rep,g2,ctrl]:
  for name,s in group.items():lines.append(f"| {name} | {s['correct']} | {s['primary_pair_correct']} | {s['corrected']}/{s['damaged']} |")
 lines+=["","数值检查仅一次：真实动作词元定位正确，分数偏置均值实际为 +1 和 −1；相应注意力质量分别从 0.001838 增至 0.005040、从 0.007348 降至 0.002755。零强度选中行重算与原 SDPA 的最大候选 logit 差为 0.125（BF16），预测一致。弱小净收益须结合该数值差异谨慎解释。","",
 "每例记录实际分数变化、attention softmax 前后质量、完整提示、目标字符/词元位置、答案候选 logits 和正确答案 margin。所有 U、无操作、技术失败均保留分母；没有置信度或逐例方向门控。未目标行沿用原 SDPA，末16行仍受因果可见性限制。","",
 "最佳 E 的确认集成绩为 77/192，较原始基线多对 6 例，主要配对仍为 0/24。反号为 71/192；规则状态错配也达到 77/192，且 dev 错配40/96高于正确状态39/96。错配控制保持相同患者证据增强，只循环错配规则状态，因此当前小幅改善没有显示出正确规则状态的特异作用。E 加结构提示为158/192、12/24，低于仅结构提示160/192、15/24。注意力和决策确实改变，但尚未打通所要求的完整因果链条。","",
 "复现：source /home/gcx/gcx261014/bin/activate 后，运行 python experiments/causal_attention_20261019/reproduce_attention.py --output /home/gcx/causal-attention-reproduction-NEW --gpus 0 1 2 3。输出目录必须尚不存在。固定配置见 configs/；全部指标见 results/group12_metrics.json 和 .csv；逐例结果及命令、环境、显存见各阶段目录。"]
 (HERE/"STAGE12_REPORT.zh.md").write_text("\n".join(lines)+"\n")
 print(json.dumps({"arms":len(flat),"prediction_rows":sum(s["cases"] for group in metrics.values() for s in group.values()),"report":str(HERE/"STAGE12_REPORT.zh.md")}))
if __name__=="__main__":main()

