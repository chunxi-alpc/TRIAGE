"""Generate server-side reports from observed records; pending stages remain explicit."""
import json
from collections import defaultdict
import numpy as np
from .common import *
from .statistics import interval,pair_metrics
def macro_difference(pairs,a,b):
 sources=sorted({p["source_id"] for p in pairs});loc={s:i for i,s in enumerate(sources)}
 kinds=["A","P","C","F"];num=np.zeros((len(sources),4));den=np.zeros_like(num)
 def ok(p,ss):
  x,y=ss.get(p["left"]),ss.get(p["right"])
  return int(x is not None and y is not None and (x>0)==bool(p["left_label"]) and (y>0)==bool(p["right_label"]))
 for p in pairs:
  if p["kind"] not in kinds:continue
  i,j=loc[p["source_id"]],kinds.index(p["kind"]);num[i,j]+=ok(p,a)-ok(p,b);den[i,j]+=1
 if not len(sources) or (den.sum(0)==0).any():return dict(difference=None,ci95=None,reason="missing specification class")
 point=float(np.mean(num.sum(0)/den.sum(0)));rng=np.random.default_rng(SEED);vals=[]
 for _ in range(2000):
  idx=rng.integers(len(sources),size=len(sources));d=den[idx].sum(0)
  if (d==0).any():continue
  vals.append(float(np.mean(num[idx].sum(0)/d)))
 return dict(difference=point,ci95=np.quantile(vals,[.025,.975]).tolist() if vals else None,valid_draws=len(vals),requested_draws=2000,sources=len(sources))
def model_report(model):
 root=ROOT/model;frozen=read(root/"frozen.json")
 if not frozen:return dict(state="pending",frozen=False)
 pairs=[p for p in rows(ROOT/"data/pairs.jsonl") if p["split"]=="test"]
 views=[v for v in rows(ROOT/"data/views.jsonl") if v["split"]=="test"]
 scores={};paths={}
 for name in frozen["readouts"]:
  ss=rows(root/"readout_scores"/(name+"_test.jsonl"))
  if ss:scores[name]={r["view_id"]:r["ell"] for r in ss}
  rs=rows(root/"path_results"/(name+"_test.jsonl"))
  if rs:paths[name]=rs
 for name in frozen["baseline_families"].values():
  ss=rows(root/"baselines"/(name+"_test.jsonl"))
  if ss:scores[name]={r["view_id"]:r["ell"] for r in ss}
  rs=rows(root/"baselines"/(name+"_paths_test.jsonl"))
  if rs:paths[name]=rs
 comp=[r for p in (root/"comparators").glob("test_*.jsonl") for r in rows(p)]
 for r in comp:
  for name,s in r["scores"].items():scores.setdefault(name,{})[r["view_id"]]=s
 for r in [r for p in (root/"comparators").glob("paths_test_*.jsonl") for r in rows(p)]:
  for name,value in r["methods"].items():paths.setdefault(name,[]).append(dict(case_id=r["case_id"],source_id=r["source_id"],**value))
 metrics={name:pair_metrics(views,pairs,ss,ci=True) for name,ss in scores.items()}
 paths={name:list({r["case_id"]:r for r in rs}.values()) for name,rs in paths.items()}
 full={r["case_id"]:r for r in paths.get("full",[])};pstats={}
 for name,rs in paths.items():
  vals=[int(r["correct"])-int(full[r["case_id"]]["correct"]) for r in rs if r["case_id"] in full]
  sources=[r["source_id"] for r in rs if r["case_id"] in full];diff=interval(vals,sources)
  pstats[name]=dict(correct=sum(r["correct"] for r in rs),n=len(rs),accuracy=float(np.mean([r["correct"] for r in rs])),
   difference_from_full=diff,noninferiority_95=bool(diff["ci95"] and diff["ci95"][0]>=-.005),
   corrected=sum(v==1 for v in vals),new_errors=sum(v==-1 for v in vals),
   first_errors=dict(__import__("collections").Counter(r["first_error"] for r in rs if r.get("first_error"))),
   total_node_forwards=sum(r.get("node_forward_count",0) for r in rs))
 main=frozen["development"]["aster_cr"];comparisons={}
 for name in {frozen["development"]["calibration"],frozen["development"]["ordinary"],frozen["strongest_baseline"],"full"}:
  if name and main in scores and name in scores:comparisons[name]=macro_difference(pairs,scores[main],scores[name])
 expected=set(frozen["readouts"])|set(frozen["baseline_families"].values())
 complete=expected<=set(scores) and expected<=set(paths)
 required_paths=read(ROOT/"data/frozen.json")["original_paths"]["test"]
 complete=complete and all(len(paths.get(name,[]))==required_paths for name in expected)
 p0=rows(root/"diagnostics/test/p0_nodes.jsonl")
 e2=[r for r in p0 if r["category"]=="E2_candidate" and r["original_error"]]
 e2_result={}
 for name in {main,frozen["development"]["calibration"],frozen["development"]["ordinary"]}:
  ss=scores.get(name,{})
  good=sum(r["group_id"]+":full" in ss and (ss[r["group_id"]+":full"]>0)==bool(r["gold"]) for r in e2)
  e2_result[name]=dict(candidate_original_errors=len(e2),corrected_by_readout=good)
 out=dict(E2_readout_check=e2_result,H3b=read(root/"diagnostics/test/p0.json"),state="complete" if complete else "in_progress",frozen=True,main_method=main,pairs=metrics,paths=pstats,
   main_macro_differences=comparisons,selection=frozen,h2=read(root/"h2.json"),h3=read(root/"h3.json"))
 write(root/"summary.json",out);return out
def report(final=False):
 data=read(ROOT/"data/frozen.json");variants=read(ROOT/"data/variants.json")
 models={model:model_report(model) for model in ("qwen","medgemma")}
 complete=all(v["state"]=="complete" for v in models.values())
 if final and not complete:raise RuntimeError("Final report requested before all frozen evaluations finished")
 stages={p.stem:read(p) for p in (ROOT/"status").glob("*.json") if not p.stem.startswith("annotate_")}
 write(ROOT/"summary.json",dict(state="complete" if final and complete else "in_progress",data=data,variants=variants,models=models,stages=stages))
 lines_out=["# ASTER-CR 新实验报告","",
  "状态："+("全部已冻结评价完成。" if final and complete else "运行中；未完成阶段不作为结果。"),"",
  "执行位置："+str(ROOT),"",
  "GPU："+"、".join(map(str,GPUS))+"。使用 repair-v2 数据；历史结果保留。","",
  "## 数据","",
  "来源数："+json.dumps(data.get("sources",{}),ensure_ascii=False)+"；独立节点数："+json.dumps(data.get("nodes",{}),ensure_ascii=False)+"。",
  "原始病例路径："+json.dumps(data.get("original_paths",{}),ensure_ascii=False)+"。",
  "MEGA 原始病例是基准提供的病例，不能解释为医院病例的自然患病或错误比例。历史使用记录见 data/history_usage.json。","",
  "各类配对数、独立节点数、来源数分别见 data/variants.json；P/C/F 独立节点仍少，不能用配对总数推断证据充足。",
  "规则抽取与原标签不一致的记录归入抽取复核，不自动重标。新 test 评分前须完成各模型的 frozen.json。","",
  "## 阶段",""]
 for name in ("prepare","preparation","qwen","medgemma","pipeline"):
  st=read(ROOT/"status"/(name+".json"));lines_out.append("- "+name+"："+st.get("state","pending")+(("，"+st["phase"]) if st.get("phase") else ""))
 for name,result in models.items():
  lines_out+=["","## "+name,"","状态："+result["state"]+"。"]
  if result.get("main_method"):lines_out+=["主配置："+result["main_method"]+"。","",
    "| 方法 | 四类 PA 宏平均 | 路径正确数 / 总数 | 相对 Full 非劣性 |","|---|---:|---:|---|"]
  for method,metrics in result.get("pairs",{}).items():
   macro=metrics.get("macro");pr=result.get("paths",{}).get(method,{})
   lines_out.append("| "+method+" | "+(f"{100*macro:.2f}%" if macro is not None else "不可计算")+" | "+
    (str(pr["correct"])+" / "+str(pr["n"]) if pr else "待完成")+" | "+
    ("满足" if pr.get("noninferiority_95") else "未建立")+" |")
  for kind in ("A","P","C","F","F0"):
   m=result.get("pairs",{}).get(result.get("main_method"),{}).get(kind)
   if m:lines_out.append(f"{kind}：{m['denominator']} 对，{m['nodes']} 个独立节点，{m['sources']} 个来源；95% 区间 {m.get('interval',{}).get('ci95')}。")
  if result.get("h2"):lines_out+=["","H2*："+json.dumps(result["h2"],ensure_ascii=False)]
  if result.get("h3"):lines_out+=["","H3a 错误子集证据："+result["h3"].get("error_subset_evidence","pending")+"。"]
  for baseline,comp in result.get("main_macro_differences",{}).items():
   lines_out+=["","主方法相对 "+baseline+"："+json.dumps(comp,ensure_ascii=False)]
 lines_out+=["","## 方法与成本说明","",
  "ASTER-CR 的信号基、干扰基、中心化与标准化仅用 fit；干扰投影先移除信号空间。一次未编辑前向之后修正二元分数。",
  "LoReFT 使用官方 pyreft 组件；LoFiT 保留官方两阶段流程，通过注意力输出钩子适配 Qwen3/Gemma3；LoRA 使用 PEFT。",
  "四个训练对照均使用相同 fit 输入和三个学习率，每阶段一轮；LoFiT 额外选头阶段单独计成本。MedGemma 仅采用 Qwen 选定的学习率和方法。",
  "LoReFT、LoFiT、LoRA 采用最接近的可实现参数规模，实际参数和全部阶段耗时见 baselines/*_training.json。",
  "普通读出与相应 ASTER-CR 的可训练系数数量一致；bias-only 保留单参数。",
  "ITI、SEA 的二元任务适配以 fit 中正确/错误答案补全提取特征；旧闭式参数不搜索，LEACE 与闭式方法仅在新 fit 必要重拟合。",
  "CAD、参考融合、参考版旧 M3 计入额外参考前向；对称输出平均计两次前向。推理耗时包含实际执行，诊断与缓存准备成本另报。",
  "所有成本记录保存在各模型 cost/、baselines/*_training.json 和 comparators/*_metrics.json。特征预缓存会计算额外节点，不能将其总耗时等同于单病例自主执行耗时。",
  "配对统计按指南来源进行 2,000 次 bootstrap；方法差值直接重采样。非劣性仅在路径差值 95% 区间下界不低于 -0.5 个百分点时成立。","",
  "## 结果解释","",
  "未超过校准和普通读出时，收束为规约诊断与负结果；未超过 LoReFT/LoFiT 时，不主张完整方法优于这些训练方法。最终解释以实际区间和消融为准。","",
  "官方实现：[LoReFT](https://github.com/stanfordnlp/pyreft)、[LoFiT](https://github.com/fc2869/lo-fit)、[ITI](https://github.com/likenneth/honest_llama)、[SEA](https://github.com/yfqiu-nlp/sea-llm)。"]
 (ROOT/"REPORT.zh.md").write_text("\n".join(lines_out)+"\n")
 return models
if __name__=="__main__":report()
