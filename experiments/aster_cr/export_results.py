"""Export observed results and analysis without running models or changing selections."""
import argparse,gzip,json,shutil,time
from pathlib import Path
from .common import rows,read,write
from .statistics import pair_metrics,interval
from .report import macro_difference

RUN_NAME="aster_cr_repair_v2_20261108"
def copy(source,target):
 if source.exists():
  target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
def compressed(source,target,omit=()):
 if not source.exists():return
 target.parent.mkdir(parents=True,exist_ok=True)
 with target.open("wb") as raw:
  with gzip.GzipFile(fileobj=raw,mode="wb",filename="",mtime=0) as out:
   for r in rows(source):out.write((json.dumps({k:v for k,v in r.items() if k not in omit},ensure_ascii=False,separators=(",",":"))+"\n").encode())
def pct(x):return "—" if x is None else f"{100*x:.2f}%"
def ci(value):return "—" if not value else f"[{100*value[0]:.2f}, {100*value[1]:.2f}]"
def export(root,destination):
 root=Path(root);destination=Path(destination);repair=root.parent/"aster_cr_repair_20261108";ordinary=root.parent/"ordinary_readout_20261108"
 out=destination/"results"/RUN_NAME;report=destination/"reports"/RUN_NAME
 out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
 now=time.strftime("%Y-%m-%d %H:%M:%S %Z");pipeline=read(root/"status/pipeline.json")
 frozen=read(root/"data/frozen.json");variants=read(root/"data/variants.json")
 for name in ("configuration.json","resource_plan.json","preflight.json","cache_reuse.json","verification.json","startup_observation.json","README.zh.md","REPORT.zh.md"):
  copy(root/name,out/name)
 for folder in ("verification","gpu_expansion","status","jobs"):
  for p in (root/folder).glob("*.json"):copy(p,out/folder/p.name)
 for name in ("frozen.json","variants.json","sources.jsonl"):
  copy(root/"data"/name,out/"data"/name)
 compressed(root/"data/pairs.jsonl",out/"data/pairs.jsonl.gz")
 compressed(root/"data/views.jsonl",out/"data/views_metadata.jsonl.gz",omit=("prompt",))
 compressed(root/"data/nodes.jsonl",out/"data/nodes_metadata.jsonl.gz",omit=("guideline","definitions","patient","ancestors","source_document"))
 for dataset in ("nli4ct","medpic"):
  copy(root/"external"/dataset/"manifest.json",out/"external"/dataset/"manifest.json")
  compressed(root/"external"/dataset/"pairs.jsonl",out/"external"/dataset/"pairs.jsonl.gz")
 copy(root/"external/summary.json",out/"external/summary.json")
 for model in ("qwen","medgemma"):
  mr=root/model;mt=out/model
  for p in mr.glob("*.json"):copy(p,mt/p.name)
  for folder in ("cost","diagnostics","mechanisms","path_results","readout_scores","scores","historical_h3","comparators","external"):
   for p in (mr/folder).rglob("*"):
    if not p.is_file():continue
    rel=p.relative_to(mr)
    if p.suffix==".json":copy(p,mt/rel)
    elif p.suffix==".jsonl":compressed(p,mt/(str(rel)+".gz"),omit=("prompt","feature"))
  # Only finished training/evaluation artifacts are reported as completed.
  for p in (mr/"baselines").glob("*_training.json"):copy(p,mt/"baselines"/p.name)
  for p in (mr/"baselines").glob("*_metrics.json"):
   copy(p,mt/"baselines"/p.name);stem=p.stem.removesuffix("_metrics")
   compressed(mr/"baselines"/(stem+".jsonl"),mt/"baselines"/(stem+".jsonl.gz"))
   name,split=stem.rsplit("_",1)
   compressed(mr/"baselines"/(name+"_paths_"+split+".jsonl"),mt/"baselines"/(name+"_paths_"+split+".jsonl.gz"))
 rr=destination/"results/aster_cr_repair_20261108"
 for rel in ("manifest.json","verification.json","runtime.json","status.json","data/frozen.json","data/variants.json","pilot/review_summary.json","bulk/built/summary.json","external/summary.json"):
  copy(repair/rel,rr/rel)
 copy(repair/"REPORT.zh.md",report/"DATA_REPAIR.zh.md")
 ordinary_out=destination/"results/ordinary_readout_20261108"
 for p in ordinary.glob("*.json"):
  if p.name not in ("environment.json","launch.json","processes.json"):copy(p,ordinary_out/p.name)
 for p in (ordinary/"workers").glob("*.json"):copy(p,ordinary_out/"workers"/p.name)
 for p in (ordinary/"predictions").glob("*.jsonl"):compressed(p,ordinary_out/"predictions"/(p.name+".gz"))
 compressed(ordinary/"path_changes.jsonl",ordinary_out/"path_changes.jsonl.gz")
 copy(ordinary/"REPORT.zh.md",ordinary_out/"REPORT.zh.md")
 copy(ordinary/"README.zh.md",ordinary_out/"README.zh.md")
 # Development intervals describe this development comparison, not held-out test confirmation.
 pairs=[p for p in rows(root/"data/pairs.jsonl") if p["split"]=="dev"]
 candidates=read(root/"qwen/readout_candidates.json");selection=read(root/"qwen/development_selection.json")
 scores={r["name"]:{v["view_id"]:v["ell"] for v in rows(root/"qwen/readout_scores"/(r["name"]+"_dev.jsonl"))} for r in candidates}
 paths={r["name"]:rows(root/"qwen/path_results"/(r["name"]+"_dev.jsonl")) for r in candidates}
 full_paths={r["case_id"]:r for r in paths["full"]};selected=selection["aster_cr"]
 analysis={}
 for r in candidates:
  name=r["name"];ps=paths[name];metrics=pair_metrics([],pairs,scores[name],ci=True)
  diffs=[int(v["correct"])-int(full_paths[v["case_id"]]["correct"]) for v in ps]
  path_delta=interval(diffs,[v["source_id"] for v in ps])
  analysis[name]=dict(parameters=r["parameters"],component=r.get("component"),pairs=metrics,
   macro_interval=macro_difference(pairs,scores[name],{}),macro_difference_from_full=macro_difference(pairs,scores[name],scores["full"]),
   path=dict(n=len(ps),correct=sum(v["correct"] for v in ps),corrected=sum(d==1 for d in diffs),new_errors=sum(d==-1 for d in diffs),difference_from_full=path_delta))
 comparisons={n:macro_difference(pairs,scores[selected],scores[n]) for n in ("full",selection["ordinary"],"ordinary_matched") if n in scores}
 dev=dict(snapshot_at=now,scope="Qwen development; selections not yet finally frozen",selected=selected,methods=analysis,selected_macro_comparisons=comparisons,
  interval_protocol="2000 source bootstrap draws; macro draws missing any class omitted; selection on this same development set")
 write(out/"development_analysis.json",dev)
 state=dict(snapshot_at=now,server_run=str(root),pipeline=pipeline,scope="current observed results; running work is not final",
  qwen_test_frozen=bool(read(root/"qwen/frozen.json").get("frozen")),qwen_test_complete=(root/"qwen/test_pairs.json").exists(),
  medgemma_formal_complete=read(root/"status/medgemma.json").get("state")=="complete",external_model_evaluation_complete=read(root/"status/external.json").get("state")=="complete",
  ordinary_readout_complete=read(ordinary/"status.json").get("state")=="complete",gpus=read(root/"configuration.json")["gpus"])
 write(out/"snapshot.json",state)
 h2=read(root/"qwen/h2.json");h3=read(root/"qwen/h3.json");p0=read(root/"qwen/diagnostics/development/p0.json")
 text=["# repair-v2 实验结果与分析","",f"快照时间：{now}。修复版运行阶段：`{pipeline.get('phase',pipeline.get('state'))}`。","",
 "已完成 Qwen fit/dev 特征、P0、定位、H2/H3、读出与消融。训练对照仍在进行；修复版 test、MedGemma 正式确认和外部模型评价尚未完成。这里的修复版性能数值均来自开发集。",
 "","## 当前结论","",
 "修复版选中的读出在开发配对宏平均上略高于 Full，低于普通读出；开发完整路径比 Full 少正确 1 例。当前结果尚未支持修复版读出优于普通读出。独立的原始病例普通线性读出已完成测试，其预测与单参数校准完全一致，没有额外路径收益。",
 "","## 数据规模与分母","",
 "repair-v2 固定 2,782 个候选节点、11,512 个输入、8,730 对变体，不重复抽取。fit/dev/test 来源数为 80/20/40。原始节点分母为 761/199/372，原始路径分母为 1,712/420/806。新增候选节点单列，不扩大原始分母。",
 "","| 划分 | 类别 | 配对数 | 独立节点 | 来源 |","|---|---|---:|---:|---:|"]
 for split in ("fit","dev","test"):
  for kind in ("A","P","C","F","F0"):
   v=variants[split][kind];text.append(f"| {split} | {kind} | {v['pairs']} | {v['nodes']} | {v['sources']} |")
 text += ["","P/C/F 的独立节点仍少。开发 P 只有 4 个节点、2 个来源；test F 只有 2 个节点、2 个来源。配对总数增加不能据此证明这些类别的证据充分。","",
 "## Qwen 开发结果","","两端均预测正确才计该对正确；宏平均对 A/P/C/F 四类等权，F0 单列。先要求路径正确率下降不超过 0.5 个百分点，再按开发宏平均选参。当前预选配置为 `"+selected+"`，只有干扰分量，17 个可训练参数；最终配置尚待训练对照和模板比较完成后冻结。",
 "","| 方法 | 参数量 | 配对宏平均 | 路径正确/420 | 相对 Full 纠正/新增错误 |","|---|---:|---:|---:|---:|"]
 names=["full","bias","logistic",selected,selection["ordinary"],"ordinary_matched","ce_only","no_AP","no_CF","random_heads","random_support"]
 for name in dict.fromkeys(names):
  a=analysis[name];pr=a['path'];text.append(f"| {name} | {a['parameters']} | {pct(a['pairs']['macro'])} | {pr['correct']}/{pr['n']} | {pr['corrected']}/{pr['new_errors']} |")
 text += ["","选中配置各类别的开发结果：","","| 类别 | 正确对/总对 | 正确率 | 独立节点/来源 | 95% 来源区间（%） |","|---|---:|---:|---:|---:|"]
 for kind in ("A","P","C","F","F0"):
  v=analysis[selected]['pairs'][kind];text.append(f"| {kind} | {v['numerator']}/{v['denominator']} | {pct(v['rate'])} | {v['nodes']}/{v['sources']} | {ci(v['interval']['ci95'])} |")
 for name,r in comparisons.items():text.append(f"\n选中配置相对 `{name}` 的宏平均差为 {100*r['difference']:+.3f} 个百分点，95% 来源区间 {ci(r['ci95'])} 个百分点；有效重采样 {r['valid_draws']}/2000 次。")
 pd=analysis[selected]['path']['difference_from_full'];text += [f"\n选中配置相对 Full 的路径差为 {100*pd['estimate']:+.3f} 个百分点，95% 来源区间 {ci(pd['ci95'])}。开发选择阈值通过不等于统计上已建立非劣性。",
 "","普通读出 `ordinary_r8` 的宏平均高于选中配置；参数匹配的 `ordinary_matched` 为 17 参数，其路径为 381/420，低于预设开发路径阈值。全部候选与消融都保留，未只展示最优结果。",
 "","同一开发集同时用于选参和上述比较，区间是开发集描述性统计。P 类全对时来源区间可能退化为 100%–100%，反映现有 4 个节点一致，不能据此外推为普遍可靠。宏平均重采样缺少任一类别时跳过该次，保留有效次数。",
 "","## P0 与机制诊断","",
 f"原始节点 Full 错误：fit {p0['fit']['original_errors']}/{p0['fit']['original_nodes']}，dev {p0['dev']['original_errors']}/{p0['dev']['original_nodes']}。去掉动作文本在 fit 纠正 3、引入 1 个错误，在 dev 纠正 2、未引入新错误。其他清理方式本轮没有可用输入，不将缺失结果写成零收益。",
 "","H2 各秩均通过既定检查：干扰变化的开发 R² 大于 0，信号扰动比低于 0.1。选中秩 16 的 R² 为 "+f"{h2['by_rank']['16']['dev_interference_r2']:.4f}"+"，信号扰动比为 "+f"{h2['by_rank']['16']['leakage_ratio']:.4f}"+"。这支持所构造干扰方向具有一定选择性，不直接证明路径纠错收益。H2 默认秩 8 的干预在 A/P 开发配对的基准节点上，纠正和新增错误均为 0。",
 "","H3 选中头探针在 398 个开发候选节点上的准确率为 98.74%。但其 202 个原模型错误节点（7 个当前开发错误，加 195 个历史错误）上的准确率只有 9.90%，低于打乱标签对照的 58.42%。因此信号分量未启用；不能用整体探针准确率来宣称已找到可用的独立纠错信息。历史错误子集与当前开发节点的选择方式不同，分别保留数量。",
 "","消融中，去掉 A/P 约束后路径降到 110/420，随机子空间为 258/420，去掉 C/F 项为 363/420。它们显示当前训练设置可能造成明显损伤；尚不能单凭这些开发集消融推断稳定的因果纠错机制。",
 "","## 已完成的独立普通读出","",
 "该实验只用 761 个原始 fit 节点，以最后一层判定位置残差做标准化、64 维 PCA 和线性分数修正；没有使用修复版配对。其 65 参数线性读出与 1 参数偏置校准在测试上逐节点、逐路径预测完全相同。",
 "","| 方法 | 原始测试节点正确/372 | 原始测试路径正确/806 | 相对 Full 纠正/新增错误 |","|---|---:|---:|---:|",
 "| Full | 368/372 | 768/806 | 0/0 |","| 偏置校准 | 368/372 | 767/806 | 3/4 |","| 普通线性读出 | 368/372 | 767/806 | 3/4 |",
 "","线性读出相对 Full 的路径差为 −0.124 个百分点，95% 来源区间 [−0.943, +0.674]；相对校准差为 0，区间 [0,0]。结论是未获得超过简单校准的额外预测收益。该独立实验沿用原始来源划分，不称为整个研究过程从未观察的盲测，也不将其结果冒充修复版 test 结果。",
 "","[独立普通读出完整报告](../../results/ordinary_readout_20261108/REPORT.zh.md)。","",
 "## 运行、复用与待完成结果","",
 "GPU 0、1、5、6。已复用 960 条输入及评分口径一致的 Full 分数/激活和 8,312 条开发路径节点缓存。修复影响的定位、训练、选参重新执行，旧完成标记未复制。34 项回归测试以及两模型各 7 条前向验证已通过；MedGemma 前向验证不等于正式评价。",
 "","四类训练对照（LoReFT、LoFiT、LoRA、最后一层）各三个学习率正在运行。之后冻结开发配置，再进行修复版 test、MedGemma 关键确认。NLI4CT 已有 5,500 条输入、5,000 对，MedPIC 有 650 道题、92 对；模型成绩尚待独立评价。两套外部数据不进入 MEGA 训练、选参、宏平均和路径分母。",
 "","原始数值、逐对/逐例明细、配置与状态见 [结果目录](../../results/"+RUN_NAME+"/README.md)；数据修复阶段报告见 [DATA_REPAIR.zh.md](DATA_REPAIR.zh.md)。历史 7,458 例路径结果的分母与本轮不同，不直接比较百分点。"]
 (report/"ANALYSIS.zh.md").write_text("\n".join(text)+"\n")
 (out/"README.md").write_text(f"""# repair-v2 结果快照

快照：{now}。当前阶段：{pipeline.get('phase')}。训练和后续评价未全部完成。

[分析报告](../../reports/{RUN_NAME}/ANALYSIS.zh.md) · [数据修复报告](../../reports/{RUN_NAME}/DATA_REPAIR.zh.md) · [独立普通读出](../ordinary_readout_20261108/REPORT.zh.md)

- snapshot.json：快照时间、运行阶段、各部分完成状态。
- development_analysis.json：全部 Qwen 开发读出/消融、类别区间、路径差值与宏平均差值。
- qwen/：已经产生的机制、诊断、开发预测、路径明细、成本与选参记录。
- data/：冻结分母、类别数量、配对及输入/节点元数据。
- external/：两套外部数据的构造与任务数量；尚无模型正式结果。
- status/、jobs/：当前运行记录；running 状态不是已完成结果。
- verification/：两模型前向验证；gpu_expansion/：四卡扩容记录。

JSONL 明细按 gzip 压缩，解压后可按 group_id、source_id、view_id 和 case_id 复算。未改变数据划分或开发选择。

按仓库现有发布方式，模型权重、激活缓存、完整输入、第三方代码及运行日志保留在服务器；这里同步代码、结果和分析。服务器来源：{root}。构造数据来源：{repair}。

更新快照仅做 CPU 汇总：python -m experiments.aster_cr.export_results --run-root {root} --destination /path/to/repository 。该命令不运行模型、不选参、不自动推送。
""")
 (report/"README.md").write_text("# 2026-11-08 新实验\n\n[实验结果与分析](ANALYSIS.zh.md) · [数据修复](DATA_REPAIR.zh.md) · [机器可读结果](../../results/"+RUN_NAME+"/README.md) · [独立普通读出](../../results/ordinary_readout_20261108/REPORT.zh.md)\n")
 write(out/"export_inventory.json",dict(snapshot_at=now,artifacts=[dict(path=str(p.relative_to(destination)),bytes=p.stat().st_size) for folder in (out,report,rr,ordinary_out) for p in sorted(folder.rglob("*")) if p.is_file()],excluded=["weights","activation caches","full input texts","vendor code","runtime logs"]))
 return state
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--run-root",default="/home/gcx/aster_cr_repair_v2_run_20261108");p.add_argument("--destination",required=True);a=p.parse_args()
 print(json.dumps(export(a.run_root,a.destination),ensure_ascii=False))
