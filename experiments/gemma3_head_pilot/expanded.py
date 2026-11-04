"""100-group development selection followed by frozen independent confirmation."""
import argparse
import gc
import json
import os
import random
import time
import traceback
from collections import defaultdict
from pathlib import Path
from .core import rows, journal, append, write_json
from .fourway import summarize, selection_key, candidates, paired_groups
from .expanded_data import prepare, DEFAULT_ROOT, PILOT_ROOT

def make_summaries(groups,records,specs):
    result={}
    for spec in specs:
        item=summarize(groups,records,spec["method"])
        item.update(spec)
        result[spec["method"]]=item
    return result

def frozen_choice(root,dev_summaries):
    root=Path(root);path=root/"frozen_selection.json"
    if path.exists():return json.loads(path.read_text())
    trained=[s for s in dev_summaries.values() if s["kind"] in ("whole","heads")]
    if len(trained)!=18 or any(s["missing"] for s in dev_summaries.values()):
        raise ValueError("All eighteen development candidates must finish before freezing")
    best={kind:max((s for s in trained if s["kind"]==kind),key=selection_key) for kind in ("whole","heads")}
    frozen=dict(frozen_at=time.time(),selection_split="development",
        primary="Four inputs all correct per group",
        order=["all_four_correct","original_100_case_development_PathAcc","smaller four-input edit","lower layer","earlier epoch"],
        selected={k:{a:b for a,b in v.items() if a!="by_group"} for k,v in best.items()},
        development_groups=100,development_sources=20,confirmation_groups=100,confirmation_sources=20,
        confirmation_predictions_used=False)
    write_json(path,frozen)
    return frozen

def require_frozen(root,split):
    if split=="confirmation" and not (Path(root)/"frozen_selection.json").exists():
        raise ValueError("Confirmation cannot run before development selection is frozen")

def confirmation_candidates(specs,frozen):
    names=[s["method"] for s in specs if s["kind"] in ("zero","blind")]
    names += [frozen["selected"][k]["method"] for k in ("whole","heads")]
    return [next(s for s in specs if s["method"]==name) for name in names]

def bootstrap_sources(summaries,baselines=("zero","blind"),samples=2000):
    source_groups=defaultdict(list)
    first=next(iter(summaries.values()))
    for g in first["by_group"]:source_groups[g["source_id"]].append(g["group_id"])
    sources=sorted(source_groups)
    correct={name:{g["group_id"]:int(g["all_correct"]) for g in s["by_group"]} for name,s in summaries.items()}
    values={name:[] for name in summaries}
    differences={base:{name:[] for name in summaries} for base in baselines}
    rng=random.Random(20261104)
    for _ in range(samples):
        selected=[rng.choice(sources) for _ in sources]
        ids=[gid for source in selected for gid in source_groups[source]]
        accuracy={name:100*sum(v[gid] for gid in ids)/len(ids) for name,v in correct.items()}
        for name,value in accuracy.items():values[name].append(value)
        for base in baselines:
            for name in summaries:differences[base][name].append(accuracy[name]-accuracy[base])
    def interval(v):
        a=sorted(v);return [a[int(.025*(len(a)-1))],a[int(.975*(len(a)-1))]]
    return dict(resampling_unit="guideline source",sources=len(sources),replicates=samples,seed=20261104,
                accuracy_95_percent={k:interval(v) for k,v in values.items()},
                paired_difference_95_percentage_points={base:{k:interval(v) for k,v in d.items()} for base,d in differences.items()})

def evaluate_split(engine,root,split,groups,specs,modules,zero_checked):
    import torch
    from .core import swap_actions
    from .fourway_data import make_case
    require_frozen(root,split)
    out=Path(root)/split;out.mkdir(exist_ok=True)
    path=out/"predictions.jsonl"
    key=lambda r:(r["method"],r["group_id"],r["state"],r["view"])
    saved=journal(path,key)
    expected=len(groups)*4*len(specs)
    if len(saved)==expected:return make_summaries(groups,list(saved.values()),specs)
    started=time.time()
    def save(row):
        append(path,row);saved[key(row)]=row
    with torch.inference_mode():
        for index,group in enumerate(groups,1):
            for state in (0,1):
                case=make_case(group,state);reference=None
                for view in ("original","swapped"):
                    pending=[s for s in specs if (s["method"],group["group_id"],state,view) not in saved]
                    if not pending:continue
                    context=query=None
                    preparation_error=None
                    try:
                        if reference is None:reference=engine.reference(case,group["gid"])
                        if any(s["kind"]!="blind" for s in pending):
                            context=engine.context(swap_actions(case) if view=="swapped" else case)
                            query=engine.decision_query(context,group["history"],group["gid"])
                    except Exception as exc:
                        preparation_error=type(exc).__name__+": "+str(exc)[:500]
                        torch.cuda.empty_cache()
                    zero=None
                    for spec in pending:
                        began=time.time()
                        record=dict(method=spec["method"],group_id=group["group_id"],source_id=group["source_id"],
                                    gid=group["gid"],state=state,view=view,branch=None,scores=[],magnitude=0.,gate=0.)
                        try:
                            if preparation_error and (spec["kind"]!="blind" or reference is None):
                                raise RuntimeError(preparation_error)
                            if spec["kind"]=="blind":
                                scores=reference["label_scores"][:2];diag={}
                            elif spec["kind"]=="zero":
                                scores,diag=engine.scores(query)
                                zero=scores.detach().clone()
                            else:
                                module=modules[spec["method"]]
                                if spec["kind"] not in zero_checked:
                                    baseline=zero if zero is not None else engine.scores(query)[0]
                                    check,_=engine.scores(query,layer=spec["layer"],module=module,reference=reference,force_zero=True)
                                    if not torch.equal(baseline,check):
                                        raise AssertionError("Zero edit differs from native model")
                                    zero_checked.add(spec["kind"])
                                scores,diag=engine.scores(query,layer=spec["layer"],module=module,reference=reference)
                            if not torch.isfinite(scores).all():raise ValueError("Nonfinite branch score")
                            record.update(status="complete",branch="Y" if int(scores.argmax())==0 else "N",
                                          scores=scores.detach().float().cpu().tolist(),
                                          magnitude=float(diag.get("magnitude",0.)),gate=float(diag.get("gate",0.)))
                        except AssertionError:
                            raise
                        except Exception as exc:
                            record.update(status="failed",error=type(exc).__name__+": "+str(exc)[:500])
                            torch.cuda.empty_cache()
                        record["seconds"]=time.time()-began;save(record)
                    del context,query
                del reference,case
                engine.clear_reference_memory()
            progress=dict(stage=split,groups=index,total_groups=len(groups),prediction_cells=len(saved),
                          expected_cells=expected,elapsed_seconds=time.time()-started,
                          remaining_seconds=(time.time()-started)/index*(len(groups)-index))
            write_json(Path(root)/"status.json",progress)
            if index%5==0:print(json.dumps(progress),flush=True)
    summaries=make_summaries(groups,list(saved.values()),specs)
    write_json(out/"summaries.json",summaries)
    return summaries

def clean(summary):
    return {k:v for k,v in summary.items() if k not in ("by_group","checkpoint")}

def table_row(s,confirm=None,paired=None):
    labels={"zero":"零干预","blind":"直接盲读","whole":"整层干预","heads":"选头干预"}
    layer="—" if s["layer"] is None else str(s["layer"])
    epoch="—" if s["epoch"] is None else str(s["epoch"])
    groups="—" if s["groups"] is None else ",".join(map(str,s["groups"]))
    conf="—" if confirm is None else str(confirm["all_four_correct"])+"/"+str(confirm["group_count"])
    changes="—" if paired is None else str(paired["corrections"])+"/"+str(paired["regressions"])
    return f"| {labels[s['kind']]} | {layer} | {groups} | {epoch} | {s['all_four_correct']}/{s['group_count']} | {conf} | {changes} | {s['development_path_correct']}/{s['development_path_cases']} |"

def report(root,dev=None,confirmation=None):
    root=Path(root)
    specs=json.loads((root/"candidate_manifest.json").read_text())
    data={s:rows(root/"data"/(s+".jsonl")) for s in ("development","confirmation")}
    if dev is None:
        values=list(journal(root/"development/predictions.jsonl",
                    lambda r:(r["method"],r["group_id"],r["state"],r["view"])).values())
        dev=make_summaries(data["development"],values,specs)
    frozen=json.loads((root/"frozen_selection.json").read_text())
    confirm_specs=confirmation_candidates(specs,frozen)
    if confirmation is None:
        values=list(journal(root/"confirmation/predictions.jsonl",
                    lambda r:(r["method"],r["group_id"],r["state"],r["view"])).values())
        confirmation=make_summaries(data["confirmation"],values,confirm_specs)
    complete=all(s["missing"]==0 for s in dev.values()) and all(s["missing"]==0 for s in confirmation.values())
    paired={base:{k:paired_groups(confirmation[base],s) for k,s in confirmation.items()} for base in ("zero","blind")}
    intervals=bootstrap_sources(confirmation) if complete else None
    result=dict(complete=complete,development={k:clean(s) for k,s in dev.items()},
        confirmation={k:clean(s) for k,s in confirmation.items()},paired_confirmation=paired,
        confidence_intervals=intervals,selection=frozen,
        data_manifest=json.loads((root/"data_manifest.json").read_text()),
        original_pathacc_note="The old 100-case original-development PathAcc is reused only as a tie-breaker and descriptive task metric; it is not a new confirmation path score.",
        confirmation_metric="At a fixed node and shared valid prefix, all four fact/action inputs must be correct; failures remain in all 100 groups.")
    write_json(root/"results.json",result)
    lines=["# Gemma 3 27B：100组开发选择与100组独立确认","",
           "两组各20套指南、每套5个不同节点；来源与训练及此前使用过的指南分开。病例为预先构造的单事实记录，不复制原始患者叙述。",
           "开发比较18个既有检查点及两条基线；依据开发四项全对数、原100例开发PathAcc、较小实际编辑量排序。冻结后仅评零干预、盲读及选中的整层和选头配置。",
           "四项全对要求事实真假×原/交换处置四种输入均正确；技术失败保留在100组分母中。节点固定在共同前缀下，不能等同于从根到叶自主路径成绩。",
           "两组均覆盖树深度0–4各20组；它们不是对任意深度和所有临床事实的无偏抽样。","",
           "## 同表比较","",
           "| 方法 | 层索引 | 头组 | 轮次 | 开发四项全对 | 独立确认四项全对 | 确认较零干预纠正/退化（组） | 原100例开发PathAcc |",
           "|---|---:|---|---:|---:|---:|---:|---:|"]
    for spec in confirm_specs:
        name=spec["method"];lines.append(table_row(dev[name],confirmation[name],paired["zero"][name]))
    lines += ["","原100例PathAcc来自旧开发集，只作任务效果描述及开发选参并列处理；不是本次确认集完整路径准确率。",
              "直接盲读在T/F中二选一，U只作诊断；其输入不含候选处置，所以两个处置视图复用同一盲判。",
              "层号、头组均从0编号。独立确认不参与重新选参。","",
              "## 全部18个检查点的开发结果","",
              "| 方法 | 层索引 | 头组 | 轮次 | 开发四项全对 | 确认四项全对 | 纠正/退化 | 原100例开发PathAcc |",
              "|---|---:|---|---:|---:|---:|---:|---:|"]
    for s in sorted((s for s in dev.values() if s["kind"] in ("whole","heads")),key=selection_key,reverse=True):
        lines.append(table_row(s))
    if intervals:
        lines += ["","## 独立确认的配对不确定性","",
                  "按20个指南来源进行2,000次有放回配对重采样，每次保留来源内全部5组。",
                  "| 方法 | 相对零干预净增加（组/100） | 95%区间（百分点） | 相对盲读净增加（组/100） | 95%区间（百分点） |",
                  "|---|---:|---|---:|---|"]
        for spec in confirm_specs:
            name=spec["method"]
            z=intervals["paired_difference_95_percentage_points"]["zero"][name]
            b=intervals["paired_difference_95_percentage_points"]["blind"][name]
            lines.append(f"| {name} | {paired['zero'][name]['net']} | [{z[0]:.1f}, {z[1]:.1f}] | {paired['blind'][name]['net']} | [{b[0]:.1f}, {b[1]:.1f}] |")
    (root/"REPORT.zh.md").write_text("\n".join(lines)+"\n")
    return result

def run(root=DEFAULT_ROOT,pilot=PILOT_ROOT):
    import torch
    from safetensors.torch import load_file
    from .core import make_module
    from .engine import Engine
    root,pilot=Path(root),Path(pilot)
    data=prepare(root,pilot)
    specs=candidates(pilot)
    cp=root/"candidate_manifest.json"
    if cp.exists() and json.loads(cp.read_text())!=specs:
        raise ValueError("Candidate list changed")
    write_json(cp,specs)
    protocol=dict(development_groups=100,development_sources=20,confirmation_groups=100,confirmation_sources=20,
                  development_candidates=18,baselines=2,development_predictions=8000,
                  confirmation_methods=4,confirmation_predictions=1600,devices=[0,1],
                  order=["all-four development accuracy","prior original-development PathAcc","lower actual edit"],
                  no_confirmation_predictions_before_freeze=True,no_new_training=True)
    write_json(root/"selection_protocol.json",protocol)
    if (root/"results.json").exists() and json.loads((root/"results.json").read_text()).get("complete"):
        return json.loads((root/"results.json").read_text())
    started=time.time()
    write_json(root/"status.json",dict(stage="loading_model",devices=[0,1],started=started))
    engine=Engine.load(pilot/"model",root/"runtime")
    if torch.cuda.device_count()!=2:raise ValueError("Only physical GPU0,1 must be visible")
    for d in (0,1):torch.cuda.reset_peak_memory_stats(d)
    modules={}
    for spec in specs:
        if spec["kind"] not in ("whole","heads"):continue
        module=make_module(spec,engine.config,engine.layer_device(spec["layer"]))
        module.load_state_dict(load_file(spec["checkpoint"]))
        modules[spec["method"]]=module.eval().requires_grad_(False)
    zero_checked=set()
    dev=evaluate_split(engine,root,"development",data["development"],specs,modules,zero_checked)
    write_json(root/"development/summaries.json",dev)
    frozen=frozen_choice(root,dev)
    confirm_specs=confirmation_candidates(specs,frozen)
    write_json(root/"confirmation_candidate_manifest.json",confirm_specs)
    confirmation=evaluate_split(engine,root,"confirmation",data["confirmation"],confirm_specs,modules,zero_checked)
    write_json(root/"confirmation/summaries.json",confirmation)
    result=report(root,dev,confirmation)
    write_json(root/"costs.json",dict(seconds=time.time()-started,
               peak_allocated_gib=[torch.cuda.max_memory_allocated(d)/2**30 for d in (0,1)],
               peak_reserved_gib=[torch.cuda.max_memory_reserved(d)/2**30 for d in (0,1)],
               device_map={k:str(v) for k,v in engine.model.hf_device_map.items()},
               zero_identity_checks=sorted(zero_checked),scored_development_cells=8000,scored_confirmation_cells=1600))
    write_json(root/"status.json",dict(stage="complete" if result["complete"] else "incomplete",
               seconds=time.time()-started,results=str(root/"results.json"),report=str(root/"REPORT.zh.md")))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=["prepare","evaluate","report"],default="evaluate")
    p.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    p.add_argument("--pilot-root",type=Path,default=PILOT_ROOT)
    p.add_argument("--devices",default="0,1")
    a=p.parse_args()
    if a.devices!="0,1":p.error("This run is authorized for GPU0,1")
    os.environ["CUDA_VISIBLE_DEVICES"]=a.devices
    try:
        if a.stage=="prepare":
            data=prepare(a.root,a.pilot_root)
            print(json.dumps({s:{"groups":len(g),"sources":len({x["source_id"] for x in g})} for s,g in data.items()}))
        elif a.stage=="report":report(a.root)
        else:run(a.root,a.pilot_root)
    except Exception as exc:
        write_json(a.root/"failure.json",dict(stage=a.stage,error=type(exc).__name__+": "+str(exc)))
        traceback.print_exc()
        raise

if __name__=="__main__":main()
