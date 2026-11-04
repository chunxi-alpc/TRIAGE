"""Evaluate existing Gemma checkpoints on fixed-node four-input development groups."""
import argparse
import json
import os
import time
import traceback
from pathlib import Path
from .core import rows, journal, append, write_json

def summarize(groups, records, method):
    lookup = {(r["group_id"], r["state"], r["view"]): r for r in records if r["method"] == method}
    all_correct = correct_cells = failures = missing = 0
    action_pairs = fact_pairs = 0
    magnitudes, worst_margins = [], []
    group_results = []
    for group in groups:
        cells, margins = {}, []
        for state in (0, 1):
            for view in ("original", "swapped"):
                r = lookup.get((group["group_id"], state, view))
                valid = r is not None and r["status"] == "complete"
                correct = valid and r["branch"] == ("Y" if state else "N")
                cells[state, view] = bool(correct)
                correct_cells += bool(correct)
                missing += r is None
                failures += r is not None and not valid
                if valid:
                    magnitudes.append(float(r["magnitude"]))
                    scores = r["scores"]
                    target = 0 if state else 1
                    margins.append(scores[target]-scores[1-target])
        good = all(cells.values())
        all_correct += good
        action_pairs += sum(all(cells[s,v] for v in ("original","swapped")) for s in (0,1))
        fact_pairs += sum(all(cells[s,v] for s in (0,1)) for v in ("original","swapped"))
        if len(margins) == 4:
            worst_margins.append(min(margins))
        group_results.append(dict(group_id=group["group_id"], source_id=group["source_id"], all_correct=good))
    count = len(groups)
    return dict(group_count=count, all_four_correct=all_correct, all_four_accuracy=all_correct/count,
                correct_cells=correct_cells, total_cells=count*4, cell_accuracy=correct_cells/(count*4),
                action_pairs_correct=action_pairs, action_pairs_total=count*2,
                fact_pairs_correct=fact_pairs, fact_pairs_total=count*2,
                failures=failures, missing=missing,
                mean_magnitude=sum(magnitudes)/len(magnitudes) if magnitudes else 0.,
                mean_worst_margin=sum(worst_margins)/len(worst_margins) if worst_margins else None,
                by_group=group_results)

def selection_key(summary):
    return (summary["all_four_correct"], summary["development_path_correct"],
            -summary["mean_magnitude"], -summary["layer"], -summary["epoch"])

def candidates(pilot_root):
    pilot_root = Path(pilot_root)
    result = []
    for pattern in ("whole_layer*", "heads_layer*"):
        for job in sorted((pilot_root/"jobs").glob(pattern)):
            spec = json.loads((job/"spec.json").read_text())
            for epoch in (1,2,3):
                checkpoint = job/f"epoch{epoch}.safetensors"
                summary_path = job/f"development_epoch{epoch}.json"
                if not checkpoint.exists() or not summary_path.exists():
                    raise ValueError("Missing completed candidate: "+str(checkpoint))
                previous = json.loads(summary_path.read_text())
                result.append(dict(method=job.name+f"_epoch{epoch}", name=job.name,
                    kind=spec["kind"], layer=spec["layer"], groups=spec.get("groups"), epoch=epoch,
                    checkpoint=str(checkpoint), parameters=previous["parameters"],
                    development_path_correct=previous["original_correct"],
                    development_path_cases=previous["cases"],
                    prior_development_magnitude=previous["mean_magnitude"]))
    if len(result) != 18:
        raise ValueError(f"Expected 18 trained candidates, found {len(result)}")
    for method in ("zero","blind"):
        previous=json.loads((pilot_root/"development_baselines"/f"{method}.summary.json").read_text())
        result.insert(0 if method=="zero" else 1, dict(method=method, name=method, kind=method,
            layer=None, groups=None, epoch=None, checkpoint=None, parameters=0,
            development_path_correct=previous["original_correct"], development_path_cases=previous["cases"]))
    return result

def paired_groups(base, other):
    a={r["group_id"]:r["all_correct"] for r in base["by_group"]}
    b={r["group_id"]:r["all_correct"] for r in other["by_group"]}
    fixed=sum(not a[k] and b[k] for k in a)
    broken=sum(a[k] and not b[k] for k in a)
    return dict(corrections=fixed, regressions=broken, net=fixed-broken)

def report(output_root, groups=None, specs=None):
    output_root=Path(output_root)
    groups=groups or rows(output_root/"groups.jsonl")
    specs=specs or json.loads((output_root/"candidate_manifest.json").read_text())
    records=list(journal(output_root/"predictions.jsonl",
                 lambda r:(r["method"],r["group_id"],r["state"],r["view"])).values())
    summaries={}
    for spec in specs:
        summary=summarize(groups,records,spec["method"])
        summary.update(spec)
        summaries[spec["method"]]=summary
    expected=len(groups)*4*len(specs)
    complete=len(records)==expected and all(s["missing"]==0 for s in summaries.values())
    result=dict(complete=complete, split="development", group_count=len(groups),
                decision_cells=len(records), expected_cells=expected, summaries=summaries,
                selection_rule=["higher all-four-input group accuracy","higher prior 100-case development PathAcc",
                                "lower mean post-projection squared relative edit on four-input cells",
                                "lower layer index","earlier epoch"],
                scope="Constructed single-fact records under development-source guidelines at a fixed node and shared prefix.",
                pathacc_scope="Previously measured autonomous execution on the original 100 development cases; no new full-path evaluation.",
                independent_confirmation_run=False)
    if complete:
        choices=[s for s in summaries.values() if s["kind"] in ("whole","heads")]
        result["best_whole"]=max((s for s in choices if s["kind"]=="whole"),key=selection_key)
        result["best_heads"]=max((s for s in choices if s["kind"]=="heads"),key=selection_key)
        result["best_intervention"]=max(choices,key=selection_key)
        result["paired_groups_vs_zero"]={k:paired_groups(summaries["zero"],v) for k,v in summaries.items()}
        write_json(output_root/"selected_development.json",
                   {k:result[k] for k in ("best_whole","best_heads","best_intervention","selection_rule","independent_confirmation_run")})
    write_json(output_root/"results.json",result)
    labels={"zero":"零干预","blind":"直接盲读","whole":"整层干预","heads":"选头干预"}
    lines=["# Gemma 3 27B：四种输入开发补评","",
           f"状态：{'已完成' if complete else '未完成，尚未选参'}。共{len(groups)}组、{len({g['source_id'] for g in groups})}个开发来源，每组四种输入。",
           "病例是按开发指南预先构造的单事实最小对，不复制原病例叙述。固定目标节点及共同前缀；四项全对才计一组成功。",
           "原100例PathAcc复用先前完整路径评测，和本次节点级指标分列。以下均为开发选择结果，不是独立确认成绩。","",
           "| 方法 | 层索引 | 头组 | 轮次 | 四项全对 | 原100例PathAcc | 相对编辑量 | 失败/缺失 |",
           "|---|---:|---|---:|---:|---:|---:|---:|"]
    order=[summaries["zero"],summaries["blind"]]+sorted(
        (s for s in summaries.values() if s["kind"] in ("whole","heads")),key=selection_key,reverse=True)
    for s in order:
        h="—" if s["groups"] is None else ",".join(map(str,s["groups"]))
        layer="—" if s["layer"] is None else str(s["layer"])
        epoch="—" if s["epoch"] is None else str(s["epoch"])
        lines.append(f"| {labels[s['kind']]} | {layer} | {h} | {epoch} | {s['all_four_correct']}/{len(groups)} | {s['development_path_correct']}/{s['development_path_cases']} | {s['mean_magnitude']:.6g} | {s['failures']}/{s['missing']} |")
    lines += ["","排序：四项全对率优先，其次原100例开发PathAcc，再比较本次四输入上投影后相对编辑量。失败保留在全部组的分母内。",
              "处置不变、事实翻转分项正确数及逐组配对纠正/退化见 results.json。直接盲读在T/F分数中二选一，U分数只留作诊断。",
              "零干预及所有干预使用相同的完整指南、原生聊天模板、合法Y/N评分；直接盲读复用同一条件提示及内部参考。",
              "本轮只重选已有18个检查点；没有重新扫描全部层头，也没有新增训练。"]
    if complete:
        lines += ["","## 选中配置"]
        for key in ("best_whole","best_heads"):
            s=result[key]
            lines.append(f"- {labels[s['kind']]}：层索引{s['layer']}，头组{s['groups']}，第{s['epoch']}轮，四项全对{s['all_four_correct']}/{len(groups)}，PathAcc {s['development_path_correct']}/{s['development_path_cases']}。")
    (output_root/"REPORT.zh.md").write_text("\n".join(lines)+"\n")
    return result

def evaluate(pilot_root, output_root):
    import torch
    from safetensors.torch import load_file
    from .core import make_module, swap_actions
    from .engine import Engine
    from .fourway_data import prepare, make_case
    pilot_root,output_root=Path(pilot_root),Path(output_root)
    groups=prepare(pilot_root,output_root)
    specs=candidates(pilot_root)
    manifest=output_root/"candidate_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())!=specs:
        raise ValueError("Candidate specification changed")
    write_json(manifest,specs)
    write_json(output_root/"selection_protocol.json",dict(
        primary="All four counterfactual inputs correct per group; any failed cell makes the group fail.",
        secondary="Prior original-guideline PathAcc on the same registered 100 development cases.",
        tertiary="Mean squared relative post-o_proj edit on all four-input cells.",
        final_ties="Lower zero-based layer index, then earlier epoch.",
        groups=len(groups),sources=10,checkpoints=18,baselines=2,expected_decisions=1600,
        independent_confirmation=False,devices=[0,1]))
    key=lambda r:(r["method"],r["group_id"],r["state"],r["view"])
    saved=journal(output_root/"predictions.jsonl",key)
    if len(saved)==len(groups)*4*len(specs):
        return report(output_root,groups,specs)
    started=time.time()
    write_json(output_root/"status.json",dict(stage="loading_model",started=started,devices=[0,1]))
    engine=Engine.load(pilot_root/"model",output_root)
    for device in (0,1):torch.cuda.reset_peak_memory_stats(device)
    modules={}
    for spec in specs:
        if spec["kind"] not in ("whole","heads"): continue
        module=make_module(spec,engine.config,engine.layer_device(spec["layer"]))
        module.load_state_dict(load_file(spec["checkpoint"]))
        modules[spec["method"]]=module.eval().requires_grad_(False)
    tested_zero=set()
    first_group=time.time()
    expected=len(groups)*4*len(specs)
    def save(row):
        append(output_root/"predictions.jsonl",row)
        saved[key(row)]=row
    with torch.inference_mode():
        for index,group in enumerate(groups,1):
            for state in (0,1):
                case=make_case(group,state)
                reference=None
                for view in ("original","swapped"):
                    pending=[s for s in specs if (s["method"],group["group_id"],state,view) not in saved]
                    if not pending:continue
                    if reference is None:
                        reference=engine.reference(case,group["gid"])
                    if any(s["kind"] in ("whole","heads","zero") for s in pending):
                        actual=swap_actions(case) if view=="swapped" else case
                        context=engine.context(actual)
                        query=engine.decision_query(context,group["history"],group["gid"])
                    else:context=query=None
                    zero=None
                    for spec in pending:
                        cell_started=time.time()
                        row=dict(group_id=group["group_id"],source_id=group["source_id"],
                                 gid=group["gid"],state=state,view=view,method=spec["method"],
                                 branch=None,scores=[],magnitude=0.,gate=0.)
                        try:
                            if spec["kind"]=="blind":
                                scores=reference["label_scores"][:2]
                                diag={}
                            elif spec["kind"]=="zero":
                                scores,diag=engine.scores(query)
                                zero=scores.detach().clone()
                            else:
                                module=modules[spec["method"]]
                                if spec["kind"] not in tested_zero:
                                    baseline=zero if zero is not None else engine.scores(query)[0]
                                    check,_=engine.scores(query,layer=spec["layer"],module=module,
                                                          reference=reference,force_zero=True)
                                    if not torch.equal(baseline,check):
                                        raise AssertionError("Zero intervention changed native scores")
                                    tested_zero.add(spec["kind"])
                                scores,diag=engine.scores(query,layer=spec["layer"],module=module,reference=reference)
                            if not torch.isfinite(scores).all():raise ValueError("Nonfinite branch scores")
                            row.update(status="complete",branch="Y" if int(scores.argmax())==0 else "N",
                                       scores=scores.detach().float().cpu().tolist(),
                                       magnitude=float(diag.get("magnitude",0.)),
                                       gate=float(diag.get("gate",0.)))
                        except Exception as exc:
                            row.update(status="failed",error=type(exc).__name__+": "+str(exc)[:500])
                            torch.cuda.empty_cache()
                        row["seconds"]=time.time()-cell_started
                        save(row)
                    del context,query
                del reference,case
                engine.clear_reference_memory()
            elapsed=time.time()-started
            remaining=(time.time()-first_group)/index*(len(groups)-index)
            progress=dict(stage="evaluating",completed_groups=index,total_groups=len(groups),
                          prediction_cells=len(saved),expected_cells=expected,seconds=elapsed,
                          estimated_remaining_seconds=remaining,zero_identity_checks=sorted(tested_zero))
            write_json(output_root/"status.json",progress)
            print(json.dumps(progress),flush=True)
    result=report(output_root,groups,specs)
    write_json(output_root/"costs.json",dict(seconds=time.time()-started,
               peak_allocated_gib=[torch.cuda.max_memory_allocated(i)/2**30 for i in (0,1)],
               peak_reserved_gib=[torch.cuda.max_memory_reserved(i)/2**30 for i in (0,1)],
               device_map={k:str(v) for k,v in engine.model.hf_device_map.items()},
               prediction_cells=len(saved),zero_identity_checks=sorted(tested_zero)))
    write_json(output_root/"status.json",dict(stage="complete",results=str(output_root/"results.json"),
               report=str(output_root/"REPORT.zh.md"),seconds=time.time()-started))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=["prepare","evaluate","report"],default="evaluate")
    p.add_argument("--pilot-root",type=Path,default=Path("/home/gcx/gemma3_27b_head_pilot_20261103"))
    p.add_argument("--output-root",type=Path,default=Path("/home/gcx/gemma3_27b_fourway_development_20261104"))
    p.add_argument("--devices",default="0,1")
    a=p.parse_args()
    if a.devices!="0,1":p.error("This authorized run uses physical GPU 0,1")
    os.environ["CUDA_VISIBLE_DEVICES"]=a.devices
    try:
        if a.stage=="prepare":
            from .fourway_data import prepare
            groups=prepare(a.pilot_root,a.output_root)
            specs=candidates(a.pilot_root)
            write_json(a.output_root/"candidate_manifest.json",specs)
            print(json.dumps({"groups":len(groups),"methods":len(specs),"decisions":len(groups)*4*len(specs)}))
        elif a.stage=="report":report(a.output_root)
        else:evaluate(a.pilot_root,a.output_root)
    except Exception as exc:
        write_json(a.output_root/"failure.json",dict(error=type(exc).__name__+": "+str(exc),stage=a.stage))
        traceback.print_exc()
        raise

if __name__=="__main__":main()
