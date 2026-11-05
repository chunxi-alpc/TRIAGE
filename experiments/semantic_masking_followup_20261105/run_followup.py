"""Resume a bounded semantic review, rebuild changed inputs, reuse scores, and report."""
import collections
import json
import os
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path
from common import CONFIG, DATA, LOGS, RESULTS, ROOT, read_rows, write_rows, write_json
from run_revision import launch, update, verify_inputs
from annotation_queue import queue_path
from annotate import parse_json

def prepare_review():
    prior = Path(CONFIG["prior_revision"])
    baseline = read_rows(DATA/"annotations_baseline.jsonl")
    raw_management = {}
    for path in sorted((prior/"logs").glob("annotation_raw_*.jsonl")):
        for row in read_rows(path):
            if row.get("task") != "management":
                continue
            try:
                value = parse_json(row["text"])
                if isinstance(value.get("conservative_cuts"), list):
                    raw_management[row["node_id"]] = value
            except (ValueError, TypeError, KeyError):
                pass
    for record in baseline:
        extraction = raw_management.get(record["node_id"])
        if extraction is not None:
            record["annotation"].update({k:extraction.get(k) for k in
                ("conservative_cuts", "action_swap_safe", "action_swap_spans")})
    write_rows(DATA/"annotations_pilot_0.jsonl", baseline)
    targets, critical = [], []
    for record in baseline:
        value = record["annotation"]
        intersect = any(s["block_id"] in value.get("direct_delete_block_ids", [])
                        for s in value.get("missing_definitions", []))
        if value.get("source_semantic_dependency") is not False or intersect:
            targets.append(record["node_id"])
        if intersect and not value.get("necessary_restores"):
            critical.append(record["node_id"])
    order = list(dict.fromkeys(critical+targets))
    from revision import all_annotations
    previous = all_annotations()
    pending = [nid for nid in order if previous.get(nid, {}).get("semantic_review_version") != 3]
    write_json(queue_path("full"),dict(stage="full", prepared_at=time.time(), total=len(pending),
        completed_before=len(order)-len(pending),pending=pending,claims={},finished={}))
    write_json(RESULTS/"followup_scope.json",dict(target_nodes=len(order),candidate_restore_conflicts=len(critical),
        critical_node_ids=critical,target_node_ids=order,model="MedGemma 27B",gpus=CONFIG["gpus"],
        preserved_nodes=429,statistical_nodes=379,construction_pool=50,initial_constructed_nodes=39,
        policy="Review previous positive/unknown labels and definition/deletion conflicts; preserve other records",
        prior_revision=str(prior),hash_checks=False))
    return len(pending)

def pair(script, prefix):
    args = (lambda shard: ["--shard",str(shard)]) if prefix=="annotation" else (
        lambda shard: ["--stage","full","--shard",str(shard)])
    workers = [launch(script,args(shard),f"{prefix}_full_{shard}.log",gpu)
               for shard,gpu in enumerate(CONFIG["gpus"])]
    update(prefix,stage="full",worker_pids=[p.pid for p in workers],physical_gpus=CONFIG["gpus"],
           task_distribution="shared_queue" if prefix=="annotation" else "fixed_shards")
    codes = [p.wait() for p in workers]
    if any(codes):
        raise RuntimeError(f"{prefix} exit codes {codes}; see logs")

def input_differences():
    from score import make_prompt
    old = {r["score_id"]:r for r in read_rows(Path(CONFIG["prior_revision"])/"data/inputs_full.jsonl")}
    new = read_rows(DATA/"inputs_full.jsonl")
    changes=[]
    for row in new:
        previous=old.get(row["score_id"])
        if previous is None or make_prompt(previous)!=make_prompt(row):
            changes.append(dict(score_id=row["score_id"],node_id=row["node_id"],view=row["view"],
                reason="new_input" if previous is None else "full_input_changed"))
        elif previous["gold"]!=row["gold"]:
            changes.append(dict(score_id=row["score_id"],node_id=row["node_id"],view=row["view"],
                reason="label_only_changed"))
    write_rows(RESULTS/"input_changes_vs_prior.jsonl",changes)
    current_ids={r["score_id"] for r in new}
    summary=dict(total_inputs=len(new),changed_or_new_inputs=sum(r["reason"]!="label_only_changed" for r in changes),
        label_only_changes=sum(r["reason"]=="label_only_changed" for r in changes),
        unchanged_inputs=len(new)-len(changes),
        removed_input_ids=sorted(set(old)-current_ids),
        by_view=dict(collections.Counter(r["view"] for r in changes)))
    write_json(RESULTS/"input_change_summary.json",summary)
    return summary

def verify_followup():
    from revision import all_annotations, normalize_annotation
    from dependency_rules import usable_dependency
    nodes={r["node_id"]:r for r in read_rows(DATA/"nodes.jsonl")}
    records=all_annotations()
    reviewed=[r for r in records.values() if r.get("semantic_review_version")==3]
    usable=collections.Counter()
    restored=lost_with_duplicate=0
    for record in reviewed:
        nid=record["node_id"]
        value=record["annotation"]
        again=normalize_annotation(nodes[nid],value)["annotation"]
        if again["necessary_restores"]!=value["necessary_restores"]:
            raise ValueError("Restoration derivation changed on repeat normalization: "+nid)
        if value.get("necessary_restores"):
            restored+=1
        if value.get("removed_definition_spans") and not value.get("definition_lost"):
            lost_with_duplicate+=1
        usable[str(usable_dependency(value,record["validation"],record.get("unresolved_references",[])))]+=1
    result=dict(status="passed",reviewed=len(reviewed),usable_dependency_counts=dict(usable),
        restoration_nodes=restored,removed_definition_but_retained_equivalent_nodes=lost_with_duplicate,
        review_failures=sum(r.get("review_status")=="failed" for r in reviewed),
        checks=["current-condition binding", "restoration derived from source positions",
                "retained equivalent definitions distinguished", "repeat normalization preserves restoration"])
    write_json(RESULTS/"verification_followup.json",result)
    return result

def finish_report(metrics, changes, verified):
    records=read_rows(DATA/"annotations.jsonl")
    scope=json.loads((RESULTS/"followup_scope.json").read_text())
    score_status=[json.loads((RESULTS/f"scoring_full_{s}_status.json").read_text()) for s in (0,1)]
    q=metrics["Q_auto"]["all"]["Q_auto"]
    a=metrics["A_by_split"]["test"]["all"]
    stat_edits=[r for r in read_rows(DATA/"edits_full.jsonl") if r["role"]=="statistical"]
    if metrics["A"]["dependent"]["n"]!=q["numerator"]:
        raise ValueError("Q and masking dependency groups still differ")
    summary=dict(automatic_pilot=True,prior_revision=CONFIG["prior_revision"],target_nodes=scope["target_nodes"],
        review_failures=verified["review_failures"],input_changes=changes,
        reused_scores=sum(r["reused"] for r in score_status),new_inference=sum(r["inferred"] for r in score_status),
        restored_statistical_nodes=sum(r["changes"]["restoration_available"] for r in stat_edits),
        matched_irrelevant_statistical_nodes=sum(r["changes"]["matched_irrelevant_available"] for r in stat_edits),
        Q_auto=q,masking_dependency_nodes=metrics["A"]["dependent"]["n"],
        complete_scores=metrics["complete_score_count"],failed_scores=metrics["failed_score_count"],
        annotation_label_changes=sum(r.get("semantic_review_version")==3 and
            r.get("previous_source_semantic_dependency") is not r["annotation"].get("source_semantic_dependency") for r in records))
    write_json(RESULTS/"followup_summary.json",summary)
    p=RESULTS/"REPORT_full.zh.md"
    content=p.read_text()
    intro=[
        "本轮为定点补跑，保留原样本与病例标签。此前补回列表遗漏、依赖标注错配及分组口径不一致已按本轮程序重算。",
        f"重审 {scope['target_nodes']} 个此前为是／未知或存在补回冲突的节点；自动标注失败 {verified['review_failures']} 个，失败者记为未知。",
        f"完整输入改变 {changes['changed_or_new_inputs']} 条，实际新增推理 {summary['new_inference']} 条，其余评分按完整渲染输入和模型设置复用。",
        f"379个统计节点中，必要定义补回可用 {summary['restored_statistical_nodes']} 个，无关补回对照可用 {summary['matched_irrelevant_statistical_nodes']} 个。",
        "补回由确认后的定义原文和实际删除位置自动导出；若同一定义仍有保留副本，则不计为定义丢失。补回文本只来自原文，不包含病例答案。",
        f"Q的依赖正例与遮挡依赖分组均为 {q['numerator']} 个；缺少原文证据或仍有未解决引用的节点均归未知。",
        "“原文依赖定义”与“本次删除丢失定义”仍分别统计，不以是否出现正例决定是否接续实验。",
        "关键事实及融合保留原方案；参数仍只用开发来源选择。详细变更见input_changes_vs_prior.jsonl及followup_summary.json。",
        "主指标P_b为每节点一对跨阈值病例中至少一例判错的比例；单例错误率还包含不跨阈值病例。",
        "",
        "| 测试集比较 | 配对数 | 修改前准确率 | 修改后准确率 |",
        "|---|---:|---:|---:|",
    ]
    for label,key in [("完整→直接遮挡","full_vs_direct"),("直接遮挡→必要定义补回","necessary_recovery"),
                      ("直接遮挡→无关说明补回","irrelevant_recovery"),("无关补回→必要补回","necessary_vs_irrelevant")]:
        row=a[key]
        fmt=lambda v:"不可估计" if v is None else f"{100*v:.2f}%"
        intro.append(f"| {label} | {row['n']} | {fmt(row['first_accuracy'])} | {fmt(row['second_accuracy'])} |")
    if a["necessary_recovery"]["n"]==0:
        intro.append("\n本轮测试集中仍无可用的定义丢失配对；该比较无法估计，不能推断不存在其他依赖。")
    p.write_text("# 定点补跑结果\n\n"+"\n".join(intro)+"\n\n"+content)
    return summary

def export_results():
    archive=ROOT/"exports"/"followup_results_20261105.zip"
    files=[]
    for folder in ("data","results","logs","tests"):
        files.extend(p for p in (ROOT/folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix!=".lock")
    files.extend(p for p in ROOT.iterdir() if p.is_file() and p.suffix in (".py",".json",".md"))
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files):
            z.write(p,str(p.relative_to(ROOT)))

def run(skip_tests=False):
    started=time.time()
    try:
        update("cpu_checks",stage="followup")
        if not skip_tests:
            with (LOGS/"tests_controller.log").open("w") as log:
                subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-v"],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        pending=prepare_review()
        if pending:
            pair("review_semantics.py","annotation")
        update("building_inputs",stage="full")
        from revision import build
        build("full")
        verify_inputs("full")
        verified=verify_followup()
        changes=input_differences()
        pair("score.py","scoring")
        update("analyzing",stage="full")
        from analyze_revision import analyze
        metrics=analyze("full")
        if metrics["failed_score_count"]:
            raise RuntimeError("Missing or failed scores; inspect missing_scores_full.jsonl")
        summary=finish_report(metrics,changes,verified)
        write_json(RESULTS/"run_full_status.json",dict(status="complete",started_at=started,
            finished_at=time.time(),elapsed_seconds=time.time()-started,
            complete_score_count=metrics["complete_score_count"],failed_score_count=0))
        update("complete",stage="full",elapsed_seconds=time.time()-started,
            report=str(RESULTS/"REPORT_full.zh.md"),new_inference=summary["new_inference"])
        export_results()
    except Exception as exc:
        traceback.print_exc()
        update("failed",error=str(exc),elapsed_seconds=time.time()-started)
        raise

if __name__=="__main__":
    run(skip_tests="--skip-tests" in sys.argv)
