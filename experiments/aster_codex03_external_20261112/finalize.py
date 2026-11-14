"""Export only completed new-method predictions and statistics."""
import csv, fcntl, json, re, statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"reports/aster_codex03_external_20261112"
CORE=("ap_rms","ap_vr","cf_er","cf_correction_rms","cf_both_correct_accuracy",
      "natural_path_accuracy","interference_path_accuracy")
def csv_write(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    tmp=path.with_name(path.name+".tmp")
    with tmp.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys)
        if keys:w.writeheader();w.writerows(rows)
    tmp.replace(path)
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/".finalize.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        rows=[]
        for path in sorted((OUT/"evaluation/test").glob("*/section4_5.csv")):
            with path.open(encoding="utf-8-sig",newline="") as f:
                rows.extend(r for r in csv.DictReader(f)
                            if r.get("complete")=="True" and r.get("result_kind")=="actual_experiment")
        groups=defaultdict(list)
        for r in rows:groups[re.sub(r"-seed[0-9]+$","",r["method"])].append(r)
        finished_groups=[name for name,rr in groups.items() if not name.startswith(("DoRA", "RePS", "PSR")) or len(rr)==3]
        expected_methods={"CAA-E1-task-adapted","DoRA-E1-SFT","INLP-task-adapted",
                          "PSR-MSE-E1-task-adapted","RePS-E1-task-adapted"}
        all_complete=expected_methods.issubset(finished_groups)
        summary=[]
        for name,rr in sorted(groups.items()):
            row=dict(method=name,completed_runs=len(rr),runs=";".join(r["method"] for r in rr))
            for key in CORE:
                vals=[float(r[key]) for r in rr if r.get(key) not in (None,"")]
                row[key+"_mean"]=statistics.mean(vals) if vals else None
                row[key+"_seed_sd"]=statistics.stdev(vals) if len(vals)>1 else None
            summary.append(row)
        if rows:
            csv_write(OUT/"section4_5_new_methods.csv",rows)
            csv_write(OUT/"section4_5_method_summary.csv",summary)
        controllers={}
        for path in (OUT/"controllers").glob("gpu*/status.json"):
            controllers[path.parent.name]=json.loads(path.read_text())
        handoff_path=OUT/"scheduling_handoff.json"
        handoff_active=handoff_path.exists() and json.loads(handoff_path.read_text()).get("active",False)
        released=[]
        for key,v in controllers.items():
            if not handoff_active and v["stage"] in ("complete","failed") and all(j["status"] in ("complete","failed") for j in v["jobs"]):
                gpu=int(v["gpu"]);released.append(gpu)
                release=dict(complete=True,success=v["stage"]=="complete",released_gpus=[gpu],
                             no_gpu_jobs_inflight=True,controller_status=str(OUT/"controllers"/key/"status.json"))
                (OUT/"controllers"/key/"gpu_release.json").write_text(json.dumps(release,indent=2)+"\n")
        if not handoff_active and set(released)=={5,6}:
            (OUT/"gpu_release.json").write_text(json.dumps(dict(complete=True,released_gpus=[5,6],
                no_gpu_jobs_inflight=True,success=all(v["stage"]=="complete" for v in controllers.values())),indent=2)+"\n")
        state_path=OUT/"status.json";state=json.loads(state_path.read_text())
        state.update(status="experiments_complete" if all_complete else "unified_queue_handoff" if handoff_active else "experiments_running" if any(x["stage"]=="running" for x in controllers.values())
                     else "experiments_partially_completed",
                     experiments_started=bool(controllers),completed_new_methods=sorted(finished_groups),available_result_methods=sorted(groups),
                     completed_new_method_runs=[r["method"] for r in rows],
                     result_files=["section4_5_new_methods.csv","section4_5_method_summary.csv"] if rows else [],
                     controllers={k:dict(stage=v["stage"],current=v.get("current")) for k,v in controllers.items()})
        state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+"\n")
        lines=["# Codex 03 新增方法运行与结果","",
               "仅收录新方法真实完成的冻结评价；开发结果、smoke和CPU夹具不进入此表。","",
               "| GPU | 队列状态 | 当前任务 |","|---|---|---|"]
        for key,v in sorted(controllers.items()):
            lines.append("|"+key+"|"+v["stage"]+"|"+str(v.get("current") or "—")+"|")
        if all_complete:
            lines += ["", "本轮5种新增方法的11次正式实验和冻结评价全部完成。"]
        elif handoff_active:
            lines += ["", "后续任务已移交统一队列 /home/gcx/aster-gpu-queue-20261113/codex03_jobs.json；原GPU固定队列不再派发。"]
        if rows:
            lines+=["","完成的测试运行："+", ".join(r["method"] for r in rows)+"。",
                    "","逐运行数据：[section4_5_new_methods.csv](section4_5_new_methods.csv)。",
                    "跨种子均值与标准差：[section4_5_method_summary.csv](section4_5_method_summary.csv)。",
                    "各方法的逐例预测、训练配置、成本和来源级区间位于 evaluation/test/<method>/。"]
        else:lines+=["","当前还没有完成冻结评价的新方法，尚未生成正式主表。"]
        (OUT/"RUN_STATUS.zh.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
        if rows:
            doc=ROOT/"docs/ASTER/experiments.zh.md"
            content=doc.read_text(encoding="utf-8").splitlines()
            running=any(v["stage"]=="running" for v in controllers.values())
            stage="Codex 03新增方法全部完成" if all_complete else "后续任务转入统一队列" if handoff_active else "新增方法仍在运行" if running else "本轮队列结束"
            names=", ".join(sorted(groups))
            for i,line in enumerate(content):
                if line.startswith("| E5 |"):
                    content[i]="| E5 | Qwen3-8B完整外部基线比较 | **"+stage+"** | Codex 03已有"+str(len(rows))+"个新方法/种子完成冻结评价："+names+" | 逐例、成本和第4.5节数据见[运行报告](../../reports/aster_codex03_external_20261112/RUN_STATUS.zh.md) |"
            doc.write_text("\n".join(content)+"\n",encoding="utf-8")

if __name__=="__main__":main()
