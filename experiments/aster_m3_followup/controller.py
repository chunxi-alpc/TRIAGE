"""Three-card sequential phases; completion produces reports without human polling."""
import fcntl,subprocess,sys,traceback,time
from pathlib import Path
from .common import *
def active_workers(name):
    found={}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():continue
        try:
            args=(entry/"cmdline").read_bytes().decode().split("\0")
            if "experiments.aster_m3_followup.worker" not in args:continue
            if args[args.index("--phase")+1]!=name:continue
            if str(REPO) not in str((entry/"cwd").resolve()):continue
            found[int(args[args.index("--gpu")+1])]=int(entry.name)
        except (OSError,ValueError,IndexError):continue
    return found
def process_alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ",1)[1][0]!="Z"
    except OSError:return False
def phase(name,jobs):
    pending=[j for j in jobs if not (ROOT/"jobs"/f"{j['id']}.json").exists()
             or read(ROOT/"jobs"/f"{j['id']}.json").get("state")!="complete"]
    if not pending:return
    existing=active_workers(name)
    if not existing:write(ROOT/"queues"/f"{name}.json",dict(jobs=pending,next=0))
    status("pipeline",state="running",phase=name,pending=len(pending),gpus=[0,1,6])
    processes=[]
    for gpu in [0,1,6]:
        if gpu in existing:continue
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="4",TOKENIZERS_PARALLELISM="false")
        log=(ROOT/"logs"/f"{name}_{gpu}.log").open("a")
        process=subprocess.Popen([sys.executable,"-m","experiments.aster_m3_followup.worker","--phase",name,"--gpu",str(gpu)],
            cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
        processes.append((process,log))
    codes=[]
    for process,log in processes:codes.append(process.wait());log.close()
    while any(process_alive(pid) for pid in existing.values()):time.sleep(5)
    failed=[j["id"] for j in jobs if read(ROOT/"jobs"/f"{j['id']}.json").get("state")!="complete"]
    if any(codes) or failed:raise RuntimeError(f"Phase {name} failed: {codes}; incomplete jobs: {failed}")
def eval_job(name,dataset):return dict(id=f"{dataset}_{name}",kind="evaluate",name=name,dataset=dataset)
def main():
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/"logs").mkdir(exist_ok=True)
    with (ROOT/"controller.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        start=time.time()
        try:
            cfg=read(ROOT/"configuration.json")
            from .reuse_history import run as reuse_history
            reuse_history()
            if read(ROOT/"smoke.json").get("state")!="passed":raise RuntimeError("Functional smoke run has not passed")
            names=[v["name"] for v in cfg["variants"]]
            old=["full","aster","learned_reference","learned_self","bias"]
            first=[dict(id="train_paired_ns",kind="train",name="paired_ns")]
            first += [dict(id=f"p0_{s}",kind="p0",shard=s) for s in [0,1]]
            first += [dict(id="calibration",kind="calibration")]
            first += [dict(id="train_"+n,kind="train",name=n) for n in names if n!="paired_ns"]
            phase("training_and_p0",first)
            phase("pilot_old",[eval_job(n,"medguide_pilot") for n in old])
            from .report import run
            interim=run()
            pilot=interim["medguide_pilot"]["methods"]
            promoted=[n for n in ["learned_reference","paired_n","paired_s","paired_ns"]
                if n in pilot and pilot[n]["correct"]>pilot["full"]["correct"]]
            write(ROOT/"confirmation_decision.json",dict(run=bool(promoted),triggered_by=promoted,
                rule="frozen learned_reference or paired variant improves pilot count over full; exploratory trigger, no significance claim; all methods evaluated if triggered"))
            jobs=[eval_job(n,"mega") for n in names+["bias"]]
            jobs += [eval_job(n,"pairs") for n in ["full","aster","learned_reference","learned_self","bias"]+names]
            if promoted:jobs += [eval_job(n,"medguide_confirm") for n in old+names]
            phase("evaluation",jobs)
            final_report=run(final=True)
            if final_report["state"]!="complete":raise RuntimeError("Evaluation records are incomplete; see summary.json")
            status("pipeline",state="complete",elapsed_seconds=time.time()-start,confirmation_run=bool(promoted),
                report=str(ROOT/"REPORT.zh.md"))
            finish_note()
            try:
                from .publish import publish
                publish(final=True)
            except Exception as exc:
                status("publish",state="failed",error=str(exc),scientific_results_complete=True)
        except Exception as exc:
            status("pipeline",state="failed",error=str(exc),traceback=traceback.format_exc())
            try:
                from .report import run
                run(final=False)
            except Exception:pass
            raise
def finish_note():
    summary=read(ROOT/"summary.json")
    table=summary["mega"];c=table["comparisons"].get("paired_ns minus full")
    if c:
        lo,hi=c["source_bootstrap_95ci"]
        conclusion=f"新版 paired_ns 相对完整阅读纠正 {c['corrected']}、新增错误 {c['degraded']}，净变化 {c['net']:+d}；差值的来源 95% 区间 [{100*lo:+.3f}, {100*hi:+.3f}] 个百分点。"
        if lo<=0<=hi:conclusion+="当前区间包含零。"
    else:conclusion="请见完整结果表。"
    note="\n## P0–P3 后续实验完成\n\n"+conclusion+" 结果见 [后续报告](results/aster_m3_followup_20261107/REPORT.zh.md)。\n"
    p=REPO/"README.md";s=p.read_text()
    if "## P0–P3 后续实验完成" not in s:p.write_text(s+note)
    replacements={
        "P0–P3 后续实验已实现，训练与评测尚未完成":"P0–P3 后续实验已完成本次训练与评测",
        "P0–P3 后续新版已实现，训练与全量结果尚未完成":"P0–P3 后续新版已完成本次训练与评测",
        "后续代码和数据准备已完成，训练与全量评测尚未完成。本节是预先记录的比较，不提供未经运行的成绩。":"后续训练与本次评测已完成。本节保留预先记录的比较，实测结果及扩展集是否触发见后续报告。",
        "其代码已实现，训练与全量实验尚未完成":"其训练与本次评测已完成",
        "P0–P3 后续实验（另列，尚未完成）":"P0–P3 后续实验（已完成，另列）",
        "P0 已补算冻结预测的直接配对统计；固定金标准路径节点、8 例改变案例的 gate 与编辑量正在补评。":"P0 已完成冻结预测的直接配对统计、固定金标准路径节点及 8 例改变案例的 gate 与编辑量补评。",
        "P1–P3 的代码和数据准备已完成，训练与全量评测由两卡任务继续执行":"P1–P3 的训练与本次评测已完成"
    }
    for name in ["README.md","docs/ASTER/README.md","docs/ASTER/manuscript.zh.md","docs/ASTER/method-clarifications.zh.md"]:
        p=REPO/name;text=p.read_text()
        for old,new in replacements.items():text=text.replace(old,new)
        if name!="README.md":
            text+="\n### P0–P3 本次运行结果\n\n"+conclusion+" 详见 [后续报告](../../results/aster_m3_followup_20261107/REPORT.zh.md)。\n"
        p.write_text(text)
    write(ROOT/"completion.json",dict(state="complete",summary=conclusion,auto_publish=True))
if __name__=="__main__":main()
