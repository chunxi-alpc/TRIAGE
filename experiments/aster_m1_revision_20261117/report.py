"""Preserve every outcome and report engineering and performance separately."""
import argparse
import fcntl
import json
import shutil
import subprocess
from pathlib import Path
from experiments.aster_cr_retrain_paired_v2_20261111.common import read,write
from .prepare import OUT,PRIVATE,REPO,KEYS
from .queue import QUEUE

def report(publish=False):
    queue=read(QUEUE/"status.json")
    records=[]
    body=["# ASTER 自适应 M1 修订：服务器执行报告","",
          "主配置使用反对称配对效应选头：候选累计比例 95%，确认保留比例 90%；80%/95% 为固定敏感性分析。",
          "M2 秩上限 16，M3 逐对预算 0.2923349、正则 0.01，截距由 fit 自然节点估计。","",
          "fit 为 80 来源、2,540 对，A/P/C/F 各 635 对；候选与确认来源各 40 个。",
          "开发结果标记为模型选择结果，40 来源评价标记为历史复用留出评价。","",
          "| 模型 | 状态 | 实际结果 |","|---|---|---|"]
    for key in KEYS:
        root=OUT/"models"/key
        done=read(root/"completion.json");failure=read(root/"failure.json")
        job=queue.get("jobs",{}).get("m1_revision_"+key,{})
        status=done.get("status",failure.get("status",job.get("state","not_started")))
        detail=failure.get("error","端到端评价完成" if done.get("complete") else
                           "等待 Qwen 端到端工程验收" if key!="qwen" else "等待队列")
        records.append(dict(model=key,status=status,detail=detail,complete=bool(done.get("complete"))))
        body.append("| "+key+" | "+status+" | "+detail.replace("|","/")+" |")
        log=Path(job.get("log",str(QUEUE/"logs"/("m1_revision_"+key+".log"))))
        if log.is_file():
            shutil.copyfile(log,root/"queue.log")
        preflight=read(root/"resource_preflight.json")
        if preflight:
            body += ["",f"## {key} 资源预检","",
                     f"完整候选为 {preflight['candidate_counts']}；确认阶段需要 {preflight['estimated_confirmation_forwards']:,} 次前向，上限为 {preflight['limit']:,} 次。"]
        for scope in ("dev","test"):
            metrics=read(root/"evaluation"/scope/"metrics.json")
            if metrics:
                body += ["",f"## {key} {scope} 点估计","",
                         "| 方法 | A/P RMS | A/P 方差降低率 | C/F 修正 RMS | 自然节点准确率 |",
                         "|---|---:|---:|---:|---:|"]
                for method,m in metrics["models"].items():
                    def fmt(name):return "NA" if m.get(name) is None else f"{m[name]:.6f}"
                    body.append("| "+method+" | "+" | ".join(fmt(n) for n in
                        ("ap_score_change_rms","ap_vr","cf_correction_rms","natural_node_accuracy"))+" |")
                full=metrics["models"]["adaptive_full"]
                base=metrics["models"]["frozen_backbone"]
                cal=metrics["models"]["independent_intercept"]
                def difference(name,ref):
                    return None if full.get(name) is None or ref.get(name) is None else full[name]-ref[name]
                body += ["", "完整修正相对原模型的自然节点准确率变化："+str(difference("natural_node_accuracy",base))+
                         "；相对独立截距的变化："+str(difference("natural_node_accuracy",cal))+"。",
                         "A/P 方差降低率为 "+str(full.get("ap_vr"))+
                         "，C/F 最大预算超额为 "+str(full.get("cf_max_budget_excess"))+"。"]
    complete=sum(r["complete"] for r in records)
    body += ["","## 完成情况","",f"七个检查点中 {complete} 个完成规定评价。其余状态按上表保留。",
             "软件合成测试与真实检查点实验分别记录。全部正负结果保留，主参数固定，未执行性能置信区间或固定 51 头重跑。",
             "原始输入、患者文本和特征缓存保存在服务器私有运行目录；本目录保存匿名数值及复现材料。","",
             "## 复现","",
             "环境：/home/gcx/gcx261014；服务器工作目录：/home/gcx/aster-m1-revision-20261117。",
             "CPU 测试：python -m unittest experiments.aster_m1_revision_20261117.test_revision experiments.aster_m1_revision_20261117.test_server -v。",
             "队列接口与完整命令见 queue_manifest.json；每个模型只由统一调度器启动。"]
    (OUT/"REPORT.zh.md").write_text("\n".join(body)+"\n",encoding="utf-8")
    write(OUT/"summary.json",dict(models=records,complete_models=complete,total_models=7))
    write(OUT/"queue_status.json",queue)
    pre=read(OUT/"models/qwen/resource_preflight.json")
    if pre:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(5.5,3.4))
        ax.bar(["Estimated forwards","Allowed ceiling"],[pre["estimated_confirmation_forwards"],pre["limit"]],color=["#4477aa","#cc6677"])
        ax.set_ylabel("Confirmation forwards")
        ax.ticklabel_format(style="plain",axis="y")
        for i,n in enumerate([pre["estimated_confirmation_forwards"],pre["limit"]]):ax.text(i,n,f"{n:,}",ha="center",va="bottom")
        ax.set_ylim(0,max(pre["estimated_confirmation_forwards"],pre["limit"])*1.18)
        fig.tight_layout();fig.savefig(OUT/"qwen_resource_preflight.png",dpi=180);plt.close(fig)
    if publish:
        PRIVATE.mkdir(parents=True,exist_ok=True)
        with (PRIVATE/"publish.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            subprocess.run(["git","add","-f","--","reports/aster_m1_revision_20261117"],cwd=REPO,check=True)
            if subprocess.run(["git","diff","--cached","--quiet"],cwd=REPO).returncode:
                subprocess.run(["git","commit","-m","reports(aster): preserve all M1 revision outcomes"],cwd=REPO,check=True)
            subprocess.run(["git","push","origin","HEAD:fix/aster-m1-estimand-20261117"],cwd=REPO,check=True)
    write(OUT/"summary_complete.json",dict(report_complete=True,complete_models=complete))
    return records

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--publish",action="store_true");a=p.parse_args()
    report(a.publish)
