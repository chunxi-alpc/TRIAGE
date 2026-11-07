"""Publish completed M1 results to main using an isolated checkout and explicit paths."""
import gzip,json,os,shutil,subprocess,tempfile,time
from pathlib import Path
from .common import ROOT,REPO,PRECISION,MODELS,read,write

RESULT_PATH="results/m1_diagnostics_20261106"
START="<!-- m1-diagnostics:start -->"
END="<!-- m1-diagnostics:end -->"
BLOCK="""<!-- m1-diagnostics:start -->
## 冻结 M1 诊断与原文试测（2026-11-07，已完成）

两模型各完成64对头修补、64对层内外贡献诊断、12对残差扫描，以及MedGUIDE和MEGA-CDP各200例发布原文试测。

|基础模型|MedGUIDE正确/错误|MEGA完整路径正确/错误|MEGA原标注节点正确/总数|
|---|---|---|---|
|Qwen3-8B|103/97|193/7|1074/1081|
|MedGemma 27B|111/89|196/4|1077/1081|

两模型在MedGUIDE共同错71题，在MEGA共同错1例。冻结专用头的旧测试A/P恢复率分别为Qwen 84.21%/74.96%、MedGemma 74.17%/82.10%；区间、分母和精度见报告。结果支持M1定位部分分数变化通路；候选外贡献与组合不可加提示后续改进方向。本轮没有纠错训练，恢复率不能换算为准确率收益。原文200例是探索样本，不是全数据集成绩；完整前向与缓存前向的数值差异已单列。

[最终结论](results/m1_diagnostics_20261106/README.md) · [完整诊断表](results/m1_diagnostics_20261106/REPORT.zh.md) · [代码与复现](experiments/m1_diagnostics/README.zh.md)

实验完成后自动合并上传本仓库主分支。入口为 `python -m experiments.m1_diagnostics.publish`；完整实验的报告阶段已接入该入口。仅提交本实验代码、汇总和压缩明细，保留已有结果及README其他章节。上传失败会保留本地结果并写入状态文件，可重试；模型权重、激活缓存、原始数据和运行日志留在服务器。后续实验沿用这一完成即上传要求。
<!-- m1-diagnostics:end -->"""

def git(*args,cwd=REPO):
    return subprocess.check_output(["git",*args],cwd=cwd,stderr=subprocess.STDOUT,text=True).strip()

def compress_rows(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw,mode="wb",mtime=0,filename="") as z:
            for row in rows:z.write((json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n").encode())

def snapshot(checkout):
    code=checkout/"experiments/m1_diagnostics"
    code.mkdir(parents=True,exist_ok=True)
    for p in (REPO/"experiments/m1_diagnostics").iterdir():
        if p.suffix in [".py",".md"]:shutil.copy2(p,code/p.name)
    target=checkout/RESULT_PATH
    target.mkdir(parents=True,exist_ok=True)
    for src,dst in [("FINAL_ANALYSIS.zh.md","README.md"),("REPORT.zh.md","REPORT.zh.md"),("QWEN_ANALYSIS.zh.md","QWEN_ANALYSIS.zh.md"),
                    ("configuration.json","configuration.json"),("resource_plan.json","resource_plan.json")]:
        if (ROOT/src).exists():shutil.copy2(ROOT/src,target/dst)
    for name in ["patch_summary.json","baseline_summary.json","validation.json","completion.json","qwen_analysis.json"]:
        if (ROOT/"results"/name).exists():shutil.copy2(ROOT/"results"/name,target/name)
    for name in ["frozen_heads.json","diagnostic_pairs.jsonl","diagnostic_sampling.json","baseline_sampling.json","pair_distances.jsonl"]:
        dest=target/"data"/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/"data"/name,dest)
    for name in ["medguide_200.jsonl","mega_200.jsonl"]:
        samples=[{k:r[k] for k in ["sample_id","row_index","source_id"]} for r in
                 (json.loads(line) for line in (ROOT/"data"/name).read_text().splitlines())]
        write(target/"data"/name.replace("_200.jsonl","_sample_ids.json"),samples)
    (target/"cpu").mkdir(exist_ok=True)
    shutil.copy2(ROOT/"cpu/summary.json",target/"cpu/summary.json")
    compress_rows(target/"cpu/legacy_effects.jsonl.gz",(json.loads(line) for line in (ROOT/"cpu/legacy_effects.jsonl").read_text().splitlines()))
    for model in MODELS:
        records=[]
        for category in ["pairs","complements","residual"]:
            for p in sorted((ROOT/"patch"/model/PRECISION/category).glob("*.json")):
                records.append(dict(category=category,record=read(p)))
        compress_rows(target/(model+"_patch_details.jsonl.gz"),records)
        records=[]
        for dataset in ["medguide","mega"]:
            for p in sorted((ROOT/"baseline"/model/PRECISION/dataset).glob("*.json")):
                row=read(p)
                row.pop("prompt",None)
                records.append(row)
        compress_rows(target/(model+"_baseline_details.jsonl.gz"),records)
    p=checkout/"README.md";text=p.read_text()
    if START in text:
        first=text.index(START);last=text.index(END,first)+len(END)
        text=text[:first]+BLOCK+text[last:]
    else:
        first=text.find("\n## ")
        if first<0:first=len(text)
        text=text[:first]+"\n\n"+BLOCK+"\n"+text[first:]
    p.write_text(text)

def run():
    if not read(ROOT/"status/report.json").get("all_stages_complete"):
        raise RuntimeError("Publish requires completed experiment and report.")
    import fcntl
    with (ROOT/"publish.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        checkout=Path(tempfile.mkdtemp(prefix="m1-publish-",dir="/home/gcx"))
        added=False
        try:
            git("fetch","origin","main")
            git("worktree","add","--detach",str(checkout),"origin/main");added=True
            snapshot(checkout)
            git("add","--","README.md","experiments/m1_diagnostics",RESULT_PATH,cwd=checkout)
            changed=git("diff","--cached","--name-only",cwd=checkout)
            if changed:
                git("commit","-m","Publish frozen M1 diagnostics and original-text baselines with automatic result upload",cwd=checkout)
                for attempt in range(3):
                    try:
                        git("push","origin","HEAD:main",cwd=checkout)
                        break
                    except subprocess.CalledProcessError:
                        if attempt==2:raise
                        git("fetch","origin","main",cwd=checkout)
                        git("rebase","origin/main",cwd=checkout)
            commit=git("rev-parse","HEAD",cwd=checkout)
            git("fetch","origin","main",cwd=checkout)
            git("merge-base","--is-ancestor",commit,"origin/main",cwd=checkout)
            write(ROOT/"status/publish.json",dict(state="complete",commit=commit,remote="https://github.com/chunxi-alpc/TRIAGE",branch="main",updated=time.time()))
            return commit
        except Exception as exc:
            detail=exc.output if isinstance(exc,subprocess.CalledProcessError) else str(exc)
            write(ROOT/"status/publish.json",dict(state="failed",error=detail,updated=time.time()))
            raise
        finally:
            if added:git("worktree","remove",str(checkout))
            elif checkout.exists():checkout.rmdir()
if __name__=="__main__":print(run())
