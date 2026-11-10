"""Finalize only a completed experiment; preserve existing work outside this directory."""
import os,sys,json,time,subprocess,traceback
from pathlib import Path
H=Path(__file__).resolve().parent;REPO=H.parents[3]
def js(p):return json.loads(Path(p).read_text()) if Path(p).exists() else {}
def run(*args):return subprocess.check_output(args,cwd=REPO,text=True,stderr=subprocess.STDOUT).strip()
def main():
 while True:
  s=js(H/"pipeline_status.json")
  if s.get("phase")=="failed":raise RuntimeError("Experiment pipeline failed: "+s.get("error",""))
  if s.get("phase")=="complete":break
  time.sleep(20)
 if js(H/"training/api_validation.json").get("status")!="passed":raise RuntimeError("Parameter replay is incomplete")
 if not (H/"REPORT.zh.md").exists() or not (H/"evaluation/final/coverage.json").exists():raise RuntimeError("Final outputs are incomplete")
 rel=str(H.relative_to(REPO))
 run("git","add","--",rel)
 changed=run("git","diff","--cached","--name-only").splitlines()
 if any(not p.startswith(rel+"/") for p in changed):raise RuntimeError("Staged changes outside requested directory")
 if changed:run("git","commit","--quiet","-m","Evaluate trace-based judgment repair on locked sources")
 run("git","fetch","origin","aster-cr-20261108")
 # This worktree began on origin; our only new paths are the follow-up experiment.
 run("git","merge","--no-edit","origin/aster-cr-20261108")
 run("git","push","origin","HEAD:aster-cr-20261108")
 commit=run("git","rev-parse","HEAD")
 p=Path("/home/gcx/aster_trace_repair_20261110/publication.json")
 p.write_text(json.dumps(dict(state="published",commit=commit,commit_url="https://github.com/chunxi-alpc/TRIAGE/commit/"+commit,
  report_url="https://github.com/chunxi-alpc/TRIAGE/blob/"+commit+"/"+rel+"/REPORT.zh.md",time=time.strftime("%Y-%m-%d %H:%M:%S")),indent=2)+"\n")
if __name__=="__main__":
 try:main()
 except Exception as e:
  Path("/home/gcx/aster_trace_repair_20261110/publication_failed.json").write_text(json.dumps(dict(error=str(e),traceback=traceback.format_exc()),indent=2)+"\n")
  raise

