"""Record execution context without reading credentials or computing checksums."""
import importlib.metadata,json,platform,subprocess
from pathlib import Path
def main():
 root=Path("/home/gcx/TRIAGE-worktrees/aster-abc-diagnostics-20261030")
 out=Path("/home/gcx/aster_abc_diagnostics_20261030/d")
 packages={}
 for name in ["torch","transformers","tokenizers","numpy","scipy","safetensors"]:
  try:packages[name]=importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:packages[name]=None
 def command(args):return subprocess.check_output(args,cwd=root,text=True).strip()
 d={"python":platform.python_version(),"packages":packages,"branch":command(["git","branch","--show-current"]),
 "base_commit":command(["git","rev-parse","HEAD"]),"base_PR":42,
 "PR_status_observed":"open draft","PR_title":"ASTER：固定模型修订、临床确认与四卡官方测试",
 "changed_files_against_main":command(["git","diff","--name-only","origin/main...HEAD"]).splitlines(),
 "gpu_snapshot":command(["nvidia-smi","--query-gpu=index,name,memory.total,memory.used","--format=csv"]),
 "official_run":"/home/gcx/aster_manuscript_20261030/clinical01",
 "diagnostic_run":"/home/gcx/aster_abc_diagnostics_20261030",
 "read_chat_reference":"read_thread timed out; current code, data and the user's explicit specification used",
 "model_upstream_revision":"26028140be3ee69b82b1d1450179ab71bb1121b9","new_checksums_computed":False}
 (out/"reproduction_manifest.json").write_text(json.dumps(d,indent=2,ensure_ascii=False)+"\n")
 print(json.dumps({"packages":packages,"changed_files":len(d["changed_files_against_main"])}))
if __name__=="__main__": main()
