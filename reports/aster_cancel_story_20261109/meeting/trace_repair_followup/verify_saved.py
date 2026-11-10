"""Replay saved parameters after completed real evaluation; no extra model forwards."""
import json,subprocess,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(H/"data/data_checks.py")],check=True)
subprocess.run([sys.executable,str(H/"training/replay_parameters.py")],check=True)
a=json.loads((H/"training/api_validation.json").read_text())
if a.get("status")!="passed":raise RuntimeError("Parameter replay failed")
coverage=json.loads((H/"evaluation/final/coverage.json").read_text())
frozen=json.loads((H/"evaluation/frozen.json").read_text())
if not frozen.get("frozen") or frozen.get("final_scores_used"):raise RuntimeError("Final freeze provenance missing")
if any(coverage["missing_predictions"].values()):raise RuntimeError("Incomplete final predictions")
(H/"replay_check.json").write_text(json.dumps(dict(status="passed",parameter_replay=a,final_sources=coverage["sources"],
 final_cases=coverage["cases"],final_tasks=coverage["tasks"],freeze=frozen["created_at"],
 protocol="Actual serialized-parameter replay plus complete predeclared final coverage; no new backbone forwards, no final tuning.",
 completed_at=time.strftime("%Y-%m-%d %H:%M:%S")),ensure_ascii=False,indent=2)+"\n")


subprocess.run([sys.executable,str(H/"report.py")],check=True)
