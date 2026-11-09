"""Recompute archived transmission diagnostics from real per-pair caches."""
import json,csv
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
ROOT=Path("/home/gcx/aster_cr_repair_v2_run_20261108")
def rms(x):return float(np.sqrt(np.mean(np.square(x))))
def r2(y,p):return float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2))
def main():
 patch=[json.loads(s) for s in (ROOT/"qwen/mechanisms/h2_dev.jsonl").read_text().splitlines() if s]
 pred=list(csv.DictReader((REPO/"reports/aster_h2_alignment_20261108/pair_results.csv").open()))
 pred={r["pair_id"]:float(r["predicted_nuisance_difference"]) for r in pred if r["split"]=="dev" and r["method"]=="ridge_r16"}
 ps=[r for r in patch if r["kind"]=="A" and r["pair_id"] in pred]
 y=np.array([r["total_effect"] for r in ps]);ie=np.array([r["leakage_by_rank"]["16"] for r in ps]);yp=np.array([pred[r["pair_id"]] for r in ps])
 result=dict(data_kind="real_cache_recomputation",source_patch=str(ROOT/"qwen/mechanisms/h2_dev.jsonl"),
  source_prediction=str(REPO/"reports/aster_h2_alignment_20261108/pair_results.csv"),
  n=len(ps),heads=51,rank=16,transmission_rms_ratio=rms(ie)/rms(y),direct_patch_r2=r2(y,ie),
  trace_prediction_r2=r2(y,yp),trace_prediction_rmse=rms(y-yp),direct_patch_rms=rms(ie),target_rms=rms(y),
  archived_script="reports/aster_cancel_story_20261109/trace/summarize_trace.py",
  definitions=dict(prediction="Ridge alpha=0.01, fit_intercept=False; fit A/P=4887 pairs; dev A=1194. R2=1-SSE/sum((y-y.mean())**2). No extra trace feature standardization.",
    paired_correction="The APF0 correction is a different fit. A scalar node-calibration intercept cancels from pair differences.",
    direct_patch="0.0786 compares IE directly with TE using 1-sum((TE-IE)**2)/sum((TE-mean(TE))**2). There is NO fitted IE-to-TE regression and NO intercept. RMS ratio=RMS(IE)/RMS(TE).",
    intervention="One-way left-to-right patch at decision position. A trace predictive association and a direct patch effect are different quantities."))
 (HERE/"reference_audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
if __name__=="__main__":main()

