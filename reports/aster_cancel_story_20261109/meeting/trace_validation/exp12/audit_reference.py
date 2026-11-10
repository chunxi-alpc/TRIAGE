"""Recompute historical transmission/reference metrics from original cached records."""
import csv,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
ROOT=Path("/home/gcx/aster_cr_repair_v2_run_20261108")
patch=[json.loads(s) for s in (ROOT/"qwen/mechanisms/h2_dev.jsonl").read_text().splitlines() if s]
pp=list(csv.DictReader((REPO/"reports/aster_h2_alignment_20261108/pair_results.csv").open()))
pred={(r["method"],r["pair_id"]):r for r in pp if r["split"]=="dev"}
rr=[r for r in patch if r["kind"]=="A" and ("ridge_r16",r["pair_id"]) in pred]
y=np.array([r["total_effect"] for r in rr])
ie=np.array([r["leakage_by_rank"]["16"] for r in rr])
yp=np.array([float(pred["ridge_r16",r["pair_id"]]["predicted_nuisance_difference"]) for r in rr])
sst=float(np.sum((y-y.mean())**2))
rms=lambda x:float(np.sqrt(np.mean(x*x)))
with np.load(ROOT/"qwen/supports_r16.npz") as a:
 shape=dict(heads=int(len(a["heads"])),rank=int(a["N"].shape[1]),raw_dimension=int(len(a["mu"])))
report=dict(
 data_origin="real_cache_recomputation",
 patch_source=str(ROOT/"qwen/mechanisms/h2_dev.jsonl"),
 prediction_source=str(REPO/"reports/aster_h2_alignment_20261108/pair_results.csv"),
 calculation_source=str(REPO/"reports/aster_cancel_story_20261109/trace/summarize_trace.py"),
 n=len(rr),sources=len(set(r["source_id"] for r in rr)),support=shape,
 patch_rms=rms(ie),total_rms=rms(y),patch_rms_ratio=rms(ie)/rms(y),
 patch_as_prediction_r2=1-float(np.sum((y-ie)**2))/sst,
 coordinate_prediction_r2=1-float(np.sum((y-yp)**2))/sst,
 prediction_rmse=rms(y-yp),
 reference_definitions=dict(
  target="Original total logit difference ell(right)-ell(left), not the selected-head intervention effect",
  ridge="Fit A/P 4887 pairs, alpha=.01, fit_intercept=False; dev A1194 primary and AP1198 secondary",
  patch="One-way left-to-right trace-direction patch at the decision position",
  r2="1 - sum((target-prediction)^2)/sum((target-mean(target))^2)",
  rms_ratio="RMS(trace patch effect) / RMS(total effect)",
  intercept="No paired regression intercept. Independent node calibration b disappears in pair differences.",
  legacy_mechanisms="mechanisms.h2 uses Ridge(alpha=.01) with default fit_intercept=True; historical h2_alignment changes to no intercept",
  paired_fit="Paired_AP_only reproduces .40633266 on A. Paired_APF0 trains 4969 A/P/F0 pairs and produces .40925484 on A; these are distinct estimators.",
  normalization="Fit full views estimate coordinate mu/scale; fit A/P determines nuisance basis, fit C/F signal basis; nuisance orthogonalized against signal",
  archive="The .07859726 reference is already present in trace/metrics.json as transmission_r2; not solely an unarchived prose result"))
(HERE/"reference_audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")

