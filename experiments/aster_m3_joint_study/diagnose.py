"""Post-fit linear-program bounds; no model fitting or candidate selection.

For each R1 natural-path first error, maximize its signed centered correction
over the ORIGINAL TRAINING CF constraints. The principal bound fixes b0 and
therefore isolates whether any allowed linear correction can cross its boundary.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from . import data, training as tr


def correction_bound(direction, cf, delta=tr.DELTA):
    result=linprog(-np.asarray(direction,float),A_ub=np.vstack([cf,-cf]),
                   b_ub=np.full(2*len(cf),delta),bounds=[(None,None)]*len(direction),method="highs")
    if result.status==3:
        return dict(status="unbounded",maximum=None,solver_status=int(result.status),max_constraint_violation=None)
    if not result.success:
        raise RuntimeError("Correction bound optimization failed: "+result.message)
    return dict(status="optimal",maximum=float(-result.fun),solver_status=int(result.status),
                max_constraint_violation=max(0.,float(np.max(np.abs(cf@result.x)))-delta))


def diagnose(output):
    output=Path(output)
    config=data.read(output/"configuration.json")
    pairs=data.rows(Path(config["historical_root"])/"data/E1_pairs.jsonl")
    first=[r for r in data.rows(output/"evaluation/natural_first_error.jsonl")
           if r["comparison"]=="J_independent minus R1" and r["before_first_error"] is not None
           and r["first_error_view_id"] is not None and r["gold_label"] is not None]
    records,fold_rows=[],[]
    for k in range(1,6):
        fold=f"fold{k}"
        model=data.read(output/"models"/fold/"J.json")
        allowed=set(model["train_sources"])
        cf=[p for p in pairs if p["source_id"] in allowed and p["kind"] in ("C","F")]
        with np.load(output/"models"/fold/"projected_cache.npz",allow_pickle=False) as z:
            ids=z["ids"].tolist();t=dict(zip(ids,z["traces"].copy()));ell=dict(zip(ids,z["ell"].tolist()))
        E=np.stack([t[p["right"]]-t[p["left"]] for p in cf])
        mu,w=np.asarray(model["mu"]),np.asarray(model["w"])
        active=np.abs(E@w)>=tr.DELTA-1e-6
        raw_direction=np.array([(p["right_label"]-p["left_label"])*(ell[p["right"]]-ell[p["left"]]) for p in cf])
        fold_rows.append(dict(fold=fold,training_CF_pairs=len(cf),active_constraints=int(active.sum()),
                             raw_wrong_direction_CF=int(np.sum(raw_direction<=0)),
                             active_raw_wrong_direction_CF=int(np.sum(active&(raw_direction<=0)))))
        for row in first:
            if not row["first_error_view_id"].startswith(fold+"::"):
                continue
            key=row["first_error_view_id"].split("::",1)[1]
            sign=2*int(row["gold_label"])-1
            bound=correction_bound(sign*(t[key]-mu),E)
            margin=row["before_gold_margin"]
            maximum=bound["maximum"]
            direction_norm=float(np.linalg.norm(t[key]-mu))
            fixed_shift=sign*(model["b_independent"]-model["b_shared"])
            required_ratio=(-margin/maximum) if maximum is not None and maximum>1e-12 else None
            records.append(dict(fold=fold,case_id=row["case_id"],source_id=row["source_id"],view_id=key,
                gold_label=row["gold_label"],R1_gold_margin=margin,
                actual_J_signed_correction=float(sign*(t[key]-mu)@w),
                actual_J_calibration_shift=float(fixed_shift),actual_J_gold_margin=row["after_gold_margin"],
                bound_status=bound["status"],bound_solver_status=bound["solver_status"],
                bound_constraint_violation=bound["max_constraint_violation"],target_direction_norm=direction_norm,
                max_signed_correction_shared_b0=maximum,
                max_gold_margin_shared_b0=None if maximum is None else margin+maximum,
                impossible_at_delta_shared_b0=bool(maximum is not None and margin+maximum < -1e-7),
                impossible_at_twice_delta_shared_b0=bool(maximum is not None and margin+2*maximum < -1e-7),
                minimum_budget_multiplier_for_this_node=required_ratio,
                max_gold_margin_with_current_fixed_bw=None if maximum is None else margin+fixed_shift+maximum))
    summary=dict(exploratory=True,models_refitted=False,records=len(records),
                 unique_first_error_inputs=len({r["fold"]+"::"+r["view_id"] for r in records}),
                 zero_trace_directions=sum(r["target_direction_norm"]<1e-10 for r in records),
                 finite_bounds=sum(r["bound_status"]=="optimal" for r in records),
                 impossible_at_delta_shared_b0=sum(r["impossible_at_delta_shared_b0"] for r in records),
                 impossible_at_twice_delta_shared_b0=sum(r["impossible_at_twice_delta_shared_b0"] for r in records),
                 newly_reachable_with_at_most_double_budget=sum(r["impossible_at_delta_shared_b0"] and not r["impossible_at_twice_delta_shared_b0"] for r in records),
                 folds=fold_rows,
                 interpretation="Per-node upper bound over all 16D linear weights under training CF constraints and fixed b0. Ignores all loss penalties, so an infeasible node cannot be repaired by any admissible weight at this fixed intercept. Feasible nodes need not be jointly repairable. The 2x bound follows exact homogeneity, not a trained budget scan. Recalibrating b can change the bound; current-bw numbers hold that bw fixed.")
    data.lines(output/"diagnostics/correction_bounds.jsonl",records)
    data.write(output/"diagnostics/correction_bounds_summary.json",summary)
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("output");a=p.parse_args()
    print(__import__("json").dumps(diagnose(a.output),ensure_ascii=False,indent=2))
