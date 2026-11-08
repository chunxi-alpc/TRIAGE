"""CPU synthetic validation of evidence-constrained paired score cancellation.

This is generated-data training, not a language-model benchmark.
Run: python mixed_trace_control.py
"""
from pathlib import Path
import csv
import json
import time
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize, LinearConstraint

OUT = Path(__file__).resolve().parent
SETTINGS = {
    "seeds": [11,22,33],
    "trace_dimension": 16,
    "trace_noise_sd": 0.03,
    "ridge_lambda_sum_loss": 0.01,
    "evidence_score_coefficient": 2.0,
    "nuisance_score_coefficient": 1.0,
    "constraint_budget_fraction_fit_ap_rms": 0.1,
    "sizes": {
        "fit": {"AP":5000,"CF":200},
        "dev": {"AP":1000,"CF":500},
        "test": {"AP":2000,"CF":1000},
    },
    "scenarios": {
        "mixed_separable": {"D":[1.0,0.0],"H":[0.8,1.0]},
        "mixed_inseparable": {"D":[1.0,0.0],"H":[1.0,0.0]},
    },
}

def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))

def make_split(rng, n_ap, n_cf, D, H):
    r = len(D)
    e = rng.choice([-1.0,1.0],size=n_ap)
    nua,nub = rng.normal(size=(2,n_ap))
    ta = nua[:,None]*D + e[:,None]*H + rng.normal(0,0.03,size=(n_ap,r))
    tb = nub[:,None]*D + e[:,None]*H + rng.normal(0,0.03,size=(n_ap,r))
    ap_x = tb-ta
    ap_y = nub-nua
    nu = rng.normal(size=n_cf)
    tc = nu[:,None]*D-H + rng.normal(0,0.03,size=(n_cf,r))
    tf = nu[:,None]*D+H + rng.normal(0,0.03,size=(n_cf,r))
    cf_x = tf-tc
    cf_y = np.full(n_cf,4.0)
    return {"AP_X":ap_x,"AP_y":ap_y,"CF_X":cf_x,"CF_y":cf_y}

def ridge(X,y):
    return np.linalg.solve(X.T@X+0.01*np.eye(X.shape[1]),-X.T@y)

def fit_constrained(X,y,B,budget):
    n=len(y)
    Q=(X.T@X+0.01*np.eye(X.shape[1]))/n
    q=X.T@y/n
    objective=lambda w: float(w@Q@w+2*q@w)
    jac=lambda w: 2*(Q@w+q)
    result=minimize(
        objective,np.zeros(X.shape[1]),jac=jac,
        method="SLSQP",
        constraints=[LinearConstraint(B,-budget,budget)],
        options={"ftol":1e-12,"maxiter":2000},
    )
    if not result.success:
        raise RuntimeError(str(result.message))
    violation=max(0.0,float(np.max(np.abs(B@result.x)))-budget)
    if violation>1e-7:
        raise RuntimeError("Constraint violation: "+str(violation))
    return result.x, {
        "success": bool(result.success),
        "iterations":int(result.nit),
        "max_fit_constraint_violation":violation,
        "message":str(result.message),
    }

def measure(data,w,budget,fit_ap_rms):
    ap=data["AP_y"]
    corr=data["AP_X"]@w
    residual=ap+corr
    cf=data["CF_y"]
    cf_corr=data["CF_X"]@w
    new_cf=cf+cf_corr
    return {
        "ap_raw_rms":rms(ap),
        "ap_corrected_rms":rms(residual),
        "ap_variance_removed_pct":float(100*(1-np.var(residual)/np.var(ap))),
        "cf_mean_evidence_response_retention_pct":float(100*np.mean(new_cf)/np.mean(cf)),
        "cf_correction_rms":rms(cf_corr),
        "cf_correction_rms_over_ap_raw_rms_pct":float(100*rms(cf_corr)/rms(ap)),
        "cf_max_absolute_correction":float(np.max(np.abs(cf_corr))),
        "cf_max_absolute_correction_over_budget":float(np.max(np.abs(cf_corr))/budget),
        "cf_fraction_within_fit_budget":float(np.mean(np.abs(cf_corr)<=budget+1e-7)),
        "cf_direction_reversals":int(np.count_nonzero(new_cf*cf<0)),
        "fit_budget_logit":float(budget),
        "fit_ap_raw_rms":float(fit_ap_rms),
    }

def main():
    start=time.perf_counter()
    rows=[]
    fits=[]
    for scenario,config in SETTINGS["scenarios"].items():
        D=np.zeros(16);H=np.zeros(16)
        D[:2]=config["D"];H[:2]=config["H"]
        for seed in SETTINGS["seeds"]:
            rng=np.random.default_rng(seed)
            data={name:make_split(rng,count["AP"],count["CF"],D,H)
                  for name,count in SETTINGS["sizes"].items()}
            X=data["fit"]["AP_X"];y=data["fit"]["AP_y"]
            B=data["fit"]["CF_X"]
            scale=rms(y)
            budget=0.1*scale
            base=ridge(X,y)
            constrained,solver=fit_constrained(X,y,B,budget)
            Z=null_space(H.reshape(1,-1))
            oracle=Z@ridge(X@Z,y)
            methods={
                "raw":np.zeros(16),
                "AP_ridge":base,
                "evidence_constrained_ridge":constrained,
                "oracle_H_nullspace_ridge":oracle,
            }
            fits.append({
                "scenario":scenario,"seed":seed,"fit_budget_logit":budget,
                "fit_cf_matrix_rank":int(np.linalg.matrix_rank(B)),
                "constrained_solver":solver,
                "weights":{name:w.tolist() for name,w in methods.items()},
                "analytical_evidence_coefficients":{name:float(H@w) for name,w in methods.items()},
                "analytical_nuisance_residual_coefficients":{name:float(1+D@w) for name,w in methods.items()},
            })
            for method,w in methods.items():
                for split,part in data.items():
                    rows.append({"scenario":scenario,"seed":seed,"method":method,"split":split,
                                 **measure(part,w,budget,scale)})
    columns=list(rows[0])
    with (OUT/"mixed_trace_results.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=columns)
        writer.writeheader();writer.writerows(rows)
    metrics=[k for k in rows[0] if k not in {"scenario","seed","method","split"}]
    summary=[]
    for scenario in SETTINGS["scenarios"]:
        for method in ["raw","AP_ridge","evidence_constrained_ridge","oracle_H_nullspace_ridge"]:
            subset=[r for r in rows if r["scenario"]==scenario and r["method"]==method and r["split"]=="test"]
            summary.append({"scenario":scenario,"method":method,"split":"test",
                            "n_seeds":len(subset),
                            **{key:{"mean":float(np.mean([r[key] for r in subset])),
                                    "sample_sd":float(np.std([r[key] for r in subset],ddof=1))}
                               for key in metrics}})
    result={
        "experiment_type":"Actual CPU fitting on synthetic data; not language-model results",
        "settings":SETTINGS,
        "formulas":{
            "trace":"t=D*nu+H*e+epsilon; epsilon independent N(0,0.03^2 I)",
            "score":"ell=2*e+nu",
            "AP":"same e, independent nu; pair difference is right minus left",
            "CF":"same nu, e:-1 -> +1, independently sampled trace noise",
            "objective":"sum((AP_X*w+AP_y)^2)+0.01*||w||^2",
            "constraint":"abs(CF_X*w)<=0.1*fit_AP_raw_RMS, individually on fit CF",
            "VR":"100*(1-var(AP_y+AP_X*w)/var(AP_y))",
            "ER":"100*mean(CF_y+CF_X*w)/mean(CF_y)",
            "oracle":"Known H, w constrained exactly by H^T*w=0; training still uses AP pairs only",
        },
        "interpretation":{
            "trace_noise":"Even oracle evidence-independent correction varies across noisy CF traces.",
            "fit_vs_test":"Per-pair hard bounds apply to fitting constraints; dev/test report out-of-sample violations.",
            "inseparable":"D=H means exact nuisance cancellation and exact evidence preservation are incompatible.",
            "delta_zero":"Fit CF matrices are full column rank with independent noise, so an exact zero empirical CF constraint permits w=0 only.",
            "uncertainty":"SD over three generated-data seeds, not LM run uncertainty",
        },
        "fits":fits,"test_summary":summary,
        "runtime_seconds":time.perf_counter()-start,
    }
    (OUT/"mixed_trace_metrics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    for item in summary:
        print(item["scenario"],item["method"],
              "VR",round(item["ap_variance_removed_pct"]["mean"],3),
              "ER",round(item["cf_mean_evidence_response_retention_pct"]["mean"],3),
              "CF/AP",round(item["cf_correction_rms_over_ap_raw_rms_pct"]["mean"],3),
              "max/budget",round(item["cf_max_absolute_correction_over_budget"]["mean"],3))
    print("runtime_seconds",round(result["runtime_seconds"],3))

if __name__=="__main__":
    main()
