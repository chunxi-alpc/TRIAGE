"""Read existing M1 artifacts; never load a model, fit M2/M3, or choose a new rule."""
from __future__ import annotations
import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import subprocess

KINDS = ("A", "P", "C", "F")
DEFAULT = dict(gradient_threshold=2.0, candidate_frequency=0.8, fdr=0.05,
               direction_consistency=0.8, min_valid_sources=5)
GRID = dict(gradient_threshold=[1.5,2.0,2.5,3.0], candidate_frequency=[0.6,0.8,1.0],
            fdr=[0.01,0.05,0.10], direction_consistency=[0.7,0.8,0.9])

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def lines(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n",encoding="utf-8")

def table(path, rows, compressed=False):
    rows=list(rows)
    if not rows:
        return
    opener=gzip.open if compressed else open
    with opener(path, "wt", encoding="utf-8", newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

def git(repo, *args):
    return subprocess.check_output(["git","-C",str(repo),*args],text=True).strip()

def jaccard(a,b):
    a,b=set(a),set(b)
    return len(a&b)/len(a|b) if a|b else None

def audit(repo, old_repo, data, out):
    if out.exists():
        raise FileExistsError("Use a new output directory; historical audit outputs are immutable")
    out.mkdir(parents=True)
    root=repo/"reports/aster_m1_revision_20261117"
    q=root/"models/qwen"
    plan=read(q/"selection_plan.json")
    cfg=read(q/"effective_configuration.json")
    sources=read(root/"data_audit.json")
    pairs=list(lines(data/"data/E1_pairs.jsonl"))
    screening=set(plan["screening_sources"])
    confirmation=set(plan["confirmation_sources"])
    confirmation_pairs={p["pair_id"]:p for p in pairs if p["source_id"] in confirmation}
    by_pair=defaultdict(set)
    by_source=defaultdict(lambda: [0,0.,0.,0.,0,0,0])
    category_rows=Counter()
    errors=[]
    expected={k:set(v["heads"]) for k,v in plan["candidates"].items()}
    numerical_failures=0
    schema=None
    patch_path=q/"models/E1/full/paired_effects.jsonl.gz"
    with gzip.open(patch_path,"rt",encoding="utf-8") as f:
        for line in f:
            r=json.loads(line)
            if schema is None: schema=sorted(r)
            pid=r["pair_id"]
            if pid not in confirmation_pairs:
                errors.append("unexpected_pair:"+pid)
                continue
            p=confirmation_pairs[pid]
            key=(r["kind"],r["source_id"],int(r["head"]))
            if any(r[k]!=p[k] for k in ["kind","source_id","left_label","right_label"]):
                errors.append("metadata_mismatch:"+pid)
            if r["head"] in by_pair[pid]: errors.append("duplicate_pair_head:"+pid)
            by_pair[pid].add(r["head"])
            category_rows[r["kind"]]+=1
            z=by_source[key]
            z[0]+=1
            z[1]+=r["magnitude"]
            z[2]+=r["c"]
            z[3]+=r["label_aligned"]
            z[4]+=int(r["c"]>0)
            z[5]+=int(r["label_aligned"]>0)
            z[6]+=int(r["label_aligned"]<0)
            numerical_failures+=int(r["identity_error_left"]>r["tolerance_left"] or
                                    r["identity_error_right"]>r["tolerance_right"])
    for pid,p in confirmation_pairs.items():
        if by_pair[pid]!=expected[p["kind"]]:
            errors.append("incomplete_or_unexpected_heads:"+pid)
    source_rows=[dict(kind=k,source_id=s,head=h,pairs=v[0],magnitude_mean=v[1]/v[0],
                      signed_mean=v[2]/v[0],label_aligned_mean=v[3]/v[0],
                      positive_signed_pairs=v[4],positive_aligned_pairs=v[5],
                      negative_aligned_pairs=v[6],
                      matched_null_mean=None)
                 for (k,s,h),v in sorted(by_source.items())]
    table(out/"existing_source_head_effects.csv.gz",source_rows,True)
    coverage=[]
    for k in KINDS:
        ps=[p for p in confirmation_pairs.values() if p["kind"]==k]
        coverage.append(dict(kind=k,candidate_heads=len(expected[k]),unmeasured_heads=1152-len(expected[k]),
                             pairs=len(ps),sources=len({p["source_id"] for p in ps}),
                             observed_patch_rows=category_rows[k],expected_patch_rows=len(ps)*len(expected[k]),
                             matched_null_rows=0))
    table(out/"cache_coverage.csv",coverage)
    existing={}
    for name,relative,mass in [("retention_80","sensitivity_80",.8),("retention_90","E1",.9),
                               ("retention_95","sensitivity_95",.95)]:
        d=read(q/("models/"+relative+"/full/m1.json"))
        if "by_kind" not in d:
            raise ValueError("Unexpected historical selection schema")
        sets={k:set(d["by_kind"][k]) for k in KINDS}
        n,e=sets["A"]|sets["P"],sets["C"]|sets["F"]
        existing[name]=dict(retained_mass=mass,by_kind={k:sorted(v) for k,v in sets.items()},
                            nuisance=sorted(n),evidence=sorted(e),shared=sorted(n&e),union=sorted(n|e))
    table(out/"existing_selection_counts.csv",
          [dict(configuration=k,retained_mass=v["retained_mass"],**{c:len(v["by_kind"][c]) for c in KINDS},
                AP=len(v["nuisance"]),CF=len(v["evidence"]),overlap=len(v["shared"]),union=len(v["union"]))
           for k,v in existing.items()])
    table(out/"existing_selection_jaccard.csv",
          [dict(configuration_a=a,configuration_b=b,union_jaccard=jaccard(x["union"],y["union"]),
                AP_jaccard=jaccard(x["nuisance"],y["nuisance"]),
                CF_jaccard=jaccard(x["evidence"],y["evidence"]))
           for a,x in existing.items() for b,y in existing.items()])
    write(out/"existing_head_sets.json",existing)
    old=old_repo/"reports/aster_e678_adaptive_20261116/models/qwen/models/E1/full"
    oldplan=read(old/"m1_plan.json")
    oldpairs={p["pair_id"]:p for p in oldplan["dev_pairs"]}
    donors=read(old/"donors.json")
    donor_stats=Counter()
    for group in donors:
        target=oldpairs[group["target_pair"]]
        for d in group["donors"]:
            donor=oldpairs[d["pair_id"]]
            donor_stats["comparisons"]+=1
            donor_stats["same_source"]+=int(target["source_id"]==donor["source_id"])
            donor_stats["same_category"]+=int(target["kind"]==donor["kind"])
            donor_stats["same_label_transition"]+=int(
                (target["left_label"],target["right_label"])==(donor["left_label"],donor["right_label"]))
    old_effects=list(lines(old/"source_effects.jsonl"))
    oldaudit=dict(commit=git(old_repo,"rev-parse","HEAD"),configuration=oldplan["config"],
                  effect_records=len(old_effects),
                  sources=sorted({r["source_id"] for r in old_effects}),donor_matching=dict(donor_stats),
                  split="dev",estimand="mean absolute endpoint perturbation",
                  reusable_as_current_training_matched_null=False,
                  reasons=["development sources","cross-source donors","label transitions not explicitly matched",
                           "different intervention estimand and candidate universe"])
    write(out/"historical_null_audit.json",oldaudit)
    configs=[dict(name="default",**DEFAULT)]
    for parameter,values in GRID.items():
        for value in values:
            if value!=DEFAULT[parameter]:
                configs.append(dict(name=parameter+"_"+str(value),**{**DEFAULT,parameter:value}))
    for c in configs:
        c.update(status="pending_protocol_confirmation",
                 reason="requested statistical selector differs from the executed cumulative-mass selector")
    write(out/"requested_configurations.json",dict(default=DEFAULT,one_factor=configs,
               joint_configurations="pending protocol confirmation; no outcome-driven selection"))
    cache=Path(cfg["private_runtime"])/"cache"
    fit_scores=list(lines(cache/"scores_fit.jsonl"))
    result=dict(audited_at=datetime.now(timezone.utc).isoformat(),model=cfg["model"],
                source_commit=git(repo,"log","-1","--format=%H","--",str(patch_path.relative_to(repo))),
                audit_base_commit=git(Path.cwd(),"rev-parse","HEAD"),
                current_selector=cfg["M1"],requested_selector=DEFAULT,protocol_conflict=True,
                complete_requested_sensitivity=False,new_model_forwards=0,new_activation_patches=0,
                m2_fits=0,m3_fits=0,m4_modified=False,
                fit_pairs=len(pairs),source_counts=sources["source_counts"],
                screening_sources=sorted(screening),confirmation_sources=sorted(confirmation),
                screening_confirmation_disjoint=not bool(screening&confirmation),
                fit_dev_test_disjoint=all(not(set(sources["sources"][a])&set(sources["sources"][b]))
                    for a,b in [("fit","dev"),("fit","test"),("dev","test")]),
                patch_rows=sum(category_rows.values()),paired_count=len(by_pair),
                patch_schema=schema,coverage_complete=not errors,coverage_errors=errors,
                recorded_self_patch_failures=numerical_failures,
                fit_cached_inputs=len(fit_scores),fit_cached_gradient_inputs=sum(bool(r.get("has_grads")) for r in fit_scores),
                feature_coverage_evidence=read(q/"gradient_completion.json"),
                matched_training_null_available=False,
                no_dev_or_test_task_accuracy_read=True,
                legacy_dev_interventions_used_for_selection=False,
                existing_retention_results="reused previously completed 80/90/95 selections",
                requested_rule_recommendation="not assessable until protocol/version conflict is resolved")
    write(out/"audit.json",result)
    return {k:result[k] for k in ["model","patch_rows","paired_count","coverage_complete",
                                  "fit_cached_inputs","fit_cached_gradient_inputs","protocol_conflict"]}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--old-repo",type=Path,required=True)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(audit(a.repo,a.old_repo,a.data,a.out),ensure_ascii=False))
