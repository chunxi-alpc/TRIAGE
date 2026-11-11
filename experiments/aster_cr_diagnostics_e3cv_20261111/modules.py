"""Saved-parameter diagnostics in the original head coordinate system."""
import itertools, traceback
import numpy as np
from .common import *
from .score_variant import parameters
from experiments.aster_cr_retrain_paired_v2_20261111 import evaluation as ev,training as tr

def principal_angles(a,b,basis):
    if a[basis] is None or b[basis] is None:return None
    union=sorted(set(a["heads"])|set(b["heads"]));index={int(h):i for i,h in enumerate(union)}
    qq=[]
    for model in (a,b):
        mat=model[basis]/model["scale"][:,None]
        embedded=np.zeros((len(union)*128,mat.shape[1]))
        for j,h in enumerate(model["heads"]):
            embedded[index[int(h)]*128:(index[int(h)]+1)*128]=mat[j*128:(j+1)*128]
        qq.append(np.linalg.qr(embedded,mode="reduced")[0])
    cos=np.linalg.svd(qq[0].T@qq[1],compute_uv=False)
    return dict(angles_degrees=np.degrees(np.arccos(np.clip(cos,0,1))).tolist(),
        mean_squared_cosine=float(np.mean(cos*cos)),maximum_cosine=float(cos.max()))

def main():
    store=Store(BASE);overlaps=[];margins=[];summaries=[];source_errors=[]
    for fold in ("full","fold1","fold2","fold3","fold4","fold5"):
        models=parameters(fold)
        for a,b in itertools.combinations(GROUPS,2):
            ma,mb=models[a],models[b];shared=len(set(ma["heads"])&set(mb["heads"]))
            row=dict(fold=fold,lhs=a,rhs=b,heads_shared=shared,head_overlap_fraction=shared/51,
                coordinates="common raw head-output coordinates, standardization scales included")
            for basis in ("S","U","N"):
                stats=principal_angles(ma,mb,basis);row[basis]=stats
            overlaps.append(row)
        for g,m in models.items():
            target=Path(m["directory"]);support=dict(np.load(target/"m2.npz"))
            coeff=dict(np.load(target/"m3.npz"));w=coeff["w"].reshape(-1)
            allowed=set(read(BASE/"folds.json")["all_sources"] if fold=="full" else
                read(BASE/"folds.json")["folds"][int(fold[-1])-1]["train_sources"])
            pairs=[p for p in ev.data_rows(BASE,g+"_pairs.jsonl")
                if p["source_id"] in allowed and p.get("split","fit")=="fit"]
            cache={}
            def coord(key):
                if key not in cache:cache[key]=tr.coordinates(key,store,support)
                return cache[key]
            local=[];source_rows=[]
            for p in pairs:
                corr=float((coord(p["right"])-coord(p["left"]))@w)
                raw=store.ell(p["right"])-store.ell(p["left"])
                source_rows.append(dict(source=p["source_id"],kind=p["kind"],raw=raw,correction=corr))
                if p["kind"] in ("C","F"):
                    local.append(dict(group=g,fold=fold,source_id=p["source_id"],pair_id=p["pair_id"],
                        kind=p["kind"],correction_difference=corr,margin=tr.DELTA-abs(corr)))
            mmin=min(r["margin"] for r in local)
            # Reused historical E0 retains its original arithmetic; mandatory bound check applies to new E3 fits.
            if g=="E3" and fold!="full":assert mmin>=-1e-7,(g,fold,mmin)
            margins.extend(local)
            summaries.append(dict(group=g,fold=fold,CF_pairs=len(local),minimum_margin=mmin,
                active_constraints=sum(abs(r["margin"])<=1e-6 for r in local),
                violations_above_1e7=sum(r["margin"] < -1e-7 for r in local),
                standardized_SN_overlap_fro=float(np.linalg.norm(support["S"].T@support["N"])),
                b=m["b"],w_norm=float(np.linalg.norm(w)),S_rank=support["S"].shape[1],N_rank=support["N"].shape[1]))
            for source in sorted({r["source"] for r in source_rows}):
                for kindset,name in [(("A","P"),"AP"),(("C","F"),"CF")]:
                    rr=[r for r in source_rows if r["source"]==source and r["kind"] in kindset]
                    raw=np.array([r["raw"] for r in rr]);c=np.array([r["correction"] for r in rr])
                    source_errors.append(dict(group=g,fold=fold,source_id=source,pair_group=name,n=len(rr),
                        raw_rms=float(np.sqrt(np.mean(raw*raw))),correction_rms=float(np.sqrt(np.mean(c*c))),
                        corrected_difference_rms=float(np.sqrt(np.mean((raw+c)**2)))))
    write(OUT/"m2_subspace_overlap.json",overlaps)
    ev.write_csv(OUT/"m3_constraint_margins.csv",margins)
    ev.write_csv(OUT/"module_summary.csv",summaries)
    ev.write_csv(OUT/"training_source_errors.csv",source_errors)
    write(OUT/"status/modules.json",dict(state="complete",models=len(summaries)))

if __name__=="__main__":
    try:main()
    except Exception as e:
        write(OUT/"status/failed_modules.json",dict(error=str(e),traceback=traceback.format_exc()));raise
