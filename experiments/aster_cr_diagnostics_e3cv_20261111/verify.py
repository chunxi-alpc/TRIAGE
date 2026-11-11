"""Independent parameter-to-score replay for the saved ablation predictions."""
import numpy as np
from .common import *
from .score_variant import parameters,score_variant
def main():
    store=Store(BASE);intercepts=read(OUT/"independent_intercepts.json");reports=[]
    for p in sorted((OUT/"evaluation").glob("*/scores_by_variant.npz")):
        scope=p.parent.name;marker=read(p.parent/"score_manifest.json");fold=marker["fold"]
        models=parameters(fold);b0=intercepts[fold]["b0"];maximum=0.;count=0
        with np.load(p) as arrays:
            ids=arrays["view_ids"]
            indices=np.unique(np.linspace(0,len(ids)-1,min(24,len(ids)),dtype=int))
            for i in indices:
                key=str(ids[i]);ell=store.ell(key)
                for g,model in models.items():
                    vector=store.heads(key,model["heads"])
                    full=ell+float(vector@model["alpha"])+model["offset"]
                    for variant in ("pair_only","original_intercept_only","full"):
                        measured=float(score_variant(ell,full,model["b"],b0,variant))
                        error=abs(measured-float(arrays[g+"_"+variant][i]))
                        maximum=max(maximum,error);count+=1
                    assert abs(float(arrays["independent_intercept"][i])-(ell+b0))<1e-10
        assert maximum<1e-8,(scope,maximum)
        reports.append(dict(scope=scope,fold=fold,recomputed_scores=count,max_absolute_error=maximum))
    write(OUT/"independent_score_verification.json",dict(complete=True,checks=reports))
    print("Independent score replay passed",sum(r["recomputed_scores"] for r in reports))
if __name__=="__main__":main()
