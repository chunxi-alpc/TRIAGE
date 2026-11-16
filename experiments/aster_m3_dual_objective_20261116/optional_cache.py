"""Check required official paths using only cached nodes; stop at the first cache miss."""
import json
from pathlib import Path
import numpy as np
from experiments.aster_closed_form.data import canonical_guide
from experiments.aster_cr_retrain_paired_v2_20261111.common import write
from .run import OUT,CACHE
def main():
    root=Path("/home/gcx/aster-final-gap-completion-20261114/reports/aster_final_gap_completion_20261114/e6/qwen/evaluation/mega_official")
    inventory=Path("/home/gcx/aster_next_20261109/inventory/official_all_path_cases.jsonl")
    records={}
    for file in root.glob("shard*/nodes.jsonl"):
        with file.open() as f:
            for line in f:
                r=json.loads(line)
                if r.get("status")=="ok" and r.get("feature") and Path(r["feature"]).is_file():
                    records[r["id"]]=r
    m=json.loads((OUT/"models/dual/full/m3.json").read_text())
    support=dict(np.load(CACHE/"models/E1/full/m2.npz"))
    # The optional source already freezes this exact E1/full parameter directory.
    h=support["heads"].astype(int)
    alpha=(support["N"].astype(float)@np.array(m["w"]))/support["scale"]
    offset=m["b"]-m["mu_c"]-support["mu"]@alpha
    seen={};outcomes=[];missing=[]
    with inventory.open() as f:
        for line in f:
            case=json.loads(line);nodes=canonical_guide(case["guideline"])["nodes"]
            lookup={n["id"]:n for n in nodes};gid=nodes[0]["id"];actual=[]
            while gid in lookup:
                key=case["case_id"]+":"+gid
                if key not in records:
                    missing.append(dict(id=key,case_id=case["case_id"],source_id=case["source_id"],
                        reason="selected eta=0 path reaches a node without an activation/score cache"))
                    break
                if key not in seen:
                    r=records[key]
                    with np.load(r["feature"],allow_pickle=False) as z:
                        if not np.array_equal(z["head_ids"],h):
                            raise ValueError("Official cache selected heads differ from E1/full")
                        a=z["selected_heads"].astype(float).reshape(-1)
                        if not np.all(np.isfinite(a)): raise ValueError("Nonfinite optional activation")
                        seen[key]=r["ell"]+a@alpha+offset
                label=int(seen[key]>0);actual.append(dict(gid=gid,label=label))
                gid=lookup[gid]["yes" if label else "no"]
            if missing:break
            outcomes.append(dict(case_id=case["case_id"],source_id=case["source_id"],
                correct=actual==case["gold_path"] and gid==case["gold_leaf"],path=actual,leaf=gid))
    if not missing:
        raise RuntimeError("Official required coverage complete; supplement must be evaluated before finalizing")
    write(OUT/"optional_official_cache.json",dict(
        status="skipped_missing_required_node",cache_root=str(root),cases_file=str(inventory),
        cached_nodes=len(records),completed_paths_before_first_miss=len(outcomes),
        scored_cached_nodes=len(seen),missing_nodes=len(missing),missing=missing,
        reason="Actual selected-model traversal encountered a missing node; stopped without model execution",
        GPU_tasks=0,model_forward_count=0))
    print("optional official traversal stopped at missing required cached node",missing[0]["id"])
if __name__=="__main__":main()
