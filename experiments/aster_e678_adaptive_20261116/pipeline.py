"""Common adaptive fit entry, shared by training.fit_group and the new E6 runner."""
from pathlib import Path
from .adaptive_m1 import ScopedStore, pair_ids, representatives, select, Config
from experiments.aster_cr_retrain_paired_v2_20261111 import training
from experiments.aster_cr_retrain_paired_v2_20261111.common import write

def fit_group(group, pairs, features, views, natural_views, outdir, *, dev_pairs,
              dev_features, patcher, total_heads, heads_per_layer, config=Config()):
    if group != "E1":
        raise ValueError("Adaptive experiment is defined on E1")
    representatives(pairs, "fit")
    representatives(dev_pairs, "dev")
    fit_sources = {p["source_id"] for p in pairs}
    if fit_sources & {p["source_id"] for p in dev_pairs}:
        raise ValueError("Training/development source overlap")
    if any(v.get("split") != "fit" or v["source_id"] not in fit_sources for v in [*views,*natural_views]):
        raise ValueError("M2/M3/calibration accept only training-source records")
    store = ScopedStore(features, pair_ids(pairs) | {v["view_id"] for v in natural_views})
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    heads, m1 = select(pairs, dev_pairs, store, dev_features, patcher, outdir,
                       total_heads, heads_per_layer, config)
    summary = dict(group=group, M1=m1, heads=heads, status=m1["status"],
                   training_sources=sorted(fit_sources), train_sources=sorted(fit_sources),
                   training_pair_count=len(pairs), delta=.2923349, regularization=.01, rank=16,
                   fresh_modules=["M1"], complete=False)
    if m1["status"] == "selection_incomplete":
        write(outdir/"model.json",summary)
        return summary
    support, m2 = training._m2(group,heads,pairs,store,views,outdir)
    m3 = training.refit_m3(group,pairs,store,natural_views,support,outdir,
                           delta=.2923349,regularization=.01)
    summary.update(M2=m2,M3=m3,complete=True,fresh_modules=["M1","M2","M3"])
    write(outdir/"model.json",summary)
    return summary
