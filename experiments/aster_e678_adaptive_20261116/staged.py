"""Two-stage adapter: identical statistics for synchronous and staged callers."""
from dataclasses import asdict
from pathlib import Path
from .adaptive_m1 import (Config, ScopedStore, pair_ids, representatives, screen_candidates,
                          measure, analyze)
from experiments.aster_cr_retrain_paired_v2_20261111.common import write, lines
from experiments.aster_cr_retrain_paired_v2_20261111.evaluation import write_csv

def make_plan(group, pairs, features, outdir, *, dev_pairs, total_heads, heads_per_layer,
              config=Config()):
    if group != "E1":
        raise ValueError("Adaptive mode requires E1")
    representatives(dev_pairs, "dev")
    if {p["source_id"] for p in pairs} & {p["source_id"] for p in dev_pairs}:
        raise ValueError("Training/development source overlap")
    screen = screen_candidates(pairs, ScopedStore(features, pair_ids(pairs)), config)
    if len(screen) != 4*total_heads:
        raise ValueError("Head geometry does not match screening")
    plan=dict(mode="adaptive", group=group, screen=screen, dev_pairs=dev_pairs,
              total_heads=total_heads, heads_per_layer=heads_per_layer, config=asdict(config))
    write(Path(outdir)/"m1_plan.json",plan)
    write(Path(outdir)/"screen.json",screen)
    write_csv(Path(outdir)/"screen.csv",screen)
    return plan

def finish_plan(plan, patcher, outdir):
    outdir=Path(outdir)
    cfg=Config(**plan["config"])
    effects,donors=measure(plan["dev_pairs"],plan["screen"],patcher,cfg)
    lines(outdir/"source_effects.jsonl",effects)
    write_csv(outdir/"source_effects.csv",effects)
    write(outdir/"donors.json",donors)
    result=analyze(plan["screen"],effects,sorted({p["source_id"] for p in plan["dev_pairs"]}),
                   plan["total_heads"],plan["heads_per_layer"],outdir,cfg)
    write(outdir/"m1.json",result)
    return result["heads"],result
