"""Server-only input and numerical artifact access for Table 2."""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from experiments.aster_cr_retrain_paired_v2_20261111.common import load_inputs, read, rows, write

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports/aster_table2_20261123"
RUN = Path("/home/gcx/aster-table2-runtime-20261123")
SOURCE = Path("/home/gcx/aster-cr-retrain-paired-v2-20261111/reports/aster_cr_retrain_paired_v2_20261111")
RAW = Path("/home/gcx/aster-m1-revision-runtime-20261117/qwen")
RULES = Path("/home/gcx/aster_cr_repair_v2_run_20261108/data/rules.jsonl")
OLD_SCORES = Path("/home/gcx/aster-m1-sensitivity-runtime-20261117")

def datasets():
    fit = rows(SOURCE / "data/E1_pairs.jsonl")
    historical = rows(SOURCE / "data/historical_pairs.jsonl")
    return {"fit": fit, **{s: [p for p in historical if p["split"] == s and p["kind"] in ("A","P","C","F")] for s in ("dev","test")}}

class Features:
    def __init__(self, stages=("fit","dev","test")):
        self.records = {}
        for s in stages:
            for r in rows(RAW / "cache" / ("scores_" + s + ".jsonl")):
                self.records[r["uid"]] = r
        for p in sorted((RUN / "features").glob("scores_*.jsonl")):
            for r in rows(p):
                self.records[r["uid"]] = r
    def ell(self, uid):
        return float(self.records[uid]["ell"])
    @lru_cache(maxsize=128)
    def get(self, uid):
        r = self.records[uid]
        with np.load(r["feature"], allow_pickle=False) as a:
            out = {k:a[k].astype(np.float32).reshape(1152,128) for k in ("heads","grads") if k in a}
        out["ell"] = float(r["ell"])
        return out
    def heads(self, uid, selected=None):
        h = self.get(uid)["heads"]
        return h if selected is None else h[np.asarray(selected,dtype=int)].reshape(-1)

def node_type_map():
    out = {}
    for r in rows(RULES):
        p = r.get("predicate", {})
        op = p.get("op")
        if op in ("and","or","not"):
            label = "复合条件"
        elif op in (">",">=","<","<=") or (op == "==" and isinstance(p.get("value"),(int,float)) and not isinstance(p.get("value"),bool)):
            label = "数值条件"
        else:
            label = "类别或状态条件"
        out[r["group_id"]] = label
    return out
