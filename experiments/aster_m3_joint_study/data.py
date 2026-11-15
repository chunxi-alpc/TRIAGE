"""Read-only access to archived E1 parameters, labels and activation caches.

Every path may be supplied explicitly. Old absolute paths are tried only when
they exist; --path-map OLD=NEW relocates cached feature references.
"""
from __future__ import annotations
import gzip
import json
from pathlib import Path
from functools import lru_cache
import numpy as np

from experiments.aster_cr_retrain_paired_v2_20261111 import evaluation as old
from .training import ProtocolConflict, check_sources, uid, DELTA

REPO = Path(__file__).resolve().parents[2]
E1_NAME = "aster_cr_retrain_paired_v2_20261111"
DIAGNOSTIC_NAME = "aster_cr_diagnostics_e3cv_20261111"


def read(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(s) for s in f if s.strip()]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def lines(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")


class MissingAssets(FileNotFoundError):
    def __init__(self, assets):
        self.assets = assets
        super().__init__("Required historical assets unavailable; see missing_assets.json")


def resolve_roots(e1_root=None, cases_file=None, diagnostic_root=None):
    candidates = [Path(e1_root)] if e1_root else [
        REPO/"reports"/E1_NAME,
        Path("/home/gcx/aster-cr-retrain-paired-v2-20261111/reports")/E1_NAME]
    found = next((p for p in candidates if (p/"data/E1_pairs.jsonl").exists()), candidates[0])
    config = read(found/"configuration.json") if (found/"configuration.json").exists() else {}
    cases = [Path(cases_file)] if cases_file else [
        found/"data/path_cases.jsonl",
        Path(config.get("historical_root", "/home/gcx/aster_cr_repair_v2_run_20261108"))/"data/path_cases.jsonl"]
    cp = next((p for p in cases if p.exists()), cases[0])
    diagnostics = [Path(diagnostic_root)] if diagnostic_root else [
        REPO/"reports"/DIAGNOSTIC_NAME,
        Path("/home/gcx/aster-cr-diagnostics-e3cv-20261111/reports")/DIAGNOSTIC_NAME]
    dp = next((p for p in diagnostics if (p/"independent_intercepts.json").exists()), diagnostics[0])
    required = [found/"folds.json", found/"configuration.json", cp,
                *[found/"data"/name for name in ("E1_pairs.jsonl", "E2_pairs.jsonl", "E2_views.jsonl",
                      "natural_views.jsonl", "path_features.jsonl", "path_interference_views.jsonl")],
                *[found/"models/E1"/f"fold{k}"/name for k in range(1,6)
                  for name in ("m1.json", "m2.npz", "m3.npz", "model.json")]]
    missing = [dict(path=str(p), reason="file not found") for p in required if not p.is_file()]
    if missing:
        missing.append(dict(searched_e1_roots=[str(p) for p in candidates],
                            searched_case_files=[str(p) for p in cases]))
        raise MissingAssets(missing)
    return found.resolve(), cp.resolve(), dp.resolve()


class CacheStore:
    """Selected heads only, with optional read-only mmap; gradients are unnecessary."""
    def __init__(self, root, path_maps=()):
        self.root = Path(root)
        self.maps = list(path_maps)
        self.records = {}
        files = [self.root/"data/reused_scores.jsonl", *sorted((self.root/"cache").glob("scores_*.jsonl"))]
        for path in files:
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("status", "ok") == "ok" and "ell" in row and "feature" in row:
                        self.records[uid(row)] = row
        if not self.records:
            raise MissingAssets([dict(path=str(p),reason="no usable cached score records") for p in files])
        runtime = self.root/"runtime"
        names = ["fit_heads.npy", "fit_present.npy", "fit_uids.json", "fit_sources.json"]
        self.runtime = None
        if all((runtime/n).exists() for n in names):
            self.runtime = dict(heads=np.load(runtime/"fit_heads.npy",mmap_mode="r"),
                                present=np.load(runtime/"fit_present.npy",mmap_mode="r"),
                                index={u:i for i,u in enumerate(read(runtime/"fit_uids.json"))},
                                sources=read(runtime/"fit_sources.json"))

    def feature_path(self, value):
        original = Path(value)
        if original.exists():
            return original
        for before, after in self.maps:
            try:
                tail = original.relative_to(Path(before))
            except ValueError:
                continue
            mapped = Path(after)/tail
            if mapped.exists():
                return mapped
        return original

    def ell(self, key):
        if key not in self.records:
            raise MissingAssets([dict(uid=key, reason="missing cached original score")])
        return float(self.records[key]["ell"])

    def heads(self, key, selected):
        if key not in self.records:
            raise MissingAssets([dict(uid=key, reason="missing cached feature record")])
        record = self.records[key]
        selected = np.asarray(selected, int)
        if self.runtime is not None:
            rt = self.runtime
            idx = rt["index"].get(key)
            if (idx is not None and idx < len(rt["sources"]) and
                rt["sources"][idx] == record["feature"] and int(rt["present"][idx]) & 1):
                h = np.asarray(rt["heads"][idx])
                h = h.reshape(-1, h.shape[-1])
                result = h[selected].astype(np.float32).reshape(-1)
                if np.all(np.isfinite(result)):
                    return result
        path = self.feature_path(record["feature"])
        if not path.is_file():
            raise MissingAssets([dict(uid=key,path=str(path),reason="cached feature file not found")])
        with np.load(path, allow_pickle=False) as f:
            h = np.asarray(f["heads"], np.float32)
            h = h.reshape(-1,h.shape[-1])
            if "head_ids" in f and h.shape[0] != 1152:
                lookup = {int(k):i for i,k in enumerate(f["head_ids"].reshape(-1))}
                absent = [int(k) for k in selected if int(k) not in lookup]
                if absent:
                    raise MissingAssets([dict(uid=key,path=str(path),missing_heads=absent,
                                              reason="required fold heads absent")])
                result = h[[lookup[int(k)] for k in selected]].reshape(-1)
            else:
                result = h[selected].reshape(-1)
        if not np.all(np.isfinite(result)):
            raise MissingAssets([dict(uid=key,path=str(path),reason="nonfinite selected cached heads")])
        return result


class StudyData:
    def __init__(self, root, cases_file, diagnostics, path_maps=()):
        self.root, self.diagnostics = Path(root), Path(diagnostics)
        self.config = read(self.root/"configuration.json")
        self.folds = read(self.root/"folds.json")
        self.natural = old.data_rows(self.root, "natural_views.jsonl")
        self.train_pairs = old.data_rows(self.root, "E1_pairs.jsonl")
        self.eval_pairs = old.data_rows(self.root, "E2_pairs.jsonl")
        self.eval_views = [dict(r,original=False) for r in old.data_rows(self.root,"E2_views.jsonl")]
        self.path_views = [r for name in ("path_features.jsonl","path_interference_views.jsonl")
                           for r in old.data_rows(self.root,name) if r.get("split")=="fit"]
        self.cases = [r for r in rows(cases_file) if r.get("split")=="fit"]
        self.check_protocol()
        self.store = CacheStore(self.root, path_maps)

    def check_protocol(self):
        c = self.config
        checks = dict(E1_pairs=len(self.train_pairs)==2540, natural_nodes=len(self.natural)==761,
                      heads=c.get("heads")==51, rank=c.get("rank")==16,
                      delta=abs(c.get("delta",0)-DELTA)<1e-12,
                      regularization=c.get("regularization")==0.01,
                      model="Qwen3-8B" in c.get("model", ""),
                      all_sources=len(self.folds["all_sources"])==80,
                      folds=len(self.folds["folds"])==5)
        if not all(checks.values()):
            raise ProtocolConflict("Formal E1 configuration conflict: "+str({k:v for k,v in checks.items() if not v}))
        all_sources = set(self.folds["all_sources"])
        if set(r["source_id"] for r in self.train_pairs) != all_sources:
            raise ProtocolConflict("E1 pair sources differ from official 80-source list")
        held = []
        for fold in self.folds["folds"]:
            tr, va = fold["train_sources"], fold["validation_sources"]
            check_sources(tr,va)
            if len(tr)!=64 or len(va)!=16 or set(tr)|set(va)!=all_sources:
                raise ProtocolConflict("Outer folds differ from the fixed 64/16 split")
            held.extend(va)
            meta = read(self.root/"models/E1"/f"fold{fold['fold']}"/"model.json")
            sources = meta.get("train_sources",meta.get("training_sources",[]))
            if set(sources)!=set(tr) or set(meta.get("validation_sources",[]))!=set(va):
                raise ProtocolConflict("Saved fold M1/M2 provenance differs from folds.json")
        if len(held)!=len(set(held)) or set(held)!=all_sources:
            raise ProtocolConflict("Outer validation sets do not partition 80 sources")

    def scope(self, sources):
        allow = set(sources)
        select = lambda rr:[r for r in rr if r["source_id"] in allow]
        return dict(views=select(self.eval_views)+select(self.natural), pairs=select(self.eval_pairs),
                    cases=select(self.cases), path_views=select(self.path_views))

    def parameters(self, fold):
        p = self.root/"models/E1"/fold
        with np.load(p/"m2.npz",allow_pickle=False) as f:
            m2 = {k:np.asarray(f[k]) for k in ("heads","mu","scale","N")}
        with np.load(p/"m3.npz",allow_pickle=False) as f:
            w,b = np.asarray(f["w"],float),float(f["b"])
            if "delta" in f and abs(float(f["delta"])-DELTA)>1e-12:
                raise ProtocolConflict("Saved M3 budget differs from fixed budget")
        m1 = read(p/"m1.json")
        if len(m2["heads"])!=51 or m2["N"].shape[1]!=16 or list(m2["heads"])!=m1["heads"]:
            raise ProtocolConflict("Fold head selection or trace dimension mismatch")
        return m2,w,b

    def coordinates(self, m2, ids, output_file):
        """Only project existing heads: no attention extraction or M2 refitting."""
        ids = sorted(set(ids))
        coordinates, ell, missing = {}, {}, []
        mu,scale,N = (np.asarray(m2[k],float) for k in ("mu","scale","N"))
        for key in ids:
            try:
                raw = np.asarray(self.store.heads(key,m2["heads"]),float)
                ell[key] = self.store.ell(key)
                coordinates[key] = ((raw-mu)/scale) @ N
            except MissingAssets as e:
                missing.extend(e.assets)
        if missing:
            raise MissingAssets(missing)
        np.savez_compressed(output_file, ids=np.array(ids), ell=np.array([ell[k] for k in ids]),
                            traces=np.stack([coordinates[k] for k in ids]))
        return ell,coordinates

    def verify_archived_scores(self, fold, r0, r1, r2, b0, natural_ids):
        """Checks saved score artifacts, not a second implementation of the same formula."""
        intercept_file = self.diagnostics/"independent_intercepts.json"
        if not intercept_file.exists():
            raise MissingAssets([dict(path=str(intercept_file),reason="archived R1 calibration required for replay test")])
        intercept = read(intercept_file)[fold]
        if set(intercept["calibration_ids"]) != set(natural_ids) or abs(intercept["b0"]-b0)>1e-8:
            raise ProtocolConflict("R1 calibration inputs or fitted intercept differ from archive")
        candidates = [self.diagnostics/"evaluation"/f"cv_{fold}_common"/"predictions.jsonl.gz",
                      self.root/"predictions"/f"cv_E1_{fold}_common_frozen_pool"/"predictions.jsonl.gz",
                      self.root/"evaluation"/f"cv_E1_{fold}_common_frozen_pool"/"predictions.jsonl"]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise MissingAssets([dict(path=str(p),reason="archived E1 score predictions required") for p in candidates])
        maxima = dict(R0=0.,R1=0.,R2=0.)
        compared = dict(R0=0,R1=0,R2=0)
        for row in rows(path):
            key = uid(row)
            if key not in r2:
                continue
            saved = row["scores"]
            values = [("R0",r0,saved.get("raw")),
                      ("R1",r1,saved.get("independent_intercept")),
                      ("R2",r2,saved.get("E1_full",saved.get("E1")))]
            for name,current,expected in values:
                if expected is not None:
                    maxima[name] = max(maxima[name],abs(current[key]-expected))
                    compared[name] += 1
        if compared["R2"] == 0 or max(maxima.values())>1e-8:
            raise ProtocolConflict("Archived E1 score replay mismatch: "+str(maxima))
        return dict(passed=True, archive=str(path), compared=compared, max_abs_difference=maxima,
                    intercept_difference=abs(intercept["b0"]-b0), tolerance=1e-8)

    def verify_archived_paths(self, fold, scope, scores):
        candidates = [self.diagnostics/"evaluation"/f"cv_{fold}_common"/"path_predictions.jsonl.gz",
                      self.root/"predictions"/f"cv_E1_{fold}_common_frozen_pool"/"path_predictions.jsonl.gz",
                      self.root/"evaluation"/f"cv_E1_{fold}_common_frozen_pool"/"path_predictions.jsonl"]
        path=next((p for p in candidates if p.exists()),None)
        if path is None:
            raise MissingAssets([dict(path=str(p),reason="archived dynamic paths required for replay test") for p in candidates])
        aliases={"raw":"R0","independent_intercept":"R1","E1_full":"R2_independent","E1":"R2_independent"}
        archived={(aliases[r["method"]],r["case_id"],r["variant"]):r for r in rows(path) if r["method"] in aliases}
        names={key[0] for key in archived}
        fresh=old.path_predictions(scope["cases"],scope["path_views"],{n:scores[n] for n in names})
        current={(r["method"],r["case_id"],r["variant"]):r for r in fresh}
        if current.keys()!=archived.keys():
            raise ProtocolConflict("Dynamic path case/variant denominator differs from archived evaluation")
        differences=[dict(method=k[0],case_id=k[1],variant=k[2]) for k in current
                     if any(current[k].get(field)!=archived[k].get(field)
                            for field in ("correct","path","leaf","first_error","error"))]
        if differences:
            raise ProtocolConflict("Dynamic path replay differs from archive: "+str(differences[:5]))
        return dict(passed=True,archive=str(path),compared=len(current),differences=0)
