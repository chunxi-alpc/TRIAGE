"""CPU evaluation of independently estimated paired-data methods.

All parameters come from source-task fitting. Target labels are consumed only
inside metric calculation. Existing records and results remain read-only.
"""
from pathlib import Path
from collections import defaultdict
import csv
import itertools
import json
import math
import time
import numpy as np
from experiments.aster_cr_retrain_paired_v2_20261111.common import HIST, Store, rows, read, write, lines
from prepare import ROOT as OUT
SEED=20261112

TRACE = Path("/home/gcx/aster_trace_repair_20261110")
TARGET = Path("/home/gcx/aster_cross_task_transfer_20261109")
GROUPS = ("E0", "E1", "E2", "E3")
DELTA = .2923349
DRAWS = 2000


def uid(row):
    return row.get("uid", row.get("view_id", row.get("id")))


def finite(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def write_csv(path, records, replace_scope=None):
    path = Path(path)
    old = []
    if replace_scope is not None and path.exists():
        with path.open(encoding="utf-8-sig", newline="") as f:
            old = [r for r in csv.DictReader(f) if r.get("scope") != replace_scope]
    records = old + list(records)
    keys = list(dict.fromkeys(k for r in records for k in r))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        if keys:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)
    tmp.replace(path)


def model_parameters(out, group, fold="full"):
    p = Path(out) / "models" / group / fold
    m2 = dict(np.load(p / "m2.npz", allow_pickle=False))
    m3 = dict(np.load(p / "m3.npz", allow_pickle=False))
    heads = np.asarray(m2["heads"], int).reshape(-1)
    mu = np.asarray(m2["mu"], float).reshape(-1)
    scale = np.asarray(m2["scale"], float).reshape(-1)
    N = np.asarray(m2["N"], float)
    w = np.asarray(m3["w"], float).reshape(-1)
    alpha = (N @ w) / scale
    b = float(np.asarray(m3["b"]).reshape(-1)[0])
    return dict(group=group, fold=fold, heads=heads, mu=mu, scale=scale,
                N=N, S=np.asarray(m2.get("S", np.empty((len(mu), 0))), float),
                U=np.asarray(m2["U"], float) if "U" in m2 else None,
                alpha=alpha, offset=b - float(mu @ alpha), b=b,
                metadata=read(p / "model.json"), directory=str(p))


_LINEAR_SCORE_CACHE = {}
def score_views(store, views, models):
    """One feature read per unique input; reuse immutable model scores across scopes."""
    unique = list(dict.fromkeys(uid(v) for v in views))
    scores = {g: {} for g in ("raw", *models)}
    caches = {name:_LINEAR_SCORE_CACHE.setdefault(obj["directory"],{})
              for name,obj in models.items()}
    for key in unique:
        ell = store.ell(key)
        scores["raw"][key] = ell
        missing = []
        for name in models:
            if key in caches[name]:
                scores[name][key] = caches[name][key]
            else:
                missing.append(name)
        if not missing:
            continue
        full = store.get(key)["heads"]
        for name in missing:
            obj = models[name]
            raw = full[obj["heads"]].reshape(-1)
            if not np.all(np.isfinite(raw)):
                absent = obj["heads"][~np.all(np.isfinite(full[obj["heads"]]),axis=1)]
                raise ValueError("Missing required cached heads for input "+key+
                                 ": "+",".join(map(str,absent.tolist())))
            value = ell + float(raw @ obj["alpha"]) + obj["offset"]
            scores[name][key] = value
            caches[name][key] = value
    return scores

class SourceBootstrap:
    """Shared draws make every reported method difference paired by source."""
    def __init__(self, sources, draws=DRAWS):
        self.sources = sorted(set(sources))
        self.index = {s: i for i, s in enumerate(self.sources)}
        n = len(self.sources)
        rng = np.random.default_rng(SEED)
        W = rng.multinomial(n, np.full(n, 1/n), size=draws) if n else np.empty((draws, 0))
        self.weights = np.vstack([np.ones((1, n)), W])
        self.draws = draws

    def sums(self, values, sources):
        v = np.asarray(values, float)
        if not len(v):
            return np.zeros(self.draws + 1)
        sums = np.bincount([self.index[s] for s in sources], weights=v,
                           minlength=len(self.sources))
        return self.weights @ sums

    def mean(self, values, sources):
        return ratio(self.sums(values, sources), self.sums(np.ones(len(values)), sources))

    def aggregate(self, columns, sources):
        """Aggregate all moments once, then apply the same 2,000 source draws."""
        columns = np.asarray(columns,float)
        if columns.ndim == 1:
            columns = columns[:,None]
        ids = np.fromiter((self.index[source] for source in sources),
                          dtype=np.int64,count=len(sources))
        grouped = np.empty((len(self.sources),columns.shape[1]),float)
        for j in range(columns.shape[1]):
            grouped[:,j] = np.bincount(ids,weights=columns[:,j],
                                       minlength=len(self.sources))
        totals = self.weights @ grouped
        counts = self.weights @ np.bincount(ids,minlength=len(self.sources))
        return totals,ratio(totals,counts[:,None]),ids

    def summary(self, values):
        vals = np.asarray(values, float)
        good = vals[1:][np.isfinite(vals[1:])]
        return dict(estimate=finite(vals[0]),
                    low=finite(np.quantile(good, .025)) if len(good) else None,
                    high=finite(np.quantile(good, .975)) if len(good) else None,
                    defined_draws=int(len(good)))


def ratio(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    out = np.full(a.shape, np.nan)
    np.divide(a, b, out=out, where=np.abs(b) > 1e-12)
    return out


def decode_medpic(item, scores):
    letters = list(item["options"])
    none = [k for k in letters if "none of" in item["options"][k].lower()]
    if not none:
        return [k for k in letters if scores[k] > 0]
    others = [k for k in letters if k not in none]
    candidates = [none] + [list(c) for n in range(1, len(others)+1)
                           for c in itertools.combinations(others, n)]
    return max(candidates, key=lambda c: sum(scores[k] for k in c))


def decisions_for(views, scores, task=None):
    decision = {name: {key: int(value > 0) for key, value in ss.items()}
                for name, ss in scores.items()}
    questions = []
    if task != "medpic":
        return decision, questions
    items = {r["id"]: r for r in rows(TARGET / task / "items.jsonl")}
    grouped = defaultdict(dict)
    for view in views:
        if view.get("item_id") and view.get("option_letter"):
            grouped[(view["item_id"], view.get("variant", "full"))][view["option_letter"]] = view
    for (qid, variant), options in grouped.items():
        item = items[qid]
        if set(options) != set(item["options"]):
            raise ValueError("Incomplete MedPIC option set: " + qid + "/" + variant)
        for name, ss in scores.items():
            chosen = decode_medpic(item, {letter: ss[uid(v)] for letter, v in options.items()})
            for letter, view in options.items():
                decision[name][uid(view)] = int(letter in chosen)
            sample = next(iter(options.values()))
            questions.append(dict(method=name, item_id=qid, variant=variant,
                                  source_id=sample["source_id"], split=sample.get("split"),
                                  predicted=chosen, gold=item["answer"],
                                  correct=set(chosen) == set(item["answer"])))
    return decision, questions


def native_relation_group(pair, task):
    """Keep existing synthetic MedPIC backgrounds separate from released pairs."""
    if task == "medpic" and pair.get("kind") == "P":
        return "synthetic_preserving"
    if pair.get("left_label") is None or pair.get("right_label") is None:
        return "native_unknown"
    return ("native_preserving" if int(pair["left_label"]) == int(pair["right_label"])
            else "native_changing")


def pair_group(pair, native=False):
    if native:
        return native_relation_group(pair, native)
    kind = pair.get("kind")
    return "AP" if kind in ("A", "P") else "CF" if kind in ("C", "F") else kind


def observations(views,pairs,scores,decisions,native=False,budgets=None):
    byid={uid(v):v for v in views}
    records=[]
    for name,ss in scores.items():
        for pair in pairs:
            left,right=pair["left"],pair["right"]
            if left not in ss or right not in ss:
                raise ValueError("Missing pair input: "+str(pair.get("pair_id")))
            a,b=ss[left],ss[right]
            ar,br=scores["raw"][left],scores["raw"][right]
            yl=pair.get("left_label")
            yr=pair.get("right_label")
            if yl is None:yl=byid[left].get("label")
            if yr is None:yr=byid[right].get("label")
            yl=None if yl is None else int(yl)
            yr=None if yr is None else int(yr)
            known=yl is not None and yr is not None
            preserves=(yl==yr) if known else (True if pair.get("kind") in ("A","P","F0") else None)
            d,dp=br-ar,b-a
            dc=dp-d
            pl,pr=decisions[name][left],decisions[name][right]
            budget=(budgets or {}).get(name,DELTA)
            records.append(dict(method=name,pair_id=pair.get("pair_id",left+"|"+right),
                source_id=pair["source_id"],kind=pair.get("kind"),
                group=pair_group(dict(pair,left_label=yl,right_label=yr),native),
                left=left,right=right,left_label=yl,right_label=yr,
                label_available=known,left_score=float(a),right_score=float(b),
                raw_difference=float(d),difference=float(dp),correction_difference=float(dc),
                paired_correct=bool(pl==yl and pr==yr) if known else None,
                predicted_preserving=bool(pl==pr),
                native_relation="preserving" if preserves is True else "changing" if preserves is False else None,
                relation_correct=bool((pl==pr)==preserves) if preserves is not None else None,
                direction_degraded=bool((yr-yl)*d>0 and (yr-yl)*dp<=0) if known else None,
                evidence_budget=budget,over_budget=bool(abs(dc)>budget+1e-9)))
    return records

def pair_statistics(records, boot, native=False):
    metrics, counts, per_source = {}, {}, []
    bygroup = defaultdict(list)
    for record in records:
        if record["group"]:
            bygroup[record["group"]].append(record)
    for group in sorted(bygroup):
        rr = bygroup[group]
        sg = [r["source_id"] for r in rr]
        raw = np.fromiter((r["raw_difference"] for r in rr),float,count=len(rr))
        adjusted = np.fromiter((r["difference"] for r in rr),float,count=len(rr))
        correction = adjusted-raw
        # Missing gold labels remain null in records. Numerators and available
        # denominators are accumulated separately; every pair keeps its scores.
        paired_valid=np.fromiter((r["paired_correct"] is not None for r in rr),float,count=len(rr))
        relation_valid=np.fromiter((r["relation_correct"] is not None for r in rr),float,count=len(rr))
        direction_valid=np.fromiter((r["left_label"] is not None and r["right_label"] is not None
                                     for r in rr),float,count=len(rr))
        paired=np.fromiter((float(r["paired_correct"]) if r["paired_correct"] is not None else 0.
                            for r in rr),float,count=len(rr))
        relation=np.fromiter((float(r["relation_correct"]) if r["relation_correct"] is not None else 0.
                              for r in rr),float,count=len(rr))
        degraded=np.fromiter((float(r["direction_degraded"]) if r["direction_degraded"] is not None else 0.
                              for r in rr),float,count=len(rr))
        over=np.fromiter((r["over_budget"] for r in rr),float,count=len(rr))
        direction=np.fromiter((r["right_label"]-r["left_label"]
                               if r["left_label"] is not None and r["right_label"] is not None else 0.
                               for r in rr),float,count=len(rr))
        # Column order is shared by every statistic and source-level summary.
        columns = np.column_stack((raw*raw,adjusted*adjusted,correction*correction,
                                   paired,relation,raw,adjusted,
                                   direction*adjusted,direction*raw,degraded,over,
                                   paired_valid,relation_valid,direction_valid))
        total,mean,ids = boot.aggregate(columns,sg)
        prefix = group.lower()
        metrics[prefix+"_raw_rms"] = np.sqrt(mean[:,0])
        metrics[prefix+"_score_change_rms"] = np.sqrt(mean[:,1])
        metrics[prefix+"_correction_rms"] = np.sqrt(mean[:,2])
        metrics[prefix+"_paired_accuracy"] = ratio(total[:,3],total[:,11])
        metrics[prefix+"_relation_accuracy"] = ratio(total[:,4],total[:,12])
        counts[prefix+"_pairs"] = len(rr)
        counts[prefix+"_sources"] = len(set(sg))
        counts[prefix+"_paired_labeled_count"] = int(paired_valid.sum())
        counts[prefix+"_paired_unlabeled_count"] = len(rr)-int(paired_valid.sum())
        if group in ("AP","native_preserving","synthetic_preserving"):
            raw_variance = mean[:,0]-mean[:,5]**2
            adjusted_variance = mean[:,1]-mean[:,6]**2
            metrics[prefix+"_vr"] = 1-ratio(adjusted_variance,raw_variance)
        if group in ("CF","native_changing"):
            metrics[prefix+"_er"] = ratio(mean[:,7],mean[:,8])
            metrics[prefix+"_direction_degradation_rate"] = ratio(total[:,9],total[:,13])
            counts[prefix+"_direction_degradation_count"] = int(degraded.sum())
            metrics[prefix+"_direction_degradation_count"] = total[:,9]
            if not native:
                metrics[prefix+"_over_budget_rate"] = mean[:,10]
                counts[prefix+"_over_budget_count"] = int(over.sum())
                counts[prefix+"_max_abs_correction"] = float(np.max(np.abs(correction)))
        # Index each source once instead of scanning the complete pair list for it.
        source_rows = defaultdict(list)
        for j,source in enumerate(sg):
            source_rows[source].append(j)
        for source in sorted(source_rows):
            idx = np.asarray(source_rows[source],int)
            d,dp = raw[idx],adjusted[idx]
            per_source.append(dict(method=rr[0]["method"],source_id=source,group=group,
                n=len(idx),correction_prediction_rmse=float(np.sqrt(np.mean(
                    (dp if group in ("AP","native_preserving","synthetic_preserving") else dp-d)**2))),
                prediction_residual_mean=float(np.mean(dp)),
                raw_rms=float(np.sqrt(np.mean(d*d))),response_rms=float(np.sqrt(np.mean(dp*dp))),
                correction_rms=float(np.sqrt(np.mean((dp-d)**2))),
                vr=finite(1-np.var(dp)/np.var(d)) if np.var(d)>1e-12 else None,
                paired_correct=int(paired[idx].sum()),
                paired_labeled_count=int(paired_valid[idx].sum()),
                paired_unlabeled_count=len(idx)-int(paired_valid[idx].sum()),
                direction_degradation_count=int(degraded[idx].sum()),
                over_budget_count=int(over[idx].sum())))
    return metrics,counts,per_source

_PATH_TREES = {}
def _execute_prepared_path(case,node_scores,nodes,root):
    gid = root
    actual = []
    first = None
    gold = case["gold_path"]
    error = None
    while gid in nodes:
        if len(actual)>len(nodes):
            error="cycle"
            break
        score=node_scores.get(gid)
        if score is None:
            error="unavailable_score"
            break
        label=int(score>0)
        actual.append(dict(gid=gid,label=label))
        i=len(actual)-1
        if first is None and (i>=len(gold) or gold[i]!=actual[-1]):
            first=gid
        gid=nodes[gid]["yes" if label else "no"]
    return dict(correct=not error and actual==gold and gid==case["gold_leaf"],
                path=actual,leaf=gid,first_error=first,error=error,node_forward_count=len(actual))

def path_predictions(cases,path_views,scores):
    """Index variants by case and parse each unchanged tree only once."""
    from experiments.aster_closed_form.data import canonical_guide
    grouped=defaultdict(dict)
    for view in path_views:
        if view.get("gid") is not None:
            variant=view.get("variant","original")
            if variant=="full":
                variant="original"
            grouped[(view["case_id"],variant)][view["gid"]]=uid(view)
    bycase=defaultdict(list)
    for (cid,variant),node_uids in grouped.items():
        bycase[cid].append((variant,node_uids))
    results=[]
    for case in cases:
        cid=case["case_id"]
        if cid not in _PATH_TREES:
            tree=canonical_guide(case["guideline"])
            _PATH_TREES[cid]=({n["id"]:n for n in tree["nodes"]},tree["nodes"][0]["id"])
        nodes,root=_PATH_TREES[cid]
        for variant,node_uids in bycase[cid]:
            if variant not in ("original","action_presence","action_neutral","action_swap",
                                "remove_action","neutral_action","swap_action"):
                continue
            for name,ss in scores.items():
                outcome=_execute_prepared_path(case,{gid:ss[key] for gid,key in node_uids.items()
                                                if key in ss},nodes,root)
                results.append(dict(method=name,case_id=cid,source_id=case["source_id"],
                                    variant=variant,interference=variant!="original",**outcome))
    return results

def scope_evaluate(scope, views, pairs, store, models, out=OUT, cases=None,
                   path_views=None, native_task=None):
    out = Path(out)
    evaldir = out / "evaluation" / scope
    evaldir.mkdir(parents=True, exist_ok=True)
    path_views = list({uid(v): v for v in (path_views or [])}.values())
    views = list({uid(v): v for v in views}.values())
    scoring_views = list({uid(v): v for v in [*views, *path_views]}.values())
    scores = score_views(store, scoring_views, models)
    decisions, questions = decisions_for(views, scores, native_task)
    budgets = {name:float(obj["metadata"].get("delta",obj["metadata"].get("M3",{}).get("delta",DELTA)))
               for name,obj in models.items()}
    pair_records = observations(views,pairs,scores,decisions,native=native_task or False,budgets=budgets)
    path_records = path_predictions(cases or [], path_views, scores) if cases else []
    sources = [v["source_id"] for v in views] + [r["source_id"] for r in path_records]
    boot = SourceBootstrap(sources)
    natural = [v for v in views if v.get("kind", "original") == "original" and
               (v.get("original") or ("original" not in v and
                v.get("variant", "original") in ("full", "original")))]
    metrics = {}
    counts = {}
    source_records = []
    table = []
    path_table = []
    for name in scores:
        prs = [r for r in pair_records if r["method"] == name]
        mm, cc, per_source = pair_statistics(prs, boot, native=bool(native_task))
        if native_task == "medpic":
            qq = [r for r in questions if r["method"] == name and r["variant"] in ("full", "original")]
            mm["natural_native_accuracy"] = boot.mean([r["correct"] for r in qq],
                                                       [r["source_id"] for r in qq])
            cc["natural_native_count"] = len(qq)
            cc["natural_native_correct"] = sum(r["correct"] for r in qq)
        if natural:
            labeled_natural=[v for v in natural if v.get("label") is not None]
            ok=[decisions[name][uid(v)]==int(v["label"]) for v in labeled_natural]
            mm["natural_node_accuracy"]=boot.mean(ok,[v["source_id"] for v in labeled_natural])
            cc["natural_node_count"]=len(ok)
            cc["natural_node_correct"]=sum(ok)
            cc["natural_node_total_count"]=len(natural)
            cc["natural_node_unlabeled_count"]=len(natural)-len(labeled_natural)
            if native_task == "nli4ct":
                mm["natural_native_accuracy"] = mm["natural_node_accuracy"]
                cc["natural_native_count"] = len(ok)
                cc["natural_native_correct"] = sum(ok)
        for interference in (False, True):
            rr = [r for r in path_records if r["method"] == name and r["interference"] == interference]
            prefix = "interference_path" if interference else "natural_path"
            if rr:
                mm[prefix+"_accuracy"] = boot.mean([r["correct"] for r in rr],
                                                    [r["source_id"] for r in rr])
                cc[prefix+"_count"] = len(rr)
                cc[prefix+"_correct"] = sum(r["correct"] for r in rr)
                for variant in sorted({r["variant"] for r in rr}):
                    vr = [r for r in rr if r["variant"] == variant]
                    path_table.append(dict(scope=scope, method=name, variant=variant,
                                           n=len(vr), correct=sum(r["correct"] for r in vr),
                                           accuracy=float(np.mean([r["correct"] for r in vr]))))
        metrics[name], counts[name] = mm, cc
        source_records.extend(dict(scope=scope, **r) for r in per_source)
        summary_row = dict(scope=scope, method=name, **cc)
        summary_row.update({k: finite(v[0]) for k, v in mm.items() if k not in cc})
        table.append(summary_row)
    paired = []
    comparisons = [(g, "raw") for g in models]
    comparisons += [(g, "E0") for g in models if g != "E0" and "E0" in models]
    comparisons += [(a, b) for a, b in (("E2", "E1"), ("E3", "E2"), ("E4", "E3"), ("E3", "E1"), ("E4", "E1"))
                    if a in models and b in models]
    for name, mm in metrics.items():
        for metric, values in mm.items():
            paired.append(dict(scope=scope, comparison=name, metric=metric, **boot.summary(values),
                               sources=len(boot.sources), draws=DRAWS))
    for lhs, rhs in comparisons:
        for metric in metrics[lhs].keys() & metrics[rhs].keys():
            diff = metrics[lhs][metric] - metrics[rhs][metric]
            paired.append(dict(scope=scope, comparison=lhs+" minus "+rhs, metric=metric,
                               **boot.summary(diff), sources=len(boot.sources), draws=DRAWS))
    lines(evaldir / "pair_predictions.jsonl", pair_records)
    lines(evaldir / "path_predictions.jsonl", path_records)
    lines(evaldir / "question_predictions.jsonl", questions)
    lines(evaldir / "predictions.jsonl",
          [dict(view_id=uid(v), original_view_id=v.get("original_view_id"), source_id=v["source_id"],
                split=v.get("split"),label=v.get("label"),label_available=v.get("label") is not None,
                original=bool(v.get("kind", "original") == "original" and (v.get("original") or ("original" not in v and v.get("variant", "original") in ("full", "original")))),
                kind=v.get("kind"), scores={name: float(ss[uid(v)]) for name, ss in scores.items()},
                predictions={name: int(ds[uid(v)]) for name, ds in decisions.items()}) for v in views])
    result = dict(scope=scope, complete=True,
                  model_directories={name:obj["directory"] for name,obj in models.items()},
                  models={r["method"]: {k: v for k, v in r.items()
                   if k not in ("scope", "method")} for r in table}, bootstrap=paired,
                  source_count=len(boot.sources), unique_inputs=len(scores["raw"]),
                  native_label_policy=("Released native label equality/change only; no A/P or C/F "
                                       "semantic interpretation" if native_task else None),
                  test_history="Previously evaluated source sets; not new independent confirmation")
    write_csv(out / "pair_metrics.csv", table, replace_scope=scope)
    write_csv(out / "path_metrics.csv", path_table, replace_scope=scope)
    write_csv(out / "source_bootstrap.csv", paired, replace_scope=scope)
    write_csv(out / "source_metrics.csv", source_records, replace_scope=scope)
    if scope in ("dev","test","new60","nli4ct","medpic","sensitivity_dev"):
        selected = read(out/"development_selection.json").get("frozen_group")
        write_csv(out/"data_group_comparison.csv",
                  [dict(row, selected=row["method"]==selected) for row in table],
                  replace_scope=scope)
    # Publish completion only after all per-sample and aggregate outputs exist.
    write(evaldir / "metrics.json", result)
    return result


def load_full_models(out=OUT, groups=GROUPS):
    return {g: model_parameters(out, g) for g in groups
            if (Path(out) / "models" / g / "full" / "m3.npz").exists()}


_DATA_CACHE = {}
def data_rows(out, name):
    # Evaluation does not need prompt strings already held in Store.inputs.
    path = Path(out) / "data" / name
    key = str(path)
    if key not in _DATA_CACHE:
        keep = {"uid","view_id","original_view_id","id","source_id","split","label",
                "variant","kind","original","case_id","gid","group_id","item_id",
                "option_letter","pair_id","left","right","left_label","right_label",
                "original_left","original_right","relation","native_intervention"}
        parsed = []
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        parsed.append({k:v for k,v in row.items() if k in keep})
        _DATA_CACHE[key] = parsed
    return _DATA_CACHE[key]


_CASE_CACHE = {}
def case_rows(path):
    key=str(path)
    if key not in _CASE_CACHE:
        _CASE_CACHE[key]=rows(path)
    return _CASE_CACHE[key]


def historical_scope(split, models, store, out=OUT, sources=None, scope=None):
    vv = [v for v in data_rows(out, "historical_views.jsonl") if v.get("split") == split]
    pp = [p for p in data_rows(out, "historical_pairs.jsonl") if p.get("split") == split]
    pv = [v for v in data_rows(out, "path_features.jsonl") if v.get("split") == split]
    pv += [v for v in data_rows(out, "path_interference_views.jsonl") if v.get("split") == split]
    cases = [c for c in case_rows(HIST / "data" / "path_cases.jsonl") if c["split"] == split]
    if sources is not None:
        sources = set(sources)
        vv = [v for v in vv if v["source_id"] in sources]
        pp = [p for p in pp if p["source_id"] in sources]
        pv = [v for v in pv if v["source_id"] in sources]
        cases = [c for c in cases if c["source_id"] in sources]
    return scope_evaluate(scope or split, vv, pp, store, models, out,
                          cases=cases, path_views=pv)


def evaluate_cv(out=OUT, store=None):
    out = Path(out)
    store = store or Store(out)
    fold_rows = []
    source_ids = sorted({v["source_id"] for v in data_rows(out, "natural_views.jsonl")})
    for group in ("E0", "E1", "E2"):
        for k in range(1, 6):
            fold = "fold"+str(k)
            path = out / "models" / group / fold / "m3.npz"
            if not path.exists():
                continue
            model = model_parameters(out, group, fold)
            meta = model["metadata"]
            heldout = meta.get("validation_sources", meta.get("heldout_sources"))
            if heldout is None and meta.get("train_sources"):
                heldout = sorted(set(source_ids)-set(meta["train_sources"]))
            if heldout is None:
                folds = read(out / "configuration.json").get("folds", [])
                if isinstance(folds, dict):
                    entry = folds.get(fold, folds.get(str(k), {}))
                else:
                    entry = folds[k-1] if len(folds) >= k else {}
                heldout = entry.get("validation_sources", entry.get("heldout_sources"))
            if not heldout:
                raise ValueError("Fold lacks held-out source metadata: " + group+"/"+fold)
            valid_sources = set(heldout)
            bv = [dict(v, original=False) for v in data_rows(out, "E2_views.jsonl") if v["source_id"] in valid_sources]
            bp = [p for p in data_rows(out, "E2_pairs.jsonl") if p["source_id"] in valid_sources]
            nv = [v for v in data_rows(out, "natural_views.jsonl") if v["source_id"] in valid_sources]
            pv = [v for v in data_rows(out, "path_features.jsonl") if
                  v.get("split") == "fit" and v["source_id"] in valid_sources]
            pv += [v for v in data_rows(out, "path_interference_views.jsonl") if
                   v.get("split") == "fit" and v["source_id"] in valid_sources]
            cases = [c for c in case_rows(HIST / "data" / "path_cases.jsonl") if
                     c["split"] == "fit" and c["source_id"] in valid_sources]
            result = scope_evaluate("cv_"+group+"_"+fold+"_common_frozen_pool",
                                    [*bv, *nv], bp, store, {group:model}, out,
                                    cases=cases, path_views=pv)
            # The historical paired pool remains an additional, separately labeled check.
            hv = [v for v in data_rows(out,"historical_views.jsonl") if
                  v.get("split") == "fit" and v["source_id"] in valid_sources]
            hp = [p for p in data_rows(out,"historical_pairs.jsonl") if
                  p.get("split") == "fit" and p["source_id"] in valid_sources]
            scope_evaluate("cv_"+group+"_"+fold+"_historical_pool", hv, hp, store,
                           {group:model}, out)
            fold_rows.append(dict(group=group, fold=k, heldout_sources=len(heldout),
                                  validation_pool="common_frozen_Dataset_B",
                                  **result["models"][group]))
    write_csv(out / "fold_results.csv", fold_rows)
    write(out / "cv_evaluation.json", dict(folds=fold_rows, complete=len(fold_rows) == 15,
                                          protocol="Every fold independently fits M1/M2/M3 and intercept"))
    return fold_rows


def evaluate_development(out=OUT, store=None, include_cv=False):
    out = Path(out)
    store = store or Store(out)
    if include_cv:
        evaluate_cv(out, store)
    models = load_full_models(out)
    result = historical_scope("dev", models, store, out)
    candidates = [g for g in ("E1", "E2") if g in result["models"]]
    if len(candidates) != 2:
        raise ValueError("Both E1 and E2 must finish before selecting the main data scheme")
    def selection_key(group):
        m = result["models"][group]
        return (m.get("natural_path_accuracy", -1), m.get("natural_node_accuracy", -1),
                -m.get("cf_direction_degradation_count", 10**9),
                -m.get("cf_over_budget_rate", 1), m.get("ap_vr", -1),
                int(group == "E1"))
    selected = max(candidates, key=selection_key)
    selection = dict(frozen_group=selected, dataset="A" if selected == "E1" else "B",
                     stage="development_selection", candidates=candidates,
                     selection_order=["natural_path_accuracy", "natural_node_accuracy",
                                      "fewer_cf_direction_degradations", "lower_cf_over_budget_rate",
                                      "higher_ap_vr", "smaller_balanced_dataset_tie_break"],
                     development_results=result["models"],
                     test_labels_used_for_selection=False)
    write(out / "development_selection.json", selection)
    write_csv(out / "data_group_comparison.csv",
              [dict(scope="dev", method=g, selected=g == selected,
                    **result["models"][g]) for g in result["models"]], replace_scope="dev")
    model_comparison(out)
    generate_report(out)
    return selection


def model_comparison(out=OUT):
    models = load_full_models(out)
    records = []
    for lhs, rhs in itertools.combinations(models, 2):
        a, b = models[lhs], models[rhs]
        ha, hb = set(a["heads"]), set(b["heads"])
        union = sorted(ha | hb)
        lookup = {h: i for i, h in enumerate(union)}
        angles = {}
        for basis in ("N", "S", "U"):
            if any(model[basis] is None for model in (a,b)):
                angles[basis] = None
                continue
            bases = []
            for model in (a, b):
                mat = model[basis] / model["scale"][:, None]
                embedded = np.zeros((len(union)*128, mat.shape[1]))
                for j, h in enumerate(model["heads"]):
                    embedded[lookup[int(h)]*128:(lookup[int(h)]+1)*128] = mat[j*128:(j+1)*128]
                bases.append(np.linalg.qr(embedded, mode="reduced")[0])
            cosine = np.linalg.svd(bases[0].T @ bases[1], compute_uv=False)
            angles[basis] = np.degrees(np.arccos(np.clip(cosine, 0, 1))).tolist()
        records.append(dict(lhs=lhs, rhs=rhs, overlapping_heads=len(ha & hb),
                            overlap_fraction=len(ha & hb)/51, jaccard=len(ha & hb)/len(ha | hb),
                            N_principal_angles_degrees=json.dumps(angles["N"]),
                            S_principal_angles_degrees=json.dumps(angles["S"]),
                            U_principal_angles_degrees=json.dumps(angles["U"]),
                            U_status="available" if angles["U"] is not None else "historical_U_not_saved",
                            principal_angle_coordinates="common raw head-output coordinates; scale accounted"))
    write_csv(Path(out) / "model_comparison.csv", records)
    details = []
    for name, model in models.items():
        m2 = dict(np.load(Path(model["directory"])/"m2.npz", allow_pickle=False))
        m1 = read(Path(model["directory"])/"m1.json")
        details.append(dict(method=name,
                            selected_heads=model["heads"].tolist(),
                            layer_distribution=dict(sorted((str(layer), int(np.sum(model["heads"]//32 == layer))) for layer in np.unique(model["heads"]//32))),
                            projected_feature_entropy_effective_rank=finite(m2.get("projected_feature_entropy_effective_rank")),
                            projected_feature_singular_values=np.asarray(m2.get("projected_feature_singular_values",[])).tolist(),
                            S_effective_rank=int(np.linalg.matrix_rank(model["S"])),
                            U_effective_rank=int(np.linalg.matrix_rank(model["U"])) if model["U"] is not None else None,
                            N_effective_rank=int(np.linalg.matrix_rank(model["N"])),
                            M1_cost=m1.get("cost",m1.get("compute_cost")),
                            historical_reuse=model["metadata"].get("historical_reuse",False)))
    write(Path(out)/"feature_rank_and_heads.json", details)
    return records


def reusable_scope(scope, models, out=OUT):
    """Reuse only a completed scope from the same run and model directories."""
    try:
        result = read(Path(out)/"evaluation"/scope/"metrics.json")
    except (json.JSONDecodeError, OSError):
        return None
    expected = {"raw", *models}
    directories = {name:obj["directory"] for name,obj in models.items()}
    if (result.get("complete") and result.get("scope")==scope
            and set(result.get("models",{}))==expected
            and result.get("model_directories")==directories):
        return result
    return None


def evaluate_all(frozen_group=None, out=OUT, store=None, include_cv=False,
                 scopes=("test", "new60", "nli4ct", "medpic")):
    out = Path(out)
    store = store or Store(out)
    selection = read(out / "development_selection.json")
    frozen_group = frozen_group or selection.get("frozen_group")
    frozen = read(out / "frozen.json")
    if not frozen_group:
        raise ValueError("Development main-data selection must precede held-out evaluation")
    if not frozen.get("frozen"):
        raise ValueError("Write frozen.json before held-out evaluation")
    models = load_full_models(out)
    results = {}
    if "test" in scopes:
        results["test"] = (reusable_scope("test",models,out)
                           or historical_scope("test",models,store,out))
    sensitivity = {}
    for relative in ("0.05", "0.25"):
        fold = "budget_"+relative
        if (out/"models"/frozen_group/fold/"m3.npz").exists():
            name = frozen_group+"_"+fold
            sensitivity[name] = model_parameters(out, frozen_group, fold)
    if sensitivity:
        sensitivity_models = {frozen_group:models[frozen_group], **sensitivity}
        results["sensitivity_dev"] = (reusable_scope("sensitivity_dev",sensitivity_models,out)
            or historical_scope("dev",sensitivity_models,store,out,scope="sensitivity_dev"))
    if "new60" in scopes:
        existing = reusable_scope("new60",models,out)
        if existing is not None:
            results["new60"] = existing
        else:
            vv = data_rows(out, "new60_views.jsonl")
            pp = data_rows(out, "new60_pairs.jsonl")
            if not vv:
                raise ValueError("60-source evaluation input table is absent")
            # cases_final contains each case once per original/action variant.
            # Expand cached path variants only from the single original case.
            cases = [case for case in case_rows(TRACE/"data"/"cases_final.jsonl")
                     if case.get("variant")=="original"]
            results["new60"] = scope_evaluate("new60",vv,pp,store,models,out,
                                              cases=cases,path_views=vv)
    for task in ("nli4ct", "medpic"):
        if task not in scopes:
            continue
        existing = reusable_scope(task,models,out)
        if existing is not None:
            results[task] = existing
            continue
        vv = [v for v in data_rows(out, task+"_views.jsonl") if v.get("split") == "test"]
        pp = [p for p in data_rows(out, task+"_pairs.jsonl") if p.get("split") == "test"]
        if not vv:
            raise ValueError("Cross-task input table absent: " + task)
        results[task] = scope_evaluate(task, vv, pp, store, models, out, native_task=task)
    if include_cv:
        evaluate_cv(out, store)
    model_comparison(out)
    write(out / "evaluation_summary.json", dict(frozen_group=frozen_group,
                                               results={k:v["models"] for k,v in results.items()},
                                               completed_scopes=list(results)))
    generate_report(out)
    return results


def generate_report(out=OUT):
    out = Path(out)
    selection = read(out / "development_selection.json")
    cv = read(out / "cv_evaluation.json")
    metrics = {}
    for scope in ("dev", "test", "new60", "nli4ct", "medpic", "sensitivity_dev"):
        p = out / "evaluation" / scope / "metrics.json"
        if p.exists():
            metrics[scope] = read(p)
    def fmt(value, percent=False):
        if value is None:
            return "\u672a\u8bc4\u4ef7"
        return f"{value*(100 if percent else 1):.2f}" + ("%" if percent else "")
    text = ["# ASTER-CR \u65b0\u65e7\u914d\u5bf9\u6570\u636e\u5bf9\u7167\u5b9e\u9a8c", "",
            "\u57fa\u7840\u6a21\u578b\u4e3a\u51bb\u7ed3 BF16 Qwen3-8B\uff0c\u4fdd\u7559\u539f\u63d0\u793a\u3001\u5206\u8bcd\u3001\u6700\u7ec8 token \u8bc4\u5206\u53ca FP32 \u6807\u7b7e\u6295\u5f71\u3002"
            "\u65b0\u7ec4\u72ec\u7acb\u91cd\u4f30 M1 \u7684 51 \u4e2a\u5934\u3001M2 \u7684 16 \u7ef4\u5b50\u7a7a\u95f4\u548c M3 \u7684\u4fee\u6b63\u53c2\u6570\uff1b"
            "E0 \u5168\u91cf\u7ed3\u679c\u5f15\u7528\u65e2\u6709\u5b9e\u9a8c\uff0c\u907f\u514d\u91cd\u590d\u8bad\u7ec3\u3002", "",
            "Dataset A \u4e3a 2,540 \u5bf9\u5747\u8861\u6570\u636e\uff1bDataset B \u4e3a 6,345 \u5bf9\u5168\u90e8\u5408\u683c\u6570\u636e\u3002"
            "E1/E2 \u4f7f\u7528\u7c7b\u522b\u53ca\u6765\u6e90\u5e73\u8861\uff0cE3 \u4f7f\u7528\u81ea\u7136\u6743\u91cd\uff1b"
            "A/P \u56de\u5f52\u6743\u91cd\u548c\u5f52\u4e00\u4e3a 4,887\uff0clambda=0.01\uff0c"
            "\u6240\u6709 C/F \u62df\u5408\u5bf9\u4f7f\u7528\u540c\u4e00 0.2923349 logit \u786c\u7ea6\u675f\u9884\u7b97\u3002", ""]
    if selection:
        text += ["\u5f00\u53d1\u96c6\u9009\u62e9\u7684\u4e3b\u65b9\u6848\u4e3a **"+selection["frozen_group"]+" / Dataset "
                 +selection["dataset"]+"**\u3002\u9009\u62e9\u4f9d\u6b21\u6bd4\u8f83\u81ea\u7136\u5b8c\u6574\u8def\u5f84\u3001\u81ea\u7136\u8282\u70b9\u6b63\u786e\u7387\u3001"
                 "\u8bc1\u636e\u65b9\u5411\u9000\u5316\u6570\u3001\u8d85\u9884\u7b97\u6bd4\u4f8b\u53ca\u5e72\u6270\u65b9\u5dee\u964d\u4f4e\u7387\u3002", ""]
    else:
        text += ["\u5f53\u524d\u5c1a\u672a\u5b8c\u6210\u5f00\u53d1\u96c6\u4e3b\u65b9\u6848\u9009\u62e9\uff1b\u4e0b\u5217\u4ec5\u5217\u5df2\u5b9e\u9645\u5b8c\u6210\u7684\u9636\u6bb5\u3002", ""]
    for scope, result in metrics.items():
        label = {"dev":"\u5f00\u53d1\u96c6", "test":"\u5386\u53f240\u6765\u6e90\u7559\u51fa\u96c6", "new60":"\u5df2\u670960\u6765\u6e90\u8bc4\u4ef7\u96c6",
                 "nli4ct":"NLI4CT-2024", "medpic":"MedPIC",
                 "sensitivity_dev":"\u4e3b\u65b9\u6848\u9884\u7b97\u654f\u611f\u6027\uff08\u5f00\u53d1\u96c6\uff09"}[scope]
        text += ["## "+label, ""]
        if scope in ("nli4ct", "medpic"):
            text += ["\u51bb\u7ed3 MEGA \u65b9\u6cd5\u76f4\u63a5\u8fc1\u79fb\uff0c\u4e0d\u4f7f\u7528\u76ee\u6807\u6807\u7b7e\u62df\u5408\u5934\u3001\u5b50\u7a7a\u95f4\u3001\u7cfb\u6570\u6216\u622a\u8ddd\u3002"
                     "\u539f\u751f\u4fdd\u6301\u548c\u7ffb\u8f6c\u4ec5\u8868\u793a\u53d1\u5e03\u6807\u7b7e\u7684\u5173\u7cfb\uff0c\u4e0d\u76f4\u63a5\u89e3\u91ca\u4e3a A/P \u6216 C/F\u3002", "",
                     "|\u65b9\u6cd5|\u539f\u751f\u6b63\u786e\u7387|\u4fdd\u6301\u5bf9\u5168\u5bf9\u7387|\u7ffb\u8f6c\u5bf9\u5168\u5bf9\u7387|\u4fdd\u6301\u5173\u7cfb\u6b63\u786e\u7387|\u7ffb\u8f6c\u5173\u7cfb\u6b63\u786e\u7387|",
                     "|---|---:|---:|---:|---:|---:|"]
            for name,m in result["models"].items():
                text.append("|"+name+"|"+ "|".join(fmt(m.get(k),True) for k in
                    ("natural_native_accuracy","native_preserving_paired_accuracy",
                     "native_changing_paired_accuracy","native_preserving_relation_accuracy",
                     "native_changing_relation_accuracy"))+"|")
        else:
            text += ["|\u65b9\u6cd5|A/P\u5deeRMS|VR|C/F\u4fee\u6b63RMS|ER|\u65b9\u5411\u9000\u5316|\u8d85\u9884\u7b97|\u81ea\u7136\u8282\u70b9|\u81ea\u7136\u8def\u5f84|\u5e72\u6270\u8def\u5f84|",
                     "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for name,m in result["models"].items():
                text.append("|"+name+"|"+ "|".join([
                    fmt(m.get("ap_score_change_rms")),fmt(m.get("ap_vr"),True),
                    fmt(m.get("cf_correction_rms")),fmt(m.get("cf_er"),True),
                    str(m.get("cf_direction_degradation_count","\u672a\u8bc4\u4ef7")),
                    fmt(m.get("cf_over_budget_rate"),True),fmt(m.get("natural_node_accuracy"),True),
                    fmt(m.get("natural_path_accuracy"),True),fmt(m.get("interference_path_accuracy"),True)])+"|")
        text += [""]
    text += ["## \u4e94\u6298\u6765\u6e90\u4ea4\u53c9\u9a8c\u8bc1", ""]
    if cv:
        text += [f"\u5df2\u5b8c\u6210 {len(cv.get('folds',[]))}/15 \u4e2a\u7ec4\u522b\u6298\u3002\u6bcf\u6298\u4ec5\u752864\u4e2a\u6765\u6e90\u91cd\u65b0\u4f30\u8ba1\u5168\u90e8\u6a21\u5757\uff0c"
                 "16\u4e2a\u6765\u6e90\u7528\u4e8e\u9a8c\u8bc1\uff1b\u81ea\u7136\u622a\u8ddd\u4ec5\u4f7f\u7528\u8bad\u7ec3\u6765\u6e90\u7684\u539f\u59cb\u81ea\u7136\u8282\u70b9\u3002", ""]
    else:
        text += ["\u4e94\u6298\u6765\u6e90\u4ea4\u53c9\u9a8c\u8bc1\u7ed3\u679c\u5c1a\u672a\u5199\u5165\u3002", ""]
    text += ["## \u7edf\u8ba1\u4e0e\u6307\u6807", "",
             "\u6309\u6307\u5357\u6765\u6e90\u6267\u884c2,000\u6b21\u914d\u5bf9\u81ea\u52a9\u91cd\u91c7\u6837\uff1b\u65b9\u6cd5\u5dee\u503c\u4f7f\u7528\u5b8c\u5168\u76f8\u540c\u7684\u6765\u6e90\u62bd\u6837\u3002"
             "\u5dee\u503c\u53ca95%\u7f6e\u4fe1\u533a\u95f4\u89c1 source_bootstrap.csv\uff0c\u9010\u6765\u6e90\u8bef\u5dee\u89c1 source_metrics.csv\u3002"
             "\u96f6\u65b9\u5dee\u6216\u96f6\u54cd\u5e94\u5206\u6bcd\u65f6\uff0c\u76f8\u5e94\u6bd4\u503c\u8bb0\u4e3a\u7a7a\u503c\uff0c\u5e76\u4fdd\u7559\u6709\u6548\u91cd\u91c7\u6837\u6b21\u6570\u3002", "",
             "A/P\u5deeRMS\u4e3a\u4fee\u6b63\u540e\u7684\u914d\u5bf9\u5206\u6570\u5dee\u5747\u65b9\u6839\uff1b\u53e6\u4fdd\u5b58\u539f\u59cbRMS\u548c\u4fee\u6b63\u91cfRMS\u3002"
             "VR=1\u2212\u4fee\u6b63\u540e\u5dee\u503c\u65b9\u5dee/\u539f\u5dee\u503c\u65b9\u5dee\u3002ER\u4e3a\u771f\u503c\u65b9\u5411\u52a0\u6743\u7684\u4fee\u6b63\u540e\u4e0e\u539f\u8bc1\u636e\u54cd\u5e94\u4e4b\u6bd4\u3002"
             "\u65b9\u5411\u9000\u5316\u7edf\u8ba1\u539f\u771f\u503c\u65b9\u5411\u54cd\u5e94\u4e3a\u6b63\u3001\u4fee\u6b63\u540e\u4e0d\u4e3a\u6b63\u7684\u914d\u5bf9\u3002"
             "\u8d85\u9884\u7b97\u6309\u4fee\u6b63\u91cf\u5728\u4e24\u4fa7\u7684\u7edd\u5bf9\u5dee\u662f\u5426\u8d85\u8fc7\u56fa\u5b9adelta\u7edf\u8ba1\u3002", "",
             "\u5386\u53f240\u4e2a\u6d4b\u8bd5\u6765\u6e90\u3001\u5df2\u670960\u4e2a\u8bc4\u4ef7\u6765\u6e90\u53ca\u76ee\u6807\u4efb\u52a1\u5747\u6709\u5386\u53f2\u4f7f\u7528\u8bb0\u5f55\uff0c"
             "\u672c\u8f6e\u7ed3\u679c\u5c5e\u4e8e\u65e2\u6709\u6570\u636e\u4e0a\u7684\u5bf9\u7167\u8bc4\u4ef7\uff0c\u4e0d\u4f5c\u4e3a\u5168\u65b0\u72ec\u7acb\u786e\u8ba4\u3002"
             "\u5b8c\u6574\u8def\u5f84\u6309\u4fee\u6b63\u540e\u5206\u6570\u81ea\u4e3b\u904d\u5386\uff0c\u6240\u5230\u8282\u70b9\u7f3a\u5c11\u7f13\u5b58\u8bb0\u4e3a\u4e0d\u53ef\u7528\u5e76\u5224\u9519\u3002", "",
             "\u9010\u8f93\u5165\u5206\u6570\u3001\u9010\u5bf9\u9884\u6d4b\u3001\u5b8c\u6574\u8def\u5f84\u3001\u6a21\u578b\u53c2\u6570\u548c\u6765\u6e90\u6307\u6807\u4fdd\u5b58\u5728\u672c\u76ee\u5f55\u3002"
             "\u65e7\u6570\u636e\u548c\u5386\u53f2\u62a5\u544a\u53ea\u8bfb\u3002\u5f53\u524d\u62a5\u544a\u53ea\u5217\u5df2\u8ba1\u7b97\u7684\u7ed3\u679c\u3002", ""]
    text += ["## \u590d\u7528\u4e0e\u65b0\u589e\u8ba1\u7b97", ""]
    for group in GROUPS:
        p = out / "models" / group / "full" / "model.json"
        if not p.exists():
            continue
        model = read(p)
        if model.get("historical_reuse"):
            text.append(group+"\uff1a\u590d\u7528\u5df2\u5b8c\u6210\u7684\u5386\u53f2\u5168\u91cf\u6a21\u578b\uff1b\u4e0d\u91cd\u590d\u9009\u5934\u3001\u5b50\u7a7a\u95f4\u4f30\u8ba1\u548c\u4fee\u6b63\u62df\u5408\u3002")
        else:
            text.append(group+"\uff1a\u672c\u8f6e\u91cd\u65b0\u4f30\u8ba1\u4e09\u4e2a\u6a21\u5757\u3002")
    reused_folds = []
    for group in ("E0","E1","E2"):
        for k in range(1,6):
            model = read(out/"models"/group/("fold"+str(k))/"model.json")
            if any(model.get(key) for key in ("historical_reuse","historical_m1_m2_reuse","reuse_m1_m2")) or model.get("reused_modules"):
                reused_folds.append(group+"/fold"+str(k))
    if reused_folds:
        text.append("\u5df2\u6709\u6298\u4ec5\u8865\u7f3a\u5931\u6a21\u5757\uff1a"+", ".join(reused_folds)+"\u3002")
    text += ["", "## \u4e3b\u8981\u5dee\u503c\u53ca95%\u7f6e\u4fe1\u533a\u95f4", "",
             "|\u8bc4\u4ef7\u96c6|\u6bd4\u8f83|\u6307\u6807|\u5dee\u503c|95%\u533a\u95f4|",
             "|---|---|---|---:|---|"]
    selected = selection.get("frozen_group","E1")
    comparisons = {selected+" minus E0","E2 minus E1","E2 minus E3"}
    selected_metrics = {"natural_path_accuracy","interference_path_accuracy",
                        "natural_node_accuracy","ap_vr","cf_correction_rms","cf_er",
                        "cf_over_budget_rate"}
    for scope in ("dev","test","new60","nli4ct","medpic"):
        if scope not in metrics:
            continue
        for row in metrics[scope].get("bootstrap",[]):
            use = row["comparison"] in comparisons and row["metric"] in selected_metrics
            if scope in ("nli4ct","medpic"):
                use = row["comparison"] == selected+" minus raw" and row["metric"] == "natural_native_accuracy"
            if not use:
                continue
            percent = row["metric"] != "cf_correction_rms"
            text.append("|"+scope+"|"+row["comparison"]+"|"+row["metric"]+"|"+
                        fmt(row.get("estimate"),percent)+"|["+
                        fmt(row.get("low"),percent)+", "+fmt(row.get("high"),percent)+"]|")
    details = read(out/"feature_rank_and_heads.json",[])
    if details:
        text += ["", "## \u5934\u9009\u62e9\u4e0e\u7279\u5f81\u79e9", "",
                 "|\u65b9\u6cd5|S\u6709\u6548\u79e9|N\u6709\u6548\u79e9|\u6295\u5f71\u7279\u5f81\u6709\u6548\u79e9|", "|---|---:|---:|---:|"]
        for row in details:
            text.append("|"+row["method"]+"|"+str(row["S_effective_rank"])+"|"+
                        str(row["N_effective_rank"])+"|"+
                        fmt(row.get("projected_feature_entropy_effective_rank"))+"|")
        text += ["", "\u5404\u7ec451\u4e2a\u5934\u3001\u5c42\u5206\u5e03\u53ca\u53ef\u7528\u8ba1\u7b97\u6210\u672c\u89c1 feature_rank_and_heads.json\uff1b"
                 "\u5934\u91cd\u5408\u7387\u4e0e\u5b50\u7a7a\u95f4\u4e3b\u5939\u89d2\u89c1 model_comparison.csv\u3002"
                 "\u4e3b\u5939\u89d2\u5728\u540c\u4e00\u539f\u59cb\u5934\u8f93\u51fa\u5750\u6807\u4e2d\u8ba1\u7b97\uff0c\u5e76\u8ba1\u5165\u5404\u7ec4\u6807\u51c6\u5316\u5c3a\u5ea6\u3002", ""]
    speed = read(out/"speed_optimization.json")
    eval_performance = read(out/"evaluation_performance.json")
    if speed or eval_performance:
        text += ["## \u8ba1\u7b97\u6548\u7387", ""]
        efficiency = []
        patch_speedup = speed.get("patch_speedup_measured")
        if patch_speedup is not None:
            efficiency.append("\u5355\u8f93\u5165\u524d\u5411\u4fee\u8865\u5c40\u90e8\u52a0\u901f"+f"{patch_speedup:.2f}"+"\u500d\uff0c\u5b9e\u6d4b\u8bc4\u5206\u4fdd\u6301\u4e00\u81f4\u3002")
        single = speed.get("single_extraction_per_second")
        if single is not None:
            efficiency.append("\u5355\u5361\u5355\u8fdb\u7a0b\u63d0\u53d6\u5b9e\u6d4b\u7ea6"+f"{single:.2f}"+"\u4e2a\u65b0\u8f93\u5165/\u79d2\uff1b"
                              "\u53cc\u8fdb\u7a0b\u672a\u63d0\u9ad8\u5408\u8ba1\u541e\u5410\uff0c\u4fdd\u7559\u6bcf\u5361\u4e00\u4e2a\u8fdb\u7a0b\u3002")
        if speed:
            efficiency.append("\u8f93\u5165\u7d22\u5f15\u6309\u9700\u8bfb\u53d6\uff0c\u6fc0\u6d3b\u7f13\u5b58\u4f7f\u7528FP16\u5185\u5b58\u6620\u5c04\uff0c\u907f\u514d\u53cd\u590d\u89e3\u538b\u3002")
        metric_speed = eval_performance.get("pair_statistics_speedup",speed.get("metric_speedup_measured"))
        path_speed = eval_performance.get("path_speedup",speed.get("path_replay_speedup_measured"))
        if metric_speed is not None and path_speed is not None:
            efficiency.append("CPU\u57fa\u51c6\u4e2d\u6307\u6807\u6c47\u603b\u52a0\u901f"+f"{metric_speed:.2f}"+"\u500d\uff0c"
                              "\u8def\u5f84\u91cd\u653e\u52a0\u901f"+f"{path_speed:.2f}"+"\u500d\uff1b\u4ecd\u6267\u884c2,000\u6b21\u6765\u6e90\u91cd\u91c7\u6837\u3002")
        text += ["".join(efficiency),
                 "\u4e0a\u8ff0\u500d\u6570\u9488\u5bf9\u76f8\u5e94\u5c40\u90e8\u6b65\u9aa4\uff0c\u4e0d\u4ee3\u8868\u6574\u8f6e\u5b9e\u9a8c\u603b\u65f6\u957f\u7684\u52a0\u901f\u500d\u6570\u3002"
                 "\u6d4b\u91cf\u8303\u56f4\u4e0e\u5b8c\u6574\u8bb0\u5f55\u89c1 speed_optimization.json \u548c evaluation_performance.json\u3002", ""]
    (out / "REPORT.zh.md").write_text("\n".join(text), encoding="utf-8")
    return str(out / "REPORT.zh.md")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("dev","cv","all","report"), default="report")
    p.add_argument("--frozen-group")
    args = p.parse_args()
    if args.stage == "dev":
        evaluate_development()
    elif args.stage == "cv":
        evaluate_cv()
    elif args.stage == "all":
        evaluate_all(args.frozen_group)
    else:
        generate_report()
