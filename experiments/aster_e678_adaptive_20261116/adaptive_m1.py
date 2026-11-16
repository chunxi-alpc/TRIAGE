"""Source-level adaptive head selection. All head IDs are checkpoint-local."""
from dataclasses import dataclass, asdict
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import binom
from experiments.aster_cr_retrain_paired_v2_20261111.common import write, lines
from experiments.aster_cr_retrain_paired_v2_20261111.evaluation import write_csv

KINDS = ("A", "P", "C", "F")

@dataclass(frozen=True)
class Config:
    seed: int = 20261116
    permutations: int = 2000
    bootstrap: int = 2000
    screen_q: float = .20
    confirm_q: float = .05
    donors: int = 5
    min_sources: int = 16

def adjust(p, method="BH"):
    """Step-up adjusted p values; include every hypothesis, including p=1."""
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p values must be finite probabilities")
    if method not in ("BH", "BY"):
        raise ValueError(method)
    m = len(p)
    if not m:
        return p.copy()
    order = np.argsort(p, kind="stable")
    factor = np.sum(1 / np.arange(1, m + 1)) if method == "BY" else 1.
    ranked = p[order] * m * factor / np.arange(1, m + 1)
    result = np.empty(m)
    result[order] = np.minimum(1., np.minimum.accumulate(ranked[::-1])[::-1])
    return result

def representatives(pairs, split):
    if any(p.get("split") != split for p in pairs):
        raise ValueError("Selection received records outside " + split)
    by = {}
    for p in pairs:
        if p["kind"] in KINDS:
            key = p["kind"], p["source_id"]
            if key not in by or p["pair_id"] < by[key]["pair_id"]:
                by[key] = p
    return {k: [by[t] for t in sorted(by) if t[0] == k] for k in KINDS}

class ScopedStore:
    """Reject out-of-split access before touching the underlying feature store."""
    def __init__(self, store, ids):
        self.store, self.ids = store, frozenset(ids)
    def _check(self, uid):
        if uid not in self.ids:
            raise ValueError("Feature access outside declared split: " + uid)
    def get(self, uid):
        self._check(uid)
        return self.store.get(uid)
    def ell(self, uid):
        self._check(uid)
        return self.store.ell(uid)
    def heads(self, uid, selected=None):
        a = np.asarray(self.get(uid)["heads"])
        a = a.reshape(-1, a.shape[-1])
        return a if selected is None else a[np.asarray(selected, int)].reshape(-1)

def pair_ids(pairs):
    return {p[key] for p in pairs for key in ("left", "right")}

def screen_arrays(gradients, differences, config=Config()):
    """Permute source correspondence. Chunking changes memory use, never head coverage."""
    g, d = np.asarray(gradients, np.float64), np.asarray(differences, np.float64)
    if g.shape != d.shape or g.ndim != 3 or len(g) < 2:
        raise ValueError("Expected matching [source, head, dimension] arrays")
    if not np.isfinite(g).all() or not np.isfinite(d).all():
        raise ValueError("Nonfinite attribution inputs")
    n, h, _ = g.shape
    rng = np.random.default_rng(config.seed)
    perms = np.stack([rng.permutation(n) for _ in range(config.permutations)])
    observed, p = np.empty(h), np.empty(h)
    for start in range(0, h, 32):
        stop = min(h, start + 32)
        cross = np.einsum("shd,thd->sth", g[:, start:stop], d[:, start:stop], optimize=True)
        observed[start:stop] = np.abs(cross[np.arange(n), np.arange(n)]).mean(0)
        exceed = np.zeros(stop - start, int)
        for batch in range(0, config.permutations, 100):
            null = np.abs(cross[np.arange(n)[None, :], perms[batch:batch + 100]]).mean(1)
            exceed += (null >= observed[None, start:stop]).sum(0)
        p[start:stop] = (1 + exceed) / (config.permutations + 1)
    return observed, p

def screen_candidates(pairs, store, config=Config()):
    reps = representatives(pairs, "fit")
    records = []
    for kind in KINDS:
        g, d = [], []
        for pair in reps[kind]:
            l, r = store.get(pair["left"]), store.get(pair["right"])
            left = np.asarray(l["heads"], np.float32)
            right = np.asarray(r["heads"], np.float32)
            g.append(np.asarray(l["grads"], np.float32).reshape(-1, left.shape[-1]))
            d.append((right - left).reshape(-1, left.shape[-1]))
        observed, p = screen_arrays(g, d, config)
        records.extend(dict(kind=kind, head=h, attribution=float(observed[h]),
                            p_screen=float(p[h]), fit_sources=len(g)) for h in range(len(p)))
    q = adjust([r["p_screen"] for r in records])
    for r, v in zip(records, q):
        r.update(q_screen=float(v), candidate=bool(v <= config.screen_q))
    return records

def donor_plan(reps, maximum=10, seed=20261116):
    """One deterministic representative per source; nested donor sets 3/5/10."""
    rng = np.random.default_rng(seed)
    plan = {}
    for kind in KINDS:
        for target in reps[kind]:
            pool = [p for p in reps[kind] if p["source_id"] != target["source_id"]]
            order = rng.permutation(len(pool))[:maximum]
            plan[(kind, target["source_id"])] = [pool[i] for i in order]
    return plan

def sign_p(d):
    d = np.asarray(d, float)
    valid = np.isfinite(d)
    positive = ((d > 0) & valid).sum(axis=0)
    nonzero = ((d != 0) & valid).sum(axis=0)
    # Exact zeros are ties, discarded in the exact binomial sign test.
    return binom.sf(positive - 1, nonzero, .5)

def bootstrap_weights(sources, draws, seed):
    sources = list(sources)
    if len(set(sources)) != len(sources):
        raise ValueError("Bootstrap rows must be independent sources")
    n = len(sources)
    return np.random.default_rng(seed).multinomial(n, np.full(n, 1/n), size=draws)

def combine_sets(records, total_heads, heads_per_layer):
    selected = {k: sorted({int(r["head"]) for r in records if r["kind"] == k and r["confirmed"]})
                for k in KINDS}
    n = set(selected["A"]) | set(selected["P"])
    e = set(selected["C"]) | set(selected["F"])
    heads = sorted(n | e)
    if heads_per_layer <= 0 or any(h < 0 or h >= total_heads for h in heads):
        raise ValueError("Invalid checkpoint head geometry")
    return dict(sets=selected, H_N=sorted(n), H_E=sorted(e), H_shared=sorted(n & e),
                nuisance_dominant=sorted(n - e), evidence_dominant=sorted(e - n),
                heads=heads, selected_count=len(heads), total_heads=total_heads,
                selection_ratio=len(heads)/total_heads,
                nuisance_count=len(n), evidence_count=len(e), shared_count=len(n & e),
                final_by_kind={k:len(selected[k]) for k in KINDS},
                layer_distribution={str(i):sum(h//heads_per_layer == i for h in heads)
                                    for i in range((total_heads + heads_per_layer - 1)//heads_per_layer)},
                status="complete" if n and e else "selection_incomplete")

def confirm(matrix, sources, hypotheses, total_heads, heads_per_layer, config=Config(), stability=True):
    """One row per source, one column per candidate (category, head)."""
    d = np.asarray(matrix, float)
    if d.shape != (len(sources), len(hypotheses)):
        raise ValueError("Source/effect/hypothesis shape mismatch")
    if len(set((r["kind"], r["head"]) for r in hypotheses)) != len(hypotheses):
        raise ValueError("Duplicate hypotheses")
    if not len(sources):
        raise ValueError("No development sources")
    valid = np.isfinite(d)
    n = valid.sum(0)
    filled = np.where(valid, d, 0.)
    means = np.divide(filled.sum(0), n, out=np.zeros(len(n)), where=n>0)
    p = np.asarray(sign_p(d))
    by, bh = adjust(p, "BY"), adjust(p, "BH")
    w = bootstrap_weights(sources, config.bootstrap, config.seed + 1)
    counts = w @ valid.astype(float)
    sums = w @ filled
    boot = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts>0)
    ci = np.zeros((2, len(hypotheses)))
    for j in range(len(hypotheses)):
        a = boot[:, j]
        a = a[np.isfinite(a)]
        if len(a):
            ci[:, j] = np.quantile(a, [.025, .975])
    records = []
    for j, hypothesis in enumerate(hypotheses):
        adequate = bool(n[j] >= config.min_sources)
        records.append(dict(hypothesis, valid_sources=int(n[j]),
            positive_sources=int(np.sum(d[:, j] > 0)),
            nonzero_sources=int(np.sum(valid[:, j] & (d[:, j] != 0))),
            excess_mean=float(means[j]) if n[j] else None, p_confirm=float(p[j]),
            q_BY=float(by[j]), q_BH=float(bh[j]),
            ci_low=float(ci[0, j]) if n[j] else None, ci_high=float(ci[1, j]) if n[j] else None,
            evidence_status="adequate" if adequate else "insufficient_statistical_evidence",
            confirmed=bool(adequate and by[j] <= config.confirm_q and ci[0, j] > 0),
            confirmed_BH=bool(adequate and bh[j] <= config.confirm_q and ci[0, j] > 0)))
    summary = combine_sets(records, total_heads, heads_per_layer)
    summary["BH_sensitivity"] = combine_sets(
        [dict(r, confirmed=r["confirmed_BH"]) for r in records], total_heads, heads_per_layer)
    summary["candidate_by_kind"] = dict(Counter(r["kind"] for r in hypotheses))
    for k in KINDS:
        summary["candidate_by_kind"].setdefault(k, 0)
    summary["N_eff"] = {}
    for k in KINDS:
        x = np.array([r["excess_mean"] for r in records if r["kind"] == k
                      and r["excess_mean"] is not None and r["excess_mean"] > 0])
        summary["N_eff"][k] = float(x.sum()**2 / (x @ x)) if len(x) else 0.
    summary["N_eff_population"] = "screened candidates with positive mean excess, by category"
    if stability and len(hypotheses):
        # Conditional stability of BY sign-test support; CI is reported separately.
        pos = w @ ((d > 0) & valid).astype(float)
        nz = w @ ((d != 0) & valid).astype(float)
        bp = binom.sf(pos - 1, nz, .5)
        chosen = np.stack([(adjust(row, "BY") <= config.confirm_q) for row in bp])
        chosen &= (boot > 0) & (counts >= config.min_sources)
        frequencies = chosen.mean(0)
        for r, f in zip(records, frequencies):
            r["bootstrap_BY_positive_mean_frequency"] = float(f)
        reference = set(summary["heads"])
        jaccard = []
        for row in chosen:
            hh = {hypotheses[j]["head"] for j in np.flatnonzero(row)}
            jaccard.append(len(hh & reference)/len(hh | reference) if hh | reference else 1.)
        summary["stability"] = dict(draws=config.bootstrap,
            definition="source-resampled BY sign rejection plus positive mean; fixed screening; CI not reapplied",
            mean_union_jaccard=float(np.mean(jaccard)), reference="primary confirmed head union")
    return records, summary

def measure(pairs, screen, patcher, config=Config()):
    reps = representatives(pairs, "dev")
    donors = donor_plan(reps, maximum=10, seed=config.seed)
    # q=.30 is measured for the prespecified sensitivity only. Main family stays q<=.20.
    candidates = {k: [r["head"] for r in screen if r["kind"] == k and r["q_screen"] <= .30]
                  for k in KINDS}
    effects, plans = [], []
    for k in KINDS:
        for p in reps[k]:
            dd = donors[(k, p["source_id"])]
            plans.append(dict(kind=k, target_source=p["source_id"], target_pair=p["pair_id"],
                              donors=[dict(source_id=x["source_id"], pair_id=x["pair_id"]) for x in dd]))
            if not candidates[k]:
                continue
            true = patcher.effects(p, p, candidates[k], sham=False)
            shams = [patcher.effects(p, donor, candidates[k], sham=True) for donor in dd]
            for j, h in enumerate(candidates[k]):
                effects.append(dict(kind=k, head=h, source_id=p["source_id"], pair_id=p["pair_id"],
                                    true_effect=float(true[j]), sham_effects=[float(a[j]) for a in shams]))
    return effects, plans

def analyze(screen, effects, dev_sources, total_heads, heads_per_layer, outdir, config=Config()):
    outdir = Path(outdir)
    scenarios = [("main", .20, .05, 5), ("screen_010", .10, .05, 5),
                 ("screen_030", .30, .05, 5), ("confirm_001", .20, .01, 5),
                 ("confirm_010", .20, .10, 5), ("donors_3", .20, .05, 3),
                 ("donors_10", .20, .05, 10)]
    lookup = {(r["kind"], r["head"], r["source_id"]):r for r in effects}
    summaries = {}
    for name, sq, cq, donor_count in scenarios:
        hypotheses = [r for r in screen if r["q_screen"] <= sq]
        d = np.full((len(dev_sources), len(hypotheses)), np.nan)
        for i, s in enumerate(dev_sources):
            for j, h in enumerate(hypotheses):
                r = lookup.get((h["kind"], h["head"], s))
                if r and len(r["sham_effects"]) >= donor_count:
                    d[i, j] = r["true_effect"] - np.mean(r["sham_effects"][:donor_count])
        cfg = Config(**dict(asdict(config), screen_q=sq, confirm_q=cq, donors=donor_count))
        records, summary = confirm(d, dev_sources, hypotheses, total_heads, heads_per_layer,
                                   cfg, stability=name == "main")
        for record in records:
            measured=[lookup[(record["kind"],record["head"],source)] for source in dev_sources
                      if (record["kind"],record["head"],source) in lookup
                      and len(lookup[(record["kind"],record["head"],source)]["sham_effects"])>=donor_count]
            record["true_effect_mean"]=float(np.mean([r["true_effect"] for r in measured])) if measured else None
            record["sham_effect_mean"]=float(np.mean([np.mean(r["sham_effects"][:donor_count])
                                                       for r in measured])) if measured else None
        summary["category_excess_effect"]={}
        for k in KINDS:
            vals=[r["excess_mean"] for r in records if r["kind"]==k and r["excess_mean"] is not None]
            selected_vals=[r["excess_mean"] for r in records if r["kind"]==k and r["confirmed"]]
            summary["category_excess_effect"][k]=dict(
                candidate_mean=float(np.mean(vals)) if vals else None,
                confirmed_mean=float(np.mean(selected_vals)) if selected_vals else None)
        summary.update(configuration=asdict(cfg), records=records)
        write(outdir / (name + ".json"), summary)
        write_csv(outdir / (name + "_heads.csv"), records)
        summaries[name] = summary
    main = summaries["main"]
    confirmed_lookup={(r["kind"],r["head"]):r for r in main["records"]}
    all_heads=[dict(r, **{k:v for k,v in confirmed_lookup.get((r["kind"],r["head"]),{
        "evidence_status":"not_screened","confirmed":False,"p_confirm":None,
        "q_BY":None,"q_BH":None,"ci_low":None,"ci_high":None}).items()
        if k not in r}) for r in screen]
    write(outdir/"per_head_statistics.json",all_heads)
    write_csv(outdir/"per_head_statistics.csv",all_heads)
    main_set = set(main["heads"])
    for name, summary in summaries.items():
        hs = set(summary["heads"])
        summary["jaccard_with_main"] = len(hs & main_set)/len(hs | main_set) if hs | main_set else 1.
    write(outdir / "sensitivity.json", summaries)
    write_csv(outdir / "sensitivity.csv", [dict(scenario=k, selected_count=v["selected_count"],
        status=v["status"], jaccard_with_main=v["jaccard_with_main"]) for k,v in summaries.items()])
    return main

def select(fit_pairs, dev_pairs, fit_store, dev_store, patcher, outdir,
           total_heads, heads_per_layer, config=Config()):
    from .staged import make_plan, finish_plan
    plan = make_plan("E1", fit_pairs, fit_store, outdir, dev_pairs=dev_pairs,
                     total_heads=total_heads, heads_per_layer=heads_per_layer, config=config)
    return finish_plan(plan, patcher, outdir)
