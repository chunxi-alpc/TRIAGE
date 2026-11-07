"""Source-cluster paired bootstrap; ratios are never clipped."""
from .common import *
def estimates(t,e):
    t=np.asarray(t,dtype=float);e=np.asarray(e,dtype=float);valid=np.abs(t)>=EPSILON
    def ratio(a,b):return float(a/b) if b!=0 else None
    return dict(original_effective=float(np.mean(e[valid]/t[valid])) if valid.any() else None,
      weighted_all=ratio(np.sign(t)@e,np.abs(t).sum()),regression_all=ratio(t@e,t@t),
      weighted_effective=ratio(np.sign(t[valid])@e[valid],np.abs(t[valid]).sum()),regression_effective=ratio(t[valid]@e[valid],t[valid]@t[valid]))
def cluster_counts(sources,n=BOOTSTRAPS,seed=SEED):
    keys=sorted(set(sources));ix={k:i for i,k in enumerate(keys)};rng=np.random.default_rng(seed)
    counts=rng.multinomial(len(keys),np.ones(len(keys))/len(keys),size=n) if keys else np.zeros((n,0),int)
    return keys,np.array([ix[s] for s in sources],dtype=int),counts
def summarize(records,bootstrap=True):
    if not records:return dict(pairs=0,estimates=estimates([],[]))
    t=np.array([r["te"] for r in records]);e=np.array([r.get("e",(r["ie_forward"]-r["ie_reverse"])/2) for r in records])
    result=dict(pairs=len(t),effective_pairs=int((np.abs(t)>=EPSILON).sum()),zero_denominators=int((t==0).sum()),
      small_nonzero_denominators=int(((np.abs(t)<EPSILON)&(t!=0)).sum()),
      nodes=len({r.get("group_id",r["pair_id"]) for r in records}),sources=len({r["source_id"] for r in records}),
      abs_TE=distribution(np.abs(t)),estimates=estimates(t,e),
      mean_forward=float(np.mean([r["ie_forward"] for r in records])),mean_reverse=float(np.mean([r["ie_reverse"] for r in records])))
    if bootstrap:
        keys,ix,counts=cluster_counts([r["source_id"] for r in records]);valid=np.abs(t)>=EPSILON
        terms=np.column_stack([np.where(valid,e/np.where(t!=0,t,1),0),valid,np.sign(t)*e,np.abs(t),t*e,t*t,
          np.where(valid,np.sign(t)*e,0),np.where(valid,np.abs(t),0),np.where(valid,t*e,0),np.where(valid,t*t,0)])
        sums=np.zeros((len(keys),terms.shape[1]));np.add.at(sums,ix,terms);vals=counts@sums;ci={}
        for name,(a,b) in zip(result["estimates"],[(0,1),(2,3),(4,5),(6,7),(8,9)]):
            ok=vals[:,b]!=0
            ci[name]=dict(low=float(np.quantile(vals[ok,a]/vals[ok,b],.025)) if ok.any() else None,
              high=float(np.quantile(vals[ok,a]/vals[ok,b],.975)) if ok.any() else None,estimable_replicates=int(ok.sum()))
        result["ci95"]=ci
    return result
def ranking_stability(matrix,sources,heads,k):
    matrix=np.asarray(matrix,dtype=float);heads=np.asarray(heads,int);keys,ix,counts=cluster_counts(sources)
    sums=np.zeros((len(keys),matrix.shape[1]));np.add.at(sums,ix,matrix);sizes=np.bincount(ix,minlength=len(keys))
    scores=counts@sums/(counts@sizes)[:,None];k=min(k,len(heads))
    order=np.argsort(heads);scores=scores[:,order];heads=heads[order];matrix=matrix[:,order]
    ranks=np.argsort(-scores,axis=1,kind="stable")[:,:k];original=np.argsort(-matrix.mean(0),kind="stable")[:k]
    hits=np.bincount(ranks.ravel(),minlength=len(heads))/BOOTSTRAPS
    overlap=np.array([len(set(row)&set(original)) for row in ranks])
    random_overlap=np.random.default_rng(SEED).hypergeometric(k,len(heads)-k,k,size=BOOTSTRAPS);weights=np.abs(matrix.mean(0))
    return dict(pairs=len(matrix),sources=len(keys),candidate_count=len(heads),k=k,
      inclusion_frequency=[dict(head=int(h),frequency=float(f)) for h,f in zip(heads,hits)],
      overlap_with_original=distribution(overlap),jaccard=distribution(overlap/(2*k-overlap)),
      random_overlap=distribution(random_overlap),random_expected_overlap=k*k/len(heads),
      random_jaccard=distribution(random_overlap/(2*k-random_overlap)),
      effective_heads=float(weights.sum()**2/(weights@weights)) if weights@weights else None,
      scope="仅梯度排名或固定候选内精确排名重采样；不重新选择前缀，不验证完整选头流程。")
