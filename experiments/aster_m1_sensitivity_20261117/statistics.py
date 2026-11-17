"""Fixed statistical rules; CPU only and no downstream fitting."""
import itertools
import numpy as np
from scipy.stats import binomtest
KINDS=("A","P","C","F")

def scenarios(cfg):
    d=cfg["defaults"]
    out=[dict(name="default",**d)]
    for k,vs in cfg["grid"].items():
        out += [dict(name=k+"_"+str(v),**{**d,k:v}) for v in vs if v!=d[k]]
    return out+cfg["joint"]

def robust_z(scores):
    a=np.asarray(scores,float)
    med=np.median(a)
    scale=1.4826*np.median(np.abs(a-med))
    if scale==0:
        return np.where(a>med,np.inf,np.where(a<med,-np.inf,0.))
    return (a-med)/scale

def candidates(attribution,folds,selected_sources,settings):
    """[category, source, head] input; source blocks never cross stages."""
    a=np.asarray(attribution,float)
    selected=np.asarray(selected_sources,int)
    folds=np.asarray(folds,int)
    hits=np.zeros((len(KINDS),a.shape[2]),int)
    for f in range(5):
        ix=selected[folds[selected]!=f]
        if not len(ix): raise ValueError("Empty screening training fold")
        for k in range(len(KINDS)):
            z=robust_z(a[k,ix].mean(0))
            hits[k] += z>=settings["gradient_threshold"]
    frequency=hits/5
    return frequency>=settings["candidate_frequency"],frequency

def bh(p):
    p=np.asarray(p,float)
    if p.ndim!=1 or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError("Invalid probabilities")
    if not len(p): return p.copy()
    order=np.argsort(p,kind="stable")
    q=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    out=np.empty_like(q);out[order]=np.minimum(q,1)
    return out

def flip_p(matrix,two_sided,draws,seed,exact_max=16):
    """One random sign per unique source, jointly across heads."""
    x=np.asarray(matrix,float)
    valid=np.isfinite(x);n=valid.sum(0)
    a=np.where(valid,x,0.)
    two=np.asarray(two_sided,bool)
    observed=a.sum(0)
    target=np.where(two,np.abs(observed),observed)
    counts=np.zeros(a.shape[1],np.int64)
    rng=np.random.default_rng(seed)
    # Rows with no measurement are removed so exact enumeration reflects actual units.
    a=a[valid.any(1)]
    exact=len(a)<=exact_max
    total=2**len(a) if exact else draws
    for start in range(0,total,2048):
        size=min(2048,total-start)
        if exact:
            ints=np.arange(start,start+size,dtype=np.uint64)
            signs=2*((ints[:,None]>>np.arange(len(a),dtype=np.uint64))&1).astype(float)-1
        else:
            signs=2*rng.integers(0,2,size=(size,len(a))).astype(float)-1
        perm=signs@a
        perm[:,two]=np.abs(perm[:,two])
        counts += (perm>=target[None,:]-1e-12).sum(0)
    p=counts/total if exact else (counts+1)/(total+1)
    p[n==0]=1
    return p

def source_statistics(matrix,two_sided,cfg,seed):
    x=np.asarray(matrix,float)
    valid=np.isfinite(x);n=valid.sum(0)
    z=np.where(valid,x,0.)
    mean=np.divide(z.sum(0),n,out=np.full(x.shape[1],np.nan),where=n>0)
    p=flip_p(x,two_sided,cfg["sign_flip_draws"],seed,cfg["exact_sign_flip_max_sources"])
    rng=np.random.default_rng(seed+1)
    weights=rng.multinomial(len(x),np.full(len(x),1/len(x)),size=cfg["bootstrap_draws"])
    denominator=weights@valid.astype(float)
    boot=np.divide(weights@z,denominator,out=np.full((len(weights),x.shape[1]),np.nan),where=denominator>0)
    low,high=np.nanquantile(boot,[.025,.975],axis=0)
    agree=np.mean(np.sign(boot)==np.sign(mean)[None,:],axis=0)
    positive=(x>0).sum(0);negative=(x<0).sum(0)
    sign_test=np.array([binomtest(int(a),int(a+b),.5,alternative="two-sided" if t else "greater").pvalue
                       if a+b else 1. for a,b,t in zip(positive,negative,two_sided)])
    return dict(mean=mean,p=p,ci_low=low,ci_high=high,direction_consistency=agree,
                valid_sources=n,positive_sources=positive,negative_sources=negative,
                source_direction_consistency=np.maximum(positive,negative)/np.maximum(n,1),
                sign_test_p=sign_test)

def select(mask,stats,settings):
    mask=np.asarray(mask,bool)
    q=np.full(mask.shape, np.nan)
    ix=np.flatnonzero(mask.ravel())
    p=np.asarray(stats["p"]).copy().ravel()
    n=stats["valid_sources"].ravel()
    p[n<settings["min_valid_sources"]]=1
    q.ravel()[ix]=bh(p[ix])
    chosen=mask & (q<=settings["fdr"]) & (stats["valid_sources"]>=settings["min_valid_sources"])
    chosen &= stats["direction_consistency"]>=settings["direction_consistency"]
    chosen[:2] &= stats["mean"][:2]>0
    return chosen,q

def sets(mask):
    ss={k:set(np.flatnonzero(mask[i]).tolist()) for i,k in enumerate(KINDS)}
    a,e=ss["A"]|ss["P"],ss["C"]|ss["F"]
    return {**ss,"AP":a,"CF":e,"overlap":a&e,"union":a|e}

def jaccard(a,b):
    return len(a&b)/len(a|b) if a|b else None
