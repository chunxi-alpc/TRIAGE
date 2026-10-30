"""Reanalyse saved keys without touching official output, fitting, or tuning."""
import argparse
import json
import math
from pathlib import Path
from collections import Counter,defaultdict
import torch
from src.aster.manuscript_core import PairMoments,FitConfig,ExcludedHead,subspace_overlap
from .common import BASE,SEED,load_records,mean_difference,unit,cosine,source_mean_ci,write_json,holm

def fit(rows,head,rank):
    moments=PairMoments()
    for r in rows:
        moments.update(r['offline']['positive'][head],r['offline']['negative'][head])
    return moments.estimate(FitConfig(rank,4))

def energy(rows,head,basis):
    numerator=denominator=0.
    for r in rows:
        d=(r['offline']['positive'][head]-r['offline']['negative'][head]).double()
        numerator+=float((d@basis).square().sum(-1).mean())
        denominator+=float(d.square().sum(-1).mean())
    return None if denominator==0 else numerator/denominator

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',default=str(BASE));p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    base=Path(args.base);rows=load_records(base);expert=json.loads((base/'analysis/expert.json').read_text());legacy=json.loads((base/'analysis/statistics.json').read_text())
    train=[r for r in rows if r['row']['split']=='fit'];test=[r for r in rows if r['row']['split']=='confirm']
    rank=expert['config']['rank'];sources=sorted({r['row']['source_id'] for r in train}); per_head={};raw=[];pvalues={}
    for name in expert['selected_heads']:
        head=tuple(map(int,name.split(':')));c=expert['candidates'][name];v=torch.tensor(c['direction'],dtype=torch.float64);mu=torch.tensor(c['mean'],dtype=torch.float64);basis=torch.tensor(c['basis'],dtype=torch.float64)
        observed=[]
        for r in test:
            d=mean_difference(r,head);row={'head':name,'pair_id':r['row']['pair_id'],'source_id':r['row']['source_id'],
                'directed_projection':float(d@v),'mean_projection':float(d@unit(mu)),'cosine':cosine(d,v),
                'spectral_minus_mean_projection':float(d@(v-unit(mu)))};observed.append(row);raw.append(row)
        rng=torch.Generator().manual_seed(SEED);random_rows=[]
        for index in range(32):
            direction=unit(torch.randn(v.shape,generator=torch.Generator().manual_seed(SEED+index),dtype=torch.float64))
            scores=[dict(source_id=r['row']['source_id'],value=cosine(mean_difference(r,head),direction)) for r in test]
            random_rows.append({'seed':SEED+index,'index':index,'mean_cosine':source_mean_ci(scores,'value')['estimate']})
        real=source_mean_ci(observed,'cosine');pvalue=(1+sum(r['mean_cosine']>=real['estimate'] for r in random_rows))/33;pvalues[name]=pvalue
        halves=[]
        for repeat in range(20):
            permutation=torch.randperm(len(sources),generator=torch.Generator().manual_seed(SEED+repeat)).tolist();left={sources[i] for i in permutation[:len(sources)//2]}
            a=[r for r in train if r['row']['source_id'] in left];b=[r for r in train if r['row']['source_id'] not in left]
            try:
                ca,cb=fit(a,head,rank),fit(b,head,rank);ba,bb=(torch.tensor(x['basis'],dtype=torch.float64) for x in [ca,cb]);va,vb=(torch.tensor(x['direction'],dtype=torch.float64) for x in [ca,cb])
                halves.append({'seed':SEED+repeat,'sources_left':sorted(left),'sources_right':sorted(set(sources)-left),'subspace_overlap':subspace_overlap(ba,bb),'direction_cosine':cosine(va,vb)})
            except ExcludedHead as exc:
                halves.append({'seed':SEED+repeat,'technical_failure':str(exc)})
        null_overlap=[]
        rng=torch.Generator().manual_seed(SEED)
        for index in range(64):
            random_basis=torch.linalg.qr(torch.randn(len(v),rank,generator=rng,dtype=torch.float64)).Q
            null_overlap.append(subspace_overlap(basis,random_basis))
        per_head[name]={'rank':rank,'train_energy':energy(train,head,basis),'heldout_energy':energy(test,head,basis),
            'mean_norm_retention':c['mean_norm_retention'],'spectral_mean_cosine':cosine(v,mu),
            'identity_expected_cosine':c['mean_norm_retention'],'train_spectrum':c['eigenvalues'],
            'heldout_source_cosine':real,'heldout_spectral_minus_mean':source_mean_ci(observed,'spectral_minus_mean_projection'),
            'random_directions':random_rows,'random_one_sided_p':pvalue,'split_half':halves,'random_subspace_overlap':null_overlap,
            'old_single_half_overlap':legacy[name].get('guideline_half_overlap')}
    adjusted=holm(pvalues)
    for name,value in adjusted.items():per_head[name]['random_p_holm']=value
    all_candidates=expert['candidates'];summary={'status':'complete','kind':'CPU analysis of existing real-model key tensors; no new model inference',
        'accepted_pairs':len(rows),'accepted_source_clusters':len({r['row']['source_id'] for r in rows}),
        'fit_pairs':len(train),'fit_sources':len(sources),'confirm_pairs':len(test),'confirm_sources':len({r['row']['source_id'] for r in test}),
        'physician_adjudicated_pairs':0,'rank':rank,'selected_heads':expert['selected_heads'],
        'selected_head_mean_norm_retention_range':[min(x['mean_norm_retention'] for x in per_head.values()),max(x['mean_norm_retention'] for x in per_head.values())],
        'all_candidate_train_energy_mean':sum(c['spectral_energy_retention'] for c in all_candidates.values())/len(all_candidates),
        'all_candidate_heldout_energy_mean':sum(s['heldout_all_difference_energy'] for s in legacy.values() if s['heldout_all_difference_energy'] is not None)/sum(s['heldout_all_difference_energy'] is not None for s in legacy.values()),
        'random_directions_per_head':32,'split_half_repeats':20,'random_subspaces_per_head':64,'independent_unit':'source guideline cluster',
        'claim':'supports reproducible age-counterfactual direction structure; does not demonstrate non-age clinical applicability specificity',
        'official_results_used_for_tuning':False,'new_frozen_parameters_selected':False,'technical_failures':sum('technical_failure' in x for c in per_head.values() for x in c['split_half'])}
    write_json(out/'summary.json',summary);write_json(out/'per_head.json',per_head);write_json(out/'source_projection.json',raw)
    print(json.dumps({'status':'complete','pairs':len(rows),'sources':summary['accepted_source_clusters'],'output':str(out)}))
if __name__=='__main__':main()
