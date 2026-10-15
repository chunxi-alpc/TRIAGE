#!/usr/bin/env python3
"""Analyze saved real model outputs, grouped inference, export paper figures."""
from __future__ import annotations
import hashlib, itertools, json, pathlib, sys, time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parents[1]
R=ROOT/'results'; F=ROOT/'figures'
F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
    'svg.fonttype':'none','pdf.fonttype':42,'savefig.dpi':300,'axes.titleweight':'normal'})
COLORS=['#333333','#0072B2','#D55E00']
RNG=np.random.default_rng(42)
def dump(path,obj):
    pathlib.Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False,default=str))
def save(fig,name):
    for ext in ['png','pdf','svg']: fig.savefig(F/(name+'.'+ext),bbox_inches='tight',dpi=300)
    plt.close(fig)
def ci(vals):
    vals=np.asarray(vals,float); vals=vals[np.isfinite(vals)]
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]
def group_ci(values,reps=5000):
    values=np.asarray(values,float)
    ix=RNG.integers(0,len(values),(reps,len(values)))
    return float(values.mean()),ci(values[ix].mean(1))
def fdr(p):
    p=np.asarray(p); order=np.argsort(p); out=np.empty(len(p))
    out[order]=np.minimum(1,np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1])
    return out
def boot_auc(y,pred,reps=3000):
    n=pred.shape[0]; vals=[]
    for _ in range(reps):
        ix=RNG.integers(0,n,n)
        vals.append(roc_auc_score(np.tile(y,n),pred[ix].reshape(-1)))
    return ci(vals)
def exact_auc_p(y,pred):
    n=len(pred); ranks=stats.rankdata(pred.reshape(-1)).reshape(n,4)
    npos=2*n; nneg=2*n
    flips=np.array(list(itertools.product([0,1],repeat=n)))
    contrib0=ranks@np.asarray(y); contrib1=ranks@(1-np.asarray(y))
    sums=flips@contrib1+(1-flips)@contrib0
    auc=(sums-npos*(npos+1)/2)/(npos*nneg)
    observed=roc_auc_score(np.tile(y,n),pred.reshape(-1))
    return float(np.mean(auc>=observed-1e-12))
def paired_permutation(values):
    # Float64 keeps the observed mean and permutation means in the same
    # precision, including the identity permutation in the exact tail count.
    values=np.asarray(values,dtype=np.float64)
    signs=2*np.array(list(itertools.product([0,1],repeat=len(values))))-1
    return float(np.mean(np.abs(signs@values/len(values))>=abs(values.mean())-1e-12))
def analyze_o1():
    df=pd.read_csv(R/'obs1/predictions.csv'); cfg=json.loads((R/'obs1/selected_hyperparameters.json').read_text())
    df=df[(df.split=='test')&(df.variant=='modified')]
    df=df[(df.method=='baseline')|df.apply(lambda r:r.strength==cfg['strengths'].get(r.method,-999),axis=1)]
    gids=sorted(df.group_id.unique()); n=len(gids); methods=['baseline','seka','pasta']; measures={}
    metrics=[]
    idx=RNG.integers(0,n,(5000,n))
    for method in methods:
        d=df[df.method==method]
        assert len(d)==2*n
        app=d[d.subset=='app'].set_index('group_id').loc[gids]
        non=d[d.subset=='non'].set_index('group_id').loc[gids]
        a=app.correct.to_numpy(); b=non.correct.to_numpy(); mis=non.misapplied.to_numpy()
        revert=(app.prior_reversion.to_numpy()+non.prior_reversion.to_numpy())/2
        h=2*a.mean()*b.mean()/(a.mean()+b.mean()) if a.mean()+b.mean() else 0.
        aa=a[idx].mean(1); bb=b[idx].mean(1); hh=np.divide(2*aa*bb,aa+bb,out=np.zeros(len(idx)),where=aa+bb>0)
        measures[method]={'A_app':a,'A_non':b,'M_non':mis,'P_revert':revert}
        for name,v in measures[method].items():
            point=float(v.mean()); interval=ci(v[idx].mean(1))
            metrics.append({'method':method,'metric':name,'estimate':point,'ci_low':interval[0],'ci_high':interval[1],
                'n_groups':n,'n_cases':n if name!='P_revert' else 2*n,'subset':'synthetic conditional-rule following'})
        interval=ci(hh); metrics.append({'method':method,'metric':'H_Acc','estimate':h,'ci_low':interval[0],'ci_high':interval[1],'n_groups':n,'n_cases':2*n})
    pd.DataFrame(metrics).to_csv(R/'obs1/metrics.csv',index=False)
    contrasts=[]; status={}
    for method in methods[1:]:
        for metric in ['A_app','A_non','M_non','P_revert']:
            delta=measures[method][metric]-measures['baseline'][metric]; interval=ci(delta[idx].mean(1))
            contrasts.append({'method':method,'metric':metric,'estimate':float(delta.mean()),'ci_low':interval[0],'ci_high':interval[1],
                'p_exact_two_sided':paired_permutation(delta),'n_groups':n})
        da=measures[method]['A_app']-measures['baseline']['A_app']; dm=measures[method]['M_non']-measures['baseline']['M_non']
        cda=ci(da[idx].mean(1)); cdm=ci(dm[idx].mean(1))
        status[method]='supported' if cda[0]>0 and cdm[0]>0 else 'not_observed'
    pd.DataFrame(contrasts).to_csv(R/'obs1/contrasts.csv',index=False)
    prior=pd.read_csv(R/'obs1/prior_screen.csv')
    summary={'clinical_prior_conflict_status':'not_testable','clinically_validated_eligible':0,'conditional_rule_status':status,
        'test_groups':n,'test_cases_per_method':2*n,'closedbook_cases_all_splits':len(prior),
        'stable_closedbook':int(prior.stable.sum()),'synthetic_prior_disagreements':int(prior.prior_disagreement.sum()),
        'selected_strengths':cfg['strengths'],'ci':'5000 paired group bootstrap, percentile 95%'}
    dump(R/'obs1/summary.json',summary)
    mdf=pd.DataFrame(metrics); cdf=pd.DataFrame(contrasts)
    fig,axs=plt.subplots(1,2,figsize=(9.4,3.5))
    for j,method in enumerate(methods):
        a=mdf[(mdf.method==method)&(mdf.metric=='A_app')].iloc[0]
        b=mdf[(mdf.method==method)&(mdf.metric=='A_non')].iloc[0]
        axs[0].errorbar(a.estimate,b.estimate,xerr=[[max(0,a.estimate-a.ci_low)],[max(0,a.ci_high-a.estimate)]],
            yerr=[[max(0,b.estimate-b.ci_low)],[max(0,b.ci_high-b.estimate)]],fmt=['o','s','^'][j],capsize=3,color=COLORS[j],label=method.upper(),ms=7)
    axs[0].set(xlabel='Applicable-rule accuracy',ylabel='Nonapplicable-rule accuracy',xlim=(-.04,1.04),ylim=(-.04,1.04))
    axs[0].legend(frameon=False,fontsize=8); axs[0].set_title('(a) Conditional-rule following')
    if all(np.array_equal(measures['baseline'][k],measures[m][k]) for m in methods[1:] for k in ['A_app','A_non']):
        axs[0].text(.97,.48,'All three methods coincide',ha='right',transform=axs[0].transAxes,fontsize=8)
    for j,method in enumerate(methods[1:]):
        dd=cdf[(cdf.method==method)&cdf.metric.isin(['A_app','A_non','M_non'])].set_index('metric').loc[['A_app','A_non','M_non']]
        yy=np.arange(3)+(j-.5)*.16
        axs[1].errorbar(dd.estimate,yy,xerr=[np.maximum(0,dd.estimate-dd.ci_low),np.maximum(0,dd.ci_high-dd.estimate)],
            fmt=['s','^'][j],capsize=3,color=COLORS[j+1],label=method.upper())
    axs[1].axvline(0,color='.6',lw=.8)
    axs[1].set(yticks=np.arange(3),yticklabels=['Applicable accuracy','Nonapplicable accuracy','Rule misuse rate'],xlabel='Change from baseline')
    axs[1].invert_yaxis(); axs[1].set_title('(b) Paired 95% confidence intervals'); axs[1].legend(frameon=False,fontsize=8)
    fig.suptitle(f'Synthetic rules only | {n} test groups, {2*n} cases per method | clinical eligible n = 0',fontsize=10)
    fig.tight_layout(); save(fig,'O1_faithfulness_misuse')
    return summary
def probe(keys,split,groups,L,h):
    X=keys[:,:,L,h,:].reshape(-1,128); tr=np.repeat(split=='train',4); te=np.repeat(split=='test',4)
    outputs={}
    for label,y in [('relevance',np.array([1,1,0,0])),('applicability',np.array([1,0,1,0]))]:
        Y=np.tile(y,len(keys))
        pipe=make_pipeline(StandardScaler(),LogisticRegression(C=1.,solver='liblinear',max_iter=1500,random_state=42))
        scores=cross_val_predict(pipe,X[tr],Y[tr],groups=np.repeat(groups[split=='train'],4),cv=GroupKFold(5),method='predict_proba')[:,1]
        cv_auc=roc_auc_score(Y[tr],scores)
        pipe.fit(X[tr],Y[tr]); pred=pipe.predict_proba(X[te])[:,1].reshape(-1,4)
        outputs[label]={'cv_auc':float(cv_auc),'test_auc':float(roc_auc_score(Y[te],pred.reshape(-1))),
            'test_p':exact_auc_p(y,pred),'pred':pred}
    return outputs
def analyze_o2():
    dat=np.load(R/'obs2/key_embeddings.npz'); keys=dat['keys']; controls=dat['same_side_keys']
    split=dat['split']; groups=dat['group_ids']; n=int((split=='test').sum())
    tr=split=='train'; dev=split=='dev'; te=split=='test'
    rel=np.stack([keys[:,0]-keys[:,2],keys[:,1]-keys[:,3]],1)
    app=np.stack([keys[:,0]-keys[:,1],keys[:,2]-keys[:,3]],1)
    inter=app[:,0]-app[:,1]; same=np.stack([keys[:,0]-controls[:,0],keys[:,1]-controls[:,1]],1)
    scale=np.linalg.norm(keys[tr],axis=-1).mean((0,1)); scale=np.maximum(scale,1e-12)
    def consistency(v): return np.linalg.norm(v.sum((0,1)),axis=-1)/np.maximum(np.linalg.norm(v,axis=-1).sum((0,1)),1e-12)
    dev_cons=consistency(rel[dev]); chosen=np.unravel_index(dev_cons.argmax(),dev_cons.shape)
    dump(R/'obs2/selected_head.json',{'layer':int(chosen[0]),'kv_head':int(chosen[1]),'criterion':'max development relevance displacement consistency; no test selection',
        'development_value':float(dev_cons[chosen]),'n_dev_groups':int(dev.sum())})
    relmag=np.linalg.norm(rel[te],axis=-1).mean((0,1))/scale
    appmag=np.linalg.norm(app[te],axis=-1).mean((0,1))/scale
    samemag=np.linalg.norm(same[te],axis=-1).mean((0,1))/scale
    intermag=np.linalg.norm(inter[te],axis=-1).mean(0)/scale
    test_consr=consistency(rel[te]); test_consa=consistency(app[te])
    rows=[]; predictions=[]
    for L in range(36):
        for h in range(8):
            p=probe(keys,split,groups,L,h)
            # Cross-threshold distance covers four units; same-side distance covers one.
            cross=np.linalg.norm(keys[te,0,L,h]-keys[te,1,L,h],axis=-1)/(4*scale[L,h])
            side=np.linalg.norm(same[te,:,L,h],axis=-1).mean(1)/scale[L,h]
            delta=cross-side
            rows.append({'layer':L,'kv_head':h,'relevance_shift':float(relmag[L,h]),'applicability_shift':float(appmag[L,h]),
                'same_side_shift':float(samemag[L,h]),'interaction_shift':float(intermag[L,h]),
                'relevance_consistency':float(test_consr[L,h]),'applicability_consistency':float(test_consa[L,h]),
                'cross_per_unit_minus_same_side_per_unit':float(delta.mean()),'numeric_control_p':paired_permutation(delta),
                **{f'{label}_{k}':v[k] for label,v in p.items() for k in ['cv_auc','test_auc','test_p']}})
            for label,v in p.items():
                for i,g in enumerate(groups[te]):
                    for s,state in enumerate(['RA','RN','IA','IN']):
                        predictions.append({'group_id':g,'layer':L,'kv_head':h,'state':state,'label':label,'probability':float(v['pred'][i,s])})
        print(f'analysis O2 layer {L+1}/36',flush=True)
    df=pd.DataFrame(rows)
    for col in ['relevance_test_p','applicability_test_p','numeric_control_p']: df[col+'_fdr']=fdr(df[col].to_numpy())
    df.to_csv(R/'obs2/layer_head_stats.csv',index=False)
    pd.DataFrame(predictions).to_csv(R/'obs2/probe_predictions.csv',index=False)
    L,h=map(int,chosen); p=probe(keys,split,groups,L,h)
    selected=df[(df.layer==L)&(df.kv_head==h)].iloc[0].to_dict()
    for label,y in [('relevance',[1,1,0,0]),('applicability',[1,0,1,0])]:
        selected[label+'_auc_ci']=boot_auc(y,p[label]['pred'])
    values={}
    for name,v in [('relevance',np.linalg.norm(rel[te,:,L,h],axis=-1).mean(1)/scale[L,h]),
                   ('applicability',np.linalg.norm(app[te,:,L,h],axis=-1).mean(1)/scale[L,h]),
                   ('same_side',np.linalg.norm(same[te,:,L,h],axis=-1).mean(1)/scale[L,h])]:
        m,c=group_ci(v); values[name]={'estimate':m,'ci95':c}
    summary={'status':'completed_exploratory','n_groups':len(groups),'n_test_groups':n,'test_states':4,
        'layers':36,'kv_heads':8,'selected_head':selected,'displacement':values,
        'n_heads_app_auc_significant_fdr':int((df.applicability_test_p_fdr<.05).sum()),
        'n_heads_relevance_auc_significant_fdr':int((df.relevance_test_p_fdr<.05).sum()),
        'limitations':['single fictional protocol grammar','question wording and length confounded with relevance',
            'numeric perturbations only; same-side per-unit comparison is exploratory, not a causal applicability estimate',
            'PCA fitted on train only; two-component projection is not a causal test']}
    dump(R/'obs2/summary.json',summary)
    # Fit one shared PCA on train states; only held-out groups are drawn.
    pc=PCA(2,svd_solver='full'); pc.fit(keys[tr,:,L,h].reshape(-1,128))
    points=pc.transform(keys[te,:,L,h].reshape(-1,128)).reshape(n,4,2)
    np.savez_compressed(R/'obs2/pca_coordinates.npz',points=points,group_ids=groups[te],components=pc.components_,
        train_mean=pc.mean_,explained_variance_ratio=pc.explained_variance_ratio_,layer=L,kv_head=h)
    fig,axs=plt.subplots(1,2,figsize=(9,3.7),sharex=True,sharey=True)
    for ax,pairs,title in [(axs[0],[(2,0),(3,1)],'(a) Question relevance'),(axs[1],[(1,0),(3,2)],'(b) Patient applicability')]:
        for i in range(n):
            for a,b in pairs:
                x=points[i,a]; delta=points[i,b]-x
                ax.annotate('',xy=points[i,b],xytext=x,arrowprops={'arrowstyle':'->','color':'.65','lw':.65,'alpha':.6})
        for s,c,m in [(0,'#0072B2','o'),(1,'#D55E00','o'),(2,'#0072B2','^'),(3,'#D55E00','^')]:
            ax.scatter(points[:,s,0],points[:,s,1],s=22,c=c,marker=m,label=['RA','RN','IA','IN'][s],alpha=.85)
        ax.set(xlabel=f'PC1 ({pc.explained_variance_ratio_[0]:.1%})',title=title)
    axs[0].set_ylabel(f'PC2 ({pc.explained_variance_ratio_[1]:.1%})')
    axs[1].legend(frameon=False,ncol=2,fontsize=8)
    fig.suptitle(f'Pre-RoPE action Keys | layer {L}, KV head {h} (0-based, dev-selected) | n = {n} groups',fontsize=10)
    fig.tight_layout(); save(fig,'O2_pca_pairwise')
    fig,axs=plt.subplots(2,3,figsize=(11,5.6),sharex=True,sharey=True)
    panels=[('relevance_shift','Relevance displacement','viridis',None,None),('applicability_shift','Applicability displacement','viridis',None,None),
        ('same_side_shift','Same-side numeric displacement','viridis',None,None),('relevance_consistency','Relevance consistency','viridis',0,1),
        ('relevance_test_auc','Held-out relevance AUROC','coolwarm',0,1),('applicability_test_auc','Held-out applicability AUROC','coolwarm',0,1)]
    for ax,(col,title,cmap,vmin,vmax) in zip(axs.flat,panels):
        im=ax.imshow(df.pivot(index='kv_head',columns='layer',values=col),aspect='auto',origin='lower',cmap=cmap,vmin=vmin,vmax=vmax)
        ax.set_title(title,fontsize=9); ax.set_xticks([0,8,17,26,35]); fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
    for ax in axs[-1]: ax.set_xlabel('Layer (0-based)')
    for ax in axs[:,0]: ax.set_ylabel('KV head (0-based)')
    fig.suptitle(f'All 36 layers × 8 KV heads | held-out n = {n} groups | displacements normalized by train Key norm',fontsize=10)
    fig.tight_layout(); save(fig,'O2_layer_head_heatmaps')
    # Paired per-group control comparison and selected-head probe uncertainty.
    fig,axs=plt.subplots(1,2,figsize=(8.5,3.4))
    names=['relevance','applicability','same_side']
    vals=[values[x]['estimate'] for x in names]; lows=[values[x]['ci95'][0] for x in names]; highs=[values[x]['ci95'][1] for x in names]
    axs[0].errorbar(range(3),vals,yerr=[np.maximum(0,np.array(vals)-lows),np.maximum(0,np.array(highs)-vals)],fmt='o',color='#0072B2',capsize=4)
    axs[0].set(xticks=range(3),xticklabels=['Relevance','Applicability','Same-side'],ylabel='Normalized Key displacement',title='(a) Group bootstrap 95% CI')
    av=[selected[x+'_test_auc'] for x in ['relevance','applicability']]
    ac=[selected[x+'_auc_ci'] for x in ['relevance','applicability']]
    axs[1].errorbar(range(2),av,yerr=[np.maximum(0,np.array(av)-np.array(ac)[:,0]),np.maximum(0,np.array(ac)[:,1]-av)],fmt='s',color='#D55E00',capsize=4)
    axs[1].axhline(.5,color='.6',ls='--'); axs[1].set(xticks=range(2),xticklabels=['Relevance','Applicability'],ylim=(0,1.05),ylabel='Held-out AUROC',title='(b) Linear probes trained on 36 groups')
    fig.suptitle(f'Dev-selected layer {L}, KV head {h} | test n = {n} groups',fontsize=10)
    fig.tight_layout(); save(fig,'O2_controls_and_probes')
    return summary
def analyze_o3():
    df=pd.read_csv(R/'obs3/logits.csv')
    assert df.formula_abs_error.max()<=1e-4
    assert np.max(np.abs(df.delta_a-.8*df.omega*df.z))<=1e-4
    rows=[]
    for (variant,weight),d in df.groupby(['projection_variant','weight_type']):
        valid=d[np.abs(d.z)>1e-6].copy(); valid['mismatch']=(np.sign(valid.delta_a)!=np.sign(valid.omega)).astype(float)
        group=valid.groupby('group_id').mismatch.mean(); estimate,interval=group_ci(group.values)
        valid['bf16_opposite']=(valid.delta_a_bf16*valid.omega < -1e-6).astype(float)
        actual_rate,actual_ci=group_ci(valid.groupby('group_id').bf16_opposite.mean().values)
        rows.append({'projection_variant':variant,'weight_type':weight,'n_records':len(d),'n_valid':len(valid),
            'near_zero_records':len(d)-len(valid),'n_groups':len(group),'mismatch_rate':estimate,'ci_low':interval[0],'ci_high':interval[1],
            'negative_z_rate':float((valid.z<0).mean()),'max_formula_abs_error':float(d.formula_abs_error.max()),
            'actual_bf16_opposite_rate':actual_rate,'actual_bf16_ci_low':actual_ci[0],'actual_bf16_ci_high':actual_ci[1],
            'actual_bf16_near_zero_delta_records':int((valid.delta_a_bf16.abs()<=1e-6).sum()),
            'median_abs_delta_a':float(d.delta_a.abs().median()),'max_abs_bf16_delta_discrepancy':float((d.delta_a_bf16-d.delta_a).abs().max()),
            'probability_direction_mismatch_rate':float((np.sign(valid.delta_p)!=np.sign(valid.omega)).mean())})
    pd.DataFrame(rows).to_csv(R/'obs3/metrics.csv',index=False)
    main=next(x for x in rows if x['projection_variant']=='official_posneg' and x['weight_type']=='unit')
    summary={'status':'supported' if main['ci_low']>0 else 'inconclusive','primary':main,'sensitivity':rows,
        'fixed_query':True,'coordinate':'QK-normalized pre-RoPE keys; realized RoPE applied once; float32 local algebra',
        'limitations':['fixed baseline query, not a full multi-layer causal effect','all measurements cluster within 12 rule groups',
            'oracle signs and designed magnitudes, not learned confidence or clinical negative weights',
            'BF16 arithmetic diagnostics reported separately from float32 identity']}
    dump(R/'obs3/summary.json',summary)
    d=df[(df.projection_variant=='official_posneg')&(df.weight_type=='unit')].copy()
    valid=d[np.abs(d.z)>1e-6].copy(); valid['mismatch']=(np.sign(valid.delta_a)!=np.sign(valid.omega)).astype(float)
    fig,ax=plt.subplots(figsize=(5.8,3.6))
    bound=np.quantile(np.abs(d.z),.999); bins=np.linspace(-bound,bound,75)
    ax.hist(d.z,bins=bins,color='#0072B2',alpha=.8); ax.axvline(0,color='k',lw=.9)
    ax.set(xlabel='z = q_rotated · RoPE(Πk) / sqrt(d)',ylabel='Token–query-head records',
        title=f'Negative z: {main["mismatch_rate"]:.1%} [95% CI {main["ci_low"]:.1%}, {main["ci_high"]:.1%}]')
    ax.text(.98,.96,f'n = {len(d):,} records\\n{main["n_groups"]} independent rule groups\\nRange shows 99.9% quantile',transform=ax.transAxes,ha='right',va='top',fontsize=8)
    fig.tight_layout(); save(fig,'O3_z_histogram')
    fig,axs=plt.subplots(1,2,figsize=(8.8,3.5))
    for j,weight in enumerate(['unit','continuous']):
        t=df[(df.projection_variant=='official_posneg')&(df.weight_type==weight)]
        # Deterministic thinning for display only; all records enter statistics.
        t=t.iloc[::max(1,len(t)//12000)]
        axs[j].scatter(t.omega,t.delta_a_bf16,c=np.where(t.z<0,'#D55E00','#0072B2'),s=4,alpha=.2,rasterized=True)
        axs[j].axhline(0,color='.6',lw=.8); axs[j].axvline(0,color='.6',lw=.8)
        axs[j].set(xlabel='Signed oracle weight ω',ylabel='Actual BF16 local logit change Δa',title=['(a) Unit magnitude','(b) Designed continuous magnitude'][j])
    fig.suptitle('Fixed-query local key edits | orange: z < 0 | λ = 0.8 | all last 10 layers and 32 query heads',fontsize=10)
    fig.tight_layout(); save(fig,'O3_weight_delta_scatter')
    fig,axs=plt.subplots(1,2,figsize=(9,3.7))
    for ax,variant in zip(axs,['official_posneg','positive_only']):
        t=df[(df.projection_variant==variant)&(df.weight_type=='unit')&(np.abs(df.z)>1e-6)].copy()
        t['mismatch']=(t.z<0).astype(float)
        hm=t.groupby(['query_head','layer']).mismatch.mean().unstack()
        im=ax.imshow(hm,aspect='auto',origin='lower',vmin=0,vmax=1,cmap='magma')
        ax.set_xticks(range(10),list(range(26,36)),fontsize=7); ax.set(xlabel='Layer (0-based)',ylabel='Query head (0-based)',title=variant.replace('_',' '))
        fig.colorbar(im,ax=ax,label='Direction mismatch rate',fraction=.046,pad=.04)
    fig.suptitle('Every query head retained; KV head = floor(query head / 4) | n = 12 groups',fontsize=10)
    fig.tight_layout(); save(fig,'O3_mismatch_heatmap')
    # Histogram is a dependent token distribution; inference uses groups only.
    return summary
def report(o1,o2,o3):
    manifest=json.loads((R/'manifest.json').read_text()); selected=o2['selected_head']; p=o3['primary']
    metrics=pd.read_csv(R/'obs1/metrics.csv')
    metric_lines=['|方法|指标|估计|95% CI|','|---|---|---:|---|']
    for row in metrics.itertuples():
        metric_lines.append(f'|{row.method}|{row.metric}|{row.estimate:.3f}|[{row.ci_low:.3f}, {row.ci_high:.3f}]|')
    metric_table='\n'.join(metric_lines)
    text=f"""# CARE 实测先导实验报告

所有模型观测来自服务器上的 Qwen/Qwen3-4B-Base 真正前向计算。病例是虚构规则数据（synthetic-only），不是临床效果证据。无模拟模型结果。图只读取保存的数据。

## 观察一：忠实性与不适用规则误用

临床参数先验冲突：**not_testable**，合格样本 **0**。条件规则测试状态：{o1['conditional_rule_status']}。
测试 {o1['test_groups']} 个配对规则组、每方法 {o1['test_cases_per_method']} 个病例；开发集选择的强度为 {o1['selected_strengths']}。
详细准确率、误用率、先验回退和配对变化的 95% CI 见 obs1/metrics.csv 与 obs1/contrasts.csv。
\n{metric_table}\n
闭卷共 {o1['closedbook_cases_all_splits']} 例，其中改写后选择稳定 {o1['stable_closedbook']} 例，和虚构规则答案不一致 {o1['synthetic_prior_disagreements']} 例。这不构成临床参数知识冲突。
A_app 与误用同时升高且两个组级区间均排除零，才视为在本先导测试中支持权衡。不得仅凭不适用准确率下降下结论。
候选空间保持原顺序，用单词元“空格+A/B/C/D”概率选择；报告完整词表中的候选概率质量，未将受限选择称为自然生成能力。

## 观察二：Key 表示

抽取 {o2['n_groups']} 组×6条件（四主条件及两个 same-side 数字对照），36层×8 KV头×128维 pre-RoPE Key，并保存目标 token 级表示。
开发集相关性位移一致性选定第 {int(selected['layer'])} 层、KV头 {int(selected['kv_head'])}（均从0计）。PCA只在训练组拟合，图展示12个独立测试组。
选中头的测试 relevance AUROC={selected['relevance_test_auc']:.4f}，95% CI {selected['relevance_auc_ci']}；
applicability AUROC={selected['applicability_test_auc']:.4f}，95% CI {selected['applicability_auc_ci']}。
288个头中 applicability 组内标签置换检验经 BH FDR<0.05 的头数：{o2['n_heads_app_auc_significant_fdr']}。
若适用性可解码，结论应是内部包含可解码适用性信息，不能声称完全未编码；单个选中头的阴性结果也不能否定其他头的信息。
训练集 GroupKFold(5)交叉验证和独立测试均保存，线性探针固定 C=1，训练标准化只用训练数据。
cross-threshold值差为4，同侧值差为1，另报告每单位数值的距离比较；不能把原始距离差直接解释为适用性语义差异。

## 观察三：带符号权重与注意力分数方向

**{o3['status']}（仅本虚构规则的局部机制实验）**。
主分析采用实际官方正负组合算子 Π=(P_pos+0.25P_neg)/2；
n={p['n_records']:,} 个 token×query-head×layer记录，来自 {p['n_groups']} 个独立规则组、24个病例。
方向不一致率 {p['mismatch_rate']:.2%}，组级bootstrap 95% CI [{p['ci_low']:.2%}, {p['ci_high']:.2%}]。
实际 BF16 Key 编辑和 RoPE 舍入后，反向 logit 变化比例 {p['actual_bf16_opposite_rate']:.2%}，95% CI [{p['actual_bf16_ci_low']:.2%}, {p['actual_bf16_ci_high']:.2%}]；近零实际变化 {p['actual_bf16_near_zero_delta_records']} 条。
|z|≤1e-6 排除 {p['near_zero_records']} 条；最大 float32 公式误差 {p['max_formula_abs_error']:.3g}（容差1e-4）。
实际计算 a_after-a_before，并独立核对 λωz；同时保存 softmax 前后概率和 BF16 有限精度差异。
正投影单独分析与设计的连续幅度灵敏度分析见 obs3/metrics.csv。连续幅度为 c=距离/(距离+0.1×阈值)、r=0.75，跨组非退化；它是预先定义的相对阈值距离函数，不是真实置信度变化的验证。
GQA映射为query_head//4；QK-norm后、RoPE前采样，Q和干预后的K各旋转一次。各query头分别记录。
负z意味着同号权重未必产生预期方向的局部logit变化，不代表完整多层模型的答案必然变化。

## 方法与可重复性

模型 revision: {manifest['model_revision']}
SEKA commit: {manifest['seka_commit']}；官方源码未修改：{manifest['seka_unmodified']}。
SEKA采用官方 builder 的传统SVD算法（top_pct=0.9，min_diff=0），官方 k_norm 干预算子；自定义部分仅是项目提示组装与流式Key提取，和官方extract_keys做数值对照。
官方PASTA直接调用未修改的编辑hook。全部方法使用同一个显式因果mask和SDPA；没有把等价重实现冒充官方实现。
谱投影仅用36个训练组72个相关病例的三元上下文构建，选头与选强度只看12个开发组。测试组12组。
随机seed=42；不训练语言模型权重。置信区间以完整规则组重采样，token、head、layer不当作独立病例。
GPU {manifest['gpu']}，物理卡 {manifest['physical_gpu']}，BF16；模型阶段 {manifest['seconds']/60:.1f}分钟，
实际峰值 allocated {manifest['peak_memory_allocated_GiB']:.2f} GiB，reserved {manifest['peak_memory_reserved_GiB']:.2f} GiB。
详细版本、配置、投影训练组、源码及SHA256见manifest与checks。

## 局限与未完成的临床验证

1. 未提供经过来源、授权和条件核验的真实临床反事实规则对；MedGUIDE的path不保证完整互斥规则，缺少明确许可字段；MedCalc为计算题，不能直接视为规则适用性反事实对。来源审查记录独立保存。
2. 60组是阈值/标识符变体，采用同一虚构规则语法，不能据此宣称泛化到未见规则家族。按group拆分防止相同病例变体跨集合，但不能消除共享语法。
3. 相关问题与无关问题使用不同措辞与长度，相关性位移可能包含词汇和位置影响。Key本身在RoPE前，但历史因果上下文仍受位置影响。不得把PCA分离称为语义或因果证明。
4. 数字大小和适用性标签关联；同侧对照只部分控制此混杂。探针结果是描述性可解码性。
5. n=12独立测试组，无预设统计功效，bootstrap在边界可退化；小样本下[0,0]是经验bootstrap区间，不意味着真实误差概率为0。
6. 仅一个Base模型、固定候选选择、有限强度网格；结果不能外推到自然生成、其他模型、真实临床安全性。
7. O3为固定query的单层局部算子，使用oracle符号及人为幅度；不是完整CARE系统或治疗建议。

## 复现命令

在服务器 /home/gcx/care_20261016/CARE_Codex_Pilot，source /home/gcx/gcx261014/bin/activate 后：
python scripts/build_minimal_pairs.py --n-groups 60 --seed 42 --out data/derived/pilot.jsonl
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_experiments.py
python scripts/analyze_results.py
python -m pytest -q

主实验前已执行独立2组烟雾测试（日志在results/smoke_run/logs），正式产物不含烟雾测试数据。
"""
    (R/'observations_summary.md').write_text(text)
    dump(R/'blockers.json',{'clinical_O1':'0 source-verified clinical eligible cases; conditional-rule results reported separately',
        'no_model_execution_blocker':True})
    (R/'STATUS.md').write_text('Model execution, saved-data analyses and paper plots completed. Clinical O1 not testable; see observations_summary.md.\n')
    dump(R/'analysis_summary.json',{'obs1':o1,'obs2':o2,'obs3':o3})
if __name__=='__main__':
    o1=analyze_o1(); o2=analyze_o2(); o3=analyze_o3(); report(o1,o2,o3)
    print('ANALYSIS COMPLETE',flush=True)

