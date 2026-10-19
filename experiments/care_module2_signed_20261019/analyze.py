#!/usr/bin/env python3
"""Recompute metrics and paired family confidence intervals from saved real outputs."""
from __future__ import annotations
import argparse,csv,gzip,itertools,json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diagnostic import METHODS,TAUS,read,rows,save,digest
METRICS=['paired_accuracy','accuracy','app_accuracy','nonapp_accuracy','gold_logit_margin','gold_probability_margin',
         'misuse','no_conflict_accuracy','no_conflict_retention','eligible_accuracy','eligible_pair_accuracy']
def annotate(rs,screen,baseline):
    for r in rs:
        q=screen[r['case_id']]
        r['eligible']=q['eligible']=='True'
        r['no_conflict']=q['stable']=='True' and q['prior_prediction']==q['gold_answer']
        r['baseline_correct']=baseline[r['case_id']]['correct']
    return rs
def nd(rs,metric):
    if metric in ['paired_accuracy','eligible_pair_accuracy']:
        pp=defaultdict(list)
        for r in rs:pp[r['pair_id']].append(r)
        pp=[p for p in pp.values() if len(p)==2 and (metric=='paired_accuracy' or all(r['eligible'] for r in p))]
        return sum(all(r['correct'] for r in p) for p in pp),len(pp)
    if metric=='app_accuracy':rs=[r for r in rs if r['applicable']]
    elif metric in ['nonapp_accuracy','misuse']:rs=[r for r in rs if not r['applicable']]
    elif metric=='no_conflict_accuracy':rs=[r for r in rs if r['no_conflict']]
    elif metric=='no_conflict_retention':rs=[r for r in rs if r['no_conflict'] and r['baseline_correct']]
    elif metric=='eligible_accuracy':rs=[r for r in rs if r['eligible']]
    key=metric if metric in ['gold_logit_margin','gold_probability_margin','misuse'] else 'correct'
    return sum(r[key] for r in rs),len(rs)
def table(rs):
    return {m:dict(value=nd(rs,m)[0]/nd(rs,m)[1] if nd(rs,m)[1] else None,n=nd(rs,m)[1]) for m in METRICS}
def vectors(rs,metric,families):
    return np.array([nd([r for r in rs if r['group_id']==f],metric) for f in families],float)
def dist(v,weights):
    x=weights@v
    return np.divide(x[:,0],x[:,1],out=np.full(len(x),np.nan),where=x[:,1]>0)
def interval(v):
    return [float(x) for x in np.nanpercentile(v,[2.5,97.5])]
def writecsv(p,rs):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def analyze(args):
    root=args.run;out=args.output;out.mkdir(parents=True,exist_ok=True)
    assert read(root/'launch.json')['status']=='completed_real_inference'
    assert read(root/'validation/validation.json')['cases']==384
    allrows=[];selected={};status={}
    for f in range(3):
        folder=root/f'fold{f}';assert read(folder/'status.json')['status']=='completed_real_inference'
        selected[f]=read(folder/'selection.json');status[f]=read(folder/'status.json')
        allrows+=rows(folder/'predictions.jsonl')
    test=[r for r in allrows if r['role']=='test']
    baseline={r['case_id']:r for r in test if r['method']=='baseline'}
    assert len(baseline)==384
    screen_path=args.artifacts/'results/obs1_clinical/prior_screen.csv'
    if not screen_path.exists():screen_path=Path(__file__).parent/'inputs/prior_screen.csv'
    screen={r['case_id']:r for r in csv.DictReader(screen_path.open())}
    test=annotate(test,screen,baseline);families=sorted({r['group_id'] for r in test})
    primary={'baseline':list(baseline.values())}
    for m in METHODS:
        primary[m]=[r for r in test if r['method']==m and r['tau']==selected[r['fold']]['common_tau']]
    for rs in primary.values():assert len(rs)==384 and len({r['case_id'] for r in rs})==384
    summary={m:table(rs) for m,rs in primary.items()}
    rng=np.random.default_rng(261019);weights=rng.multinomial(len(families),np.full(len(families),1/len(families)),size=10000)
    ci=[];paired=[]
    for m,rs in primary.items():
        for metric in METRICS:
            dd=dist(vectors(rs,metric,families),weights)
            bd=dist(vectors(primary['baseline'],metric,families),weights)
            item=dict(method=m,metric=metric,value=summary[m][metric]['value'],n=summary[m][metric]['n'],
                      ci_low=interval(dd)[0],ci_high=interval(dd)[1],delta_vs_baseline=summary[m][metric]['value']-summary['baseline'][metric]['value'],
                      delta_ci_low=interval(dd-bd)[0],delta_ci_high=interval(dd-bd)[1])
            ci.append(item)
    for a,b in itertools.combinations(primary,2):
        for metric in METRICS:
            av=dist(vectors(primary[a],metric,families),weights);bv=dist(vectors(primary[b],metric,families),weights)
            paired.append(dict(method_a=a,method_b=b,metric=metric,delta_b_minus_a=summary[b][metric]['value']-summary[a][metric]['value'],
                               ci_low=interval(bv-av)[0],ci_high=interval(bv-av)[1]))
    family=[]
    for f in families:
        for m,rs in primary.items():
            item=dict(family=f,method=m)
            for k,v in table([r for r in rs if r['group_id']==f]).items():item[k]=v['value']
            family.append(item)
    sensitivity=[]
    for tau in TAUS:
        for m in METHODS:
            vals=table([r for r in test if r['method']==m and r['tau']==tau])
            sensitivity.append(dict(tau=tau,method=m,**{k:v['value'] for k,v in vals.items()}))
    secondary={m:table([r for r in test if r['method']==m and r['tau']==selected[r['fold']]['method_specific_tau'][m]]) for m in METHODS}
    cal=[]
    for m in METHODS:
        for tau in TAUS[1:]:
            rr=[r for r in allrows if r['method']==m and r['tau']==tau]
            actual=[r['disturbance']['actual_rms'] for r in rr]
            err=[abs(v/tau-1) for v in actual]
            cal.append(dict(method=m,tau=tau,n=len(rr),actual_rms_mean=float(np.mean(actual)),max_relative_error=max(err),
                p95_relative_error=float(np.quantile(err,.95)),mean_negative_score_fraction=float(np.mean([r['disturbance']['delta_score_negative_fraction'] for r in rr]))))
    summary_obj=dict(status='completed_real_inference_and_analysis',patient_data='simulated; authentic clinical rule paraphrases',
        cases=384,pairs=192,families=12,bootstrap_replicates=10000,bootstrap_seed=261019,
        confidence_intervals='paired clinical-family cluster percentile bootstrap, conditional on fitted folds; no simultaneous multiplicity correction',
        gold_applicability_oracle=True,selected=selected,primary=summary,method_specific_secondary=secondary,status_per_fold=status,
        main_records=len(allrows),main_actual_forwards=3*(192+192*13),validation_actual_forwards=769,
        retention_definition='accuracy after intervention among stable closedbook-agreement cases answered correctly by contextual baseline',
        no_conflict_definition='stable closedbook/paraphrase argmax equal to gold; archival pre-intervention screen',
        no_conflict_cases=sum(r['no_conflict'] for r in primary['baseline']),eligible_cases=sum(r['eligible'] for r in primary['baseline']),
        source_sha256=digest(__file__))
    save(out/'summary.json',summary_obj);writecsv(out/'metrics_ci.csv',ci);writecsv(out/'paired_comparisons.csv',paired)
    writecsv(out/'per_family.csv',family);writecsv(out/'sensitivity.csv',sensitivity);writecsv(out/'calibration.csv',cal)
    with gzip.open(out/'selected_predictions.jsonl.gz','wt') as f:
        for rs in primary.values():
            for r in rs:f.write(json.dumps(r,allow_nan=False)+'\n')
    # All plots read saved statistics rather than generating example data.
    saved=list(csv.DictReader((out/'metrics_ci.csv').open()))
    labels=['Baseline','B4 positive','B4 + sign','Signed mean shift','Direct score bias']
    colors=['#555555','#b5651d','#0072b2','#009e73','#cc79a7']
    fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    for ax,metric,title,scale in zip(axs,['paired_accuracy','gold_logit_margin'],['Both pair members correct','Gold-answer logit margin'],[100,1]):
        rr=[next(r for r in saved if r['method']==m and r['metric']==metric) for m in primary]
        val=np.array([float(r['value']) for r in rr])*scale;lo=np.array([float(r['ci_low']) for r in rr])*scale;hi=np.array([float(r['ci_high']) for r in rr])*scale
        ax.bar(range(5),val,color=colors);ax.errorbar(range(5),val,yerr=[val-lo,hi-val],fmt='none',color='black',capsize=4)
        ax.set_xticks(range(5),labels,rotation=24,ha='right');ax.set_title(title);ax.set_ylabel('%' if scale==100 else 'Logit difference');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Gold applicability; 384 simulated cases / 192 pairs / 12 families\n95% family bootstrap CI; common development-selected score RMS')
    fig.tight_layout();fig.savefig(out/'primary_comparison.png',dpi=300);fig.savefig(out/'primary_comparison.svg');plt.close(fig)
    mat=np.array([[next(r for r in family if r['family']==f and r['method']==m)['paired_accuracy']-next(r for r in family if r['family']==f and r['method']=='baseline')['paired_accuracy'] for m in METHODS] for f in families])*100
    fig,ax=plt.subplots(figsize=(8,7));lim=max(1,float(abs(mat).max()));im=ax.imshow(mat,cmap='RdBu',vmin=-lim,vmax=lim)
    ax.set_xticks(range(4),labels[1:],rotation=20,ha='right');ax.set_yticks(range(12),families)
    for i in range(12):
        for j in range(4):ax.text(j,i,f'{mat[i,j]:+.1f}',ha='center',va='center',fontsize=9)
    fig.colorbar(im,ax=ax,label='Paired accuracy change (percentage points)');ax.set_title('Every held-out clinical rule family; 16 pairs each')
    fig.tight_layout();fig.savefig(out/'per_family.png',dpi=300);fig.savefig(out/'per_family.svg');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for j,m in enumerate(METHODS):
        rr=[r for r in sensitivity if r['method']==m]
        for ax,metric,scale in zip(axs,['paired_accuracy','gold_logit_margin'],[100,1]):
            ax.plot([r['tau'] for r in rr],[r[metric]*scale for r in rr],marker='o',label=labels[j+1],color=colors[j+1])
            ax.set_xlabel('Target RMS of actual attention-score change');ax.grid(alpha=.2)
    axs[0].set_ylabel('Paired accuracy (%)');axs[1].set_ylabel('Gold-answer logit margin')
    axs[0].legend(fontsize=8);fig.suptitle('Predeclared strength grid; all test results shown without test selection')
    fig.tight_layout();fig.savefig(out/'strength_sensitivity.png',dpi=300);fig.savefig(out/'strength_sensitivity.svg');plt.close(fig)
    # Paired uncertainty is more informative than overlapping absolute intervals.
    fig,axs=plt.subplots(1,2,figsize=(10,4.5))
    for ax,metric,title,scale in zip(axs,['paired_accuracy','gold_logit_margin'],['Paired accuracy change','Gold-answer logit margin change'],[100,1]):
        rr=[next(r for r in saved if r['method']==m and r['metric']==metric) for m in METHODS]
        val=np.array([float(r['delta_vs_baseline']) for r in rr])*scale
        lo=np.array([float(r['delta_ci_low']) for r in rr])*scale
        hi=np.array([float(r['delta_ci_high']) for r in rr])*scale
        for j in range(4):
            ax.errorbar(val[j],j,xerr=[[val[j]-lo[j]],[hi[j]-val[j]]],fmt='o',color=colors[j+1],capsize=4)
        ax.axvline(0,color='#555555',linestyle='--',linewidth=1)
        ax.set_yticks(range(4),labels[1:]);ax.invert_yaxis();ax.set_title(title)
        ax.set_xlabel('Percentage points vs baseline' if scale==100 else 'Logit difference vs baseline');ax.grid(axis='x',alpha=.2)
    fig.suptitle('Paired 95% family bootstrap CI; 192 pairs / 12 held-out families')
    fig.tight_layout();fig.savefig(out/'paired_changes.png',dpi=300);fig.savefig(out/'paired_changes.svg');plt.close(fig)
    report=['# CARE module two: gold-applicability diagnostic','',
        'Completed real Qwen3-4B-Base inference on GPU 0–3. 384 simulated cases, 192 pairs, 12 clinical rule families; exact archived three family splits.','',
        '| Method | Pair accuracy | Change vs baseline; paired 95% CI | App | Nonapp | Misuse | Gold logit margin | No-conflict retention |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m,label in zip(primary,labels):
        v=summary[m];c=next(x for x in ci if x['method']==m and x['metric']=='paired_accuracy')
        report.append(f"| {label} | {100*v['paired_accuracy']['value']:.2f}% | {100*c['delta_vs_baseline']:+.2f} pp [{100*c['delta_ci_low']:+.2f}, {100*c['delta_ci_high']:+.2f}] | {100*v['app_accuracy']['value']:.2f}% | {100*v['nonapp_accuracy']['value']:.2f}% | {100*v['misuse']['value']:.2f}% | {v['gold_logit_margin']['value']:.6f} | {100*v['no_conflict_retention']['value']:.2f}% |")
    report+=['','![Primary comparison](primary_comparison.png)','','![Paired changes](paired_changes.png)','',
        'No learned-direction advantage over direct score bias is established by these comparisons. Direct bias has the largest paired-accuracy point estimate, while its gold logit margin is slightly below baseline. Full paired comparisons, rather than rankings alone, should guide interpretation.', '', 'B4 is a positive semidefinite orthogonal projector P=UUᵀ followed by additive amplification along Pk. This is not a clamp of negative coordinates. Original code has no applicability sign input. A positive key coefficient does not guarantee positive QK-score change.',
        '','The new B4 signed method uses s=+1 for applicable and −1 otherwise. The signed mean shift uses s·mean(k_app−k_nonapp), estimated only from training families. The direct score bias uses s·tau. Gold applicability is supplied to interventions only and is never inserted into prompts.',
        '','All four methods share each fold’s frozen PR3 B4 development-selected location and rank, target action tokens, GQA query heads, and causal query/key support. B4 is refitted from freshly extracted training keys and checked against the archived projector. Mean shifts use the same training pairs.',
        '','Key edits are scalar-calibrated per input, without answer labels, to match actual post-RoPE fixed-query score RMS after BF16 rounding. Direct bias uses the nearest BF16 value. Calibration measures every causal entry at the edited head/span; see calibration.csv. Sign control does not force every query score to have the requested sign.',
        '','The common strength is selected before test inference using mean development pair accuracy across all four methods, then case accuracy, then smaller strength. Zero is eligible. Method-specific development selections and all fixed-strength test results are separately saved. No test outcome selects the reported strength.',
        '','Gold-answer logit margin is the gold candidate logit minus the largest other candidate logit; probability margin is also saved. All outcomes use identical single-token A/B/C/D scoring. Misuse is selection of the highlighted inapplicable action among nonapplicable cases.',
        f"","The no-conflict set contains "+str(summary_obj['no_conflict_cases'])+" cases with stable closedbook agreement, fixed by the archived screening. Retention conditions additionally on contextual-baseline correctness. All confidence intervals and denominators appear in metrics_ci.csv; all ten paired method comparisons appear in paired_comparisons.csv.",
        '','Intervals use 10,000 shared family-bootstrap resamples and condition on the three fitted outer folds. Only 12 families are available. Secondary comparisons have no multiplicity correction; sparse or zero-width empirical intervals do not establish a zero population effect.',
        '','![Per-family results](per_family.png)','','![Strength sensitivity](strength_sensitivity.png)','',
        'The experiment isolates gold-applicability module-two behavior. It does not test module-one errors or establish clinical efficacy. Cases are simulated; thresholds and answers are from the unchanged archived executable rules.',
        '','See ../../PROTOCOL.md for formulas, selection policy, reproduction, audit details and limitations. Old B4 reproduction is a separate fourth-GPU task. All 384 cached key examples and archived B4 candidate distributions must match exactly before this report is produced.']
    (out/'README.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(dict(primary={m:summary[m]['paired_accuracy'] for m in primary},selected=selected,calibration_max_error=max(x['max_relative_error'] for x in cal))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=Path(__file__).parent/'results');p.add_argument('--output',type=Path)
    p.add_argument('--artifacts',type=Path,default=Path('/home/gcx/care_20261016/CARE_Codex_Pilot'));args=p.parse_args()
    args.output=args.output or args.run/'analysis';analyze(args)

