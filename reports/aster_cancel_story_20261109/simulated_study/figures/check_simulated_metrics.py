from pathlib import Path
import csv,json,collections,math
import numpy as np
ROOT=Path('/home/gcx/aster_cancel_story_20261109/simulated_study')
OUT=ROOT/'figures'
def read(name):
    with (ROOT/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
checks=[];warnings=[]
def check(name,ok,detail):
    checks.append({'check':name,'passed':bool(ok),'detail':detail})
def val(r,k):return float(r[k])
difference_keys=['invariant_rms','variance_reduction','r2','evidence_retention','cf_correction_over_ap_rms']
pair_keys=['invariant_rms','variance_reduction','r2','evidence_retention','cf_correction_over_ap_rms','invariant_pair_accuracy','flip_pair_accuracy']
datasets={}
for name in ['main_seed_metrics.csv','ablation_seed_metrics.csv']:
    rows=read(name);datasets[name]=rows
    check(name+' simulated status',all(r.get('status')=='simulated' for r in rows),f'{len(rows)} rows')
    check(name+' finite metrics',all(math.isfinite(val(r,k)) for r in rows for k in pair_keys+['native_accuracy']), 'all numeric metrics finite')
    check(name+' variance bound',all(val(r,'variance_reduction')<=1+1e-8 for r in rows),'VR cannot exceed 1')
    check(name+' R2 bound',all(val(r,'r2')<=1+1e-8 for r in rows),'R2 cannot exceed 1')
    check(name+' VR vs R2',all(val(r,'variance_reduction')+1e-7>=val(r,'r2') for r in rows),
          'same pair residual: VR-R2 is squared residual mean divided by raw variance')
    check(name+' probability range',all(0<=val(r,k)<=1 for r in rows for k in ['native_accuracy','invariant_pair_accuracy','flip_pair_accuracy']),'correctness rates in [0,1]')
    side_results=[]
    for family,pair_key in [('invariant','invariant_pair_accuracy'),('flip','flip_pair_accuracy')]:
        keys=[family+'_left_accuracy',family+'_right_accuracy']
        if all(k in rows[0] for k in keys):
            ok=all(max(0,val(r,keys[0])+val(r,keys[1])-1)-1e-8<=val(r,pair_key)<=min(val(r,keys[0]),val(r,keys[1]))+1e-8 for r in rows)
            check(name+' '+family+' pair probability bound',ok,'pair both correct between Frechet bounds from the same two sides')
        else:
            warnings.append(name+' lacks '+family+' side accuracy; pair bound cannot be numerically checked from this CSV.')

def group(rows,keys):
    d=collections.defaultdict(list)
    for r in rows:d[tuple(r[k] for k in keys)].append(r)
    return d
for name,rows in datasets.items():
    cell=group(rows,['model','task','scenario','method'])
    check(name+' 5 seeds per cell',all(len(v)==5 and len({r['seed'] for r in v})==5 for v in cell.values()),f'{len(cell)} cells')
    scenarios={(r['model'],r['task'],r['scenario']) for r in rows}
    check(name+' nine scenarios',len(scenarios)==9,str(len(scenarios)))
for name,rows in datasets.items():
    maxgap=max(abs((val(r,'variance_reduction')-val(r,'r2'))-val(r,'invariant_mean')**2/val(r,'raw_invariant_variance')) for r in rows)
    check(name+' VR-R2 exact identity',maxgap<1e-7,f'max absolute error={maxgap}')
    maxrms=max(abs(val(r,'invariant_rms')**2-((1-val(r,'variance_reduction'))*val(r,'raw_invariant_variance')+val(r,'invariant_mean')**2)) for r in rows)
    check(name+' RMS-variance-mean identity',maxrms<1e-6,f'max absolute error={maxrms}')
    ok=all((bool(r.get('path_accuracy','').strip()) and 0<=val(r,'path_accuracy')<=1) if 'mega' in r['task'].lower() else not bool(r.get('path_accuracy','').strip()) for r in rows)
    check(name+' paths only in MEGA scenarios',ok,'NLI4CT and syllogism path fields must be blank')
main=datasets['main_seed_metrics.csv']
ml={(r['model'],r['task'],r['scenario'],r['seed'],r['method']):r for r in main}
maxbias=0.
for r in main:
    if r['method']!='raw':continue
    other=ml[tuple(r[k] for k in ['model','task','scenario','seed'])+('bias',)]
    for k in difference_keys:maxbias=max(maxbias,abs(val(r,k)-val(other,k)))
check('constant b leaves main pair metrics unchanged',maxbias<1e-7,f'max absolute difference={maxbias}')
abl=datasets['ablation_seed_metrics.csv']
al={(r['model'],r['task'],r['scenario'],r['seed'],r['method']):r for r in abl}
maxnob=0.
for r in abl:
    if r['method']!='full':continue
    other=al[tuple(r[k] for k in ['model','task','scenario','seed'])+('without_b',)]
    for k in difference_keys:maxnob=max(maxnob,abs(val(r,k)-val(other,k)))
check('without_b leaves ablation pair metrics unchanged',maxnob<1e-7,f'max absolute difference={maxnob}')
for seed_name,summary_name in [('main_seed_metrics.csv','main_summary.csv'),('ablation_seed_metrics.csv','ablation_summary.csv')]:
    groups=group(datasets[seed_name],['model','task','scenario','method'])
    maxmean=maxsd=0.
    for r in read(summary_name):
        rr=groups[tuple(r[k] for k in ['model','task','scenario','method'])]
        for k in pair_keys+['native_accuracy']:
            values=[val(x,k) for x in rr]
            maxmean=max(maxmean,abs(np.mean(values)-val(r,k+'_mean')))
            maxsd=max(maxsd,abs(np.std(values,ddof=1)-val(r,k+'_std')))
    check(summary_name+' mean/SD recomputation',maxmean<1e-7 and maxsd<1e-7,f'max mean diff={maxmean}; max sample SD diff={maxsd}')

summary=read('main_summary.csv');wins=collections.Counter();total=0
for _,rows in group(summary,['model','task','scenario']).items():
    ours=next(r for r in rows if r['method']=='aster_cr');total+=1
    wins['VR']+=int(val(ours,'variance_reduction_mean')>=max(val(r,'variance_reduction_mean') for r in rows)-1e-10)
    wins['ER_closest_to_one']+=int(abs(val(ours,'evidence_retention_mean')-1)<=min(abs(val(r,'evidence_retention_mean')-1) for r in rows)+1e-10)
    wins['native_accuracy']+=int(val(ours,'native_accuracy_mean')>=max(val(r,'native_accuracy_mean') for r in rows)-1e-10)
check('ours does not win every metric in every cell',not all(wins[k]==total for k in ['VR','ER_closest_to_one','native_accuracy']),dict(wins))
result={'status':'simulated_scenario_checks_only','checks':checks,'warnings':warnings,'ours_best_counts':dict(wins),
 'scenario_count':total,'failed':[r for r in checks if not r['passed']]}
(OUT/'numerical_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
text=['# \u6a21\u62df\u7814\u7a76\u7684\u72ec\u7acb\u68c0\u67e5','',
'\u672c\u6587\u4ef6\u68c0\u67e5\u6a21\u62df\u8868\u683c\u7684\u6570\u5b66\u4e00\u81f4\u6027\uff0c\u4e0d\u8bc1\u660e\u5b9e\u9645\u6a21\u578b\u3001\u6570\u636e\u96c6\u6216\u57fa\u7ebf\u6027\u80fd\u3002','',
'## \u6570\u503c\u68c0\u67e5','']
for c in checks:text.append(('\u901a\u8fc7\uff1a' if c['passed'] else '\u9700\u4fee\u6b63\uff1a')+c['check']+'\uff1b'+str(c['detail']))
text.extend(['','## \u89e3\u91ca\u8303\u56f4','',
'1. \u6a21\u578b\u540d\u4e0e\u4efb\u52a1\u540d\u5728\u6b64\u4ec5\u5bf9\u5e94\u53c2\u6570\u914d\u7f6e\u3002\u8868\u683c\u4e0e\u56fe\u4e2d\u6240\u6709\u7ed3\u679c\u5747\u987b\u6807\u201c\u6a21\u62df\u201d\uff0c\u4e0d\u5f97\u5199\u6210\u5b9e\u9645\u8fd0\u884cbenchmark\u7684\u7ed3\u8bba\u3002',
'2. 5\u4e2aseed\u7684\u6807\u51c6\u5dee\u53ea\u63cf\u8ff0\u751f\u6210\u8fc7\u7a0b\u4e2d\u7684\u968f\u673a\u6ce2\u52a8\uff0c\u4e0d\u662f\u5b9e\u9645\u6a21\u578b\u8bad\u7ec3\u6216\u771f\u5b9e\u4efb\u52a1\u62bd\u6837\u7684\u4e0d\u786e\u5b9a\u6027\u3002',
'3. \u5e38\u6570\u622a\u8dddb\u4e0d\u6539\u53d8\u914d\u5bf9\u5dee\u5206\uff0c\u4f46\u80fd\u6539\u53d8\u4ee50\u4e3a\u9608\u503c\u7684\u5224\u65ad\uff1b\u56e0\u6b64\u53bb\u6389b\u53ea\u5e94\u5f71\u54cd\u9608\u503c\u76f8\u5173\u6307\u6807\u3002',
'4. VR\u4e0eR\u00b2\u4f7f\u7528\u4e0d\u540c\u7684\u5747\u503c\u5904\u7406\u3002\u76f8\u540c\u6b8b\u5dee\u4e0b\uff0cVR-R\u00b2\u7b49\u4e8e\u6b8b\u5dee\u5747\u503c\u5e73\u65b9\u9664\u4ee5\u539f\u914d\u5bf9\u5dee\u65b9\u5dee\uff1b\u4e8c\u8005\u4e0d\u5e94\u88ab\u65e0\u6761\u4ef6\u5199\u6210\u5b8c\u5168\u76f8\u7b49\u3002',
'5. ER\u4ee5100%\u4e3a\u76ee\u6807\u3002\u8d85\u8fc7100%\u8bf4\u660e\u8bc1\u636e\u54cd\u5e94\u653e\u5927\uff0c\u5e76\u975e\u66f4\u597d\u7684\u8bc1\u636e\u4fdd\u7559\u3002',
'6. \u6a21\u62df\u4e2d\u7684\u8f83\u7eaf\u75d5\u8ff9\u4e0e\u63a5\u8fd1100%\u7684ER\u6765\u81ea\u989d\u5916\u8bbe\u5b9a\uff0c\u4e0d\u80fd\u66ff\u4ee3\u771f\u5b9e\u5f00\u53d1\u7ed3\u679c94.64%\u7684ER\u3002',
'7. \u8f93\u51fa\u7aef\u7ebf\u6027\u4fee\u6b63\u4e5f\u53ef\u80fd\u7531\u5408\u9002\u7684\u8868\u793a\u7f16\u8f91\u5b9e\u73b0\u3002\u6a21\u62df\u53ef\u8bf4\u660e\u7ed9\u5b9a\u5199\u56de\u566a\u58f0\u4e0e\u7ea6\u675f\u4e0b\u7684\u6743\u8861\uff0c\u4e0d\u80fd\u8bc1\u660e\u6240\u6709\u8868\u793a\u7f16\u8f91\u90fd\u5fc5\u7136\u5dee\u4e8e\u5bf9\u6d88\u3002',
'8. NLI\u4e0e\u4e09\u6bb5\u8bba\u4e0d\u62a5\u544a\u4e34\u5e8a\u6307\u5357\u8def\u5f84\u6b63\u786e\u7387\u3002'])
if warnings:text.extend(['','## \u4e0d\u80fd\u4ec5\u7531\u73b0\u6709CSV\u68c0\u67e5\u7684\u9879\u76ee','']+warnings)
(ROOT/'REVIEW_NOTES.zh.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
print(json.dumps({'checks':len(checks),'failures':result['failed'],'warnings':warnings,'best_counts':dict(wins)},ensure_ascii=False))

