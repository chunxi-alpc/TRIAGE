"""Fit the paper's frozen-r16 paired correction, using cached repair-v2 activations only."""
import csv,json,time
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.metrics import r2_score

ROOT=Path('/home/gcx/aster_cr_repair_v2_run_20261108')
REPO=Path('/home/gcx/TRIAGE-worktrees/aster-cr-20261108')
OUT=Path('/home/gcx/aster_cancel_story_20261109/meeting/paired_fit')
OUT.mkdir(parents=True,exist_ok=True)
def rows(p):
 return [json.loads(s) for s in p.read_text().splitlines() if s.strip()] if p.exists() else []
def rms(a):
 return float(np.sqrt(np.mean(np.square(a)))) if len(a) else None
def corr(x,y):
 return float(np.corrcoef(x,y)[0,1]) if len(x)>1 and np.std(x)>1e-12 and np.std(y)>1e-12 else None
def dump(p,x):
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
start=time.time()
scores={}
for file in sorted((ROOT/'qwen/scores').glob('*.jsonl')):
 for row in rows(file):
  if row.get('status')=='ok' and row.get('split') in ('fit','dev'):
   scores[row['view_id']]=row
views=[v for v in rows(ROOT/'data/views.jsonl') if v['split'] in ('fit','dev') and v['view_id'] in scores]
view_by_id={v['view_id']:v for v in views}
all_pairs=[p for p in rows(ROOT/'data/pairs.jsonl') if p['split'] in ('fit','dev')]
pairs=[p for p in all_pairs if p['left'] in view_by_id and p['right'] in view_by_id]
needed={p[k] for p in pairs for k in ('left','right')}
needed.update(v['view_id'] for v in views if v['variant']=='full')
ids=sorted(needed);ix={v:i for i,v in enumerate(ids)}
with np.load(ROOT/'qwen/supports_r16.npz') as a:
 support={k:a[k] for k in a.files}
heads=support['heads']
raw=np.empty((len(ids),len(support['mu'])),np.float32)
by_file={}
for vid in ids:by_file.setdefault(scores[vid]['feature'],[]).append(ix[vid])
for i,(file,positions) in enumerate(by_file.items()):
 with np.load(file) as a:
  h=a['heads'];value=h.reshape(-1,h.shape[-1])[heads].reshape(-1).astype(np.float32)
 raw[positions]=value
 if (i+1)%2000==0:print('loaded',i+1,'of',len(by_file),flush=True)
t=(((raw-support['mu'])/support['scale'])@support['N']).astype(np.float64)
del raw
ell=np.array([scores[vid]['ell'] for vid in ids],np.float64)
labels=np.array([view_by_id[vid]['label'] for vid in ids],np.int8)
fit_pairs=[p for p in pairs if p['split']=='fit' and p['kind'] in ('A','P','F0')]
left=np.array([ix[p['left']] for p in fit_pairs]);right=np.array([ix[p['right']] for p in fit_pairs])
dx=t[right]-t[left];dy=ell[right]-ell[left]
alpha=.01
w=-np.linalg.solve(dx.T@dx+alpha*np.eye(t.shape[1]),dx.T@dy)
fit_original=[v for v in views if v['split']=='fit' and v['variant']=='full' and v.get('original',True)]
fit_idx=np.array([ix[v['view_id']] for v in fit_original])
def calibrate(base):
 yy=labels[fit_idx];xx=base[fit_idx]
 def derivative(b):return float(np.mean(expit(xx+b)-yy))
 bound=1.
 while derivative(-bound)*derivative(bound)>0:bound*=2
 b=float(brentq(derivative,-bound,bound,xtol=1e-12))
 ce=float(np.mean(np.logaddexp(0,xx+b)-yy*(xx+b)))
 return b,ce
b,ce=calibrate(ell+t@w)
bias,bias_ce=calibrate(ell)
with np.load(REPO/'reports/aster_h2_alignment_20261108/ridge_coefficients.npz') as a:
 w_ap=-a['beta_r16'].astype(np.float64)
b_ap,ce_ap=calibrate(ell+t@w_ap)
methods={
 'raw':ell,
 'bias_only':ell+bias,
 'paired_APF0':ell+t@w+b,
 'paired_AP_only':ell+t@w_ap+b_ap,
}
selected=json.loads((ROOT/'qwen/development_selection.json').read_text())['aster_cr']
old={r['view_id']:r['ell'] for r in rows(ROOT/'qwen/readout_scores'/f'{selected}_dev.jsonl')}
oldarr=np.full(len(ids),np.nan)
for vid,value in old.items():
 if vid in ix:oldarr[ix[vid]]=value
methods['old_selected_CR']=oldarr
dev_pairs=[p for p in pairs if p['split']=='dev']
ap_dev=[p for p in dev_pairs if p['kind'] in ('A','P')]
ap_raw=np.array([ell[ix[p['right']]]-ell[ix[p['left']]] for p in ap_dev])
ap_rms=rms(ap_raw)
result=dict(
 data_version='repair-v2',model='Qwen3-8B',gpu_used=False,test_used=False,
 procedure='Fixed existing r16 N, alpha=.01. w uses fit A/P/F0 differences without intercept or labels. b uses only fit variant=full and original=True labels with w frozen.',
 support_file=str(ROOT/'qwen/supports_r16.npz'),
 fit_pairs=dict(Counter(p['kind'] for p in fit_pairs)),fit_pair_count=len(fit_pairs),
 fit_calibration_n=len(fit_idx),fit_calibration_label_counts=dict(Counter(map(int,labels[fit_idx]))),
 rank=int(t.shape[1]),alpha=alpha,b=b,bias_only_b=bias,ap_only_b=b_ap,
 fit_cross_entropy=dict(paired_APF0=ce,bias_only=bias_ce,paired_AP_only=ce_ap),
 coverage={split:{k:dict(expected=sum(p['split']==split and p['kind']==k for p in all_pairs),
 available=sum(p['split']==split and p['kind']==k for p in pairs)) for k in ('A','P','F0','C','F')} for split in ('fit','dev')},
 raw_ap_interference_rms=ap_rms,methods={},
 limitations=['No repair-v2 test scores or paths are computed in this run.',
 'P dev has four pairs; descriptive only. Correlation with labels is descriptive and does not prove absence of evidence.',
 'F0 changes evidence values while preserving truth; including F0 assumes desired within-label score stability beyond nuisance-only cancellation.',
 'No hyperparameter or support selection uses these dev results. Old selected CR was previously selected on dev.',
 'Calibration uses natural original nodes only; constructed full views are reported separately.'])
pair_rows=[]
for method,method_ell in methods.items():
 mr=dict(pair_groups={},nodes={})
 for group,kinds in [('A',('A',)),('P',('P',)),('F0',('F0',)),('C',('C',)),('F',('F',)),
                    ('AP',('A','P')),('APF0',('A','P','F0')),('CF',('C','F'))]:
  expected=[p for p in dev_pairs if p['kind'] in kinds]
  subset=[p for p in expected if np.isfinite(method_ell[ix[p['left']]]) and np.isfinite(method_ell[ix[p['right']]])]
  n=len(subset)
  if not n:
   mr['pair_groups'][group]=dict(n=0,expected=len(expected));continue
  rawdiff=np.array([ell[ix[p['right']]]-ell[ix[p['left']]] for p in subset])
  diff=np.array([method_ell[ix[p['right']]]-method_ell[ix[p['left']]] for p in subset])
  delta_correction=diff-rawdiff
  correct=np.array([int((method_ell[ix[p['left']]]>0)==bool(p['left_label']) and (method_ell[ix[p['right']]]>0)==bool(p['right_label'])) for p in subset])
  mm=dict(n=n,expected=len(expected),sources=len({p['source_id'] for p in subset}),
    paired_correct=int(correct.sum()),paired_accuracy=float(correct.mean()),
    difference_rms=rms(diff),difference_mean=float(np.mean(diff)),
    raw_difference_rms=rms(rawdiff),correction_difference_rms=rms(delta_correction),
    correction_difference_rms_over_ap_interference_rms=rms(delta_correction)/ap_rms if ap_rms else None)
  if group in ('A','P','F0','AP','APF0'):
   sst=float(np.sum((rawdiff-rawdiff.mean())**2))
   mm['nuisance_prediction_r2']=1-float(diff@diff)/sst if sst else None
   var=float(np.var(rawdiff))
   mm['difference_variance_reduction']=1-float(np.var(diff))/var if var else None
  if group in ('C','F','CF'):
   sign=np.array([p['right_label']-p['left_label'] for p in subset])
   before=float(sign@rawdiff);after=float(sign@diff)
   mm.update(signed_response_sum_before=before,signed_response_sum_after=after,
     signed_response_mean_before=before/n,signed_response_mean_after=after/n,
     signed_response_retention=after/before if before else None,
     correction_difference_rms_over_evidence_rms=rms(delta_correction)/rms(rawdiff) if rms(rawdiff) else None)
  mr['pair_groups'][group]=mm
  if group in ('A','P','F0','C','F'):
   for p,dd,cc in zip(subset,diff,correct):
    pair_rows.append(dict(method=method,pair_id=p['pair_id'],kind=p['kind'],source_id=p['source_id'],
      original_difference=float(ell[ix[p['right']]]-ell[ix[p['left']]]),corrected_difference=float(dd),paired_correct=int(cc)))
 for group,fn in [('original',lambda v:v.get('original',True)),('constructed',lambda v:not v.get('original',True)),('all_full',lambda v:True)]:
  vv=[v for v in views if v['split']=='dev' and v['variant']=='full' and fn(v) and np.isfinite(method_ell[ix[v['view_id']]])]
  inds=np.array([ix[v['view_id']] for v in vv],int)
  yy=labels[inds];ss=method_ell[inds];correction=ss-ell[inds]
  cc=(ss>0)==yy
  basecc=(ell[inds]>0)==yy
  mr['nodes'][group]=dict(n=len(vv),correct=int(cc.sum()),accuracy=float(cc.mean()) if len(cc) else None,
    corrected=int((cc&~basecc).sum()),new_errors=int((~cc&basecc).sum()),
    correction_label_pearson=corr(correction,yy),
    correction_rms=rms(correction),
    correction_mean=float(np.mean(correction)) if len(correction) else None)
 result['methods'][method]=mr
result['runtime_seconds']=time.time()-start
result['views_loaded']=len(ids);result['unique_feature_files']=len(by_file)
dump(OUT/'metrics.json',result)
np.savez_compressed(OUT/'fitted.npz',w=w,b=b,w_ap_only=w_ap,b_ap_only=b_ap,bias_only_b=bias,
 alpha=alpha,heads=heads,mu=support['mu'],scale=support['scale'],S=support['S'],N=support['N'])
with (OUT/'dev_scores.jsonl').open('w',encoding='utf-8') as f:
 for vid in ids:
  v=view_by_id[vid]
  if v['split']!='dev':continue
  i=ix[vid]
  row={k:v.get(k) for k in ('view_id','source_id','group_id','variant','original','label','split')}
  row['scores']={m:float(a[i]) for m,a in methods.items() if np.isfinite(a[i])}
  row['corrections']={m:float(a[i]-ell[i]) for m,a in methods.items() if np.isfinite(a[i])}
  f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
with (OUT/'pair_metrics.csv').open('w',newline='',encoding='utf-8') as f:
 fields=sorted({k for mr in result['methods'].values() for mm in mr['pair_groups'].values() for k in mm})
 writer=csv.DictWriter(f,fieldnames=['method','group']+fields);writer.writeheader()
 for m,mr in result['methods'].items():
  for group,mm in mr['pair_groups'].items():writer.writerow(dict(method=m,group=group,**mm))
with (OUT/'dev_pair_scores.csv').open('w',newline='',encoding='utf-8') as f:
 writer=csv.DictWriter(f,fieldnames=list(pair_rows[0]));writer.writeheader();writer.writerows(pair_rows)
lines=['# 固定秩16的配对差分拟合：实测缓存结果','',
'本次仅使用 repair-v2 已保存的模型分数和头表示；没有新增模型前向，没有使用测试集。',
'w 由 fit 的 4,869 个 A、18 个 P 和 82 个 F0 配对差分拟合，固定 λ=0.01、无配对截距。随后固定 w，仅用 761 个 fit 自然原始节点拟合校准截距 b。既有 N 子空间保持不变。','',
'F0 虽保持真值，但改变了决定性事实的数值；把它加入目标额外假设同侧证据变化应保持分数稳定，不能自动把 F0 分数差都称为纯干扰。A/P-only 对照使用已有 beta，仅按相同 761 个自然 fit 节点重估 b，主方法仍固定为 A/P/F0。','','## 开发集对比','',
'| 方法 | A/P 干扰 RMS | A/P 预测 R² | A/P 方差下降比例 | C/F 证据响应保留率 | C/F 修正差 RMS / A/P 干扰 RMS | 自然节点正确数 | 修正与标签相关系数 |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
names={'raw':'原模型','bias_only':'单参数校准','paired_APF0':'新配对差分（A/P/F0）','paired_AP_only':'A/P 诊断回归＋独立校准','old_selected_CR':'旧已选定 CR'}
for m,mr in result['methods'].items():
 ap=mr['pair_groups']['AP'];cf=mr['pair_groups']['CF'];nn=mr['nodes']['original']
 rho='NA' if nn['correction_label_pearson'] is None else f"{nn['correction_label_pearson']:.4f}"
 lines.append(f"| {names[m]} | {ap['difference_rms']:.4f} | {ap['nuisance_prediction_r2']:.4f} | {ap['difference_variance_reduction']:.2%} | {cf['signed_response_retention']:.2%} | {cf['correction_difference_rms_over_ap_interference_rms']:.2%} | {nn['correct']}/{nn['n']} | {rho} |")
lines+=['','自然节点为 original=True 且 variant=full；开发集共 199 个。另有 199 个构造边界节点，二者在 metrics.json 中单列，同时给出全部 398 个 full 视图结果。',
'R² 使用中心化原分数差的总平方和作分母、修正后差值的平方和作残差；方差下降比例使用修正前后方差之比。二者会因修正后差值均值不为零而略有差别。',
'证据响应保留率为 Σ(right_label−left_label)·Δℓ′ / Σ(right_label−left_label)·Δℓ。RMS 扰动比反映改动幅度，不等于响应损失。标签相关系数仅作描述，不能证明不含证据。','',
'## 各类配对全对率','',
'| 方法 | A（1194） | P（4） | F0（29） | C（29） | F（31） |',
'|---|---:|---:|---:|---:|---:|']
for m,mr in result['methods'].items():
 cells=[f"{mr['pair_groups'][k]['paired_correct']}/{mr['pair_groups'][k]['n']}" for k in ('A','P','F0','C','F')]
 lines.append('| '+names[m]+' | '+' | '.join(cells)+' |')
lines+=['','P 类仅 4 对，解释限于探索性观察。旧 CR 曾使用开发集选型，本次新方法参数未根据开发结果调整。测试集和完整路径仍待补。',
'全部数值为缓存上的实际拟合及实测结果；没有模拟值。完整数据见 metrics.json、pair_metrics.csv、dev_scores.jsonl、dev_pair_scores.csv；拟合参数见 fitted.npz。']
(OUT/'REPORT.zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
brief={m:dict(AP=mr['pair_groups']['AP'],CF=mr['pair_groups']['CF'],original=mr['nodes']['original'],all_full=mr['nodes']['all_full']) for m,mr in result['methods'].items()}
print(json.dumps(dict(runtime_seconds=result['runtime_seconds'],b=b,bias_only_b=bias,results=brief),ensure_ascii=True),flush=True)

