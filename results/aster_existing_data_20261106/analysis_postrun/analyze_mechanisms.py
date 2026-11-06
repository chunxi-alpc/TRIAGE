from pathlib import Path
import json,sys,math,collections,gzip
import numpy as np
ROOT=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/home/gcx/aster_existing_data_20261106')
OUT=ROOT/'analysis_postrun';OUT.mkdir(exist_ok=True)
def read(name):return json.loads((ROOT/name).read_text())
def read_text(path):
 path=Path(path)
 if not path.exists():path=Path(str(path)+'.gz')
 return gzip.open(path,'rt',encoding='utf-8').read() if path.suffix=='.gz' else path.read_text()
def rows(name):return [json.loads(l) for l in read_text(ROOT/name).splitlines() if l.strip()]
def stats(v):
 a=np.array(v,dtype=float)
 return dict(n=len(a),mean=float(a.mean()) if len(a) else None,median=float(np.median(a)) if len(a) else None,min=float(a.min()) if len(a) else None,max=float(a.max()) if len(a) else None)
main=read('results/main_effects.json');geom=read('operators/grid_11.json');sel=read('results/selection.json')
ev=dict(scope='Post-run CPU analysis of frozen existing artifacts; no new labels, model calls, or selection changes.')
energy={}
for kind in ['A','P']:
 hs=[h for h in geom['heads'].values() if kind in h['nuisance_energy']]
 sums={k:sum(h['nuisance_energy'][kind][k] for h in hs) for k in ['total','in_merged_nuisance_basis','in_retained_principal_directions','discarded_by_angle','outside_merged_nuisance_basis']}
 energy[kind]=dict(heads=len(hs),**sums,merged_fraction=sums['in_merged_nuisance_basis']/sums['total'],retained_fraction=sums['in_retained_principal_directions']/sums['total'],angle_retained_fraction_of_merged=sums['in_retained_principal_directions']/sums['in_merged_nuisance_basis'],per_type_rank_target_reached=sum(h['per_type'][kind]['target_reached'] for h in hs),per_type_required_rank=stats([h['per_type'][kind]['rank_needed_for_target'] for h in hs]))
ev['fit_nuisance_energy']=energy
ev['signal_basis_energy']={}
for kind in ['C','F']:
 hs=list(geom['heads'].values());es=[h['per_type'][kind] for h in hs]
 ev['signal_basis_energy'][kind]=dict(retained_fraction=sum(x['retained_energy'] for x in es)/sum(x['total_energy'] for x in es),head_fraction=stats([x['retained_fraction'] for x in es]),target_reached=sum(x['target_reached'] for x in es),needed_rank=stats([x['rank_needed_for_target'] for x in es]))
ev['ranks']=dict(heads=len(geom['heads']),nuisance_before_angle=sum(h['merged_nuisance_rank'] for h in geom['heads'].values()),after_angle=sum(h['edit_rank'] for h in geom['heads'].values()),signal=sum(h['merged_signal_rank'] for h in geom['heads'].values()),zero_edit_heads=[h for h,v in geom['heads'].items() if v['edit_rank']==0])
ev['algebra']=dict(max_signal_error=max(h['algebra']['signal_preservation_error'] for h in geom['heads'].values()),max_erasure_error=max(h['algebra']['erasure_error'] for h in geom['heads'].values()),max_norm=max(h['algebra']['edit_operator_norm'] for h in geom['heads'].values()))
ev['joint_repair']={}
for split in ['dev','test']:
 rs=rows('mechanisms/joint_components_'+split+'.jsonl');groups={}
 for kind in ['A','P']:
  for component in ['full_head','nuisance_orthogonal','screened_orthogonal','oblique']:
   sub=[r for r in rs if r['kind']==kind and r['component']==component]
   te=np.array([r['te'] for r in sub]);ie=np.array([.5*(r['ie_forward']-r['ie_reverse']) for r in sub])
   val=dict(pairs=len(sub),mean_abs_total_effect=float(np.abs(te).mean()),mean_abs_symmetric_effect=float(np.abs(ie).mean()),least_squares_effect_ratio=float(te@ie/(te@te)),normalized_by_threshold={})
   for threshold in [.1,.5,1,2]:
    selected=np.abs(te)>=threshold
    val['normalized_by_threshold'][str(threshold)]=stats((ie[selected]/te[selected]).tolist())
   groups[kind+':'+component]=val
 ev['joint_repair'][split]=groups
ev['comparators']={}
scores={}
for p in sorted((ROOT/'scores').glob('baseline_*.jsonl*'))+sorted((ROOT/'scores').glob('final_*.jsonl*')):
 for l in read_text(p).splitlines():
  r=json.loads(l);scores.setdefault(r['method'],{})[r['view_id']]=r.get('ell')
scores.update(read('results/fusion.json')['scores'])
views={r['view_id']:r for r in rows('data/views.jsonl') if r['split']=='test'}
def pred(z):return None if z is None else int(z>0)
for name in ['full','aster','anchor_original','anchor_zero','per_type_mean','orthogonal','leace','no_C','no_F','random_13','random_37','random_73','calibrated_full','logistic_direct']:
 m=main['methods'][name]
 ev['comparators'][name]=dict(macro_PA=m['macro_PA'],PA_A=m['PA_A']['rate'],PA_P=m['PA_P']['rate'],PA_C=m['PA_C']['rate'],PA_F=m['PA_F']['rate'],all_input_accuracy=m['all_input_accuracy'],predictions_differ_from_aster=sum(pred(scores[name].get(k))!=pred(scores['aster'].get(k)) for k in views),response_mean_abs={k:m['responses'][k]['score_response']['mean_abs'] for k in ['A','P','C','F']})
probe=read('results/h3.json')['probe']
ev['h3']={k:v for k,v in probe.items() if k not in ['coefficients','predictions']}
ev['h3']['error_label_counts']=dict(collections.Counter(r['label'] for r in probe['predictions'].values()))
ev['h3']['prediction_counts']=dict(collections.Counter(r['prediction'] for r in probe['predictions'].values()))
ev['local_theory']={}
for split in ['dev','test']:
 rs=rows('mechanisms/local_theory_'+split+'.jsonl')
 ev['local_theory'][split]=dict(summary=read('results/local_theory.json')[split],mean_abs_linear_delta=float(np.mean([abs(r['linear_delta']) for r in rs])),mean_abs_actual_edit_delta=float(np.mean([abs(r['sequential_ell']-r['baseline_ell']) for r in rs])),prediction_error_sign_reversal=sum((r['linear_delta']>0)!=(r['sequential_ell']-r['baseline_ell']>0) for r in rs if r['linear_delta']!=0 and r['sequential_ell']!=r['baseline_ell']))
ev['selection']=dict(config=sel['config'],heads=sel['heads'],rank=sel['rank'],ties=len(sel['tied_on_primary_metrics']),dev_C_pairs=sel['development_C_pairs'])
ev['ci_main_difference']=main['confidence_intervals']['methods']['aster']['differences']['full']['macro_PA']
ev['calibration'] = read('results/fusion.json')['specs']['calibrated_full']
ev['calibration']['raw_margin_threshold'] = -ev['calibration']['intercept']/ev['calibration']['coefficients'][0]
ev['main_prediction_changes'] = [dict(view_id=k,variant=views[k]['variant'],label=views[k]['label'],full=scores['full'][k],aster=scores['aster'][k]) for k in views if pred(scores['full'].get(k))!=pred(scores['aster'].get(k))]
(OUT/'mechanism_evidence.json').write_text(json.dumps(ev,ensure_ascii=False,indent=2),encoding='utf-8')
print('mechanism evidence written')
