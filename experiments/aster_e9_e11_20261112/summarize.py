import collections,csv,json
import numpy as np
from .common import *
def csvfile(name,records):
 if not records:return
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
def moments(a):
 a=np.array(a,dtype=float)
 return dict(mean=float(a.mean()),std=float(a.std(ddof=1)) if len(a)>1 else 0.,n=len(a))
def source_ci(records):
 by=collections.defaultdict(list)
 for source,value in records:by[source].append(value)
 vals=np.array([np.mean(v) for v in by.values()])
 rng=np.random.default_rng(20261112);means=vals[rng.integers(len(vals),size=(2000,len(vals)))].mean(1)
 return dict(mean=float(vals.mean()),ci95_low=float(np.quantile(means,.025)),ci95_high=float(np.quantile(means,.975)),sources=len(vals))
def e9():
 rr=rows(OUT/'e9_predictions.jsonl');cfg=read(OUT/'configs/e9.json')
 expected=cfg['pair_count']*2*2*len(cfg['interventions'])
 if len(rr)!=expected:raise ValueError(f'E9 incomplete: {len(rr)}/{expected}')
 keys=[(r['pair_id'],r['direction'],r['group'],r['intervention']) for r in rr]
 if len(set(keys))!=len(keys):raise ValueError('Duplicate E9 records')
 stats=[]
 for group in ('selected','random'):
  for mode in cfg['interventions']:
   for kind in ('all','AP','CF','A','P','C','F'):
    sub=[r for r in rr if r['group']==group and r['intervention']==mode and (kind=='all' or r['kind'] in kind)]
    if not sub:continue
    raw=np.array([r['ell']-r['base_ell'] for r in sub]);ast=np.array([r['aster_score']-r['base_aster_score'] for r in sub])
    def changes(field,base):
     before=np.array([(r[base]>0)==bool(r['label']) for r in sub]);after=np.array([(r[field]>0)==bool(r['label']) for r in sub])
     return dict(correct_before=int(before.sum()),correct_after=int(after.sum()),corrected=int((~before&after).sum()),new_errors=int((before&~after).sum()))
    stats.append(dict(group=group,intervention=mode,kind=kind,n=len(sub),sources=len({r['source_id'] for r in sub}),
     raw_delta_mean=float(raw.mean()),raw_delta_std=float(raw.std(ddof=1)),raw_abs_delta_mean=float(np.abs(raw).mean()),
     raw_delta_rms=float(np.sqrt(np.mean(raw**2))),aster_delta_mean=float(ast.mean()),aster_abs_delta_mean=float(np.abs(ast).mean()),
     **{'raw_'+k:v for k,v in changes('ell','base_ell').items()},**{'aster_'+k:v for k,v in changes('aster_score','base_aster_score').items()}))
 csvfile('e9_summary.csv',stats)
 index={(r['pair_id'],r['direction'],r['group'],r['intervention']):r for r in rr};contrasts=[]
 specifications=[('head_selection','selected','output_patch','random','output_patch'),
  ('qk_dependence_selected','selected','output_patch','selected','output_qk_frozen'),
  ('qk_dependence_random','random','output_patch','random','output_qk_frozen'),
  ('output_projection_selected','selected','projection_native','selected','projection_swap'),
  ('output_projection_random','random','projection_native','random','projection_swap'),
  ('value_vs_route_selected','selected','ov_values_qk_frozen','selected','routing_only_qk_frozen')]
 for name,ga,ma,gb,mb in specifications:
  for kind in ('all','AP','CF'):
   xs=[];signed=[];effectdiff=[]
   for a in rr:
    if a['group']!=ga or a['intervention']!=ma or not(kind=='all' or a['kind'] in kind):continue
    b=index[(a['pair_id'],a['direction'],gb,mb)];da=a['ell']-a['base_ell'];db=b['ell']-b['base_ell']
    xs.append((a['source_id'],abs(da)-abs(db)));signed.append((a['source_id'],da-db));effectdiff.append((a['source_id'],abs(da-db)))
   for metric,records in [('abs_score_change_difference',xs),('signed_score_change_difference',signed),('absolute_between_conditions_score_difference',effectdiff)]:
    contrasts.append(dict(contrast=name,kind=kind,metric=metric,**source_ci(records)))
 csvfile('e9_paired_contrasts.csv',contrasts)
 identities=rows(OUT/'e9_identity_controls.jsonl')
 info=dict(state='complete',pairs=cfg['pair_count'],directions=2,conditions_per_group=len(cfg['interventions']),
  prediction_rows=len(rr),unique_sources=len({r['source_id'] for r in rr}),kind_pairs=cfg['kind_counts'],
  alignment=moments([r['alignment_fraction'] for r in rr if r['group']=='selected' and r['intervention']=='output_patch']),
  identity_max_abs_difference=max(abs(r['self_patch']-r['baseline']) for r in identities),
  summary=stats,contrasts=contrasts,model_parameters_frozen=True,test_label_selection=False)
 write(OUT/'e9_statistics.json',info)
 def cell(mode,group,kind='all'):return next(r for r in stats if r['group']==group and r['intervention']==mode and r['kind']==kind)
 def contrast(name,kind='all',metric='abs_score_change_difference'):
  return next(r for r in contrasts if r['contrast']==name and r['kind']==kind and r['metric']==metric)
 head=contrast('head_selection');proj=contrast('output_projection_selected');qk=contrast('qk_dependence_selected',metric='absolute_between_conditions_score_difference')
 return info
def e11():
 rr=rows(OUT/'e11_measurements.jsonl');cfg=read(OUT/'configs/e11.json');infos=read(OUT/'e11_method_info.json')
 stats=[];repeats=[]
 for name in infos:
  for length in cfg['lengths']:
   sub=[r for r in rr if r['method']==name and r['length']==length]
   if len(sub)!=cfg['inputs']*cfg['repeats']:raise ValueError(f'E11 incomplete {name}/{length}')
   row=dict(method=name,length=length,n=len(sub))
   for field in ('prefill_ms','latency_ms','peak_allocated_gib','peak_reserved_gib','kv_cache_bytes','reference_kv_cache_bytes','primary_tokens','reference_tokens'):
    mm=moments([r[field] for r in sub]);row[field+'_mean']=mm['mean'];row[field+'_std']=mm['std']
   row['throughput_samples_s']=len(sub)/sum(r['latency_ms']/1000 for r in sub)
   for repeat in range(cfg['repeats']):
    rs=[r for r in sub if r['repeat']==repeat]
    repeats.append(dict(method=name,length=length,repeat=repeat,n=len(rs),
     prefill_ms=float(np.mean([r['prefill_ms'] for r in rs])),latency_ms=float(np.mean([r['latency_ms'] for r in rs])),
     throughput_samples_s=len(rs)/sum(r['latency_ms']/1000 for r in rs)))
   a=[r['throughput_samples_s'] for r in repeats if r['method']==name and r['length']==length]
   row['throughput_repeat_mean']=float(np.mean(a));row['throughput_repeat_std']=float(np.std(a,ddof=1))
   row['peak_allocated_gib_max']=max(r['peak_allocated_gib'] for r in sub)
   row.update({k:infos[name][k] for k in ('trainable_parameters','storage_bytes','deployment_tensor_bytes','logical_forwards')})
   row['model_forward_calls']=sub[0]['model_forward_calls'];stats.append(row)
 csvfile('e11_summary.csv',stats);csvfile('e11_repeat_summary.csv',repeats)
 result=dict(state='available_methods_complete',pending=['audited_ITI','completed_new_representative'],
  measurements=len(rr),summary=stats,reused_benchmarks=['extraction_performance.json','evaluation_performance.json'])
 write(OUT/'e11_statistics.json',result)
 return result
def main():
 a=e9();b=e11()
 write(OUT/'status.json',dict(E9='complete',E11=b['state'],pending=b['pending'],manuscript_modified=False,training_runs=0))
 from .render_reports import main as render
 render()
if __name__=='__main__':main()
