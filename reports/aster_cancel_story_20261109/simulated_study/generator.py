"""SIMULATED score-level study. Does not run language models or official baselines."""
import copy,csv,json,time
from pathlib import Path
from collections import defaultdict
import numpy as np

OUT=Path('/home/gcx/aster_cancel_story_20261109/simulated_study')
OUT.mkdir(parents=True,exist_ok=True)
MODELS=['Qwen3-8B','Llama-3.1-8B-Instruct','MedGemma-27B']
TASKS=['MEGA-CDP-style','NLI4CT-style','Syllogism-believability']
SEEDS=[11,22,33,44,55]
N_INV,N_FLIP,N_NATIVE=2000,1000,1500
METRICS=['invariant_rms','invariant_mean','raw_invariant_rms','raw_invariant_variance',
 'variance_reduction','r2','evidence_retention','cf_correction_over_ap_rms',
 'native_accuracy','invariant_pair_accuracy','flip_pair_accuracy',
 'invariant_left_accuracy','invariant_right_accuracy','flip_left_accuracy','flip_right_accuracy','path_accuracy']
TASK_CONFIG={
 'MEGA-CDP-style':dict(evidence_scale=12.,boundary_fraction=.14,boundary_scale=.55,nuisance_sigma=.85,base_noise_sigma=.17,raw_bias=.55),
 'NLI4CT-style':dict(evidence_scale=4.5,boundary_fraction=.32,boundary_scale=.42,nuisance_sigma=1.05,base_noise_sigma=.24,raw_bias=-.38),
 'Syllogism-believability':dict(evidence_scale=3.,boundary_fraction=.40,boundary_scale=.38,nuisance_sigma=1.15,base_noise_sigma=.20,raw_bias=.45)}
MODEL_CONFIG={
 'Qwen3-8B':dict(evidence_multiplier=1.,nuisance_multiplier=1.,noise_multiplier=1.),
 'Llama-3.1-8B-Instruct':dict(evidence_multiplier=.92,nuisance_multiplier=1.08,noise_multiplier=1.07),
 'MedGemma-27B':dict(evidence_multiplier=1.12,nuisance_multiplier=.92,noise_multiplier=.95)}
TRACE_Q={
 'Qwen3-8B':[.42,.20,.38],
 'Llama-3.1-8B-Instruct':[.26,.10,.25],
 'MedGemma-27B':[.50,.18,.22]}
TRACE_GAMMA={
 'Qwen3-8B':[.120,.050,.025],
 'Llama-3.1-8B-Instruct':[.050,.080,.035],
 'MedGemma-27B':[.040,.070,.050]}
def preset(g=1.,a=1.,trace=0.,extra=0.,bias=0.,**kw):
 return dict(g=g,a=a,trace_noise_coefficient=trace,extra_noise_coefficient=extra,bias=bias,**kw)
BASE={
 'context_calibration':preset(g=1.,a=1.,extra=0.,bias=.10),
 'pca_readout':preset(g=.90,a=.63,extra=.22,bias=.10),
 'early_closed_form':preset(g=.93,a=.82,extra=.25,bias=.80),
 'leace':preset(g=.72,a=.22,extra=.15,bias=.06),
 'inlp':preset(g=.58,a=.10,extra=.25,bias=.02),
 'iti':preset(g=1.02,a=.78,extra=.25,bias=-.03),
 'paired_n':preset(g=.99,a=.84,extra=.12,bias=0.),
 'paired_ns':preset(g=.78,a=.18,extra=.40,bias=0.),
 'loreft':preset(g=1.08,a=.66,extra=.12,bias=0.),
 'lofit':preset(g=1.03,a=.57,extra=.10,bias=0.),
 'lora':preset(g=1.12,a=.45,extra=.08,bias=0.),
}
FAMILIES={
 'raw':'unmodified','bias':'calibration','context_calibration':'calibration',
 'pca_readout':'output_readout','early_closed_form':'representation_erasure',
 'leace':'representation_erasure','inlp':'representation_erasure',
 'iti':'representation_editing','paired_n':'representation_editing','paired_ns':'representation_editing',
 'loreft':'training_adaptation','lofit':'training_adaptation','lora':'training_adaptation',
 'aster_cr':'output_cancellation','mask_action':'input_processing','semantic_extract':'input_processing'}
def condition_config(model,task):
 ti=TASKS.index(task);tc=TASK_CONFIG[task];mc=MODEL_CONFIG[model]
 Q=TRACE_Q[model][ti];gamma=TRACE_GAMMA[model][ti]
 conf=copy.deepcopy(BASE)
 conf['raw']=preset(bias=tc['raw_bias'])
 conf['bias']=preset()
 conf['aster_cr']=preset(g=1-Q*gamma,a=1-Q,trace=-np.sqrt(Q*(1-Q)),bias=0.,
  trace_predictability_Q=Q,evidence_leakage_gamma=gamma,
  trace_purity_scenario=('current-like evidence-impact assumption, not empirical replication' if model=='Qwen3-8B' and task=='MEGA-CDP-style' else 'hypothetical improved-trace-purity target scenario'),
  assumption='Fixed trace-quality and residual evidence-leakage assumptions; not estimated from the real ASTER experiment.')
 if task=='MEGA-CDP-style':
  conf['mask_action']=preset(g=.91,a=.07,extra=.05,bias=.08)
  conf['semantic_extract']=preset(g=.99,a=.20,extra=.10,bias=.05)
 for name,c in conf.items():
  c['family']=FAMILIES[name];c['status']='simulated'
 return conf
def ablations(main,task):
 full=copy.deepcopy(main['aster_cr']);Q=full['trace_predictability_Q'];gamma=full['evidence_leakage_gamma']
 def trace_cfg(q,leak):
  return preset(g=1-q*leak,a=1-q,trace=-np.sqrt(q*(1-q)),bias=0.,
    trace_predictability_Q=q,evidence_leakage_gamma=leak)
 apq=.97*Q
 return {
  'full':copy.deepcopy(full),'without_w':copy.deepcopy(main['bias']),
  'without_b':dict(copy.deepcopy(full),bias=TASK_CONFIG[task]['raw_bias']),
  'AP_only':trace_cfg(apq,max(gamma-.02,.005)),
  'APF0':copy.deepcopy(full),
  'no_orth_U':trace_cfg(Q,gamma+.32),
  'signal_only':preset(g=.78,a=.91,extra=.10),
  'random_heads':preset(g=.88,a=.99,trace=.35,extra=.20),
  'random_subspace':preset(g=.55,a=.90,trace=.45,extra=.25),
  'label_joint':preset(g=1.15,a=.65,trace=.10,extra=.15),
  'hidden_writeback':dict(copy.deepcopy(full),extra_noise_coefficient=.80,bias=.10),
  'rank8':trace_cfg(.65*Q,.75*gamma),
  'trace_leakage_current':trace_cfg(Q,(1-.9464)/Q)}
def make_data(rng,model,task):
 tc=TASK_CONFIG[task];mc=MODEL_CONFIG[model]
 ns=tc['nuisance_sigma']*mc['nuisance_multiplier']
 bs=tc['base_noise_sigma']*mc['noise_multiplier']
 def evidence(n):
  y=rng.integers(0,2,n)
  weak=rng.random(n)<tc['boundary_fraction']
  mag=np.where(weak,rng.gamma(1.4,tc['boundary_scale'],n),
        rng.lognormal(np.log(tc['evidence_scale']),.35,n))*mc['evidence_multiplier']
  return (2*y-1)*mag,y
 def primitives(E,y,shared_nuisance=False):
  shape=E.shape
  if shared_nuisance:
   nu=np.repeat(rng.normal(size=(shape[0],1)),2,axis=1)*ns
   tr=np.repeat(rng.normal(size=(shape[0],1)),2,axis=1)*ns
  else:nu=rng.normal(size=shape)*ns;tr=rng.normal(size=shape)*ns
  return dict(E=E,y=y,N=nu,T=tr,X=rng.normal(size=shape)*ns,B=rng.normal(size=shape)*bs)
 e,y=evidence(N_INV)
 invariant=primitives(np.column_stack((e,e)),np.column_stack((y,y)))
 e,y=evidence(N_FLIP)
 flip=primitives(np.column_stack((e,-e)),np.column_stack((y,1-y)),shared_nuisance=True)
 e,y=evidence(N_NATIVE)
 native=primitives(e,y)
 if task=='MEGA-CDP-style':native['path_indices']=rng.integers(0,N_NATIVE,size=(200,6))
 return invariant,flip,native
def components(data,c):
 return c['g']*data['E']+c['a']*data['N']+c['trace_noise_coefficient']*data['T']+c['extra_noise_coefficient']*data['X']+data['B']
def differences(data,c):
 # Compute the linear paired difference before adding any scalar bias.
 # Thus removing b leaves every paired-difference metric exactly unchanged.
 return (c['g']*(data['E'][:,1]-data['E'][:,0])
   +c['a']*(data['N'][:,1]-data['N'][:,0])
   +c['trace_noise_coefficient']*(data['T'][:,1]-data['T'][:,0])
   +c['extra_noise_coefficient']*(data['X'][:,1]-data['X'][:,0])
   +data['B'][:,1]-data['B'][:,0])
def rms(x):return float(np.sqrt(np.mean(x*x)))
def evaluate(data,c,raw):
 inv,flip,native=data
 d=differences(inv,c);d0=differences(inv,raw)
 df=differences(flip,c);df0=differences(flip,raw)
 var0=float(np.var(d0))
 score_inv=components(inv,c)+c['bias'];score_flip=components(flip,c)+c['bias']
 score_native=components(native,c)+c['bias']
 ic=(score_inv>0)==inv['y'];fc=(score_flip>0)==flip['y']
 sign=flip['y'][:,1]-flip['y'][:,0]
 er_denom=float(np.sum(sign*df0))
 return dict(invariant_rms=rms(d),invariant_mean=float(d.mean()),raw_invariant_rms=rms(d0),
  raw_invariant_variance=var0,variance_reduction=1-float(np.var(d))/var0,
  r2=1-float(np.mean(d*d))/var0,evidence_retention=float(np.sum(sign*df))/er_denom,
  cf_correction_over_ap_rms=rms(df-df0)/rms(d0),
  native_accuracy=float(np.mean((score_native>0)==native['y'])),
  invariant_pair_accuracy=float(np.mean(np.all(ic,axis=1))),
  flip_pair_accuracy=float(np.mean(np.all(fc,axis=1))),
  invariant_left_accuracy=float(ic[:,0].mean()),invariant_right_accuracy=float(ic[:,1].mean()),
  flip_left_accuracy=float(fc[:,0].mean()),flip_right_accuracy=float(fc[:,1].mean()),
  path_accuracy=float(np.mean(np.all(((score_native>0)==native['y'])[native['path_indices']],axis=1))) if 'path_indices' in native else None)
def write_csv(path,rr):
 fields=list(rr[0])
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rr)
def aggregate(rr):
 groups=defaultdict(list)
 for r in rr:groups[(r['model'],r['task'],r['scenario'],r['method'])].append(r)
 result=[]
 for (model,task,scenario,method),r in groups.items():
  out=dict(status='simulated',model=model,task=task,scenario=scenario,method=method,
   family=r[0]['family'],n_seeds=len(r),n_invariant=N_INV,n_flip=N_FLIP,n_native=N_NATIVE,
   n_paths=200 if task=='MEGA-CDP-style' else 0,path_length=6 if task=='MEGA-CDP-style' else None,
   path_status='synthetic_shared_node_paths' if task=='MEGA-CDP-style' else 'not_applicable')
  for metric in METRICS:
   values=[v[metric] for v in r if v[metric] is not None]
   x=np.array(values,float)
   out[metric+'_mean']=float(x.mean()) if len(x) else None
   out[metric+'_std']=float(x.std(ddof=1)) if len(x)>1 else None
  result.append(out)
 return result
main_rows=[];abl_rows=[];configs=[]
for mi,model in enumerate(MODELS):
 for ti,task in enumerate(TASKS):
  scenario=f'M{mi+1}T{ti+1}';main=condition_config(model,task);abl=ablations(main,task)
  configs.append(dict(status='simulated',model=model,task=task,scenario=scenario,
   task_latent=TASK_CONFIG[task],model_latent=MODEL_CONFIG[model],main_methods=main,ablations=abl))
  for seed in SEEDS:
   rng=np.random.default_rng(20261109+10000*mi+1000*ti+seed)
   data=make_data(rng,model,task)
   base=dict(status='simulated',model=model,task=task,scenario=scenario,seed=seed,
    n_invariant=N_INV,n_flip=N_FLIP,n_native=N_NATIVE,
    n_paths=200 if task=='MEGA-CDP-style' else 0,path_length=6 if task=='MEGA-CDP-style' else None,
   path_status='synthetic_shared_node_paths' if task=='MEGA-CDP-style' else 'not_applicable')
   for name,c in main.items():
    main_rows.append(dict(base,method=name,family=FAMILIES[name],**evaluate(data,c,main['raw'])))
   for name,c in abl.items():
    abl_rows.append(dict(base,method=name,family='ablation',**evaluate(data,c,main['raw'])))
main_summary=aggregate(main_rows);ablation_summary=aggregate(abl_rows)
write_csv(OUT/'main_seed_metrics.csv',main_rows)
write_csv(OUT/'main_summary.csv',main_summary)
write_csv(OUT/'ablation_seed_metrics.csv',abl_rows)
write_csv(OUT/'ablation_summary.csv',ablation_summary)
main_keys={(r['model'],r['task'],r['seed'],r['method']):r for r in main_rows}
abl_keys={(r['model'],r['task'],r['seed'],r['method']):r for r in abl_rows}
diffmetrics=['invariant_rms','invariant_mean','raw_invariant_rms','raw_invariant_variance',
 'variance_reduction','r2','evidence_retention','cf_correction_over_ap_rms']
nob_exact=all(abl_keys[(m,t,s,'full')][k]==abl_keys[(m,t,s,'without_b')][k]
 for m in MODELS for t in TASKS for s in SEEDS for k in diffmetrics)
apf0_exact=all(abl_keys[(m,t,s,'full')][k]==abl_keys[(m,t,s,'APF0')][k]
 for m in MODELS for t in TASKS for s in SEEDS for k in METRICS)
pair_bounds=all(r['invariant_pair_accuracy']<=min(r['invariant_left_accuracy'],r['invariant_right_accuracy'])+1e-12
 and r['flip_pair_accuracy']<=min(r['flip_left_accuracy'],r['flip_right_accuracy'])+1e-12 for r in main_rows+abl_rows)
identity_error=max(abs((r['variance_reduction']-r['r2'])-r['invariant_mean']**2/r['raw_invariant_variance']) for r in main_rows+abl_rows)
validation=dict(status='simulated',without_b_difference_metrics_exactly_equal=nob_exact,
 full_APF0_all_metrics_exactly_equal=apf0_exact,pair_accuracy_bounds_respected=pair_bounds,
 variance_R2_identity_max_error=identity_error,
 checked_rows=len(main_rows)+len(abl_rows))
configuration=dict(status='simulated',
 description='Prespecified score-level transformations. Method names are condition labels inspired by broad algorithm classes, not official implementations or measured baseline outputs.',
 score_equation='score_m = g_m E + a_m N + c_m T + d_m X + B + bias_m; E/N/T/X/B shared by all methods in each scenario and seed.',
 primitive_meaning=dict(E='label-signed evidence score with a fixed low-margin mixture',
 N='nuisance score, independently rewritten across invariant pair sides; held constant across truth-flipping pair sides',
 T='independent noisy nuisance-reference channel, independently rewritten across invariant sides and held constant across flip sides',
 X='shared standardized additional intervention/readout error draw; each method has its prespecified coefficient',
 B='shared unresolved model-score noise, unchanged across methods'),
 trace_derivation='For noise-reference quality Q and leakage gamma: a=1-Q, c=-sqrt(Q(1-Q)), g=1-Q*gamma. This presupposes a linear nuisance/reference model; Q/gamma are chosen assumptions, not empirical estimates.',
 F0_assumption='The APF0 label describes an assumed auxiliary within-label stability objective. F0 changes evidence values; its inclusion is an additional stability assumption, not automatic nuisance-only supervision. This generator does not simulate fitting actual A/P/F0 language-model features.',
 without_b_definition='Same full coefficients and primitive draws; only scalar bias changes to the original raw bias. All paired-score-difference metrics are computed before scalar bias and therefore are exactly equal.',
 main_full_relation='ablation full and APF0 exactly copy main aster_cr.',
 context_calibration_assumption='A global constant offset only; paired differences, variance reduction, R2 and evidence response are identical to raw/bias. Its residual offset is a prespecified calibration error, not a measured contextual calibration algorithm.',
 trace_current_comparator='trace_leakage_current imposes g=.9464 to illustrate an evidence-impact level similar to the real 94.64% result; this remains a synthetic assumption, not replication or transfer of real coefficients.',
 baseline_fairness='All methods see the same latent evidence, nuisance and noise draws. Differences are imposed by prespecified coefficients; no baseline official code, parameter counts, training costs or learned superiority is measured.',
 real_result_separation='This simulation does not replace the cached repair-v2 result of 41.01% invariant variance reduction and 94.64% signed evidence-response retention. Some synthetic conditions assume less trace leakage than observed in that real result.',
 seed_values=SEEDS,sample_sizes=dict(invariant_pairs=N_INV,truth_flip_pairs=N_FLIP,native_nodes=N_NATIVE),
 conditions=configs)
metadata=dict(status='simulated',not_real_model_runs=True,not_official_benchmark_results=True,
 intended_use='Labeled hypothetical result projection and figure layout for discussion; no evidence of empirical method superiority.',
 models_are_condition_labels=MODELS,tasks_are_condition_labels=TASKS,
 real_baseline_implementations_executed=False,gpu_used=False,language_models_loaded=False,
 seed_count=5,summary_std='Sample SD across five seeds, ddof=1; not a confidence interval or empirical benchmark uncertainty.',
 metric_definitions={
 'invariant_rms':'sqrt(mean((score_right-score_left)^2))',
 'variance_reduction':'1-Var(corrected invariant score difference)/Var(raw invariant score difference)',
 'r2':'1-sum(corrected invariant difference^2)/sum((raw invariant difference-mean(raw invariant difference))^2)',
 'evidence_retention':'sum((right_label-left_label)*corrected_flip_difference)/sum((right_label-left_label)*raw_flip_difference); closer to 1 is preferable, not larger without limit',
 'cf_correction_over_ap_rms':'RMS(corrected_flip_difference-raw_flip_difference)/RMS(raw_invariant_difference)',
 'native_accuracy':'mean(predicted_label==label) on a separate n=1500 native sample',
 'path_accuracy':'MEGA-style only: fraction of 200 fixed synthetic length-six paths with all six sampled native nodes correct; not real decision-tree execution',
 'invariant_pair_accuracy':'fraction with both invariant pair sides correct',
 'flip_pair_accuracy':'fraction with both truth-flipping pair sides correct'},
 recommended_main_direction='Smaller invariant_rms, larger variance_reduction/R2, evidence_retention closer to 1, smaller cf_correction_over_ap_rms; accuracy is a separate check.',
 paths='MEGA-style: 200 fixed synthetic length-six paths sampled with replacement from the same 1500 native nodes; every method uses identical path indices. This does not simulate a real decision-tree controller. NLI/syllogism paths are not applicable.',
 main_rows=len(main_rows),ablation_rows=len(abl_rows),
 files=['generator.py','configuration.json','metadata.json','main_seed_metrics.csv','main_summary.csv','ablation_seed_metrics.csv','ablation_summary.csv','validation.json'])
for name,obj in [('configuration.json',configuration),('metadata.json',metadata),('validation.json',validation)]:
 (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(dict(status='simulated',main_rows=len(main_rows),main_summary_rows=len(main_summary),
 ablation_rows=len(abl_rows),ablation_summary_rows=len(ablation_summary),validation=validation,
 examples=[{k:r[k] for k in ('model','task','method','variance_reduction_mean','evidence_retention_mean','native_accuracy_mean')} for r in main_summary if r['method'] in ('raw','aster_cr','lora')]),ensure_ascii=True))

