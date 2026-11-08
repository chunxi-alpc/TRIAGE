"""Original-label residual readout: no pairs, head selection, or internal editing."""
import argparse, json, os, pickle, subprocess, sys, time, traceback
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.decomposition import PCA
from experiments.aster_cr.common import rows, read, write, lines, append
from experiments.aster_cr.paths import execute
from experiments.aster_cr.statistics import interval
from experiments.aster_closed_form.data import canonical_guide, prompt

ROOT=Path('/home/gcx/ordinary_readout_20261108')
OLD=Path('/home/gcx/aster_cr_20261108')
REPAIRED=Path('/home/gcx/aster_cr_repair_20261108')
SEED=20261108
METHODS=['full','calibration','linear']

def fit_correction(ell,z,y,reg=0.,mode='linear'):
 ell=np.asarray(ell,dtype=np.float64);z=np.asarray(z,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
 if mode=='logistic':design=np.column_stack([np.ones(len(y)),ell]);offset=np.zeros(len(y));initial=np.array([0.,1.]);bounds=[(None,None),(0.,None)]
 else:design=np.column_stack([np.ones(len(y)),z]) if mode=='linear' else np.ones((len(y),1));offset=ell;initial=np.zeros(design.shape[1]);bounds=None
 def objective(w):
  scores=offset+design@w;pen=reg if mode=='linear' else 0.
  loss=np.mean(np.logaddexp(0,scores)-y*scores)+pen*np.dot(w[1:],w[1:])/2
  grad=design.T@(expit(scores)-y)/len(y);grad[1:]+=pen*w[1:]
  return loss,grad
 result=minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=500,ftol=1e-12,gtol=1e-8))
 if not result.success:raise RuntimeError('Readout optimization failed: '+result.message)
 return dict(mode=mode,coef=result.x,regularization=reg,parameters=len(result.x),loss=float(result.fun),iterations=int(result.nit))

def predict(model,ell,z=None):
 ell=np.asarray(ell,dtype=np.float64)
 if model['mode']=='full':return ell
 w=model['coef']
 if model['mode']=='logistic':return w[0]+w[1]*ell
 return ell+w[0]+(np.asarray(z)@w[1:] if model['mode']=='linear' else 0)

def project(transform,raw):
 return transform['pca'].transform((np.asarray(raw,dtype=np.float64)-transform['mean'])/transform['scale'])

def make_transform(raw,rank=64):
 raw=np.asarray(raw,dtype=np.float64);mean=raw.mean(0);scale=np.maximum(raw.std(0),1e-6)
 pca=PCA(n_components=min(rank,len(raw)-1,raw.shape[1]),whiten=True,svd_solver='randomized',random_state=SEED)
 pca.fit((raw-mean)/scale)
 return dict(mean=mean,scale=scale,pca=pca,fit_examples=len(raw),requested_rank=rank)

def load_residual(record):
 with np.load(record['feature']) as f:return f['residual'].astype(np.float64).reshape(-1)

def node_record(v,score):
 return dict(view_id=v['view_id'],case_id=v['case_id'],source_id=v['source_id'],label=v['label'],score=float(score),prediction=int(score>0),correct=int(score>0)==v['label'])

def summarize(records,baseline=None):
 result=dict(n=len(records),correct=sum(r['correct'] for r in records),accuracy=sum(r['correct'] for r in records)/len(records) if records else None)
 result['accuracy_by_source_bootstrap']=interval([r['correct'] for r in records],[r['source_id'] for r in records])
 if baseline is not None:
  idname='view_id' if records and 'view_id' in records[0] else 'case_id';base={r[idname]:r['correct'] for r in baseline}
  assert set(base)=={r[idname] for r in records}
  delta=[int(r['correct'])-int(base[r[idname]]) for r in records]
  result.update(corrected=sum(d==1 for d in delta),new_errors=sum(d==-1 for d in delta),difference_by_source_bootstrap=interval(delta,[r['source_id'] for r in records]))
 return result

def prepare():
 ROOT.mkdir(exist_ok=True);(ROOT/'logs').mkdir(exist_ok=True)
 # Publish the protocol before loading development outcomes or any new test scores.
 protocol=dict(model='Qwen3-8B',input='unchanged full tree, patient, current node; existing CR score protocol',representation='last transformer block output at final decision token, before final RMSNorm',training='761 original selected fit nodes, released labels only',rank=64,transform='fit-only coordinate standardization and whitened PCA',formula='corrected_logit = original_logit + intercept + PCA(residual) @ weights',regularization=[.01,.1,1.],loss='mean binary cross entropy plus lambda/2 times squared weight norm; intercept unpenalized',calibration=['bias','nonnegative-slope logistic'],selection='development autonomous path accuracy, then original-node accuracy; ties: larger regularization for linear, fewer parameters for calibration',seed=SEED,gpus=[0,1,6],node_denominators=dict(fit=761,dev=199,test=372),path_denominators=dict(dev=420,test=806),original_inputs_only=True,pair_data_used=False,head_selection=False,backbone_training=False,test_used_for_selection=False,old_m3='historical context only, different test denominator; not a matched comparison')
 write(ROOT/'protocol.json',protocol)
 views=[v for v in rows(OLD/'data/views.jsonl') if v.get('original')]
 nodes={g['group_id']:g for g in rows(REPAIRED/'data/original_nodes.jsonl')}
 for v in views:v['gid']=nodes[v['group_id']]['gid']
 cases=rows(REPAIRED/'data/path_cases.jsonl');index={c['case_id']:c for c in cases}
 for v in views:
  c=index[v['case_id']]
  assert v['prompt']==prompt(c['guideline'],c['definitions'],c['patient'],v['gid'])
 for a,b in [('fit','dev'),('fit','test'),('dev','test')]:
  assert not ({v['source_id'] for v in views if v['split']==a}&{v['source_id'] for v in views if v['split']==b})
 lines(ROOT/'original_views.jsonl',views);lines(ROOT/'path_cases.jsonl',cases)
 cached={r['view_id']:r for p in (OLD/'qwen/scores').glob('*.jsonl') for r in rows(p) if r.get('status')=='ok'}
 fit=[v for v in views if v['split']=='fit'];dev=[v for v in views if v['split']=='dev']
 assert len(fit)==761 and len(dev)==199 and all(v['view_id'] in cached for v in fit+dev)
 raw=np.stack([load_residual(cached[v['view_id']]) for v in fit]);transform=make_transform(raw)
 z=project(transform,raw);ell=np.array([cached[v['view_id']]['ell'] for v in fit]);y=np.array([v['label'] for v in fit])
 models={'full':dict(mode='full',parameters=0),'bias':fit_correction(ell,z,y,mode='bias'),'logistic':fit_correction(ell,z,y,mode='logistic')}
 for reg in protocol['regularization']:models['linear_'+str(reg)]=fit_correction(ell,z,y,reg)
 devraw=np.stack([load_residual(cached[v['view_id']]) for v in dev]);dz=project(transform,devraw);de=np.array([cached[v['view_id']]['ell'] for v in dev])
 node_results={name:[node_record(v,s) for v,s in zip(dev,predict(model,de,dz))] for name,model in models.items()}
 pathcache={r['id']:r for p in (OLD/'qwen/paths').glob('features_dev_*.jsonl') for r in rows(p) if r.get('status')=='ok'}
 devcases=[c for c in cases if c['split']=='dev'];rs=list(pathcache.values())
 assert len(devcases)==420
 for c in devcases:assert all(c['case_id']+':'+n['id'] in pathcache for n in canonical_guide(c['guideline'])['nodes'])
 # Project the cache in small blocks, without storing all head activations.
 pathscore={name:defaultdict(dict) for name in models}
 for start in range(0,len(rs),256):
  block=rs[start:start+256];br=np.stack([load_residual(r) for r in block]);bz=project(transform,br);be=np.array([r['ell'] for r in block])
  for name,model in models.items():
   for r,s in zip(block,predict(model,be,bz)):pathscore[name][r['case_id']][r['gid']]=float(s)
 path_results={name:[dict(case_id=c['case_id'],source_id=c['source_id'],**execute(c,pathscore[name][c['case_id']])) for c in devcases] for name in models}
 metrics={name:dict(nodes=summarize(node_results[name]),paths=summarize(path_results[name]),parameters=model['parameters'],regularization=model.get('regularization')) for name,model in models.items()}
 def selection_key(name):return (metrics[name]['paths']['correct'],metrics[name]['nodes']['correct'],models[name].get('regularization',0),-models[name]['parameters'])
 chosen=dict(calibration=max(['bias','logistic'],key=selection_key),linear=max([n for n in models if n.startswith('linear_')],key=selection_key))
 selected={'full':models['full'],**{name:models[ident] for name,ident in chosen.items()}}
 bundle=dict(transform=transform,models=selected,selected=chosen)
 with (ROOT/'model.pkl').open('wb') as f:pickle.dump(bundle,f)
 write(ROOT/'development.json',dict(candidates=metrics,selected=chosen))
 for name in METHODS:
  ident=chosen.get(name,name);lines(ROOT/'predictions'/f'{name}_nodes_dev.jsonl',node_results[ident]);lines(ROOT/'predictions'/f'{name}_paths_dev.jsonl',path_results[ident])
 write(ROOT/'frozen.json',dict(frozen=True,time=time.strftime('%Y-%m-%d %H:%M:%S'),selected=chosen,rank=transform['pca'].n_components_,fit_only_pca_explained_variance=float(transform['pca'].explained_variance_ratio_.sum()),test_scored=False,training_nodes=len(fit),cached_dev_tree_nodes=len(pathcache)))
 write(ROOT/'status.json',dict(state='model_frozen_test_pending',selected=chosen))
 return bundle

class ResidualScorer:
 def __init__(self):
  from experiments.aster_cr.engine import Engine
  import torch
  self.torch=torch;torch.set_num_threads(4);self.engine=Engine('qwen')
 def score(self,text):
  eng=self.engine;context=eng.prepare(text);captured={}
  def hook(module,args,out):
   value=out[0] if isinstance(out,tuple) else out
   captured['residual']=value[0,-1].detach().float().cpu().numpy().astype(np.float16)
  handle=eng.layers[-1].register_forward_hook(hook)
  try:result=eng.run(context,capture=False)
  finally:handle.remove()
  return result,captured['residual']

def reachable_scores(case,score_fn,methods=METHODS):
 # Prediction alone decides the next node. Gold branches are never passed to score_fn.
 tree=canonical_guide(case['guideline']);nodes={n['id']:n for n in tree['nodes']};scores={m:{} for m in methods}
 current={m:tree['nodes'][0]['id'] for m in methods}
 for _ in range(len(nodes)+1):
  needed={g for g in current.values() if g in nodes}
  if not needed:break
  for gid in sorted(needed):
   result=score_fn(gid)
   for m in methods:scores[m][gid]=result[m]
  for m,gid in list(current.items()):
   if gid in nodes:current[m]=nodes[gid]['yes' if scores[m][gid]>0 else 'no']
 return scores

def worker(shard):
 import torch
 torch.set_num_threads(4)
 assert read(ROOT/'frozen.json').get('frozen')
 with (ROOT/'model.pkl').open('rb') as f:bundle=pickle.load(f)
 cases=[c for c in rows(ROOT/'path_cases.jsonl') if c['split']=='test'][shard::3]
 views=[v for v in rows(ROOT/'original_views.jsonl') if v['split']=='test'];bycase=defaultdict(list)
 for v in views:bycase[v['case_id']].append(v)
 cachefile=ROOT/'test_features'/f'records_{shard}.jsonl';cached={r['id']:r for r in rows(cachefile)}
 recordsfile=ROOT/'test_results'/f'cases_{shard}.jsonl';completed={r['case_id']:r for r in rows(recordsfile)}
 scorer=None;start=time.time();forward_count=0;reused=0
 def engine():
  nonlocal scorer
  if scorer is None:scorer=ResidualScorer()
  return scorer
 # One cached fit input checks that the lean residual-only scorer matches the old protocol.
 if shard==0 and not read(ROOT/'score_protocol_check.json').get('passed'):
  fit=next(v for v in rows(ROOT/'original_views.jsonl') if v['split']=='fit')
  old=next(r for p in (OLD/'qwen/scores').glob('fit_*.jsonl') for r in rows(p) if r['view_id']==fit['view_id'])
  result,residual=engine().score(fit['prompt']);delta=abs(result['ell']-old['ell']);rd=float(np.max(np.abs(residual.astype(float)-load_residual(old))))
  check=dict(view_id=fit['view_id'],logit_abs_difference=delta,residual_max_abs_difference=rd,passed=delta<.005 and rd<.005)
  write(ROOT/'score_protocol_check.json',check)
  if not check['passed']:raise RuntimeError('Residual-only score protocol mismatch')
 for i,c in enumerate(cases):
  if c['case_id'] in completed:continue
  def score_node(gid):
   nonlocal forward_count,reused
   ident=c['case_id']+':'+gid
   if ident in cached:reused+=1;return cached[ident]['scores']
   before=time.perf_counter();result,residual=engine().score(prompt(c['guideline'],c['definitions'],c['patient'],gid))
   z=project(bundle['transform'],residual[None,:]);scores={m:float(predict(bundle['models'][m],[result['ell']],z)[0]) for m in METHODS}
   target=ROOT/'residuals'/(ident+'.npz');target.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,residual=residual)
   r=dict(id=ident,case_id=c['case_id'],source_id=c['source_id'],gid=gid,scores=scores,ell=result['ell'],feature=str(target),tokens=result['tokens'],inference_seconds=time.perf_counter()-before)
   append(cachefile,r);cached[ident]=r;forward_count+=1
   return scores
  for v in bycase[c['case_id']]:score_node(v['gid'])
  scores=reachable_scores(c,score_node)
  paths={m:dict(case_id=c['case_id'],source_id=c['source_id'],**execute(c,scores[m])) for m in METHODS}
  nodes={m:[node_record(v,cached[c['case_id']+':'+v['gid']]['scores'][m]) for v in bycase[c['case_id']]] for m in METHODS}
  append(recordsfile,dict(case_id=c['case_id'],paths=paths,nodes=nodes));completed[c['case_id']]=True
  write(ROOT/'workers'/f'{shard}.json',dict(state='running',pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),cases_done=i+1,cases_total=len(cases),new_forwards=forward_count,seconds=time.time()-start,peak_gib=round(torch.cuda.max_memory_allocated()/2**30,2)))
 write(ROOT/'workers'/f'{shard}.json',dict(state='complete',pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),cases_done=len(cases),cases_total=len(cases),new_forwards=forward_count,reused=reused,seconds=time.time()-start,peak_gib=round(torch.cuda.max_memory_allocated()/2**30,2)))

def finalize():
 rr=[r for shard in range(3) for r in rows(ROOT/'test_results'/f'cases_{shard}.jsonl')]
 assert len(rr)==806 and len({r['case_id'] for r in rr})==806
 summary={};allrecords={}
 for name in METHODS:
  paths=[r['paths'][name] for r in rr];nodes=[n for r in rr for n in r['nodes'][name]]
  assert len(nodes)==372 and len({n['view_id'] for n in nodes})==372
  lines(ROOT/'predictions'/f'{name}_paths_test.jsonl',paths);lines(ROOT/'predictions'/f'{name}_nodes_test.jsonl',nodes)
  allrecords[name]=dict(paths=paths,nodes=nodes)
 for name in METHODS:
  summary[name]={kind:summarize(allrecords[name][kind],allrecords['full'][kind]) for kind in ['nodes','paths']}
 comparisons={kind:summarize(allrecords['linear'][kind],allrecords['calibration'][kind]) for kind in ['nodes','paths']}
 write(ROOT/'results.json',dict(methods=summary,linear_vs_calibration=comparisons,selected=read(ROOT/'development.json')['selected'],node_denominator=372,path_denominator=806,test_sources=40))
 frozen=read(ROOT/'frozen.json');frozen['test_scored']=True;write(ROOT/'frozen.json',frozen)
 write(ROOT/'status.json',dict(state='complete',completed_at=time.strftime('%Y-%m-%d %H:%M:%S'),training='only small linear readout and scalar calibration; no backbone training'))
 report()
 return summary

def report():
 result=read(ROOT/'results.json');dev=read(ROOT/'development.json');state=read(ROOT/'status.json');frozen=read(ROOT/'frozen.json')
 out=['# 原始病例监督的普通线性读出实验','',f"状态：{state.get('state')}。",'', 'Qwen3-8B 冻结。输入、标签和划分沿用原始节点及病例；不使用 A/P/C/F 配对，不选注意力头、不构造干扰/信号子空间、不编辑隐藏层。', '', '## 方法与选择','', '固定最后一层、最终判定位置、最终 RMSNorm 之前的残差表示。仅用 761 个 fit 节点计算标准化和 64 维白化 PCA；训练 原分数 + 截距 + 线性修正。损失为平均交叉熵加权重 L2；截距不惩罚。正则强度仅比较 0.01/0.1/1。','', '校准比较单独截距和非负斜率逻辑校准。两类方法分别按 420 例开发自主路径正确数、199 个原始节点正确数选择；并列选更强正则或更少参数。不按测试结果调整，不用配对宏平均选参，也不因开发性能不佳而把线性方法替换成原模型。', '',f"开发选择：{dev.get('selected')}。PCA 拟合解释方差：{frozen.get('fit_only_pca_explained_variance')}。",'', '训练/开发/测试节点分母：761/199/372；开发/测试路径：420/806；来源：80/20/40。新的测试预测在 model.pkl 和 frozen.json 写入后产生。原始来源划分继承既有研究，不称为全流程盲测。', '', '## 测试结果','']
 if result:
  out+=['|方法|节点正确/372|路径正确/806|相对原模型：纠正/新增路径错误|','|---|---:|---:|---:|']
  for name,s in result['methods'].items():out.append(f"|{name}|{s['nodes']['correct']}|{s['paths']['correct']}|{s['paths']['corrected']}/{s['paths']['new_errors']}|")
  out+=['','按 40 个来源进行 2,000 次配对 bootstrap；差值区间见 results.json。','']
  for kind in ['nodes','paths']:
   comp=result['linear_vs_calibration'][kind]['difference_by_source_bootstrap'];out.append(f"linear − calibration ({kind})：差值 {comp['estimate']:.6f}，95% 来源区间 {comp['ci95']}。")
  d=result['linear_vs_calibration']['paths']['difference_by_source_bootstrap']
  out+=['', '线性读出是否优于校准以同一测试病例的路径差值及来源区间判断；可解码不等于已证明存在独立纠错信息。', '', '本次结论：'+('路径差值的来源区间全部高于零，支持本设置中线性读出优于校准。' if d['ci95'][0]>0 else '尚未建立线性读出相对简单校准的稳定路径收益。')]
 else:out+=['测试尚未完成，不填入结果。']
 out+=['','## 复用、成本与历史结果','', '复用全部训练/开发原始节点，以及 8,312 个开发树节点的相同提示、模型和分数协议缓存。新增测试只计算三种方法实际到达节点的并集，以及固定原始测试节点；路径由各方法自己的预测决定，标准路径仅用于评分。残差保存为 FP16，与旧缓存精度一致。', '', 'GPU 0/1/6 分片；每卡时间、前向数和峰值显存见 workers/；总时间见 runtime.json。训练仅在 CPU 上拟合小型读出。没有新增训练网格、配对构造或哈希检查。', '', '旧辅助参考 M3：官方 7,458 例中正确 7,072，原模型 7,068。该历史实验使用不同输入/评价划分，不能与本轮 806 例直接比较，也不作为本轮配对统计。', '', '交付：protocol.json、development.json、frozen.json、model.pkl、results.json、predictions/、residuals/、test_features/、workers/。']
 (ROOT/'REPORT.zh.md').write_text('\n'.join(out)+'\n')

def run():
 start=time.time();ROOT.mkdir(exist_ok=True);(ROOT/'logs').mkdir(exist_ok=True)
 try:
  if not read(ROOT/'frozen.json').get('frozen'):prepare()
  write(ROOT/'status.json',dict(state='test_running',pid=os.getpid(),started_at=time.strftime('%Y-%m-%d %H:%M:%S')));report()
  workers=[]
  for shard,gpu in enumerate([0,1,6]):
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
   log=open(ROOT/'logs'/f'worker_{shard}.log','a')
   p=subprocess.Popen([sys.executable,'-u','-m','experiments.ordinary_readout','--stage','worker','--shard',str(shard)],env=env,stdout=log,stderr=subprocess.STDOUT);log.close();workers.append((p,gpu))
  write(ROOT/'processes.json',[dict(pid=p.pid,gpu=gpu) for p,gpu in workers])
  pending=list(workers);codes=[]
  while pending:
   for p,gpu in list(pending):
    code=p.poll()
    if code is not None:
     codes.append(dict(pid=p.pid,gpu=gpu,exit_code=code));pending.remove((p,gpu))
     if code!=0:
      for other,_ in pending:other.terminate()
      raise RuntimeError(f'GPU {gpu} worker failed with exit code {code}; see worker log')
   if pending:time.sleep(5)
  finalize();write(ROOT/'runtime.json',dict(seconds=time.time()-start,workers=codes));report()
 except Exception as exc:
  traceback.print_exc();write(ROOT/'status.json',dict(state='failed',error=str(exc),pid=os.getpid()));raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','worker','finalize','run'],default='run');p.add_argument('--shard',type=int,default=0);a=p.parse_args()
 if a.stage=='worker':worker(a.shard)
 else:globals()[a.stage]()
