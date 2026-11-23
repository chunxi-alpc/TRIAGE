"""One-seed development experiment. Independent of locked continuation_v2 runs."""
import argparse,collections,hashlib,json,os,time,traceback
from pathlib import Path
import torch
from engine import Engine
from continue_run import Journal,atomic,key,safe,argmax
BASE=Path('/home/gcx/aster_e0_e8_20261024/private')
CORE=('NC','AC','IC','SW')
WEIGHTS=[(1.,0.,0.),(0.,1.,0.),(.75,.25,0.),(.5,.5,0.),(.5,.25,.25),(.25,.5,.25),(0.,.5,.5)]
PROMPT_VERSION='condition-three-way-v1'
def read(path):
 rows=[]
 if path.exists():
  for line in path.read_text().splitlines():
   try:rows.append(json.loads(line))
   except json.JSONDecodeError:break
 return rows
def mix(ps,w):return [sum(wi*pi for wi,pi in zip(w,col)) for col in zip(*ps)]
def judge_payload(c):
 return {'patient':c['patient'],'conditions':[{'rid':r['rid'],'condition':r['condition']} for r in c['rules']]}
def condition_key(c):return hashlib.sha256(json.dumps(judge_payload(c),sort_keys=True).encode()).hexdigest()
def judge(e,payload):
 """Only the patient and conditions are accepted; no actions/options/gold."""
 support=[];judgments=[]
 for rule in payload['conditions']:
  text=('Determine whether the patient satisfies the supplied rule condition. Evaluate only patient facts and this condition. '
        'Do not infer a treatment or choose a rule number. Respect negation, thresholds, alternatives, and temporal qualifiers. '
        'For a conjunction, every required part must hold; for an alternative, one supported alternative is sufficient. '
        'Use information insufficient when a required patient fact is missing.\n'
        'Patient:\n'+payload['patient']+'\nCondition:\n'+rule['condition']+'\n'
        'A. The patient satisfies the condition.\nB. The patient contradicts the condition.\nC. Information is insufficient.\n'
        'Return only A, B, or C.')
  rendered=e.tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  ids=e.tok.encode(rendered,add_special_tokens=False)+e.tok.encode('Assessment:',add_special_tokens=False)
  labels=[e.tok.encode(' '+x,add_special_tokens=False) for x in 'ABC']
  if any(len(x)!=1 for x in labels):raise ValueError('Expected single-token judgment labels')
  logits=e.forward(ids,[None]*len(ids),len(ids))[-1]
  probs=torch.tensor([float(logits[x[0]]) for x in labels]).softmax(0).tolist()
  judgments.append(probs);support.append(max(probs[0],1e-12))
 total=sum(support)
 return [p/total for p in support],judgments

def select_weights(rows,probabilities):
 """Cross-fit weights by tree. No held-out label enters weight selection."""
 trees=sorted({c['tree'] for c in rows});choices={}
 for heldout in trees:
  train=[c for c in rows if c['tree']!=heldout]
  scores=[]
  for w in WEIGHTS:
   by_tree=collections.defaultdict(list)
   for c in train:by_tree[c['tree']].append(argmax(mix(probabilities[key(c)],w))==c['gold_rule'])
   scores.append(sum(sum(v)/len(v) for v in by_tree.values())/len(by_tree))
  idx=max(range(len(scores)),key=lambda j:scores[j])
  choices[heldout]={'weights':WEIGHTS[idx],'training_macro_branch_accuracy':scores[idx],'candidate_scores':scores,'training_trees':[t for t in trees if t!=heldout]}
 return choices

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--name',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='6','This experiment is assigned only GPU6'
 a.out.mkdir(parents=True,exist_ok=True);start=time.time();source=BASE/'continuation_v2'/a.name
 try:
  rows=[c for c in read(BASE/'cases.jsonl') if c['split']=='dev' and c['seed']==0 and c['variant'] in CORE]
  rows.sort(key=lambda c:(hashlib.sha256(c['id'].encode()).hexdigest(),CORE.index(c['variant'])))
  manifest={'model':a.model,'seed':0,'split':'dev','rows':len(rows),'cases':len({c['id'] for c in rows}),'trees':sorted({c['tree'] for c in rows}),'gpu':6,'prompt':PROMPT_VERSION,'weights':WEIGHTS,'lambda':2,'selection':'leave-one-tree-out macro branch accuracy; sources condition, blind, external','code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'engine_sha256':hashlib.sha256(Path(__file__).with_name('engine.py').read_bytes()).hexdigest(),'data_sha256':hashlib.sha256((BASE/'cases.jsonl').read_bytes()).hexdigest()}
  path=a.out/'manifest.json'
  if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(manifest)),'Configuration changed; use another output'
  else:atomic(path,manifest)
  e=Engine(a.model);j=Journal(a.out);cachepath=a.out/'condition_cache.jsonl';cache={r['condition_key']:r for r in read(cachepath)}
  def status(stage,n,total):
   atomic(a.out/'status.json',{'stage':stage,'completed_rows':n,'total_rows':total,'elapsed_seconds':time.time()-start,'gpu_hours_this_process':(time.time()-start)/3600,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,'updated_unix':time.time(),'complete':stage=='complete'})
  with cachepath.open('a') as f:
   for i,c in enumerate(rows):
    ck=condition_key(c)
    if ck not in cache:
     t=time.time();p,assessments=judge(e,judge_payload(c));r={'condition_key':ck,'p':p,'assessments':assessments,'seconds':time.time()-t};cache[ck]=r;f.write(json.dumps(r)+'\n');f.flush()
    r=cache[ck]
    if not j.get('branches','condition',c):j.put('branches','condition',c,p=r['p'],assessments=r['assessments'],condition_key=ck)
    if (i+1)%20==0:status('condition_judgment',i+1,len(rows))
  oldblind={r['key']:r for r in read(source/'branches'/'blind.jsonl')};external={r['key']:r for r in read(BASE/'continuation_v2'/'evidence'/'branches'/'hidden.jsonl')}
  # If the original run has not reached a case, infer this missing baseline here.
  for c in rows:
   k=key(c)
   if k not in oldblind:
    r=j.get('branches','blind',c)
    if not r:r=j.put('branches','blind',c,p=e.branch(safe(c),blind=True))
    oldblind[k]=r
  probabilities={key(c):[cache[condition_key(c)]['p'],oldblind[key(c)]['p'],external[key(c)]['p']] for c in rows}
  choices=select_weights(rows,probabilities);atomic(a.out/'fusion_choices.json',choices)
  previous={}
  for p in (source/'dev_select').glob('*.jsonl'):previous[p.stem]={r['key']:r for r in read(p)}
  for p in (source/'E2').glob('*.jsonl'):previous['E2_'+p.stem]={r['key']:r for r in read(p)}
  for i,c in enumerate(rows):
   k=key(c);cond,blind,ext=probabilities[k];weighted=mix(probabilities[k],choices[c['tree']]['weights']);equal=mix([blind,ext],[.5,.5]);sources={'blind':blind,'condition':cond,'equal_external':equal,'weighted_oof':weighted};answer_cache={}
   for name,p in sources.items():
    if not j.get('branches',name,c):j.put('branches',name,c,p=p)
   def answer(name,kw=None,p=None,direct=False,reuse=None,oracle=False):
    if j.get('answers',name,c):return
    kw=kw or {};t=time.time();reused=False
    old=previous.get(reuse,{}).get(k) if reuse else None
    signature=json.dumps({'kw':kw,'direct':p if direct else None},sort_keys=True)
    if old:pred,scores=old['prediction'],old['scores'];reused=True
    elif signature in answer_cache:pred,scores=answer_cache[signature];reused=True
    else:pred,scores=e.direct_map(safe(c),p) if direct else e.answer(safe(c),**kw);answer_cache[signature]=(pred,scores)
    j.put('answers',name,c,prediction=pred,scores=scores,correct=pred==c['gold_option'],gold_option=c['gold_option'],oracle=oracle,seconds=time.time()-t,reused=reused,baseline_source=reuse if old else None)
   answer('none',reuse='none')
   answer('gold_crop',kw={'form':'crop','selected':c['gold_rule']},reuse='E2_gold_crop',oracle=True)
   for src,p in sources.items():
    reuse={'direct':'direct','crop':'crop','soft_all':'self_soft_2_all'} if src=='blind' else {'soft_all':'full_soft_2_all'} if src=='equal_external' else {}
    answer(src+'_direct',p=p,direct=True,reuse=reuse.get('direct'))
    answer(src+'_crop',kw={'form':'crop','selected':argmax(p)},reuse=reuse.get('crop'))
    answer(src+'_soft_all',kw={'mode':'gate','p':p,'lam':2,'layers':'all','gate_phase':'all'},reuse=reuse.get('soft_all'))
    answer(src+'_hard_all',kw={'mode':'hard','p':p,'layers':'all','gate_phase':'all'})
   answer('blind_soft_late',kw={'mode':'gate','p':blind,'lam':2,'layers':'last_half','gate_phase':'all'},reuse='self_soft_2_last_half')
   answer('condition_soft_late',kw={'mode':'gate','p':cond,'lam':2,'layers':'last_half','gate_phase':'all'})
   agreement=argmax(cond)==argmax(blind)
   answer('agreement_gate',kw={'mode':'gate','p':cond,'lam':2,'layers':'all','gate_phase':'all'} if agreement else {})
   answer('agreement_crop',kw={'form':'crop','selected':argmax(cond)} if agreement else {})
   if (i+1)%20==0:status('answer_comparison',i+1,len(rows))
  status('complete',len(rows),len(rows))
 except BaseException as exc:
  atomic(a.out/'FAILED.json',{'error':repr(exc),'time':time.time()});traceback.print_exc();raise
if __name__=='__main__':main()
