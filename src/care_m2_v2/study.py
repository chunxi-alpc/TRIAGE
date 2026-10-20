from pathlib import Path
from collections import defaultdict
import json, time, os, subprocess, traceback, copy
import numpy as np
import torch
from care_m2_inc.io import read_json,read_jsonl,write_json,write_jsonl,sha256_file
from .data import BASE,digest,oracle
from .engine import Engine,fit_axis,LAYER,SEED

METHODS={
'N1':['baseline','text','direct_bias','R_only','A_only','route','mean','spectral','wrong_sign','wrong_rule','wrong_position','random','repaired_pairs','oracle'],
'N2':['baseline','text','direct_bias','mean','route','random','noinfo','iia_train','iia_test','iia_opposite','iia_random','iia_same'],
'N3':['baseline','neutral_text','text','any_unknown','mean','route','random','oracle'],
'N5':['baseline','text','filter','direct_bias','route','random']}
GRID=[0.,.5,1.]
TUNED={'direct_bias','R_only','A_only','route','mean','spectral'}

def load(data,task,split=None):
    data=Path(data); m=read_json(data/'manifest.json')
    for f,h in m['files'].items(): assert sha256_file(data/f)==h,('data changed',f)
    labels={r['case_id']:r for r in read_jsonl(data/'labels.jsonl')}
    return [{**r,**labels[r['case_id']]} for r in read_jsonl(data/'inputs.jsonl') if r['task']==task and (split is None or r['split']==split)]

def policy(r,c):
    t=r['task']
    if t=='N1':return 'The target rule is '+('relevant' if c['R'] else 'irrelevant')+' to the question and '+('applicable.' if c['A'] else 'not applicable.')+' Use both conditions when choosing the action.'
    if t=='N2':return ('Use the context note.' if c['source']=='C' else 'Use your factual knowledge and disregard the context note.') if r.get('track')=='parametric_knowledge' else 'Use only the '+('Context record' if c['source']=='C' else 'Prior record')+' for the requested unit.'
    if t=='N3':return ('The action is determined. Apply the stated rule and choose the unique action.' if c['DQ'] else 'The action is not determined. Ask an observation that can change the action.')
    return 'Frozen upstream rule states: '+', '.join(s['rule_id']+'='+str(s['state']) for s in r['signals'])+'. T means applicable, F means inapplicable, U means unknown; unavailable is no signal. Use the original rules to decide.'

def append_policy(prompt,s):return prompt.rsplit('\nAnswer:',1)[0]+'\nPolicy: '+s+'\nAnswer:'
def target_positions(engine,r):
    return engine.positions(r) if r['task'] in ('N1','N5') else [[len(engine.tokenizer.encode(r['prompt']))-1]]

def train(engine,rows,out):
    assert not (Path(out)/'axes.pt').exists(), 'fit is immutable; use a new run directory'
    assert all(r['split']=='train' for r in rows); t=rows[0]['task']; captures=[]; axes={}; by=defaultdict(list)
    if t=='N1':
        for r in rows:
            h=engine.capture(r['prompt'],target_positions(engine,r)[0]); by[(r['family_id'],r['controls']['R'],r['controls']['A'])]=h
            captures.append({'case_id':r['case_id'],'positions':target_positions(engine,r)[0]})
        families=sorted({r['family_id'] for r in rows})
        axes['R']=fit_axis([by[(f,1,a)] for f in families for a in (0,1)],[by[(f,0,a)] for f in families for a in (0,1)])
        axes['A']=fit_axis([by[(f,r,1)] for f in families for r in (0,1)],[by[(f,r,0)] for f in families for r in (0,1)])
        pieces=[]
        for key in ['R','A']:
            d=axes[key]['differences']; d=d-d.mean(0); pieces.append(d/d.norm().clamp_min(1e-9)*(0.5**.5))
        _,s,v=torch.linalg.svd(torch.cat(pieces),full_matrices=False)
        axes['mix']={'basis':v[:2].T,'singular_values':s}
        # Independently re-paired centered activations, not sign flips of the same covariance.
        gen=torch.Generator().manual_seed(SEED)
        h=torch.stack(list(by.values())); perm=torch.randperm(len(h),generator=gen)
        axes['repaired']=fit_axis(list(h),list(h[perm]))
    elif t in ('N2','N3'):
        key='source' if t=='N2' else 'DQ'
        controls=['C','P'] if t=='N2' else [1,0]
        for r in rows:
            for control in controls:
                h=engine.capture(append_policy(r['prompt'],policy(r,{key:control})))
                by[control].append(h); captures.append({'case_id':r['case_id'],'policy':control})
        axes['source' if t=='N2' else 'DQ']=fit_axis(by[controls[0]],by[controls[1]])
    elif t=='N5':
        for r in rows:
            positions=engine.positions(r)
            for s,pos in zip([s for s in r['signals'] if s['span']],positions):
                if s['signal'] not in (1,-1):continue
                h=engine.capture(r['prompt'],pos);by[s['signal']].append(h);captures.append({'case_id':r['case_id'],'rule_id':s['rule_id'],'state':s['state']})
        axes['A1']=fit_axis(by[1],by[-1])
    torch.save(axes,Path(out)/'axes.pt')
    meta={'task':t,'split':'train','case_ids':[r['case_id'] for r in rows],'input_hash':digest(rows),'captures':captures,'layer':LAYER,
          'rank':2,'centered_covariance':True,'family_weights':'equal cases per family; N5 each valid predicted T/F rule equal',
          'difference_sign':'positive minus negative','spectral_lambda':.5,'seed':SEED,'axes_sha256':sha256_file(Path(out)/'axes.pt'),
          'axes':{k:{'basis_rank':a['basis'].shape[1],'singular_values':a['singular_values'].tolist()} for k,a in axes.items()}}
    if t=='N1':meta['RA_overlap_singular_values']=torch.linalg.svdvals(axes['R']['basis'].T@axes['A']['basis']).tolist()
    write_json(Path(out)/'train_fit.json',meta)
    return axes

def controls(r):return [{'source':'C'},{'source':'P'}] if r['task']=='N2' else [r['controls']]
def correct(r,c,p):return p==r['gold_by_source'][c['source']] if r['task']=='N2' else p in r.get('acceptable',[r.get('gold')])
def primary(task,records):
    groups=defaultdict(list)
    for p in records:groups[p['case_id'] if task=='N2' else p['family_id']].append(float(p['correct']))
    values=[float(all(x)) if task in ('N1','N2','N4') else float(np.mean(x)) for x in groups.values()]
    return float(np.mean(values)) if values else 0.

def spec(engine,r,c,method,beta,axes,donor=None):
    prompt=r['prompt']; edits=[]; biases=[]; positions=target_positions(engine,r);t=r['task']
    if method=='text':prompt=append_policy(prompt,policy(r,c))
    if method=='neutral_text':prompt=append_policy(prompt,'First enumerate completions of missing facts. Decide only when every completion gives the same action; otherwise ask a decisive attribute.')
    if method=='filter':
        # Remove only original F rules using predicted states, retain U/null and the denominator.
        for s in r['signals']:
            if s['state']=='F':
                lines=prompt.splitlines();prompt='\n'.join(line for line in lines if not line.startswith(s['rule_id']+':'))
    if method=='direct_bias':
        if t=='N1':biases=[(positions[0],beta*(1 if c['R'] and c['A'] else -1))]
        elif t=='N2':biases=[(engine.positions(r)[0],beta*(1 if c['source']=='C' else -1))]
        elif t=='N5':biases=[(pos,beta*s['signal']) for s,pos in zip([s for s in r['signals'] if s['span']],positions) if s['signal'] in (1,-1)]
    neural={'route','R_only','A_only','mean','spectral','wrong_sign','wrong_rule','wrong_position','random','repaired_pairs','noinfo','iia_train','iia_test','iia_opposite','iia_random','iia_same'}
    if method not in neural:return prompt,edits,biases
    if t=='N1':items=[('R',bool(c['R']),positions[0]),('A',bool(c['A']),positions[0])]
    elif t=='N2':items=[('source',c['source']=='C',None)]
    elif t=='N3':items=[('DQ',bool(c['DQ']),None)]
    else:items=[('A1',s['signal']==1,pos) for s,pos in zip([s for s in r['signals'] if s['span']],positions) if s['signal'] in (1,-1)]
    for key,positive,pos in items:
        if method=='R_only' and key!='R':continue
        if method=='A_only' and key!='A':continue
        ax=axes[key]; basis=ax['basis']; target=ax['positive' if positive else 'negative']
        if method=='spectral':basis=axes['mix']['basis']
        if method=='repaired_pairs':basis=axes['repaired']['basis']
        if method=='noinfo':target=(ax['positive']+ax['negative'])/2
        if method.startswith('iia_') and donor is not None:target=donor
        if method=='wrong_rule':pos=engine.positions({**r,'spans':[[r['prompt'].index('record the administrative status'),r['prompt'].index('record the administrative status')+len('record the administrative status')]]})[0]
        if method=='wrong_position':pos=[0]
        edits.append({'kind':'mean' if method=='mean' else 'project','positions':pos,'basis':basis,'target':target,'beta':beta/(2 if t=='N1' and method not in ('R_only','A_only') else 1),
                      'delta':ax['mean']*(1 if positive else -1), 'reverse':method=='wrong_sign','random':method=='random'})
    return prompt,edits,biases

def runone(engine,r,c,method,beta,axes,allrows):
    if method=='oracle':
        prediction=r['gold'];result={'prediction':prediction,'forward_count':0,'cache_hit':True,'diagnostic':'deterministic_oracle_upper_bound'}
    elif method=='any_unknown':
        prediction=next((k for k,v in r['options'].items() if v.startswith('ask ')),next(iter(r['options'])))
        result={'prediction':prediction,'forward_count':0,'cache_hit':True,'diagnostic':'any_unknown_heuristic'}
    else:
        donor=None;donor_meta=None
        if method.startswith('iia_'):
            src=c['source']; pool=[x for x in allrows if x['task']=='N2']; ix=int(r['case_id'].split('_')[1]);dtype='test' if method=='iia_test' else 'train'
            target_i=(12+(ix-12+1)%4) if dtype=='test' and r['split']=='dev' else r['donors'][dtype]
            if method=='iia_same':target_i=ix
            if method=='iia_random':target_i=(ix+5)%12;src='P' if ix%2 else 'C'
            if method=='iia_opposite':src='P' if src=='C' else 'C'
            dr=next(x for x in pool if int(x['case_id'].split('_')[1])==target_i)
            donor=engine.capture(append_policy(dr['prompt'],policy(dr,{'source':src})))
            donor_meta={'case_id':dr['case_id'],'split':dr['split'],'source':src,'answer':dr['source_answers'][src], 'same_fact':dr['case_id']==r['case_id']}
        prompt,edits,biases=spec(engine,r,c,method,beta,axes,donor)
        result=engine.score(prompt,r['options'],edits,biases)
        result['prompt_length_change']=len(engine.tokenizer.encode(prompt))-len(engine.tokenizer.encode(r['prompt']))
        if donor_meta:result.update(donor=donor_meta,donor_copy=r['options'][result['prediction']]==donor_meta['answer'],donor_answer_available=donor_meta['answer'] in r['options'].values())
    result={**result,'case_id':r['case_id'],'family_id':r['family_id'],'pair_id':r['pair_id'],'split':r['split'],'method':method,'control':c,
            'control_origin':r['control_origin'],'beta':beta,'correct':correct(r,c,result['prediction']),
            'expected':r.get('gold_by_source',{}).get(c.get('source')) if r['task']=='N2' else r.get('acceptable',[r['gold']]),
            'option_labels':r['options']}
    if r['task']=='N3':
        result['DQ']='D' if c['DQ'] else 'Q';result['predicted_query']=r['options'][result['prediction']].startswith('ask ')
    if r['task']=='N5':result['signals']=r['signals'];result['reference_states']=r['reference_states']
    if hasattr(engine,'log'):
        engine.log('prediction',digest([result['case_id'],c,method,beta,r['split']]),{**result,'forward_count':0,'request_forward_count':result['forward_count']})
    return result

def select(engine,rows,axes,out,allrows):
    assert not (Path(out)/'dev_selection.json').exists(), 'selection is immutable'
    assert all(r['split']=='dev' for r in rows); task=rows[0]['task']; records=[]; selected={};scores={}
    for method in METHODS[task]:
        candidates=GRID if method in TUNED else [0] if method in ('baseline','text','neutral_text','oracle','any_unknown','filter') else [selected.get('route',0.)]
        by={}
        for beta in candidates:
            preds=[runone(engine,r,c,method,beta,axes,allrows) for r in rows for c in controls(r)]
            records.extend(preds);by[beta]=primary(task,preds)
        selected[method]=max(by,key=lambda b:(by[b],-b));scores[method]=by
    write_jsonl(Path(out)/'dev_predictions.jsonl',records)
    selection={'task':task,'layer':LAYER,'grid':GRID,'selected':selected,'scores':scores,'selection_rule':'maximize registered primary metric; ties smaller beta',
               'axes_sha256':sha256_file(Path(out)/'axes.pt'),'dev_input_sha256':digest(rows),'data_manifest_sha256':sha256_file(Path(out)/'data_manifest.json')}
    selection['sha256']=digest(selection);write_json(Path(out)/'dev_selection.json',selection)
    return selection,records

def evaluate(engine,rows,axes,selection,out,allrows):
    assert all(r['split']=='test' for r in rows)
    assert selection['sha256']==digest({k:v for k,v in selection.items() if k!='sha256'})
    assert selection['axes_sha256']==sha256_file(Path(out)/'axes.pt')
    assert selection['data_manifest_sha256']==sha256_file(Path(out)/'data_manifest.json')
    path=Path(out)/'test_predictions.jsonl'; done={}
    if path.exists():
        for p in read_jsonl(path):done[digest([p['case_id'],p['control'],p['method']])]=p
    preds=[]
    for r in rows:
        for c in controls(r):
            for method,beta in selection['selected'].items():
                key=digest([r['case_id'],c,method])
                if key in done:p=done[key]
                else:
                    p=runone(engine,r,c,method,beta,axes,allrows)
                    with path.open('a') as f:f.write(json.dumps(p,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
                preds.append(p)
    return preds

def summarize(task,preds,out,split='test'):
    by=defaultdict(list)
    for p in preds:by[p['method']].append(p)
    metrics={};families=[]
    for method,rs in by.items():
        families.extend({'method':method,'family_id':f,'correct':sum(x['correct'] for x in rs if x['family_id']==f),'total':sum(x['family_id']==f for x in rs)} for f in sorted({x['family_id'] for x in rs}))
        m={'correct':sum(x['correct'] for x in rs),'total':len(rs),'primary':primary(task,rs),'cache_hits':sum(x['cache_hit'] for x in rs)}
        baseline={(x['case_id'],json.dumps(x['control'],sort_keys=True)):x for x in by.get('baseline',[])}
        paired=[(x,baseline.get((x['case_id'],json.dumps(x['control'],sort_keys=True)))) for x in rs]
        m['corrected']=sum(bool(b and not b['correct'] and x['correct']) for x,b in paired);m['damaged']=sum(bool(b and b['correct'] and not x['correct']) for x,b in paired)
        if task=='N2':
            m.update({s:{'correct':sum(x['correct'] for x in rs if x['control']['source']==s),'total':sum(x['control']['source']==s for x in rs)} for s in ('C','P')})
            m['donor_copy']=sum(x.get('donor_copy',False) for x in rs);m['donor_copy_eligible']=sum(x.get('donor_answer_available',False) for x in rs)
        if task=='N3':
            m['q_wrong_decision']=sum(x['DQ']=='Q' and not x['predicted_query'] for x in rs);m['q_total']=sum(x['DQ']=='Q' for x in rs)
            m['d_unnecessary_query']=sum(x['DQ']=='D' and x['predicted_query'] for x in rs);m['d_total']=sum(x['DQ']=='D' for x in rs)
            m['decisive_query_correct']=sum(x['DQ']=='Q' and x['correct'] for x in rs)
        metrics[method]=m
    for method,rs in by.items():
        pairs=defaultdict(list)
        for x in rs:pairs[x['pair_id']].append(x['correct'])
        realpairs=[v for v in pairs.values() if len(v)==2]
        metrics[method]['pair_both_correct']=sum(all(v) for v in realpairs);metrics[method]['pair_total']=len(realpairs)
        if task=='N1':
            sub=[x for x in rs if not x['control']['R'] and x['control']['A']]
            metrics[method]['R_negative_A_positive_errors']=sum(not x['correct'] for x in sub);metrics[method]['R_negative_A_positive_total']=len(sub)
        if task=='N5':
            metrics[method]['wrong_T_misleading_action']=sum(any(s['state']=='T' and next(z['state'] for z in x['reference_states'] if z['rule_id']==s['rule_id'])!='T' and x['option_labels'][x['prediction']]==s['action'] for s in x['signals']) for x in rs)
            metrics[method]['no_signal_cases']=sum(not any(s['signal'] in (1,-1) and s['action'] for s in x['signals']) for x in rs)
            metrics[method]['A1_correct_stratum']={str(flag):{'correct':sum(x['correct'] for x in rs if all(s['state']==next(z['state'] for z in x['reference_states'] if z['rule_id']==s['rule_id']) for s in x['signals'])==flag),'total':sum(all(s['state']==next(z['state'] for z in x['reference_states'] if z['rule_id']==s['rule_id']) for s in x['signals'])==flag for x in rs)} for flag in (True,False)}
    comparison=None
    reference='direct_bias' if task in ('N1','N5') else 'text'
    if 'route' in by and reference in by:
        fs=sorted({x['family_id'] for x in preds});diff=[]
        for f in fs:diff.append(primary(task,[x for x in by['route'] if x['family_id']==f])-primary(task,[x for x in by[reference] if x['family_id']==f]))
        a=np.array(diff);rng=np.random.default_rng(SEED);bs=np.array([rng.choice(a,len(a),replace=True).mean() for _ in range(10000)])
        flips=np.array([np.mean(a*np.array(s)) for s in __import__('itertools').product((-1,1),repeat=len(a))]) if len(a)<=16 else np.array([np.mean(a*rng.choice([-1,1],len(a))) for _ in range(10000)])
        comparison={'reference':reference,'family_differences':dict(zip(fs,diff)),'difference':float(a.mean()),'ci95':np.quantile(bs,[.025,.975]).tolist(),
                    'two_sided_signflip_p':float(np.mean(np.abs(flips)>=abs(a.mean())-1e-12)),'bootstrap':10000,'seed':SEED,'families':len(fs)}
    summary={'task':task,'split':split,'metrics':metrics,'primary_comparison':comparison,'status':'gpu_evaluated','conclusion':'exploratory; oracle mechanism controls are not a deployable predictor'}
    write_json(Path(out)/'summary.json',summary);write_jsonl(Path(out)/'failure_cases.jsonl',[x for x in preds if not x['correct']])
    import csv
    with (Path(out)/'per_family.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['method','family_id','correct','total']);w.writeheader();w.writerows(families)
    return summary

def followup_decisions(engine,rows,preds,selection,axes,out,allrows):
    records=[];lookup={r['case_id']:r for r in rows}
    for p in preds:
        r=lookup[p['case_id']]
        if r['controls']['DQ']:continue
        attribute=r['options'][p['prediction']].removeprefix('ask ')
        for value in (False,True):
            if attribute not in r['valid_attributes']:
                records.append({'case_id':r['case_id'],'method':p['method'],'answer':value,'correct':False,'failure_code':'initial_question_not_decisive','forward_count':0});continue
            v={**r['values'],attribute:value};possible,attrs=oracle(r['ast'],v)
            if len(possible)!=1:
                records.append({'case_id':r['case_id'],'method':p['method'],'answer':value,'correct':False,'failure_code':'more_questions_required','forward_count':0});continue
            action='execute processing' if next(iter(possible)) else 'withhold processing'
            gold=next(k for k,x in r['options'].items() if x==action)
            follow={**r,'case_id':r['case_id']+f'_answered_{attribute}_{value}','prompt':r['prompt'].replace(attribute+'=unknown',attribute+'='+str(value).lower()),
                    'values':v,'gold':gold,'acceptable':[gold],'controls':{'DQ':1}}
            answer=runone(engine,follow,follow['controls'],p['method'],selection['selected'][p['method']],axes,allrows)
            records.append({**answer,'original_case_id':r['case_id'],'answered_attribute':attribute,'answer':value})
    write_jsonl(Path(out)/'after_answer_predictions.jsonl',records)
    metrics={m:{'correct':sum(r['correct'] for r in records if r['method']==m),'total':sum(r['method']==m for r in records)} for m in selection['selected']}
    write_json(Path(out)/'after_answer_summary.json',metrics)
    return metrics
