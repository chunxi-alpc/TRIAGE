from pathlib import Path
from itertools import product, permutations
from collections import defaultdict
import hashlib, json, subprocess
from care_m2_inc.generate import _templates
from care_m2_inc.oracle import atoms, evaluate, enumerate_completions
from care_m2_inc.io import write_json, write_jsonl, read_jsonl, sha256_file

BASE=Path('experiments/care_module2_incremental_20261020')
SOURCE='8fb87d15576b88500f44a88f04799267d9f2b5ee'
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def split(i): return 'train' if i<12 else 'dev' if i<16 else 'test'
def rename(ast,m):
    if isinstance(ast,str): return m[ast]
    return {**ast,'args':[rename(a,m) for a in ast['args']]}
def expr(ast):
    if isinstance(ast,str): return ast
    op=ast['op']; a=[expr(x) for x in ast['args']]
    if op=='NOT': return '(NOT '+a[0]+')'
    if op=='ATLEAST': return f"(at least {ast['threshold']} true among "+', '.join(a)+')'
    if op=='XOR': return '(an odd number true among '+', '.join(a)+')'
    if op=='EQUIV': return '(all equal among '+', '.join(a)+')'
    return '('+f' {op} '.join(a)+')'
def completions(ast,v):
    names=[n for n in atoms(ast) if v[n] is None]
    return [{**v,**dict(zip(names,b))} for b in product((False,True),repeat=len(names))]
def truth(ast,v):
    if isinstance(ast,str): return v[ast]
    a=[truth(x,v) for x in ast['args']]; op=ast['op']
    return {'AND':lambda:all(a),'OR':lambda:any(a),'NOT':lambda:not a[0],
            'XOR':lambda:sum(a)%2==1,'EQUIV':lambda:all(x==a[0] for x in a),
            'ATLEAST':lambda:sum(a)>=ast['threshold']}[op]()
def oracle(ast,v):
    cs=completions(ast,v); outcomes={truth(ast,c) for c in cs}
    attrs=[]
    for n in atoms(ast):
        if v[n] is None and any(truth(ast,c)!=truth(ast,{**c,n:not c[n]}) for c in cs): attrs.append(n)
    return outcomes,attrs

def signature(ast):
    names=atoms(ast)
    return str(len(names))+':'+min(''.join(str(int(truth(ast,dict(zip(order,vs))))) for vs in product((False,True),repeat=len(names))) for order in permutations(names))
def family(i):
    f=_templates('logic_',24)[i]; ast=f['ast']; ast=rename(ast,{n:chr(97+j) for j,n in enumerate(atoms(ast))})
    return ast

def choices(text,values,rotation=0):
    ordered=values[rotation%len(values):]+values[:rotation%len(values)]
    opts={chr(65+i):v for i,v in enumerate(ordered)}
    return text+'\nOptions:\n'+'\n'.join(k+'. '+v for k,v in opts.items())+'\nReturn exactly one option letter.\nAnswer:',opts

def facttext(v): return '; '.join(n+'='+('unknown' if x is None else 'true' if x else 'false') for n,x in v.items())
def span(text,s):
    a=text.index(s); assert a<text.index('\nOptions:'); return [a,a+len(s)]
def rowbase(task,i,j,ast=None):
    f=f'logic_{i:02d}' if ast else f'fact_{i:02d}'
    return dict(task=task,case_id=f'{task}_{i:02d}_{j}',family_id=f,root_case_id=f, pair_id=f,
                fact_group_id=f,underlying_mechanism_id=digest(signature(ast)) if ast else f,
                split=split(i),control_origin='oracle_diagnostic' if task in ('N1','N3','N4') else 'external_source_policy')

def n1():
    out=[]
    for i in range(24):
        ast=family(i); names=atoms(ast)
        full=[dict(zip(names,v)) for v in product((False,True),repeat=len(names))]
        # The same one-attribute contrast is used under both queries.
        pos,neg=next((v,{**v,n:False}) for v in full for n in names if v[n] and truth(ast,v) and not truth(ast,{**v,n:False})) if any(truth(ast,v) and v[n] and not truth(ast,{**v,n:False}) for v in full for n in names) else next((v,{**v,n:not v[n]}) for v in full for n in names if truth(ast,v) and not truth(ast,{**v,n:not v[n]}))
        for r,a in product((False,True),repeat=2):
            v=pos if a else neg
            rule='Rule: If '+expr(ast)+' then perform target processing; otherwise perform alternative processing.'
            admin='Administrative rule: record the administrative status.'
            body='Synthetic routing task. Boolean facts are complete; NOT reverses truth.\n'+('Question: Which processing action applies?' if r else 'Question: Which administrative action applies?')+'\nFacts: '+facttext(v)+'\n'+ ('\n'.join([rule,admin][::(-1 if i%2 else 1)]))
            text,opts=choices(body,['perform target processing','perform alternative processing','record the administrative status'],i%3)
            action='record the administrative status' if not r else 'perform target processing' if a else 'perform alternative processing'
            out.append({**rowbase('N1',i,f'{int(r)}{int(a)}',ast),'prompt':text,'options':opts,'spans':[span(text,'perform target processing')],
                        'controls':{'R':int(r),'A':int(a)},'gold':next(k for k,x in opts.items() if x==action),'ast':ast,'values':v})
    return out

def n2():
    routes=[('Cedar','Maple'),('Quartz','Willow'),('Amber','Slate'),('Birch','Coral'),('Dawn','Frost'),('Elm','Grove'),('Harbor','Ivory'),('Jade','Linen'),('Mica','Navy'),('Olive','Pine'),('Rose','Stone'),('Teal','Umber'),('Aster','Basil'),('Cobalt','Denim'),('Flint','Glacier'),('Hazel','Indigo'),('Kelp','Larch'),('Moss','Nimbus'),('Opal','Pearl'),('Reed','Sage'),('Topaz','Violet'),('Wheat','Yarrow'),('Zinc','Copper'),('Silver','Gold')]
    out=[]
    for i,(c,p) in enumerate(routes):
        train_donor=(i+1)%12; test_donor=16+(i+1)%8
        vals=list(dict.fromkeys([c,p,*routes[train_donor],*routes[test_donor]]))
        body=f'Synthetic source task. Both records are explicit inputs, not model parameter knowledge.\nPrior record: unit {i+1} uses route {p}.\nContext record: unit {i+1} uses route {c}.\nQuestion: Which route does unit {i+1} use?'
        text,opts=choices(body,vals,i%len(vals))
        out.append({**rowbase('N2',i,'receiver'),'prompt':text,'options':opts,'spans':[span(text,f'Context record: unit {i+1} uses route {c}.')],
                    'controls':{},'gold_by_source':{s:next(k for k,v in opts.items() if v==a) for s,a in [('C',c),('P',p)]},
                    'source_answers':{'C':c,'P':p},'donors':{'train':train_donor,'test':test_donor},'track':'synthetic_prior','p_verified':False})
    return out

def n3():
    out=[]
    for i in range(24):
        ast=family(i); names=atoms(ast)
        full=[dict(zip(names,v)) for v in product((False,True),repeat=len(names))]
        # Full T/F, one decisive unknown, and a determined partial assignment when it exists.
        vs=[next(v for v in full if truth(ast,v)),next(v for v in full if not truth(ast,v))]
        partial=[{**v,n:None} for v in full for n in names]
        vs.append(next(v for v in partial if len(oracle(ast,v)[0])==2))
        determined=next((v for v in partial if len(oracle(ast,v)[0])==1),None)
        vs.append(determined or {**vs[0]})
        for j,v in enumerate(vs):
            v={**v,'irrelevant_note':None}; possible,attrs=oracle(ast,v)
            body='Synthetic decision task. Enumerate legal true/false completions of unknown observations. Execute a unique action; otherwise ask an attribute that can change the action.\nRule: If '+expr(ast)+' then execute processing; otherwise withhold processing.\nFacts: '+facttext(v)
            options=['execute processing','withhold processing']+['ask '+n for n,x in v.items() if x is None]
            text,opts=choices(body,options,(i+j)%len(options))
            valid=['execute processing' if next(iter(possible)) else 'withhold processing'] if len(possible)==1 else ['ask '+n for n in attrs]
            out.append({**rowbase('N3',i,j,ast),'prompt':text,'options':opts,'spans':[span(text,'execute processing')],
                        'controls':{'DQ':int(len(possible)==1)},'gold':next(k for k,x in opts.items() if x in valid),
                        'acceptable':[k for k,x in opts.items() if x in valid],'valid_attributes':attrs,'ast':ast,'values':v,'possible':sorted(possible)})
    return out

def n4():
    out=[]
    for i in range(24):
        ast=family(i); full=[dict(zip(atoms(ast),v)) for v in product((False,True),repeat=len(atoms(ast)))]
        for source,d,a,order in product(('C','P'),(0,1),(0,1),(0,1)):
            v=next(v for v in full if truth(ast,v)==bool(a)); v={**v,'review':False if d else None}
            rules=['If review=true, defer all processing. This rule takes precedence.','If review=false and '+expr(ast)+', use the selected source route; otherwise use the fallback route.']
            body='Synthetic joint task. Enumerate missing observations; ask when actions differ.\nPrior record: route Maple.\nContext record: route Cedar.\n'+'\n'.join(rules[::(-1 if order else 1)])+'\nFacts: '+facttext(v)
            text,opts=choices(body,['route Cedar','route Maple','fallback route','defer all processing','ask review'],i%5)
            result='ask review' if not d else ('route Cedar' if source=='C' else 'route Maple') if a else 'fallback route'
            out.append({**rowbase('N4',i,f'{source}{d}{a}{order}',ast),'prompt':text,'options':opts,'spans':[span(text,'Context record: route Cedar.')],
                        'controls':{'source':source,'DQ':d,'A':a,'order':order},'gold':next(k for k,x in opts.items() if x==result),'ast':ast,'values':v})
    return out

def gitread(path): return subprocess.check_output(['git','show',SOURCE+':'+path],text=True)
def n5():
    prefix='experiments/care_module1_semantic_v8/'
    paths=['data/new72_cases.jsonl','data/new72_gold.jsonl','results/new72_A1/predictions.jsonl']
    sets=[{r['case_id']:r for r in map(json.loads,gitread(prefix+p).splitlines())} for p in paths]
    cases,golds,preds=sets; out=[]; hashes=[]
    families=sorted({c['family_id'] for c in cases.values()})
    for cid,c in cases.items():
        p=preds[cid]; g=golds[cid]; signals=[]
        assert p['arm']=='A1'
        # Exact original input exists in the recorded A1 request, not reconstructed from gold.
        request=p['attempts'][0]['prompt']; assert c['patient'] in request
        for rule in c['rules']:
            assert rule['text'] in request
            matches=[r for r in p.get('rules',[]) if r['rule_id']==rule['rule_id']]
            state=matches[0]['state'] if p['status']=='ok' and len(matches)==1 else None
            assert state in ('T','F','U',None)
            start=c['prompt'].index(rule['text']); action=rule['action']; off=rule['text'].index(action) if action else 0
            signals.append({'rule_id':rule['rule_id'],'state':state,'signal':{'T':1,'F':-1,'U':0,None:None}[state],
                            'action':action,'span':[start+off,start+off+len(action)] if action else None,
                            'rule_sha256':hashlib.sha256(rule['text'].encode()).hexdigest()})
        ix=families.index(c['family_id']); partition='train' if ix<6 else 'dev' if ix<8 else 'test'
        out.append(dict(task='N5',case_id=cid,family_id=c['family_id'],root_case_id=g['pair_id'] or c['family_id'],pair_id=g['pair_id'] or c['family_id'],fact_group_id=c['family_id'],
                        underlying_mechanism_id='historical_A1_transfer_not_independent_mechanisms',split=partition,prompt=c['prompt'],options=c['option_map'],signals=signals,
                        controls={},control_origin='frozen_A1',gold=g['gold_answer'],reference_states=g['reference_structure']['rules'],
                        eligibility=any(s['signal'] in (1,-1) and s['action'] for s in signals),
                        patient_sha256=hashlib.sha256(c['patient'].encode()).hexdigest(),request_sha256=hashlib.sha256(request.encode()).hexdigest(),
                        source_commit=SOURCE,a1_method='A1',spans=[s['span'] for s in signals if s['span']]))
    for p in paths:
        text=gitread(prefix+p)
        hashes.append({'path':prefix+p,'commit':SOURCE,'blob':subprocess.check_output(['git','rev-parse',SOURCE+':'+prefix+p],text=True).strip(),'sha256':hashlib.sha256(text.encode()).hexdigest()})
    return out,hashes

def verify(rows):
    checks={}; sigs=[signature(family(i)) for i in range(24)]
    checks['24_permutation_distinct_truth_tables']=len(set(sigs))==24
    checks['all_label_semantics_independent']=all(evaluate(r['ast'],v)==truth(r['ast'],v) for r in rows if 'ast' in r for v in completions(r['ast'],r['values']))
    checks['options_shown']=all(all(k+'. '+v in r['prompt'] for k,v in r['options'].items()) for r in rows)
    checks['spans_outside_candidates']=all(all(s[1]<=r['prompt'].index('Options:') for s in r.get('spans',[])) for r in rows)
    checks['n2_prior_available']=all(r['source_answers']['P'] in r['prompt'].split('Context record:')[0] for r in rows if r['task']=='N2')
    checks['n3_gold']=all((set(r['acceptable'])=={k for k,v in r['options'].items() if v in (['execute processing' if next(iter(oracle(r['ast'],r['values'])[0])) else 'withhold processing'] if r['controls']['DQ'] else ['ask '+a for a in oracle(r['ast'],r['values'])[1]])}) for r in rows if r['task']=='N3')
    checks['n4_legal_completions']=True
    for r in rows:
        if r['task']!='N4': continue
        c=r['controls']; actions=set()
        for review in ([False,True] if r['values']['review'] is None else [r['values']['review']]):
            actions.add('defer all processing' if review else ('route Cedar' if c['source']=='C' else 'route Maple') if truth(r['ast'],r['values']) else 'fallback route')
        expected=next(iter(actions)) if len(actions)==1 else 'ask review'
        checks['n4_legal_completions'] &= r['options'][r['gold']]==expected
    checks['group_isolation']=True
    for task in ('N1','N2','N3','N4','N5'):
        for field in ('family_id','root_case_id','pair_id','fact_group_id'):
            seen=defaultdict(set)
            for r in rows:
                if r['task']==task: seen[r[field]].add(r['split'])
            checks['group_isolation'] &= all(len(s)==1 for s in seen.values())
    checks['no_metadata_in_prompt']=all(not any(x in r['prompt'] for x in ['gold_', 'control_target','family_id','root_case_id']) for r in rows)
    checks['n5_all72_all216_signals']=sum(r['task']=='N5' for r in rows)==72 and sum(len(r.get('signals',[])) for r in rows)==216
    return {'valid':all(checks.values()),'checks':checks}

def prepare(out):
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    a1,provenance=n5(); rows=n1()+n2()+n3()+n4()+a1
    report=verify(rows)
    inputs=[]; labels=[]; splits=[]
    hidden={'gold','acceptable','gold_by_source','reference_states','valid_attributes','possible','ast','values','source_answers'}
    for r in rows:
        inputs.append({k:v for k,v in r.items() if k not in hidden})
        labels.append({'case_id':r['case_id'],**{k:v for k,v in r.items() if k in hidden}})
        splits.append({k:r[k] for k in ('case_id','task','family_id','root_case_id','pair_id','fact_group_id','underlying_mechanism_id','split')})
    for name,obj in [('inputs',inputs),('labels',labels),('splits',splits)]: write_jsonl(out/(name+'.jsonl'),obj)
    write_json(out/'cpu_report.json',report)
    write_json(out/'manifest.json',{'revision':'v2_explicit_repair_of_invalid_v1','source_A1':provenance,'files':{p.name:sha256_file(p) for p in out.glob('*.jsonl')},
       'counts':{t:{s:sum(r['task']==t and r['split']==s for r in rows) for s in ('train','dev','test')} for t in ('N1','N2','N3','N4','N5')},
       'limitations':['N1 old invalid outputs already seen; repaired evaluation is exploratory','N5 historical family split is not independent logic-family confirmation'],
       'status':'cpu_verified' if report['valid'] else 'blocked'})
    assert report['valid'],report
    return report
if __name__=='__main__':
    import sys
    print(json.dumps(prepare(sys.argv[1]),indent=2))
