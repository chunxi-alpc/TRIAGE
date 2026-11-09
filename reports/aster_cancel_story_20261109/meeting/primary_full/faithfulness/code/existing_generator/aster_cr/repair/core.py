import copy, json, re, math
from pathlib import Path
from collections import Counter
from ..common import rows,read,write,lines,append
from experiments.aster_closed_form.data import condition,canonical_guide,prompt,ancestors
OLD=Path('/home/gcx/aster_cr_20261108')
ROOT=Path('/home/gcx/aster_cr_repair_20261108')
GRID=[-2,-1,-.5,0,.5,1,2]
def leaves(e):
 if e['op'] in ('and','or'):
  for x in e['args']:yield from leaves(x)
 elif e['op']=='not':yield from leaves(e['arg'])
 else:yield e
def evaluate(e,f):
 op=e['op']
 if op in ('and','or'):
  v=[evaluate(x,f) for x in e['args']]
  if op=='and':return False if False in v else None if None in v else True
  return True if True in v else None if None in v else False
 if op=='not':
  v=evaluate(e['arg'],f);return None if v is None else not v
 a=f.get(e['field']);b=e['value']
 if a is None:return None
 return {'<':lambda:a<b,'<=':lambda:a<=b,'>':lambda:a>b,'>=':lambda:a>=b,'==':lambda:a==b,'!=':lambda:a!=b,'in':lambda:a in b,'not_in':lambda:a not in b}[op]()
def sentences(text):
 return [dict(id=i,start=m.start(),end=m.end(),text=m.group()) for i,m in enumerate(re.finditer(r'[^。！？!?\n]+[。！？!?\n]*',text))]
def key(g,gid=None):return str(g['guideline_id'])+':'+(gid or g['gid'])
NUM=r'-?\d+(?:\.\d+)?'
CMP=r'(>=|<=|≥|≤|>|<|不低于|不超过|至少|超过|低于|高于|小于|大于|不少于|至多)\s*('+NUM+r')'
OPS={'≥':'>=','≤':'<=','不低于':'>=','不少于':'>=','至少':'>=','不超过':'<=','至多':'<=','超过':'>','高于':'>','大于':'>','低于':'<','小于':'<'}
def validate_rule(q,a):
 if not a.get('available'):raise ValueError(a.get('reason','rule unavailable'))
 e=a['predicate'];ls=list(leaves(e))
 if not ls:raise ValueError('empty predicate')
 covered=[]
 for leaf in ls:
  src=leaf['source'];name=leaf['name']
  if not src or src not in q or not name or name not in src:raise ValueError('rule anchor outside current condition')
  leaf['field']=name
  if leaf['op'] in ('<','<=','>','>='):
   ms=list(re.finditer(CMP,src))
   if len(ms)==1:
    m=ms[0];leaf['op']=OPS.get(m[1],m[1]);leaf['value']=float(m[2]);leaf['number_source']=m[2];leaf['comparator_source']=m[1]
   else:
    val=leaf['value']
    if type(val) not in (int,float) or not any(float(m.group())==val for m in re.finditer(NUM,src)):raise ValueError('numeric threshold not in condition')
   if type(leaf['value']) not in (int,float):raise ValueError('invalid threshold')
  elif leaf['op'] in ('in','not_in'):
   if not isinstance(leaf['value'],list) or not leaf['value'] or any(str(v) not in src for v in leaf['value']):raise ValueError('list not grounded')
  elif leaf['op'] not in ('==','!='):raise ValueError('unsupported leaf operator')
  covered.append(src)
 # Source anchors must cover substantive clauses, not just an unrelated age threshold.
 residue=q
 for s in sorted(covered,key=len,reverse=True):residue=residue.replace(s,'')
 residue=re.sub(r'是否|患者|病人|Does|does|Is|is|Are|are|Has|has|the|patient|and|or|且|及|或|并|同时|满足|符合|[\W_]+','',residue)
 if len(residue)>max(12,len(q)*.30):raise ValueError('current condition has uncovered clauses: '+residue[:70])
 return a
def validate_facts(g,a,r):
 ss=sentences(g['patient']);out={};obs={}
 if 'observations' not in a and any(l['field'] in a for l in leaves(r['predicate'])):a={'observations':a}
 for leaf in leaves(r['predicate']):
  f=leaf['field'];o=a.get('observations',{}).get(f,{})
  ids=o.get('sentence_ids',[]);state=o.get('state','unknown');value=o.get('value')
  if state=='unknown':value=None
  elif state not in ('observed','explicit_negative') or not ids:raise ValueError('observation missing evidence')
  if any(type(i)!=int or i<0 or i>=len(ss) for i in ids):raise ValueError('invalid evidence sentence')
  if state=='explicit_negative' and value is None:raise ValueError('negative needs an observed value')
  if value is not None and leaf['op'] in ('<','<=','>','>='):
   if type(value) not in (int,float):raise ValueError('numeric fact is not numeric')
   evidence=' '.join(ss[i]['text'] for i in ids)
   if not any(float(m.group())==value for m in re.finditer(NUM,evidence)):raise ValueError('fact number not in evidence')
  if value is not None and type(leaf['value'])==bool and type(value)!=bool:raise ValueError('boolean observation is not boolean')
  if leaf['op'] in ('in','not_in') and isinstance(value,str) and re.search(r'或|/| or ',value) and value not in leaf['value']:value=None;state='unknown'
  out[f]=value;obs[f]=dict(value=value,state=state,sentence_ids=ids,evidence=[ss[i] for i in ids])
 return dict(facts=out,observations=obs)
def replace_spans(text,edits):
 slots=[]
 for e in edits:
  old=e['old']
  if not old or old not in text:raise ValueError('edit anchor absent')
  for m in re.finditer(re.escape(old),text):slots.append((m.start(),m.end(),e['new']))
 slots=sorted(set(slots))
 if any(a[1]>b[0] for a,b in zip(slots,slots[1:])):raise ValueError('overlapping edits')
 for s,t,new in reversed(slots):text=text[:s]+new+text[t:]
 return text

def prepare():
 old=rows(OLD/'data/nodes.jsonl');paths=rows(OLD/'data/path_cases.jsonl');sources=rows(OLD/'data/sources.jsonl')
 by={(g['guideline_id'],g['gid']):copy.deepcopy(g) for g in old};allnodes=dict(by)
 for c in paths:
  for n in c['gold_path']:
   k=(c['guideline_id'],n['gid'])
   if k in allnodes:continue
   g=dict(c,group_id=f"new-{k[0]}-{k[1]}",gid=n['gid'],original_label=n['label'],language='zh' if re.search('[\u4e00-\u9fff]',c['patient']) else 'en',cohort='main')
   g['ancestors']=ancestors(g);allnodes[k]=g
 nodes=list(allnodes.values())
 for g in nodes:g['original_selected']=g['group_id'] in {x['group_id'] for x in old}
 lines(ROOT/'data/nodes.jsonl',nodes)
 # Keep the original path and baseline node denominator exactly, separately from expanded controlled nodes.
 lines(ROOT/'data/original_nodes.jsonl',old);lines(ROOT/'data/path_cases.jsonl',paths);lines(ROOT/'data/sources.jsonl',sources)
 cats={'numeric':r'[<>≥≤]|\d+.*(?:longer|less|more)', 'interval':r'至|之间|between|范围|within', 'list':r'、|e\.g\.|any of|包括', 'logic':r'且|或| and | or ', 'explicit':r'是否|Is |Are |Has |Does '}
 pilot=[];used=set()
 for cat,pat in cats.items():
  opts=[g for g in nodes if g['split']=='fit' and re.search(pat,condition(g),re.I)]
  opts.sort(key=lambda g:(len(condition(g)),len(g['ancestors']),g['group_id']))
  source_count=Counter()
  for g in opts:
   if g['group_id'] in used or source_count[g['source_id']]>=2:continue
   pilot.append(dict(g,pilot_category=cat));used.add(g['group_id']);source_count[g['source_id']]+=1
   if sum(x['pilot_category']==cat for x in pilot)>=6:break
 lines(ROOT/'pilot/nodes.jsonl',pilot)
 write(ROOT/'manifest.json',dict(version='repair-v2',nodes=dict(Counter(g['split'] for g in nodes)),original_nodes=dict(Counter(g['split'] for g in old)),paths=dict(Counter(g['split'] for g in paths)),pilot=len(pilot),downstream='paused',test_scored=False,external_role='independent validation only; no training or parameter selection',cache_policy='Reuse only byte-identical original prompts and matching scoring protocol; recompute edited inputs and dependent localization/training/selection',gpus=[0,1,6]))
 return read(ROOT/'manifest.json')

def direct_rule(q):
 """Parse explicit source syntax before any model call. Return None for mixed grammar."""
 s=q.strip().rstrip('？?')
 def clean(x):
  x=re.sub(r'^(?:患者|病人|当前|目前|相关症状)?(?:是否|年龄是否)?','',x).strip()
  return x or s
 def leaf(name,op,value,source=None,unit=''):
  return dict(op=op,name=name,field=name,value=value,source=source or q,unit=unit)
 def result(e):return dict(available=True,predicate=e,reason='Parsed directly from condition text',parser='source-syntax-v2')
 # Numeric age and laboratory questions with a single explicit comparator.
 ms=list(re.finditer(CMP,s))
 if len(ms)==1 and not re.search(r'且|或|并|\band\b|\bor\b|[()（）]|×|\^',s,re.I):
  m=ms[0];prefix=s[:m.start()];name=re.sub(r'^(?:患者|病人)','',prefix);name=re.sub(r'是否(?:为|在|达到)?$','',name).strip()
  if name and name in q:
   e=leaf(name,OPS.get(m[1],m[1]),float(m[2]),unit=s[m.end():].strip());e.update(number_source=m[2],comparator_source=m[1]);return result(e)
 # Chinese closed intervals with explicit bounds.
 m=re.fullmatch(r'(.+?)是否在\s*('+NUM+r')\s*[~～至—-]\s*('+NUM+r')\s*(.*?)(?:之间|范围内)',s)
 if m:
  name=m[1];return result(dict(op='and',args=[leaf(name,'>=',float(m[2]),unit=m[4]),leaf(name,'<=',float(m[3]),unit=m[4])],range_source=m[2]+'~'+m[3]))
 # Literal categorical lists; do not infer a list from illustrative parentheses.
 m=re.fullmatch(r'(?:患者)?是否(?:为|接受)([^()（）]+)',s)
 if m and re.search(r'或|、',m[1]) and not re.search(r'[<>≥≤]|\d',m[1]):
  vals=re.split(r'、|或',m[1]);name=m[1]
  return result(leaf(name,'in',vals))
 # Explicit symptom disjunction; nonnumeric atomic propositions retain source words.
 m=re.fullmatch(r'是否(?:出现|存在|伴有)([^()（）\d]+)',s)
 if m and '或' in m[1] and '且' not in m[1] and all(x in {'恶心','呕吐','咳嗽','喘息','休克','昏迷','惊厥','拒食','喂养困难'} for x in m[1].split('或')):return result(dict(op='or',args=[leaf(x,'==',True,source=x) for x in m[1].split('或')]))
 # Single yes/no fact, with no numeric threshold or compound connective.
 if '是否' in s and not re.search(r'\d|[<>≥≤()（）]|或|且|及|、|至|范围|之间|并|如|当|若|根据',s):
  name=re.sub(r'^(?:患者)?是否(?:存在|出现|有|为|已)?','',s);name=re.sub('是否','',name)
  if name in q and name:return result(leaf(name,'==',True))
 return None
