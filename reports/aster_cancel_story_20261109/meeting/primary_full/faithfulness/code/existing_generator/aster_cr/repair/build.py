"""Independent categories, source-position rule edits, changed-variable ancestors."""
import argparse
from collections import defaultdict
from .core import *

def annotations(scope):
 out={}
 for typ in ('rules','facts','edits','ancestor_facts'):
  out[typ]={}
  for folder in [ROOT/'pilot',ROOT/scope]:
   for p in sorted(folder.glob(typ+'_*.jsonl')):
    out[typ].update({r['id']:r for r in rows(p)})
 overrides=read(ROOT/'pilot/review_overrides.json')
 for typ,values in overrides.items():
  if typ in out:out[typ].update(values)
 return out

def action_guides(g):
 leaves_=canonical_guide(g['guideline'])['leaves'];conditions='\n'.join(n['condition'] for n in canonical_guide(g['guideline'])['nodes'])
 protected={k for k,v in leaves_.items() if re.search(r'定义|指的是|是指|defined as|definition|means that|criteria (?:are|include)',v,re.I) or re.search(r'\b'+k+r'\b',conditions)}
 editable=[k for k in leaves_ if k not in protected]
 maps={'presence':{k:'[处置已删除]' for k in editable},'neutral':{k:('记录内容。' if g['language']=='zh' else 'Administrative information.') for k in editable}}
 if len({leaves_[k] for k in editable})>1:maps['swap']={k:leaves_[editable[(i+1)%len(editable)]] for i,k in enumerate(editable)}
 for mode,mapping in maps.items():
  guide=copy.deepcopy(g['guideline'])
  for node in guide['groups']:
   for rule in node['rules']:
    m=re.fullmatch(r'(L\d+)([：:])(.*)',rule['action'],re.S)
    if m and m[1] in mapping:rule['action']=m[1]+m[2]+mapping[m[1]]
  if guide!=g['guideline']:yield mode,guide,sorted(protected)

def replace_condition(g,q):
 guide=copy.deepcopy(g['guideline']);old=condition(g)
 for n in guide['groups']:
  if n['group_id']==g['gid']:
   for r in n['rules']:r['condition']=r['condition'].replace(old,q,1)
 return guide

def rule_variants(q,e,edits):
 for j,leaf in enumerate(leaves(e)):
  value=leaf['value'];source=leaf['source'];op=leaf['op']
  if op in ('<','<=','>','>=') and type(value) in (int,float):
   matches=[m for m in re.finditer(NUM,source) if float(m.group())==value]
   if len(matches)!=1:continue
   m=matches[0];step=None
   if step is None:step=10**(-len(m.group().partition('.')[2])) if '.' in m.group() else 1
   for d in GRID:
    if not d:continue
    new=value+d*step
    if new<0:continue
    changed_source=source[:m.start()]+format(new,'.12g')+source[m.end():]
    changed_q=q.replace(source,changed_source,1)
    expr=copy.deepcopy(e);list(leaves(expr))[j]['value']=new
    yield 'threshold',changed_q,expr,dict(old=source,new=changed_source,field=leaf['field'],delta=d,step=step)
   cs=leaf.get('comparator_source')
   if cs and cs in source:
    newop={'<':'<=','<=':'<','>':'>=','>=':'>'}[op]
    token={'<=':'≤','>=':'≥','<':'<','>':'>'}[newop]
    changed_source=source.replace(cs,token,1);expr=copy.deepcopy(e);list(leaves(expr))[j]['op']=newop
    yield 'endpoint',q.replace(source,changed_source,1),expr,dict(old=source,new=changed_source,field=leaf['field'])
  elif op in ('in','not_in') and len(value)>1:
   for k,v in enumerate(value):
    # Delete one actual list member and its adjacent delimiter in the original condition.
    pattern=re.escape(str(v))+r'\s*(?:、|或|,\s*|\bor\s+)'
    new_source,n=re.subn(pattern,'',source,count=1)
    if not n:new_source,n=re.subn(r'(?:、|或|,\s*|\s+or\s+)'+re.escape(str(v)),'',source,count=1)
    if not n:continue
    expr=copy.deepcopy(e);list(leaves(expr))[j]['value']=value[:k]+value[k+1:]
    yield 'list',q.replace(source,new_source,1),expr,dict(old=source,new=new_source,field=leaf['field'])
 if e['op'] in ('and','or'):
  tokens=['且','并且',' and '] if e['op']=='and' else ['或',' or ']
  found=[t for t in tokens if q.count(t)==1]
  if len(found)==1:
   old=found[0];new=('或' if e['op']=='and' else '且') if old.strip() in ('且','并且','或') else (' or ' if e['op']=='and' else ' and ')
   expr=copy.deepcopy(e);expr['op']='or' if e['op']=='and' else 'and'
   yield 'logic',q.replace(old,new,1),expr,dict(old=old,new=new)

def edit_values(ed,e):
 if ed.get('legal_values'):
  for item in ed['legal_values']:yield item['value'],item.get('replacements'),None
 else:
  step=ed.get('step')
  if type(step) not in (int,float) or step<=0:raise ValueError('invalid numeric step')
  thresholds=sorted({l['value'] for l in leaves(e) if l['field']==ed['field'] and type(l['value']) in (int,float)})
  seen=set()
  for t in thresholds:
   for delta in GRID:
    v=t+delta*step
    if v in seen or v<ed.get('minimum',-math.inf) or v>ed.get('maximum',math.inf):continue
    if ed.get('integer_only') and abs(v-round(v))>1e-8:continue
    seen.add(v);yield v,None,dict(threshold=t,distance=delta,step=step)

def edited_fact(g,ed,value,replacements,observations):
 field=ed['field'];spans=ed['spans'];obs=observations[field];patient=g['patient']
 if not spans:raise ValueError('no located fact edits')
 if replacements is not None:
  if len(replacements)!=len(spans):raise ValueError('wrong replacement count')
 else:
  if any(s.get('template','').count('{value}')!=1 for s in spans):raise ValueError('invalid value template')
  replacements=[s['template'].replace('{value}',format(value,'.12g')) for s in spans]
 locations=[]
 for sp in spans:
  if not sp.get('old') or sp['old'] not in patient:raise ValueError('fact anchor absent')
  locations.extend((m.start(),m.end()) for m in re.finditer(re.escape(sp['old']),patient))
 for ev in obs['evidence']:
  if not any(s<ev['end'] and t>ev['start'] for s,t in locations):raise ValueError('repeated fact sentence not edited')
 # A repeated exact numeric value within a fact's evidence must not be left behind.
 oldval=obs['value']
 if type(oldval) in (int,float):
  for ev in obs['evidence']:
   for m in re.finditer(NUM,ev['text']):
    if float(m.group())==oldval and not any(s<=ev['start']+m.start() and t>=ev['start']+m.end() for s,t in locations):raise ValueError('repeated numeric fact not edited')
 edits=[dict(old=s['old'],new=v) for s,v in zip(spans,replacements)]
 return replace_spans(patient,edits),edits

def ancestors_preserved(g,ed,value,ann):
 affected=set(ed.get('affected_ancestors',[]));field=ed['field']
 for a in g['ancestors']:
  if field in a['condition']:affected.add(a['gid'])
 trace=[]
 valid_ids={a['gid'] for a in g['ancestors']}
 if not affected<=valid_ids:raise ValueError('unknown affected ancestor')
 for a in g['ancestors']:
  if a['gid'] not in affected:continue
  rr=ann['rules'].get(key(g,a['gid']),{});ff=ann['ancestor_facts'].get(g['group_id']+'@'+a['gid'],{})
  if rr.get('state')!='complete' or ff.get('state')!='complete':raise ValueError('affected ancestor extraction unavailable: '+a['gid'])
  e=rr['value']['predicate'];facts=ff['value']['facts'];names={l['field'] for l in leaves(e)}
  target=ed.get('ancestor_field_map',{}).get(a['gid']) or (field if field in names else None)
  if target not in names:raise ValueError('affected ancestor variable unresolved: '+a['gid'])
  before=evaluate(e,facts);after=evaluate(e,dict(facts,**{target:value}))
  if before is None or after is None or int(before)!=a['label'] or int(after)!=a['label']:raise ValueError('affected ancestor changes path: '+a['gid'])
  trace.append(dict(gid=a['gid'],field=target,before=int(before),after=int(after)))
 return trace

def build(scope='pilot'):
 folder=ROOT/scope;groups=rows(ROOT/('pilot/nodes.jsonl' if scope=='pilot' else 'data/nodes.jsonl'));ann=annotations(scope)
 views=[];pairs=[];failures=[];rules=[];review=[]
 for g in groups:
  base={k:g[k] for k in ('group_id','source_id','split','case_id')};base['cohort']='main';base['construction_version']='repair-v2'
  def add(name,kind,patient=None,guide=None,label=None,**extra):
   v=dict(base,view_id=g['group_id']+':'+name,variant=name,kind=kind,original=name=='full' and g.get('original_selected',True),controlled_reference=name=='full',label=g['original_label'] if label is None else int(label),prompt=prompt(guide if guide is not None else g['guideline'],g['definitions'],g['patient'] if patient is None else patient,g['gid']),**extra)
   views.append(v);return v
  def pair(kind,right,subtype,**extra):
   pairs.append(dict(base,pair_id=g['group_id']+':'+kind+':'+str(len(pairs)),left=full['view_id'],right=right['view_id'],left_label=full['label'],right_label=right['label'],kind=kind,subtype=subtype,**extra))
  def fail(kind,reason):failures.append(dict(base,kind=kind,reason=reason))
  full=add('full','original')
  # A does not depend on rule, fact, P, C or F availability.
  for mode,guide,protected in action_guides(g):pair('A',add({'presence':'remove_action','neutral':'neutral_action','swap':'swap_action'}[mode],'A',guide=guide),'action_'+mode,protected_action_ids=protected)
  rr=ann['rules'].get(key(g),{});fr=ann['facts'].get(g['group_id'],{});er=ann['edits'].get(g['group_id'],{}).get('value',{})
  if not isinstance(er,dict):er={}
  er=dict(er,edits=[x for x in er.get('edits',[]) if isinstance(x,dict)] if isinstance(er.get('edits',[]),list) else [])
  rule=rr.get('value',{});obs=fr.get('value',{}).get('observations',{});facts=fr.get('value',{}).get('facts',{})
  if rr.get('state')=='complete' and fr.get('state')=='complete':
   from .program_edits import numeric_edits
   pe=numeric_edits(g,rule,fr['value'],ann)
   er=dict(er,edits=pe+[ed for ed in er.get('edits',[]) if ed.get('legal_values')])
  # P uses directly located unrelated background; no all-fields gate.
  from .program_edits import background_from_measurements
  bg=background_from_measurements(g) or er.get('background')
  try:
   if not isinstance(bg,dict):raise ValueError('no real unrelated background identified')
   if bg.get('unrelated_to_all_conditions') is not True:raise ValueError('background relevance unresolved')
   old=bg['old'];field=bg['field'];context=condition(g)+'\n'+'\n'.join(a['condition'] for a in g['ancestors'])
   if not old or old not in g['patient'] or old==bg['new'] or field not in old or str(bg.get('before','')) not in old or str(bg.get('after','')) not in bg['new']:raise ValueError('background edit absent or unchanged')
   if re.search(r'主要临床关注|clinical focus|main focus|管理。$',old,re.I):raise ValueError('task marker is not clinical background')
   if field in context or field in obs:raise ValueError('background variable belongs to a condition')
   for o in obs.values():
    if bg.get('method')!='program-background' and any(old in ev['text'] or ev['text'] in old for ev in o['evidence']):raise ValueError('background overlaps decisive evidence')
   patient=replace_spans(g['patient'],[dict(old=old,new=bg['new'])])
   pair('P',add('P_background','P',patient=patient),'background',edit_locations=[dict(old=old,new=bg['new'])],changed_field=field,decisive_facts=facts,relevance_reason=bg.get('reason'),conditions_checked=[condition(g)]+[a['condition'] for a in g['ancestors']])
  except (ValueError,KeyError,TypeError) as exc:fail('P',str(exc))
  ready=rr.get('state')=='complete' and fr.get('state')=='complete'
  label=None
  if ready:
   try:label=evaluate(rule['predicate'],facts)
   except (ValueError,KeyError,TypeError) as exc:ready=False;fail('rule',str(exc))
  if not ready or label is None:
   reason=rr.get('error') or fr.get('error') or 'required observation unknown'
   fail('C',reason);fail('F',reason);review.append(dict(base,state='extraction_problem',reason=reason));continue
  if int(label)!=g['original_label']:
   fail('C','rule/fact interpretation disagrees with released label');fail('F','rule/fact interpretation disagrees with released label')
   review.append(dict(base,state='extraction_disagreement',program_label=int(label),released_label=g['original_label']));continue
  rules.append(dict(base,predicate=rule['predicate'],facts=facts,observations=obs,original_condition=condition(g),program_label=int(label),original_label=g['original_label'],method=rr.get('method','model-source-grounded')))
  count=0
  try:
   for i,(sub,q,e,edit) in enumerate(rule_variants(condition(g),rule['predicate'],er.get('edits',[]))):
    newlabel=evaluate(e,facts)
    if newlabel is None or int(newlabel)==int(label):continue
    pair('C',add('C_'+str(i),'C',guide=replace_condition(g,q),label=newlabel,predicate=e,facts=facts),sub,edit_locations=[edit],patient_unchanged=True);count+=1
   if not count:fail('C','no source-position rule edit flips this original case')
  except (ValueError,KeyError,TypeError,IndexError) as exc:fail('C',str(exc))
  count=0
  for j,ed in enumerate(er.get('edits',[])):
   try:
    if ed['field'] not in facts or facts[ed['field']] is None:raise ValueError('edited observation unknown')
    for k,(value,reps,grid) in enumerate(edit_values(ed,rule['predicate'])):
     if value==facts[ed['field']]:continue
     try:
      patient,edits=edited_fact(g,ed,value,reps,obs)
      trace=ancestors_preserved(g,ed,value,ann)
      nf=dict(facts,**{ed['field']:value});newlabel=evaluate(rule['predicate'],nf)
      if newlabel is None:raise ValueError('edited rule remains unknown')
      if patient==g['patient']:continue
      kind='F' if int(newlabel)!=int(label) else 'F0'
      pair(kind,add(f'F_{j}_{k}',kind,patient=patient,label=newlabel,facts=nf,field=ed['field'],value=value,grid=grid),'cross' if kind=='F' else 'same_side',edit_locations=edits,ancestor_checks=trace,changed_field=ed['field']);count+=1
     except (ValueError,KeyError,TypeError,IndexError) as exc:fail('F',str(exc))
   except (ValueError,KeyError,TypeError,IndexError) as exc:fail('F',str(exc))
  if not count:fail('F','no supported fact edits')
 summary={}
 for split in ('fit','dev','test'):
  ps=[p for p in pairs if p['split']==split]
  summary[split]={k:dict(pairs=sum(p['kind']==k for p in ps),nodes=len({p['group_id'] for p in ps if p['kind']==k}),sources=len({p['source_id'] for p in ps if p['kind']==k})) for k in ('A','P','C','F','F0')}
 summary.update(nodes=len(groups),rules=len(rules),views=len(views),pairs=len(pairs),failure_reasons={kind:dict(Counter(x['reason'] for x in failures if x['kind']==kind)) for kind in ('P','C','F','rule')},P_scope='exploratory',C_scope='exploratory',test_scored=False,downstream='paused')
 for name,data in [('views',views),('pairs',pairs),('rules',rules),('unavailable',failures),('extraction_review',review)]:lines(folder/'built'/(name+'.jsonl'),data)
 write(folder/'built/summary.json',summary)
 return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scope',default='pilot');a=p.parse_args();print(json.dumps(build(a.scope),ensure_ascii=False))
