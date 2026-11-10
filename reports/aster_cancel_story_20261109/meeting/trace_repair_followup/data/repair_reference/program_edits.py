"""Construct numeric templates from existing measurements and redundant comparisons."""
from .core import *
def numeric_edits(g,rule,fr,ann):
 edits=[];seen=set();patient=g['patient']
 for l in leaves(rule['predicate']):
  field=l['field'];value=fr['facts'].get(field)
  if field in seen or l['op'] not in ('<','<=','>','>=') or type(value) not in (int,float):continue
  seen.add(field);evidence=fr['observations'][field]['evidence'];unit=l.get('unit');unit='' if unit is None else unit;unit=unit.strip();spans=[];valid=True
  alias=field.replace(' ','')
  if not alias:continue
  for ev in evidence:
   text=ev['text'];compact=text.replace(' ','')
   if alias not in compact:valid=False;break
   # Match actual variable measurement with optional grammatical words; not quoted rule cutoffs.
   pattern=re.escape(field).replace(r'\ ',r'\s*')+r'(?:为|是|测得为|测得|水平为|约为|约|达到|达|:|：|\s)*\s*('+NUM+r')'
   matches=[m for m in re.finditer(pattern,text) if float(m[1])==value]
   if not matches:valid=False;break
   new=text
   for m in reversed(matches):new=new[:m.start(1)]+'{value}'+new[m.end(1):]
   if new.count('{value}')!=1:valid=False;break
   # Parenthetical restatements of the same threshold must not survive a crossing edit.
   def remove_redundant(match):
    body=match[1]
    others=any(n not in field and n in body for n in ['血压','体温','心率','呼吸','血糖','BMI','糖尿病','SpO','GCS'])
    if unit and unit in body and re.search(r'[<>≥≤]|高于|低于|超过|不超过',body) and not others:return ''
    return match.group()
   new=re.sub(r'[（(]([^（）()]+)[）)]',remove_redundant,new)
   # Delete redundant comparative comments ONLY when they refer to this measurement's unit.
   pieces=re.split(r'([，,；;])',new)
   clean=[]
   for i in range(0,len(pieces),2):
    piece=pieces[i];sep=pieces[i+1] if i+1<len(pieces) else ''
    has_other=any(n not in field and n in piece for n in ['血压','体温','心率','呼吸','血糖','BMI','糖尿病','SpO','GCS'])
    comparison=bool(re.search(r'不符合|符合|满足|未达到|不属于|未超过|不超过|超过|高于|低于|≥|≤|[<>]',piece))
    if '{value}' not in piece and comparison and not has_other and (unit and unit in piece):
     if clean:clean[-1]=clean[-1].rstrip('，,；;')
     if piece.endswith('。'):clean.append('。')
     continue
    clean.append(piece+sep)
   new=''.join(clean)
   # Do not leave qualitative threshold commentary in the changed sentence.
   if re.search(r'高龄|低龄|偏高|偏低|升高|降低|过快|过慢|未达|不符合|不属于',new) or (unit and re.search(r'(?:[<>≥≤]|高于|低于|超过|不超过)\s*'+NUM+r'\s*'+re.escape(unit),new)):valid=False;break
   if new.count('{value}')!=1:valid=False;break
   spans.append(dict(old=text,template=new))
  if not valid or not spans:continue
  # Unlocated repeated measurements or categorical restatements require a richer annotation.
  evidence_text=''.join(x['old'] for x in spans)
  if sum(patient.count(x['old']) for x in spans)<len(evidence):continue
  if field=='年龄':
   remainder=patient
   for sp in spans:remainder=remainder.replace(sp['old'],'')
   if re.search(r'高龄|年迈|幼儿|婴儿|儿童|成人|老年|月龄|\d+岁',remainder):continue
  affected=[];mapping={}
  for a in g['ancestors']:
   if alias in a['condition'].replace(' ','') or (field=='年龄' and re.search(r'成人|儿童|婴儿|老年|岁|月龄',a['condition'])):
    affected.append(a['gid']);record=ann['rules'].get(key(g,a['gid']),{})
    ar=record.get('value',{}).get('predicate') if record.get('state')=='complete' else None
    if ar:
     names=[x['field'] for x in leaves(ar) if x['field'].replace(' ','')==alias]
     if names:mapping[a['gid']]=names[0]
  threshold=str(l.get('number_source',l['value']));step=10**(-len(threshold.partition('.')[2])) if '.' in threshold and not threshold.endswith('.0') else 1
  edits.append(dict(field=field,spans=spans,step=step,unit=unit,minimum=0,integer_only=False,affected_ancestors=affected,ancestor_field_map=mapping,method='source-measurement-template'))
 return edits

def background_from_measurements(g):
 context=condition(g)+'\n'+'\n'.join(a['condition'] for a in g['ancestors'])
 families=[('收缩压',['收缩压','血压','循环','血流动力','低血压','高血压'],'mmHg'),('心率',['心率','心动','心律','脉搏','循环','血流动力'],'次/分'),('呼吸频率',['呼吸','气促','气短','喘','RR','肺'],'次/分'),('体温',['体温','发热','感染','热'],'℃'),('BMI',['BMI','体重','肥胖','超重','营养'],'kg/m²')]
 for field,aliases,unit in families:
  if any(a in context for a in aliases):continue
  pattern=re.escape(field)+r'(?:为|测得为|测得|约为|约|是|\s)*\s*('+NUM+r')\s*'+re.escape(unit)
  ms=list(re.finditer(pattern,g['patient']))
  if len(ms)!=1:continue
  m=ms[0];v=float(m[1]);step=.1 if '.' in m[1] else 1;new=v+step
  # Stay on the same side of all explicit comparisons of that measurement in the case.
  if any((v<t)!=(new<t) or (v<=t)!=(new<=t) for t in [float(x[1]) for x in re.finditer(r'(?:[<>≥≤]|低于|高于|超过|达到)\s*('+NUM+r')\s*'+re.escape(unit),g['patient'])]):continue
  old=m.group();updated=old[:m.start(1)-m.start()]+format(new,'.12g')+old[m.end(1)-m.start():]
  return dict(old=old,new=updated,field=field,before=m[1],after=format(new,'.12g'),reason='Located background measurement; variable and related terms absent from current and ancestor conditions; reported comparison sides preserved',unrelated_to_all_conditions=True,method='program-background')
 return None
