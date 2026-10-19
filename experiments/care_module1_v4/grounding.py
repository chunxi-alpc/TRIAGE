"""Format compatibility and exact source alignment, not a semantic theorem prover.
Semantics are assessed by a separate label-blind model pass. Every accepted
verified non-null fact needs a verbatim source span; no proximity regex veto.
"""
import math,re
from logic import compare,evaluate
def canonical_map(x):return {str(k).strip().lower():v for k,v in x.items()} if isinstance(x,dict) else {}
def number(x):
 if type(x) in [int,float] and math.isfinite(x):return x
 if isinstance(x,str) and re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)',x.strip()):return float(x)
 return None
def normalize_value(x,numeric):
 if isinstance(x,dict) and 'value' in x:x=x['value']
 if x is None:return None
 if numeric:
  if isinstance(x,dict):
   if x and set(x)<=set(['lt','le','gt','ge']) and all(number(v) is not None for v in x.values()):return {k:number(v) for k,v in x.items()}
   return None
  return number(x)
 if isinstance(x,str):
  z=x.strip().lower()
  if z in ['positive','negative']:return z
 # Never silently map 1/-1 or raw laboratory values to categories.
 return None
def entries(x):
 if x is None:return []
 if isinstance(x,list):return [z for item in x for z in entries(item)]
 if isinstance(x,dict):
  if 'sentence_ids' in x:return [dict(x,sentence_id=i) for i in x['sentence_ids']]
  return [x]
 if isinstance(x,str) and not x.strip().isdigit():return [{'quote':x}]
 return [{'sentence_id':x}]
def norm(s):return re.sub(r'\s+',' ',s).strip()
def locate(x,sentences,require_quote=False):
 ids=[];audit=[];spans=[]
 for e in entries(x):
  rawid=e.get('sentence_id',e.get('id',e.get('sentence')));n=number(rawid);idx=int(n) if n is not None and n==int(n) and 1<=n<=len(sentences) else None
  quote=e.get('quote',e.get('text',e.get('evidence_text')));q=norm(quote) if isinstance(quote,str) else ''
  matches=[i+1 for i,s in enumerate(sentences) if q and q in norm(s)] if len(q)>=8 else []
  aligned=idx if idx in matches else matches[0] if len(matches)==1 else None
  chosen=aligned if q else (None if require_quote else idx)
  if require_quote and not matches:chosen=None
  if chosen is not None:
   ids.append(chosen)
   if aligned:spans.append({'sentence_id':chosen,'normalized_start':norm(sentences[chosen-1]).index(q),'normalized_end':norm(sentences[chosen-1]).index(q)+len(q)})
  audit.append({'provided_id':rawid,'selected_id':chosen,'quote_found':bool(matches),'relocated':aligned is not None and aligned!=idx,'rejected':chosen is None})
 return sorted(set(ids)),audit,spans
def normalize(obj,sentences,fs,numeric,verified=False):
 obj=obj if isinstance(obj,dict) else {};fv=canonical_map(obj.get('facts'));ev=canonical_map(obj.get('evidence'));statuses=canonical_map(obj.get('status'));out={};oe={};os={};audit={};spans={}
 for f in fs:
  value=normalize_value(fv.get(f),f in numeric);status=statuses.get(f,('known' if value is not None else 'missing'))
  ids,au,sp=locate(ev.get(f),sentences,require_quote=verified)
  if verified:
   if status=='conflict':value=None;ids=ids if len(ids)>=2 else [];status='conflict' if ids else 'missing'
   elif status=='missing':value=None;ids=[]
   elif value is None or not ids:value=None;ids=[];status='missing'
  if value is None and status!='conflict':ids=[]
  out[f]=value;oe[f]=ids;os[f]=status;audit[f]=au;spans[f]=sp
 return {'facts':out,'evidence':oe,'status':os,'alignment':audit,'spans':spans}
def proof(rule,record):
 facts=record['facts'];ev=record['evidence'];label=evaluate(rule,facts)
 if label=='U':return sorted({i for f,s in record['status'].items() if s=='conflict' for i in ev[f]})
 cond=rule['conditions'];truth=[compare(c,facts) for c in cond]
 if (rule['mode']=='all' and label=='T') or(rule['mode']=='any' and label=='F'):return sorted({i for c in cond for i in ev.get(c['field'],[])})
 candidates=[ev.get(c['field'],[]) for c,z in zip(cond,truth) if z==label]
 return min((x for x in candidates if x),key=lambda x:(len(x),x),default=[])
