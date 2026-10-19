"""Strong Kleene evaluation, including open/closed numerical intervals."""
import math
def k_all(xs):return 'F' if 'F' in xs else 'U' if 'U' in xs else 'T'
def k_any(xs):return 'T' if 'T' in xs else 'U' if 'U' in xs else 'F'
def valid_schema(r):
 return isinstance(r,dict) and r.get('mode') in ['all','any'] and isinstance(r.get('conditions'),list) and bool(r['conditions']) and isinstance(r.get('exceptions',[]),list) and all(isinstance(c,dict) and isinstance(c.get('field'),str) and c.get('op') in ['lt','le','gt','ge','eq'] and 'value' in c for c in r['conditions']+r.get('exceptions',[]))
def compare(c,facts):
 v=facts.get(c['field']);t=c['value'];op=c['op']
 if v is None:return 'U'
 if isinstance(v,dict):
  lo=-math.inf;hi=math.inf;lc=hc=False
  for o,x in v.items():
   if o in ['ge','gt'] and x>=lo:lo=x;lc=o=='ge'
   if o in ['le','lt'] and x<=hi:hi=x;hc=o=='le'
  if lo>hi or (lo==hi and not(lc and hc)):return 'U'
  if op=='lt':yes=hi<t or (hi==t and not hc);no=lo>=t
  elif op=='le':yes=hi<=t;no=lo>t or(lo==t and not lc)
  elif op=='gt':yes=lo>t or(lo==t and not lc);no=hi<=t
  elif op=='ge':yes=lo>=t;no=hi<t or(hi==t and not hc)
  else:yes=lo==hi==t and lc and hc;no=t<lo or t>hi or(t==lo and not lc)or(t==hi and not hc)
  return 'T' if yes else 'F' if no else 'U'
 try:
  z={'lt':lambda:v<t,'le':lambda:v<=t,'gt':lambda:v>t,'ge':lambda:v>=t,'eq':lambda:v==t}[op]()
 except (TypeError,KeyError):return 'U'
 return 'T' if z else 'F'
def evaluate(rule,facts):
 if not valid_schema(rule):return 'U'
 c=[compare(x,facts) for x in rule['conditions']];e=[compare(x,facts) for x in rule.get('exceptions',[])]
 return k_all([(k_all(c) if rule['mode']=='all' else k_any(c)),{'T':'F','F':'T','U':'U'}[k_any(e)]])
