"""Source-based inspection of the 30 pilot nodes, before expansion/scoring."""
from .core import *
from .build import annotations,build

def review():
 gs={g['group_id']:g for g in rows(ROOT/'pilot/nodes.jsonl')};a=annotations('pilot');over={'rules':{},'facts':{}};notes={}
 def set_rule(ident,name,op,value,observed,needle,unit=''):
  g=gs[ident];q=condition(g);e=dict(op=op,field=name,name=name,source=q,value=value,unit=unit)
  r=dict(id=key(g),condition=q,state='complete',method='pilot-source-review',value=dict(available=True,predicate=e))
  ids=[s['id'] for s in sentences(g['patient']) if needle in s['text']]
  assert ids,(ident,needle)
  f=validate_facts(g,dict(observations={name:dict(value=observed,state='explicit_negative' if observed is False else 'observed',sentence_ids=ids)}),r['value'])
  over['rules'][key(g)]=r;over['facts'][ident]=dict(id=ident,state='complete',method='pilot-source-review',value=f)
  notes[ident]='Rule and observation corrected against the quoted current condition and original patient sentence.'
 set_rule('new-869-G7','持续','>=',6,8,'相关症状已持续约8个月','个月')
 set_rule('new-985-G46','适合转运至具备保肢能力的医院','==',True,True,'适合进行转运')
 set_rule('new-1250-G3','肝包膜不光滑、粗糙','==',True,False,'肝包膜回声表现为光滑')
 set_rule('new-130-G4','包括发现的罕见病及其临床特征','==',True,True,'典型临床特征')
 set_rule('new-110-G67','适合含铂双药化疗','==',True,True,'肾功能与听力等评估')
 set_rule('new-78-G18','舌后坠、颈短或肥胖导致上气道阻塞','==',True,True,'舌后坠、颈短或肥胖因素出现上气道阻塞')
 set_rule('new-1080-G69','淋巴结或内脏受累','==',True,True,'伴有淋巴结或内脏受累')
 set_rule('new-1126-G13','心包积液','in',['中量','大量'],'大量','存在大量心包积液')
 set_rule('new-1126-G45','破溃或感染','==',True,True,'拟治疗区域皮肤存在破溃或感染')
 set_rule('new-1080-G51','广泛','==',True,True,'皮损分布范围广泛')
 notes.update({
 'new-1080-G65':'The patient supplies a stage interval, not an exact ordinal stage; keep unsupported interval extraction, do not infer a point stage.',
 'new-2527-G20':'Explicit denial of annual follow-up supplies a bound, not an exact count. Keep unknown numeric count.',
 'new-1080-G49':'The patient says stage IB OR IIA; no unique category. Do not assign one stage or two simultaneously.',
 'new-869-G10':'The case states both marked and mild functional impairment. Keep this extraction unresolved; do not call it a confirmed label error.',
 'new-190-G18':'Measured ACT=18 contradicts the ancestral 16–19 exclusion in the released path. Current rule is valid; reject F edits that rely on that inconsistent ancestor.',
 'new-1977-G15':'Explicit wheeze is supported, while an ancestor broadly denies respiratory signs. Do not generate F without resolving that dependency.',
 'new-1284-G6':'Case explicitly says nausea AND vomiting, supporting both atoms; switching OR to AND does not flip this case.',
 'new-2423-G57':'Patient supports the whole disjunction only. Keep the disjunction as one directly observed proposition; do not infer both alternatives.',
 'new-1838-G6':'Explicit no loose stool is false, not unknown; formed-stool repetition must also change before any F edit.'})
 over['facts']['new-869-G10']=dict(id='new-869-G10',state='extraction_problem',error='contradictory descriptions of functional severity')
 write(ROOT/'pilot/review_overrides.json',over)
 a=annotations('pilot');result=[]
 for ident,g in gs.items():
  r=a['rules'].get(key(g),{});f=a['facts'].get(ident,{})
  note=notes.get(ident,'Current condition, numeric units/endpoints or explicit fact, and original evidence sentences inspected; no correction needed.')
  result.append(dict(group_id=ident,condition=condition(g),patient=g['patient'],rule=r.get('value'),facts=f.get('value'),rule_state=r.get('state'),fact_state=f.get('state'),review_note=note,reviewer='Codex source-text inspection; not physician adjudication'))
 lines(ROOT/'pilot/review.jsonl',result)
 summary=build();write(ROOT/'pilot/review_summary.json',dict(reviewed_nodes=30,corrected_rules=len(over['rules']),ambiguous_records=['new-1080-G65','new-2527-G20','new-1080-G49','new-869-G10'],counts=summary['fit'],test_scores_used=False))
 return summary
if __name__=='__main__':print(json.dumps(review(),ensure_ascii=False))
