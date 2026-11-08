import copy,json
import pytest
from .core import *
from .build import action_guides,rule_variants,edited_fact,ancestors_preserved,replace_condition
from .external import compatible,evaluate_external

def test_template_copy_rejected():
 with pytest.raises(ValueError):validate_rule('体温是否≥38℃？',dict(available=True,predicate=dict(op='>=',name='age',source='age >= 65',value=65)))
def test_other_condition_rejected():
 with pytest.raises(ValueError):validate_rule('是否存在咳嗽？',dict(available=True,predicate=dict(op='==',name='肾功能',source='是否肾功能正常？',value=True)))
def test_direct_threshold_and_unit():
 e=direct_rule('患者年龄是否≥2个月？')['predicate'];assert (e['op'],e['value'],e['unit'])==('>=',2,'个月')
def test_unknown_is_not_negative():
 e=dict(op='==',field='咳嗽',value=True)
 assert evaluate(e,{'咳嗽':None}) is None
 assert evaluate(e,{'咳嗽':False}) is False
 assert evaluate(dict(op='not',arg=e),{'咳嗽':None}) is None
 assert evaluate(dict(op='or',args=[e,dict(op='==',field='喘息',value=True)]),{'喘息':True}) is True

def test_root_fact_mapping_keeps_evidence():
 g={'patient':'患者无咳嗽。'};r=dict(predicate=dict(op='==',field='咳嗽',value=True))
 a=validate_facts(g,{'咳嗽':dict(value=False,state='explicit_negative',sentence_ids=[0])},r)
 assert a['facts']['咳嗽'] is False
 assert a['observations']['咳嗽']['evidence'][0]['text']==g['patient']
def test_repeated_fact_not_left_behind():
 pt='年龄为40岁。复述年龄为40岁。';ev=sentences(pt)
 ed=dict(field='年龄',spans=[dict(old='年龄为40岁。',template='年龄为{value}岁。')])
 # replace_spans updates every exact occurrence, including duplicates.
 out,_=edited_fact({'patient':pt},ed,41,None,{'年龄':dict(value=40,evidence=ev)})
 assert '40' not in out and out.count('41')==2
 ed['spans']=[dict(old='复述年龄为40岁。',template='复述年龄为{value}岁。')]
 with pytest.raises(ValueError,match='repeated fact'):edited_fact({'patient':pt},ed,41,None,{'年龄':dict(value=40,evidence=ev)})
def test_changed_ancestor_preserved():
 g=dict(group_id='n',guideline_id=1,ancestors=[dict(gid='G0',condition='年龄≥18',label=1)])
 ed=dict(field='年龄',affected_ancestors=['G0'])
 a=dict(rules={'1:G0':dict(state='complete',value=dict(predicate=dict(op='>=',field='年龄',value=18)))},ancestor_facts={'n@G0':dict(state='complete',value=dict(facts={'年龄':30}))})
 assert ancestors_preserved(g,ed,31,a)[0]['after']==1
 with pytest.raises(ValueError,match='changes path'):ancestors_preserved(g,ed,17,a)
def test_unchanged_ancestors_need_no_extraction():
 g=dict(ancestors=[dict(gid='G0',condition='是否诊断肺炎',label=1)])
 assert ancestors_preserved(g,dict(field='年龄',affected_ancestors=[]),30,{})==[]
def test_action_survives_failed_cf():
 g=rows(ROOT/'pilot/nodes.jsonl')[0];results=list(action_guides(g));assert len(results)==3
 for _,guide,_ in results:
  assert canonical_guide(guide)['nodes']==canonical_guide(g['guideline'])['nodes']
 # Main integration verifies every pilot node gets A despite unavailable extraction records.
 pairs=rows(ROOT/'pilot/built/pairs.jsonl');assert len({p['group_id'] for p in pairs if p['kind']=='A'})==30

def test_c_preserves_case_and_condition_context():
 q='患者年龄是否≥2个月？';e=direct_rule(q)['predicate'];vv=list(rule_variants(q,e,[]))
 assert any(s=='endpoint' and '>' in text and '≥' not in text for s,text,_,_ in vv)
 assert all('患者年龄是否' in text and '个月？' in text for _,text,_,_ in vv)
 assert all(x[3].get('step',1)==1 for x in vv)
 for p in rows(ROOT/'pilot/built/pairs.jsonl'):
  if p['kind']=='C':assert p['patient_unchanged']
def test_numeric_contradiction_uses_released_label():
 ps=[p for p in rows(ROOT/'external/nli4ct/pairs.jsonl') if p['intervention']=='Numerical_contradiction']
 assert len(ps)==276 and sum(p['relation']=='changing' for p in ps)==114

def test_medpic_permutation_is_supported():
 rr=rows(ROOT/'external/medpic/items.jsonl');a=next(x for x in rr if x['id']=='RK-CF-CIP-001-A');b=next(x for x in rr if x['id']=='RK-CF-CIP-001-B')
 ok,_,proof=compatible(a,b);assert ok and proof['option_map']['B']=='A'
 b=copy.deepcopy(b);b['options']['A']='different medication';assert not compatible(a,b)[0]
def test_medpic_groups_not_question_count():
 m=read(ROOT/'external/medpic/manifest.json');assert (m['verified_groups'],m['verified_members'],m['pairs'])==(90,181,92)
 assert any(g['member_count']==3 for g in rows(ROOT/'external/medpic/groups.jsonl'))
@pytest.mark.parametrize('dataset',['nli4ct','medpic'])
def test_external_evaluator_perfect_and_missing(dataset,tmp_path):
 rr=rows(ROOT/'external'/dataset/'items.jsonl');p=tmp_path/'synthetic_predictions.jsonl'
 lines(p,[dict(id=r['id'],prediction=r['label'] if dataset=='nli4ct' else r['answer']) for r in rr])
 result=evaluate_external(dataset,p);assert result['exact_accuracy']==1 and result['missing_or_invalid']==0
 assert all(x['both_correct']==1 for x in result['pair_metrics'].values() if x['pairs'])
 lines(p,[]);result=evaluate_external(dataset,p);assert result['exact_accuracy']==0 and result['missing_or_invalid']==len(rr)

def test_numeric_parenthetical_restatement_removed():
 from .program_edits import numeric_edits
 g=dict(patient='体温为37℃（低于38℃）。',ancestors=[])
 r=direct_rule('体温是否≥38℃？')
 fr=validate_facts(g,{'体温':dict(value=37,state='observed',sentence_ids=[0])},r)
 ed=numeric_edits(g,r,fr,dict(rules={}))
 assert len(ed)==1
 text,_=edited_fact(g,ed[0],39,None,fr['observations'])
 assert '39℃' in text and '低于38' not in text


def test_failed_ancestor_rule_keeps_numeric_edit_for_path_validation():
 from .program_edits import numeric_edits
 g=dict(group_id='n',guideline_id=1,patient='体温为37℃。',ancestors=[dict(gid='G0',condition='体温是否低于40℃？',label=1)])
 r=direct_rule('体温是否≥38℃？')
 fr=validate_facts(g,{'体温':dict(value=37,state='observed',sentence_ids=[0])},r)
 bad=dict(state='extraction_problem',value=dict(predicate=dict(op='<',name='体温',value=40)))
 ed=numeric_edits(g,r,fr,dict(rules={'1:G0':bad}))
 assert len(ed)==1 and ed[0]['affected_ancestors']==['G0']
 with pytest.raises(ValueError):ancestors_preserved(g,ed[0],39,dict(rules={'1:G0':bad},ancestor_facts={}))


def test_malformed_edit_plans_do_not_abort_ancestor_stage():
 from .extract import affected_ancestor_ids
 g=dict(ancestors=[dict(gid='G0',condition='体温低于40℃')])
 for value in (None,[],{'edits':None},{'edits':[None,'bad',{'field':None,'affected_ancestors':None}]}):
  assert affected_ancestor_ids(g,dict(value=value))==set()
 assert affected_ancestor_ids(g,dict(value=dict(edits=[dict(field='体温',affected_ancestors=[{},'unknown'])])))=={'G0'}
