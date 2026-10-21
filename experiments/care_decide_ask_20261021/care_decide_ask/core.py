"""Exhaustive possible-world semantics for small executable rule policies.

This prototype supports Boolean predicates with and/or/not, fixed rule
priority, optional constraints, and explicit defaults. Clinical guideline text
must first be converted into a separately validated policy AST; frozen A1 T/F/U
alone cannot be treated as a rule interpreter.
"""
from __future__ import annotations
from itertools import product
from collections import defaultdict

VALID={'T','F','U'}


def _expression(expr, world):
    if not isinstance(expr,dict) or len(expr)!=1:
        raise ValueError('expression must contain exactly one operator')
    op,arg=next(iter(expr.items()))
    if op=='atom':
        if not isinstance(arg,str) or arg not in world:raise ValueError('undefined atom')
        return world[arg]
    if op=='not':return not _expression(arg,world)
    if op in ('and','or'):
        if not isinstance(arg,list) or not arg:raise ValueError('nonempty operands required')
        vals=[_expression(x,world) for x in arg]
        return all(vals) if op=='and' else any(vals)
    raise ValueError(f'unknown operator: {op}')


def validate_policy(policy):
    if not isinstance(policy,dict):raise ValueError('policy must be dict')
    atoms=policy.get('atoms')
    rules=policy.get('rules')
    if not isinstance(atoms,list) or not atoms or len(atoms)>18:
        raise ValueError('1..18 Boolean atoms required (enumeration bound)')
    if len(set(atoms))!=len(atoms) or any(not isinstance(v,str) or not v for v in atoms):
        raise ValueError('duplicate/bad atom')
    if not isinstance(rules,list) or not rules:raise ValueError('nonempty rules required')
    ids=[];priorities=[]
    probe={x:False for x in atoms}
    for r in rules:
        if not isinstance(r,dict):raise ValueError('bad rule')
        rid=r.get('rule_id');act=r.get('action');priority=r.get('priority')
        if not isinstance(rid,str) or not rid or not isinstance(act,str) or not act:
            raise ValueError('rule_id and action required')
        if not isinstance(priority,int) or isinstance(priority,bool) or priority<0:
            raise ValueError('nonnegative integer priority required')
        _expression(r.get('when'),probe)
        ids.append(rid);priorities.append(priority)
    if len(set(ids))!=len(ids) or len(set(priorities))!=len(priorities):
        raise ValueError('rule ids and priority order must be unique')
    if not isinstance(policy.get('default_action'),str) or not policy['default_action']:
        raise ValueError('explicit default_action required')
    constraints=policy.get('constraints',[])
    if not isinstance(constraints,list):raise ValueError('constraints must be list')
    for c in constraints:_expression(c,probe)
    return policy


def possible_worlds(policy,observations):
    validate_policy(policy)
    if not isinstance(observations,dict) or set(observations)-set(policy['atoms']):
        raise ValueError('invalid observations')
    for v in observations.values():
        if v not in ('T','F','U',None):raise ValueError('observations require T/F/U/null')
    missing=[a for a in policy['atoms'] if observations.get(a) in ('U',None)]
    known={a:observations[a]=='T' for a in policy['atoms'] if a not in missing}
    for bits in product((False,True),repeat=len(missing)):
        world=dict(known,**dict(zip(missing,bits)))
        if all(_expression(c,world) for c in policy.get('constraints',[])):
            yield world


def action_in_world(policy,world):
    matches=sorted((r for r in policy['rules'] if _expression(r['when'],world)),
                   key=lambda r:r['priority'])
    return matches[0]['action'] if matches else policy['default_action']


def derive_truth_states(policy,observations):
    worlds=list(possible_worlds(policy,observations))
    if not worlds:raise ValueError('inconsistent evidence/constraints: no admissible completion')
    out={}
    for rule in policy['rules']:
        values={_expression(rule['when'],w) for w in worlds}
        out[rule['rule_id']]='U' if len(values)>1 else 'T' if True in values else 'F'
    return out


def _allows(label_set, truth_bool):
    if not label_set:return False
    # A possible U label must include both T and F in world propagation.
    return 'U' in label_set or ('T' if truth_bool else 'F') in label_set


def feasible_worlds(policy, observations, rule_label_sets):
    rule_ids={r['rule_id'] for r in policy['rules']}
    if not isinstance(rule_label_sets,dict) or set(rule_label_sets)!=rule_ids:
        raise ValueError('rule state sets must match all rule IDs')
    for choices in rule_label_sets.values():
        if not isinstance(choices,(list,tuple,set,frozenset)) or any(x not in VALID for x in choices):
            raise ValueError('state choices must contain T/F/U labels')
    worlds=[]
    for w in possible_worlds(policy,observations):
        if all(_allows(rule_label_sets[r['rule_id']],_expression(r['when'],w)) for r in policy['rules']):
            worlds.append(w)
    return worlds


def _question_rank(worlds,policy,observations,possible_actions):
    candidate=[];all_n=len(worlds)
    if not all_n:return []
    for attr in policy['atoms']:
        if observations.get(attr) in ('T','F'):continue
        partition={False:[],True:[]}
        for w in worlds:partition[w[attr]].append(w)
        if not partition[False] or not partition[True]:continue
        expected_actions=sum(len(group)/all_n*len({action_in_world(policy,w) for w in group})
                             for group in partition.values())
        improvement=len(possible_actions)-expected_actions
        world_reduction=all_n-sum(len(group)**2 for group in partition.values())/all_n
        candidate.append({'attribute':attr,'expected_action_set_reduction':round(improvement,8),
                          'expected_world_count_reduction':round(world_reduction,8),
                          'answer_groups':{str(k):len(v) for k,v in partition.items()}})
    candidate.sort(key=lambda z:(-z['expected_action_set_reduction'],
                                  -z['expected_world_count_reduction'],z['attribute']))
    return candidate


def decide_or_ask(policy,observations,rule_label_sets,*,assurance='uncalibrated'):
    """Safety is conditional on a correct policy and simultaneous state-set coverage.

    Never claim conditional per-decision alpha-risk. If no feasible worlds,
    return REVIEW rather than inventing a policy-compatible world.
    """
    worlds=feasible_worlds(policy,observations,rule_label_sets)
    if not worlds:
        return {'status':'REVIEW','action':None,'ask_attribute':None,
                'reason':'no_consistent_worlds_or_empty_prediction_set',
                'possible_actions':[],'world_count':0,'assurance':assurance}
    actions=sorted({action_in_world(policy,w) for w in worlds})
    if len(actions)==1:
        return {'status':'DECIDE','action':actions[0],'ask_attribute':None,
                'reason':'all_admissible_completions_agree',
                'possible_actions':actions,'world_count':len(worlds),'assurance':assurance}
    ranks=_question_rank(worlds,policy,observations,actions)
    if not ranks:
        return {'status':'REVIEW','action':None,'ask_attribute':None,
                'reason':'multiple_actions_but_no_answerable_single_attribute',
                'possible_actions':actions,'world_count':len(worlds),'assurance':assurance}
    return {'status':'ASK','action':None,'ask_attribute':ranks[0]['attribute'],
            'reason':'admissible_completions_disagree',
            'possible_actions':actions,'world_count':len(worlds),
            'question_candidates':ranks,'assurance':assurance,
            'question_selection_assumption':'uniform_over_admissible_worlds_for_ranking_only'}
