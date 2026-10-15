"""Offline atomic-condition verification; never imported by normal annotation."""
import re,operator,hashlib,json
from .span_mapping import locate
OPS={'<':operator.lt,'<=':operator.le,'>':operator.gt,'>=':operator.ge,'==':operator.eq}

def normalize_unit(unit):
    return unit.lower().replace(' ','').replace('²','2').replace('×','x').replace('μ','u').replace('µ','u')

def convert(value,source,target,substance=None):
    source,target=normalize_unit(source),normalize_unit(target)
    aliases={'岁':'years','year':'years','yearsold':'years','月':'months','个月':'months','month':'months','天':'days','day':'days'}
    source,target=aliases.get(source,source),aliases.get(target,target)
    if source==target:return value
    table={('months','years'):1/12,('years','months'):12,('days','years'):1/365.25}
    if substance=='potassium':table.update({('meq/l','mmol/l'):1,('mmol/l','meq/l'):1})
    if (source,target) not in table:raise ValueError('unsupported_or_incompatible_unit_conversion')
    return value*table[source,target]

def parse_threshold(text):
    """One natural-language or mathematical threshold, with explicit variable/unit."""
    forms=[('>=',r'>=|≥|at least|no less than|不低于|不少于|至少|大于或等于'),('<=',r'<=|≤|at most|no more than|不超过|至多|小于或等于'),('>',r'(?<![<>=])>(?!=)|more than|greater than|above|over|超过|大于|高于'),('<',r'(?<![<>=])<(?!=)|less than|below|under|小于|低于')]
    for op,form in forms:
        match=re.search('(?:'+form+r')\s*(\d+(?:\.\d+)?)',text,re.I)
        if match:return {'operator':op,'threshold':float(match[1])}
    return None

def interval(value,lower,upper,lower_inclusive=True,upper_inclusive=True):
    return (value>=lower if lower_inclusive else value>lower) and (value<=upper if upper_inclusive else value<upper)

def infer_boolean(patient,positive,negative):
    pos=list(re.finditer(positive,patient,re.I));neg=list(re.finditer(negative,patient,re.I))
    if bool(pos)==bool(neg):raise ValueError('missing_or_conflicting_boolean_evidence')
    match=(pos or neg)[0]
    if len(pos or neg)!=1:raise ValueError('repeated_boolean_evidence_requires_explicit_context')
    return bool(pos),{'text':patient[match.start():match.end()],'start':match.start(),'end':match.end()}

def infer_numeric(patient,spec):
    matches=list(re.finditer(spec['measurement_pattern'],patient,re.I))
    if len(matches)!=1:raise ValueError('numeric_measurement_not_unique')
    m=matches[0];value=convert(float(m['value']),m['unit'],spec['unit'],spec.get('substance'))
    state=OPS[spec['operator']](value,spec['threshold'])
    return state,{'text':m[0],'start':m.start(),'end':m.end(),'value':value,'unit':spec['unit']}

def verify_pair(review,source,paths):
    patient=source['patient'];old,new=review['old'],review['new'];a,b=locate(patient,old)
    counterfactual=patient[:a]+new+patient[b:]
    if review.get('numeric_spec'):
        old_state,old_ev=infer_numeric(patient,review['numeric_spec']);new_state,new_ev=infer_numeric(counterfactual,review['numeric_spec'])
    else:
        old_state,old_ev=infer_boolean(patient,review['positive_pattern'],review['negative_pattern']);new_state,new_ev=infer_boolean(counterfactual,review['positive_pattern'],review['negative_pattern'])
    if old_state==new_state:raise ValueError('condition_did_not_flip')
    # Everything outside the single explicitly reviewed condition clause is exact.
    if patient[:a]!=counterfactual[:a] or patient[b:]!=counterfactual[a+len(new):]:raise ValueError('multiple_condition_edits')
    source_path=source['path'];node=source_path[-2]
    if old_state!=(node['decision_outcome']=='yes'):raise ValueError('original_patient_disagrees_with_source_path')
    matches=[p for p in paths if p[:-2]==source_path[:-2] and p[-2]['index']==node['index'] and p[-2]['decision_outcome']==('yes' if new_state else 'no')]
    if len(matches)!=1:raise ValueError('counterfactual_path_not_unique')
    opposite=matches[0]
    proofs=[]
    if len(review['ancestor_evidence'])!=len(source_path)-2:raise ValueError('incomplete_original_path_evidence_review')
    for element,quotes in zip(source_path[:-2],review['ancestor_evidence']):
        numeric_check=next((r for r in review.get('numeric_ancestor_checks',[]) if r['node_index']==element['index']),None)
        if numeric_check:
            old=OPS[numeric_check['operator']](old_ev['value'],numeric_check['threshold']);new=OPS[numeric_check['operator']](new_ev['value'],numeric_check['threshold'])
            if old!=new or old!=(element['decision_outcome']=='yes'):raise ValueError('ancestor_numeric_condition_changed')
            proofs.append({'node_index':element['index'],'condition':element['condition'],'source_state':element['decision_outcome'],'numeric_check':numeric_check,'old_value':old_ev['value'],'new_value':new_ev['value'],'truth_unchanged':True});continue
        if not quotes:raise ValueError('missing_ancestor_evidence')
        row=[]
        for quote in quotes:
            x,y=locate(patient,quote)
            if not (y<=a or x>=b):raise ValueError('edit_overlaps_another_condition_evidence')
            xx,yy=locate(counterfactual,quote)
            row.append({'text':quote,'original_start':x,'original_end':y,'new_start':xx,'new_end':yy})
        proofs.append({'node_index':element['index'],'condition':element['condition'],'source_state':element['decision_outcome'],'source_text_reviewed':True,'unchanged_exact_evidence':row})
    if review.get('incompatible_actions') is not True or not review.get('incompatibility_reason'):raise ValueError('action_incompatibility_not_reviewed')
    if review.get('clinical_consistency') is not True or not review.get('single_factor'):raise ValueError('clinical_or_single_factor_review_missing')
    return {'original_patient':patient,'counterfactual_patient':counterfactual,'original_path':source_path,'counterfactual_path':opposite,'changed_node_index':node['index'],'changed_condition':node['condition'],'single_factor':review['single_factor'],'original_state':old_state,'new_state':new_state,'evidence':{'original':old_ev,'counterfactual':new_ev},'edit':{'source_start':a,'source_end':b,'old':old,'new':new},'ancestor_checks':proofs,'incompatible_actions':True,'incompatibility_reason':review['incompatibility_reason'],'clinical_consistency_reviewed':True,'all_other_characters_unchanged':True,'tree_route_verified':True,'references_used_only_offline':True}
