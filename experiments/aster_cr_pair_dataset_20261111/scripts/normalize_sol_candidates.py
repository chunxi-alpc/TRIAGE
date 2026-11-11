"""Execute explicit edits authored by the assigned Codex Sol/high Agent A.
No model calls, inferred clinical edits, reviewer reasoning, or semantic PASS decisions.
Author rows: candidate_id, kind, case_id, node_id, label, factor,
actual_replacements [{component,old,new, optional start/occurrence/all/expected_count}],
source_evidence (dict), label_derivation {original,modified}, counterfactual.
C may instead use target_condition_edit {old,new,synchronize_target_occurrences:true}.
Optional original/modified program semantics are preserved only as supplied.
"""
from runtime_paths import ROOT, DATA, RAW, REPO, SCRIPTS, SCHEMA, source_document_path
import argparse
import collections
import copy
import json
import pathlib
import re

from agent_a_sol_prepare import ROOT, MODEL, EFFORT, FitCatalog, read_rows, render


def match_starts(text, old):
    found = []
    start = 0
    while True:
        i = text.find(old, start)
        if i < 0:
            return found
        found.append(i)
        start = i + len(old)


def literal_replace(text, edits):
    """Apply independent replacements against the original text, never cascade."""
    slots = []
    for edit in edits:
        old, new = edit['old'], edit['new']
        if not isinstance(old,str) or not isinstance(new,str) or not old or old == new:
            raise ValueError('Literal edits require different nonempty old text and string new text')
        starts = match_starts(text,old)
        if 'expected_count' in edit and len(starts) != edit['expected_count']:
            raise ValueError('Unexpected original occurrence count')
        if 'start' in edit:
            i = edit['start']
            if type(i) is not int or i not in starts:
                raise ValueError('Explicit start does not anchor the supplied original span')
            chosen = [i]
        elif 'occurrence' in edit:
            ordinal = edit['occurrence']
            if type(ordinal) is not int or ordinal < 0 or ordinal >= len(starts):
                raise ValueError('Invalid zero-based occurrence')
            chosen = [starts[ordinal]]
        elif edit.get('all') is True:
            if 'expected_count' not in edit or not starts:
                raise ValueError('All-occurrences edit must state its nonzero expected_count')
            chosen = starts
        elif len(starts) == 1:
            chosen = starts
        else:
            raise ValueError('Literal span is absent or ambiguous; specify exact location/all count')
        slots.extend((i,i+len(old),new) for i in chosen)
    slots.sort()
    if any(a[1]>b[0] for a,b in zip(slots,slots[1:])):
        raise ValueError('Overlapping original spans')
    result = text
    for i,j,new in reversed(slots):
        result = result[:i]+new+result[j:]
    return result


def explicit_condition_edits(side, change):
    old,new=change['old'],change['new']
    if old != side['condition'] or change.get('synchronize_target_occurrences') is not True:
        raise ValueError('C requires the complete original target condition and explicit synchronization')
    line = side['node_id'] + ': ' + old
    occurrences = list(re.finditer(r'(?m)^'+re.escape(line)+r'$',side['guide']))
    if not occurrences:
        raise ValueError('Original target condition line is absent')
    return [dict(component='condition',old=old,new=new,all=True,expected_count=1)] + [
        dict(component='guide',old=line,new=side['node_id']+': '+new,start=m.start()) for m in occurrences]


def validate_scope(kind, original, modified):
    if original['definitions'] != modified['definitions']:
        raise ValueError('Definitions must remain exact')
    if type(modified['label']) is not int or modified['label'] not in (0,1):
        raise ValueError('Author label must be integer 0 or 1')
    if kind in ('P','A') and original['label'] != modified['label']:
        raise ValueError('P/A labels must be preserved')
    if kind in ('C','F') and original['label'] == modified['label']:
        raise ValueError('C/F labels must flip')
    if kind in ('P','F'):
        if original['condition']!=modified['condition'] or original['guide']!=modified['guide']:
            raise ValueError('P/F may edit only patient facts')
        if original['patient']==modified['patient']:
            raise ValueError('P/F requires an actual patient edit')
    elif kind == 'C':
        if original['patient']!=modified['patient'] or original['condition']==modified['condition']:
            raise ValueError('C preserves patient and changes target condition')
        before = original['node_id']+': '+original['condition']
        after = original['node_id']+': '+modified['condition']
        # Current released full prompts contain one condition line per target; the current
        # question is the second occurrence. No other decision node or action may change.
        expected, count = re.subn(r'(?m)^'+re.escape(before)+r'$',lambda m:after,original['guide'])
        if not count or expected != modified['guide']:
            raise ValueError('C guide differs outside synchronized occurrences of its target node')
    elif kind == 'A':
        if original['condition']!=modified['condition'] or original['patient']!=modified['patient']:
            raise ValueError('A preserves patient and target condition')
        left,right=original['guide'].splitlines(),modified['guide'].splitlines()
        if len(left)!=len(right):
            raise ValueError('A must preserve leaf lines and tree structure')
        changes=0
        for a,b in zip(left,right):
            if a==b:
                continue
            la,lb=re.match(r'^(L[0-9]+):',a),re.match(r'^(L[0-9]+):',b)
            if not la or not lb or la.group(1)!=lb.group(1):
                raise ValueError('A may change terminal action text only, preserving leaf identifiers')
            changes+=1
        if not changes:
            raise ValueError('A requires an actual terminal action edit')
    else:
        raise ValueError('Unsupported training class')


def edit_signature(kind, case_id, node_id, edits):
    return (kind,case_id,node_id,tuple(sorted((e['component'],e['old'],e['new']) for e in edits)))


def normalized_edit_signature(pair):
    return (pair['kind'],pair['case_id'],pair['node_id'],tuple((field,pair['original'][field],pair['modified'][field]) for field in ('patient','condition','guide','definitions') if pair['original'][field]!=pair['modified'][field]))


def counterfactual_rule(original_rule, kind, modified_facts, modified_predicate):
    supplied=(kind=='C' and isinstance(modified_predicate,dict) and bool(modified_predicate)) or (kind=='F' and isinstance(modified_facts,dict) and bool(modified_facts))
    return original_rule if original_rule and supplied else None


def normalize_one(author, catalog, author_file, ordinal):
    if author.get('available') is False:
        raise ValueError('ABSTAIN: '+str(author.get('reason','No supported one-factor edit')))
    kind,cid,gid=author['kind'],author['case_id'],author['node_id']
    if kind not in ('P','C','F','A'):
        raise ValueError('Unknown class')
    if author.get('counterfactual') is not (kind in ('C','F')):
        raise ValueError('Explicit counterfactual marker must agree with class')
    original = catalog.original(cid,gid)
    modified = copy.deepcopy(original)
    modified['label']=author['label']
    replacements=copy.deepcopy(author.get('actual_replacements',[]))
    if kind=='C' and 'target_condition_edit' in author:
        if replacements:
            raise ValueError('Supply either target_condition_edit or literal replacements')
        replacements=explicit_condition_edits(original,author['target_condition_edit'])
    by_component=collections.defaultdict(list)
    for replacement in replacements:
        if replacement['component'] not in ('patient','condition','guide'):
            raise ValueError('Only patient, condition, or guide literal edits are supported')
        by_component[replacement['component']].append(replacement)
    for component,edits in by_component.items():
        modified[component]=literal_replace(original[component],edits)
    for field in ('patient','condition','guide'):
        explicit=author.get('modified_'+field)
        if explicit is not None and explicit!=modified[field]:
            raise ValueError('Literal result differs from supplied modified_'+field)
    validate_scope(kind,original,modified)
    derivation=copy.deepcopy(author['label_derivation'])
    if not isinstance(derivation,dict) or not derivation.get('original') or not derivation.get('modified'):
        raise ValueError('Both author label derivations must be retained')
    evidence=copy.deepcopy(author['source_evidence'])
    if not isinstance(evidence,dict):
        raise ValueError('source_evidence must be an object')
    if not isinstance(author['factor'],str) or not author['factor'].strip():
        raise ValueError('Author must identify the edited factor')
    candidate_id=author['candidate_id']
    if not isinstance(candidate_id,str) or not candidate_id:
        raise ValueError('A stable author candidate_id is required')
    pair_id=author.get('pair_id','sol-a:'+candidate_id)
    modified['view_id']=pair_id+':modified'
    modified['prompt']=render(modified)
    meta=catalog.metadata(cid,gid)
    evidence.setdefault('document',meta['source_document'])
    evidence.setdefault('original_condition',original['condition'])
    evidence.setdefault('original_patient',original['patient'])
    original_rule=copy.deepcopy(author.get('program_rule',catalog.rules.get((cid,gid))))
    # A released original rule does not describe the counterfactual by itself.
    # Only an explicitly authored nonempty modified-side representation enables
    # program recomputation; otherwise leave semantic judgment to independent B.
    modified_facts=copy.deepcopy(author.get('program_modified_facts'))
    modified_predicate=copy.deepcopy(author.get('program_modified_predicate'))
    rule=counterfactual_rule(original_rule,kind,modified_facts,modified_predicate)
    pair=dict(schema_version='1.0',pair_id=pair_id,kind=kind,origin='generated',**meta,
              raw_patient=catalog.cases[cid]['patient'],original=original,modified=modified,
              edits=replacements,counterfactual=kind in ('C','F'),
              edit_attribution='Agent A artificial counterfactual' if kind in ('C','F') else 'Agent A controlled edit',
              source_evidence=evidence,label_derivation=derivation,generation_factor=author['factor'],
              original_label_source='MEGA-CDP released fit gold_path',
              program_rule=rule,original_executable_rule=original_rule,program_modified_facts=modified_facts,
              program_modified_predicate=modified_predicate,
              program_semantics_status='explicit_counterfactual_representation_supplied' if rule else 'counterfactual_not_program_computable_independent_B_required',
              provenance=dict(author_role='formal Codex Agent A',requested_model=MODEL,requested_reasoning_effort=EFFORT,
                              underlying_snapshot=None,model_execution='assigned Codex agent; no server model API or CLI',
                              author_file=str(author_file),author_record_index=ordinal,author_candidate_id=candidate_id,
                              original_view=original['view_id'],raw_case_path_id=cid.rsplit(':',1)[1],
                              semantic_review='pending independent Agent B'))
    return pair


def normalize_file(input_path, output_path, exclusions_path):
    if not (ROOT/'reports/completion_historical.json').is_file():
        raise RuntimeError('Historical review must complete before normalization')
    output_path,exclusions_path=pathlib.Path(output_path),pathlib.Path(exclusions_path)
    if output_path.exists() or exclusions_path.exists():
        raise FileExistsError('Existing normalized results are preserved; use explicit new output paths')
    catalog=FitCatalog()
    from jsonschema import Draft202012Validator
    schema=Draft202012Validator(json.loads((SCHEMA).read_text(encoding='utf-8')))
    # Only actual pairs are compared; the existence of another accepted case/class is
    # not grounds to discard a different target node or genuinely different edit.
    signatures=set()
    for old in read_rows(ROOT/'accepted_historical_pairs.jsonl'):
        signatures.add(normalized_edit_signature(old))
    results=[]
    exclusions=[]
    ids=set()
    for index,author in enumerate(read_rows(input_path)):
        try:
            pair=normalize_one(author,catalog,input_path,index)
            schema.validate(pair)
            if pair['pair_id'] in ids:
                raise ValueError('Duplicate candidate pair_id')
            signature=normalized_edit_signature(pair)
            if signature in signatures:
                raise ValueError('Same accepted original case/node/class and literal edit')
            signatures.add(signature)
            ids.add(pair['pair_id'])
            results.append(pair)
        except Exception as exc:
            exclusions.append(dict(candidate_id=author.get('candidate_id'),case_id=author.get('case_id'),node_id=author.get('node_id'),kind=author.get('kind'),reason=str(exc)))
    with output_path.open('x',encoding='utf-8') as stream:
        for pair in results:
            stream.write(json.dumps(pair,ensure_ascii=False)+'\n')
    with exclusions_path.open('x',encoding='utf-8') as stream:
        for row in exclusions:
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    return dict(normalized=len(results),excluded=len(exclusions),semantic_pass_decisions=0,output=str(output_path),exclusions=str(exclusions_path))


def self_test():
    original_rule={'predicate':{'op':'>','field':'x','value':5},'facts':{'x':6}}
    assert counterfactual_rule(original_rule,'C',None,None) is None
    assert counterfactual_rule(original_rule,'F',None,None) is None
    assert counterfactual_rule(original_rule,'C',None,{'op':'>','field':'x','value':7}) is original_rule
    assert counterfactual_rule(original_rule,'F',{'x':4},None) is original_rule
    assert counterfactual_rule(None,'F',{'x':4},None) is None
    assert literal_replace('12 kg and 12 days',[dict(old='12 kg',new='13 kg')])=='13 kg and 12 days'
    assert literal_replace('x x',[dict(old='x',new='y',all=True,expected_count=2)])=='y y'
    assert literal_replace('ab',[dict(old='a',new='b'),dict(old='b',new='c')])=='bc'
    assert literal_replace('x x',[dict(old='x',new='y',occurrence=1)])=='x y'
    for text,edits in [('x x',[dict(old='x',new='y')]),('abc',[dict(old='ab',new='x'),dict(old='bc',new='y')])]:
        try:
            literal_replace(text,edits)
        except ValueError:
            pass
        else:
            raise AssertionError('Ambiguous/overlapping source edit was accepted')
    original=dict(patient='fact',condition='x > 5',guide='G0: x > 5\nG1: y > 2\nL2: action',definitions='',node_id='G0',label=1)
    changed=copy.deepcopy(original)
    for edit in explicit_condition_edits(original,dict(old='x > 5',new='x > 9',synchronize_target_occurrences=True)):
        component=edit['component']
        changed[component]=literal_replace(changed[component],[edit])
    changed['label']=0
    validate_scope('C',original,changed)
    changed['guide']=changed['guide'].replace('y > 2','y > 8')
    try:
        validate_scope('C',original,changed)
    except ValueError:
        pass
    else:
        raise AssertionError('C changed an unrelated decision node')
    action=copy.deepcopy(original)
    action['guide']=action['guide'].replace('L2: action','L2: replacement action')
    validate_scope('A',original,action)
    print('normalizer pure-function tests passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--input',default=str(ROOT/'sol_author_edits.jsonl'))
    parser.add_argument('--output',default=str(ROOT/'generated_candidates.jsonl'))
    parser.add_argument('--exclusions',default=str(ROOT/'sol_conversion_exclusions.jsonl'))
    args=parser.parse_args()
    if args.self_test:
        self_test()
    else:
        print(json.dumps(normalize_file(args.input,args.output,args.exclusions),ensure_ascii=False))
