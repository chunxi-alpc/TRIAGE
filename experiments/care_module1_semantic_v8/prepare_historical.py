#!/usr/bin/env python3
"""Prepare label-free direct-semantic inputs and freeze training demonstrations."""
import collections, json
from pathlib import Path
from prompts import prediction_input, messages_for, parse_output
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

def main():
    rows = [json.loads(line) for line in open(REPO/'experiments/care_reliability_20261019/data/mechanism_cases.jsonl')]
    data = HERE/'data'; data.mkdir(exist_ok=True)
    for split in ('dev','confirmation'):
        with (data/f'semantic_{split}_inputs.jsonl').open('w') as f:
            for c in rows:
                if c['split'] == split:
                    f.write(json.dumps(dict(case_id=c['case_id'], split=split, family_id=c['family_id'], **prediction_input(c)),ensure_ascii=False)+'\n')
    ids = [['mechanism_01_conjunction-p01-canonical-app','mechanism_01_conjunction-p01-canonical-non'],
           ['mechanism_03_negation-p03-canonical-non'],
           ['mechanism_01_conjunction-u-missing-canonical-unknown']]
    descriptions = ['T/F minimal contrast: only current age changes', 'Exception and negation', 'Missing information remains U']
    byid={c['case_id']:c for c in rows}
    reasons = [
      ['The age and authorization requirements hold, and there is no suspension.', 'The required age-and-authorization combination holds, so its negation is false.', 'The patient has no suspension flag.'],
      ['The patient is younger than 18, so the combined prerequisite fails.', 'The patient is younger than 18, establishing that the age-and-authorization combination fails.', 'The patient has no suspension flag.'],
      ['The unwaived suspension blocks this rule even though the age prerequisite holds.', 'The age prerequisite for this alternative fails, and the unwaived suspension also blocks it.', 'The patient has a suspension flag without a waiver.'],
      ['Current age is missing, so applicability cannot be established despite authorization and no suspension.', 'Current age is missing, so the negation of the age-and-authorization combination is also unresolved.', 'The patient has no suspension flag.']]
    quotes = [
      ["The patient's current age is 18 years.","The patient's current age is 18 years.",'The current patient does not have a routing suspension flag.'],
      ["The patient's current age is 17 years.","The patient's current age is 17 years.",'The current patient does not have a routing suspension flag.'],
      ['The current patient has a routing suspension flag. The current patient does not have a suspension waiver.',"The patient's current age is 19 years.",'The current patient has a routing suspension flag. The current patient does not have a suspension waiver.'],
      ['The current value of age is not documented.','The current value of age is not documented.','The current patient does not have a routing suspension flag.']]
    examples = {'count_demonstrations':3,'count_patient_narratives':4,'source':'historical train split only','frozen_before_dev_inference':True,'demonstrations':[]}
    k=0
    for group, description in zip(ids, descriptions):
        demo={'description':description,'cases':[]}
        for cid in group:
            c=byid[cid]; assert c['split']=='train'
            output=[]
            for j,r in enumerate(c['reference_structure']['rules']):
                assert quotes[k][j] in c['patient']
                output.append(dict(rule_id=r['rule_id'],state=r['state'],evidence_quote=quotes[k][j],reason_short=reasons[k][j]))
            demo['cases'].append(dict(source_case_id=cid,source_family_id=c['family_id'],input=prediction_input(c),output=output));k+=1
        examples['demonstrations'].append(demo)
    eval_families={c['family_id'] for c in rows if c['split'] in ('dev','confirmation')}
    assert all(ex['source_family_id'] not in eval_families for d in examples['demonstrations'] for ex in d['cases'])
    (data/'fewshot_examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n')
    # A single lightweight parser/allowlist check; no model selection or gating.
    ex=examples['demonstrations'][0]['cases'][0]
    parsed, error = parse_output(json.dumps({'rules':ex['output']}),[r['rule_id'] for r in ex['input']['rules']])
    assert parsed and error is None
    assert parse_output('{}',[r['rule_id'] for r in ex['input']['rules']])[0] is None
    assert set(prediction_input(rows[0])) == {'question','patient','rules'}
    report={'split_counts':dict(collections.Counter(c['split'] for c in rows)), 'checks':'passed: input allowlist, three frozen training-only demonstrations, valid and failed JSON parse',
            'historical_scope':'reused development material; historical regression exploration, not a new independent confirmation',
            'historical_rule_text':'offline lossless verbalization of existing guideline logic trees; no patient facts or reference labels in prediction inputs'}
    (data/'semantic_preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
if __name__=='__main__':main()
