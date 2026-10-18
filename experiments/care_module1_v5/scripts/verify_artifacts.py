"""Check completed artifacts against raw server outputs, without model inference."""
from __future__ import annotations
import argparse,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[3]
EXP=ROOT/'experiments/care_module1_v5'

def rows(path):
 return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def check(run,labels_path,expected):
 status=json.loads((run/'run_status.json').read_text())
 assert status['status']=='completed' and status['sample_count_completed']==expected
 public=rows(run/'predictions_public.jsonl');private=rows(run/'private/predictions_private.jsonl');labels=rows(labels_path)
 assert len(public)==len(private)==len(labels)==expected
 by_id={x['sample_id']:x for x in private};assert len(by_id)==expected
 assert {x['sample_id'] for x in labels}=={x['sample_id'] for x in public}==set(by_id)
 counts={'samples':expected,'raw_outputs_verified':0,'patient_spans_verified':0,'condition_spans_verified':0,'action_spans_verified':0,'nli_score_vectors_verified':0,'nli_skipped_candidates':0}
 for pub in public:
  raw=by_id[pub['sample_id']];graph=raw['module1_output'];patient='\n'.join(raw['patient_sentences'])
  assert graph['metadata']['qwen_called'] is False and graph['metadata']['reference_answers_used'] is False
  assert graph['schema_version']=='module1.v1' and graph['implementation_version']=='module1.v5'
  rule=graph['rules'][0];assert pub['applicability']==rule['applicability']
  specs={c['condition_id']:c for r in graph['candidate_rules'] for c in r['conditions']+r.get('exceptions',[])}
  details={c['condition_id']:c for c in rule['conditions']+rule.get('exceptions',[])}
  for c in pub['conditions']:
   detail=details[c['condition_id']];completion=detail['trialmatch']['raw_output']
   assert hashlib.sha256(completion.encode()).hexdigest()==c['trialmatch_output_sha256']
   assert c['trialmatch_state']==detail['trialmatch']['state']
   counts['raw_outputs_verified']+=1
   for span in detail['support_evidence']+detail['refute_evidence']+detail['unknown_evidence']:
    assert 0<=span['start']<=span['end']<=len(patient)
    assert patient[span['start']:span['end']]==span['text']
    assert raw['patient_sentences'][span['sentence_id']-1]==span['text']
    counts['patient_spans_verified']+=1
   for candidate in detail['nli_candidates']:
    scores=candidate['scores']
    if candidate['class_id']==-1:
     assert candidate['nli_relation']=='neutral' and scores=={}
     counts['nli_skipped_candidates']+=1
    else:
     mapping={int(k):v.lower() for k,v in status['mednli_id2label'].items()}
     assert mapping[candidate['class_id']]==candidate['nli_relation']
     assert set(scores)=={'entailment','neutral','contradiction'}
     assert abs(sum(scores.values())-1)<1e-5 and all(0<=v<=1 for v in scores.values())
     counts['nli_score_vectors_verified']+=1
   spec=specs[c['condition_id']];span=spec.get('source_span')
   if span and span.get('start') is not None:
    assert spec['text'] in graph['candidate_rules'][0]['text'][span['start']:span['end']]
    counts['condition_spans_verified']+=1
  for prepared in graph['candidate_rules']:
   if prepared.get('action_span'):
    span=prepared['action_span'];assert prepared['text'][span['start']:span['end']]==prepared['action']
    counts['action_spans_verified']+=1
 assert counts['raw_outputs_verified']==status['trialmatch_condition_count']
 assert status['trialmatch_lora_parameter_count']>0 and status['trialmatch_active_adapters']
 return counts

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,default=EXP/'provenance/completed_artifact_integrity.json');a=p.parse_args()
 result={'status':'passed','model_calls':0,'qwen_calls':0,'checks':{
  'regression':check(EXP/'runs/v5_20261018',EXP/'data/labels_public.jsonl',192),
  'new_synthetic':check(EXP/'exploratory/runs/frozen_v1',EXP/'exploratory/data/labels_public.jsonl',24)}}
 freeze=json.loads((EXP/'data/RUN_FREEZE.json').read_text())
 for name,metadata in freeze['files'].items():
  assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==metadata['sha256'],name
 result['frozen_files_verified']=len(freeze['files'])
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
