"""Prepare the entire fixed benchmark; evaluation labels never enter inference files."""
import hashlib,json,collections,sys
from pathlib import Path
import pyarrow.parquet as pq
P=Path(__file__).resolve().parent
REV='1460ec478280a7295040b92f7eb985327f77dcbe'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 private=P/'private'; rows=pq.read_table(private/'train-00000-of-00001.parquet').to_pylist(); paths=json.loads((private/'all_paths.json').read_text())
 assert len(rows)==7747
 assert digest(private/'train-00000-of-00001.parquet')=='306c90e4899f8bd32c5c9911813b561aea3f2e2cb1251319a576a30b1180d943'
 root=P.parents[1];cfg=json.loads((root/'configs/care_module1_for_module2.json').read_text())
 for key in ['prompt','fixed_examples']:
  assert digest(root/cfg[key+'_path'])==cfg[key+'_sha256']
 cases=[];gold=[];coverage=[]
 for i,row in enumerate(rows):
  rid=f'medguide_{i:05d}'; rules=[]
  for j,path in enumerate(paths.get(row['disease'],[])):
   assert isinstance(path,list) and len(path)>1 and all(isinstance(s,str) and s.strip() for s in path)
   rules.append({'rule_id':f'R{j+1:02d}','text':'If the patient follows this clinical sequence: '+' -> '.join(path[:-1])+', then the recommended next step is '+path[-1]+'.','action':path[-1]})
  assert row['profile'] and row['options']
  cases.append({'case_id':rid,'patient':row['profile'],'question':'What is the appropriate next step?','rules':rules,'options':row['options']})
  gold.append({'case_id':rid,'answer':row['answer'],'disease':row['disease'],'cancer':row['disease'].split('_')[0].rstrip('0123456789'),'path_hash':hashlib.sha256(json.dumps(row['path'],ensure_ascii=False).encode()).hexdigest()})
  coverage.append({'case_id':rid,'rule_count':len(rules),'source_mapping_available':bool(rules),'blocker':None if rules else 'missing_source_guideline'})
 for k in range(4):
  with (private/f'input_{k}.jsonl').open('w') as f:
   for c in cases[k::4]:f.write(json.dumps(c,ensure_ascii=False)+'\n')
 with (private/'gold.jsonl').open('w') as f:
  for g in gold:f.write(json.dumps(g)+'\n')
 with (P/'source_coverage.jsonl').open('w') as f:
  for c in coverage:f.write(json.dumps(c)+'\n')
 write(P/'data_manifest.json',{'dataset':'MedGUIDE/MedGUIDE-MCQA-8K','revision':REV,'parquet_sha256':digest(private/'train-00000-of-00001.parquet'),'split':'train','evaluation_use':'all records as evaluation only; no fitting, tuning, training or data selection','n':len(rows),'source_guideline_url':'https://anonymous.4open.science/api/repo/Submission-MedGUIDE-187A/file/data/all_paths.json','source_guideline_sha256':digest(private/'all_paths.json'),'source_guideline_trees':len(paths),'source_mapped_cases':sum(c['source_mapping_available'] for c in coverage),'rule_requests':sum(c['rule_count'] for c in coverage),'by_disease':dict(collections.Counter(r['disease'] for r in rows)),'input_sha256':{str(k):digest(private/f'input_{k}.jsonl') for k in range(4)},'inference_fields':['case_id','patient (profile)','question (fixed generic)','rules (ALL independent published paths for disease)','options'],'excluded_fields':['answer','answer_text','path','prompt','quality scores','generation_model'],'license':'Dataset card: TBD. Intended for academic and research use. Raw data and raw text outputs stay private on research server.','limitations':['Guideline paths include sequential treatments and headings; deterministic verbalization is not expert-adjudicated rule ground truth.','No per-rule T/F/U gold labels exist; only MCQA is accuracy-scored.','Original A1 system prompt refers to invented simulation and is intentionally unchanged.','All neighboring branches retained; frozen 384-token output budget can cause parse failures.','Pretraining contamination cannot be excluded; no new training is performed.']})
 print('prepared',len(cases),'cases',sum(len(c['rules']) for c in cases),'rules')
if __name__=='__main__':main()
