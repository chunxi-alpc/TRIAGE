"""Whitelist public deliverables; never publish source profiles or quoted raw outputs."""
import hashlib,json,pathlib,shutil,zipfile
root=pathlib.Path('.');out=root/'delivery_stage/care_module1_v4';out.mkdir(parents=True,exist_ok=False)
def copy(p):
 q=out/p;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
for p in list(root.glob('*.py'))+list(root.glob('*.txt'))+[root/'FREEZE.json',root/'reproduction_check.json']:
 if p.is_file():copy(p)
for d in ['prerequisites','provenance']:
 for p in (root/d).glob('*'):
  if p.is_file():copy(p)
for p in (root/'data').glob('*'):
 if p.is_file():copy(p)
normalized={r['sample_id']:r for r in [json.loads(s) for s in (root/'data/model/normalized_records.jsonl').read_text().splitlines()]}
for p in (root/'data/model').glob('*'):
 if not p.is_file():continue
 if not p.name.endswith('_raw.jsonl'):
  if p.name=='normalized_records.jsonl':
   items=[]
   for record in normalized.values():
    record=json.loads(json.dumps(record))
    for arm in ['unverified','verified']:
     for au in record[arm]['alignment'].values():
      for entry in au:
       x=entry['provided_id']
       if isinstance(x,str) and not x.strip().replace('.','',1).isdigit():entry['provided_id']={'withheld_sha256':hashlib.sha256(x.encode()).hexdigest()}
    items.append(record)
   q=out/p;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(''.join(json.dumps(x)+'\n' for x in items))
  else:copy(p)
  continue
 rows=[]
 def sanitize(x):
  if isinstance(x,list):return [sanitize(v) for v in x]
  if isinstance(x,dict):return {k:({'withheld_source_quote_sha256':hashlib.sha256(str(v).encode()).hexdigest()} if k in ['quote','text','evidence_text'] else sanitize(v)) for k,v in x.items()}
  return x
 for line in p.read_text().splitlines():
  r=json.loads(line);r['original_record_sha256']=hashlib.sha256(line.encode()).hexdigest();r['original_raw_output_sha256']=hashlib.sha256(r['raw_output'].encode()).hexdigest()
  if p.name in ['facts_raw.jsonl','verification_raw.jsonl']:
   r['original_parsed_sha256']=hashlib.sha256(json.dumps(r['parsed'],sort_keys=True).encode()).hexdigest()
   r['raw_output']=None
   nr=normalized[r['id']]['unverified' if p.name=='facts_raw.jsonl' else 'verified']
   r['parsed']={k:nr[k] for k in ['facts','evidence','status']}
   r['source_quotes_withheld']=True;r['parsed_is_normalized_not_raw']=True
  rows.append(r)
 q=out/p;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
manifest={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
(out/'PUBLIC_SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
with zipfile.ZipFile('care_module1_v4.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in out.rglob('*'):
  if p.is_file():z.write(p,p.relative_to(out.parent))
print(json.dumps({'files':len(manifest),'zip_bytes':pathlib.Path('care_module1_v4.zip').stat().st_size,'zip_sha256':hashlib.sha256(pathlib.Path('care_module1_v4.zip').read_bytes()).hexdigest()}))
