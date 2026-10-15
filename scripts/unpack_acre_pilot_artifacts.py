#!/usr/bin/env python3
"""Restore compressed audit inputs and complete per-case records without a GPU."""
import argparse,gzip,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();d=a.directory;run=d/'development';run.mkdir(parents=True,exist_ok=True)
base=d.parent if d.name=='revision3' else d
for name in ['development_manifest.json','development_oracle.json']:
 source=base/(name+'.gz')
 if source.exists():
  data=gzip.decompress(source.read_bytes());dest=base/name
  if dest.exists() and dest.read_bytes()!=data:raise ValueError('Existing frozen source differs: '+str(dest))
  dest.write_bytes(data)
for source in (base.parent).glob('*.json.gz'):
 dest=source.with_suffix('');data=gzip.decompress(source.read_bytes())
 if dest.exists() and dest.read_bytes()!=data:raise ValueError('Existing inventory differs')
 dest.write_bytes(data)
with gzip.open(d/'case_results.jsonl.gz','rt',encoding='utf-8') as f:
 for line in f:
  row=json.loads(line);oracle={'case_id':row['case_id'],'base_prompt_sha256':row['base_prompt_sha256'],'oracle_diagnostic_only':True,'variants':{k:v for k,v in row['variants'].items() if k.startswith('B6@')},'oracle_annotations':row['oracle_annotations']}
  (run/('case_'+row['case_id']+'.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
  (run/('case_'+row['case_id']+'_oracle.json')).write_text(json.dumps(oracle,ensure_ascii=False,indent=2)+'\n')
if (d/'annotations.jsonl.gz').exists():
 (run/'annotations').mkdir(exist_ok=True)
 with gzip.open(d/'annotations.jsonl.gz','rt',encoding='utf-8') as f:
  for line in f:
   row=json.loads(line);(run/'annotations'/(row['cache_key']+'.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
if (d/'noise_results.jsonl.gz').exists():
 (d/'noise').mkdir(exist_ok=True)
 with gzip.open(d/'noise_results.jsonl.gz','rt',encoding='utf-8') as f:
  for line in f:
   row=json.loads(line);(d/'noise'/(row['name']+'_'+row['case_id']+'.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
print('Offline inputs and records restored; no model or GPU is used.')
