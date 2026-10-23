"""Acquire revision-pinned external sources, preferring canonical HF then mirror."""
import argparse,concurrent.futures,hashlib,json,time
from pathlib import Path
import requests
SOURCES={'RDBH/DriftMed':('5dcfe380f49c6f17f3c6ba5e25e0e958f9642f0e',['README.md','advice_pairs.jsonl','ConflictMed_v2.jsonl']), 'ncbi/MedCalc-Bench':('83a1a56d6b53f8d4c0e03f0ba9699becf524a595',['README.md','test_data_11_18_final.csv'])}
EXPECTED={'ConflictMed_v2.jsonl':'c391320d44c7f3bb63b5e1566fa1b64dc693beb2d56f20bb8ed2487a83dbde2a','test_data_11_18_final.csv':'c9d219a30e43f50646b73fa5fe5fe86d1ef47e7d9b2c7403a0edb7f3e5290e3a'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--private',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.private);root.mkdir(parents=True,exist_ok=True)
 def fetch(item):
  repo,rev,file=item;dest=root/(repo.replace('/','__')+'__'+file);canonical=f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{file}';errors=[]
  if dest.exists() and file in EXPECTED and hashlib.sha256(dest.read_bytes()).hexdigest()==EXPECTED[file]:return {'dataset':repo,'revision':rev,'file':file,'url':canonical,'sha256':EXPECTED[file],'bytes':dest.stat().st_size,'status':'verified private cache'}
  for host in ['https://hf-mirror.com','https://huggingface.co']:
   try:
    url=canonical.replace('https://huggingface.co',host);r=requests.get(url,timeout=(10,40));r.raise_for_status()
    if file in EXPECTED and hashlib.sha256(r.content).hexdigest()!=EXPECTED[file]:raise ValueError('source hash mismatch: '+file)
    dest.write_bytes(r.content)
    return {'dataset':repo,'revision':rev,'file':file,'url':canonical,'download_url':url,'sha256':hashlib.sha256(r.content).hexdigest(),'bytes':len(r.content),'status':'downloaded'}
   except requests.RequestException as e:errors.append(str(e))
  return {'dataset':repo,'revision':rev,'file':file,'url':canonical,'status':'unavailable','errors':errors}
 with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:result=list(ex.map(fetch,[(d,v,f) for d,(v,fs) in SOURCES.items() for f in fs]))
 Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
