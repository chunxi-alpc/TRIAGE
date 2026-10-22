#!/usr/bin/env python3
"""Acquire the real Parquet schema and original-author text paths, privately."""
import argparse
import hashlib
from pathlib import Path
import requests

REVISION='1460ec478280a7295040b92f7eb985327f77dcbe'
SOURCES={
 'train.parquet':('https://huggingface.co/datasets/MedGUIDE/MedGUIDE-MCQA-8K/resolve/'+REVISION+'/data/train-00000-of-00001.parquet',
                  '306c90e4899f8bd32c5c9911813b561aea3f2e2cb1251319a576a30b1180d943'),
 'all_paths.json':('https://anonymous.4open.science/api/repo/Submission-MedGUIDE-187A/file/data/all_paths.json','0c0af3b487bde55e87ede987be8389696f204fbdd12231549f54bd0e6078cace')}

def acquire(output_dir, parquet=None, paths_json=None):
    import shutil,json
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=True);manifest={}
    for name,(url,expected) in SOURCES.items():
        dest=root/name;local=parquet if name=='train.parquet' else paths_json
        if local:
            if Path(local).resolve()!=dest.resolve():shutil.copyfile(local,dest)
        elif not dest.exists():
            response=requests.get(url,timeout=30);response.raise_for_status()
            tmp=dest.with_suffix('.download');tmp.write_bytes(response.content);tmp.replace(dest)
        digest=hashlib.sha256(dest.read_bytes()).hexdigest()
        if expected and digest!=expected:raise ValueError(name+' differs from fixed revision')
        manifest[name]={'source_url':url,'sha256':digest,'local_copy':bool(local)}
    import pyarrow.parquet as pq
    schema=pq.read_schema(root/'train.parquet')
    needed={'disease','path','profile','prompt','options','answer','answer_text'}
    if not needed.issubset(schema.names):raise ValueError('missing MedGUIDE fields')
    paths=json.loads((root/'all_paths.json').read_text())
    if not isinstance(paths,dict) or not paths:raise ValueError('expected disease -> list of string paths')
    manifest.update({'revision':REVISION,'schema':str(schema),'guideline_sets':len(paths)})
    (root/'sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('verified actual MedGUIDE Parquet and',len(paths),'published path sets')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True)
    p.add_argument('--parquet');p.add_argument('--paths-json');a=p.parse_args()
    acquire(a.output_dir,a.parquet,a.paths_json)
