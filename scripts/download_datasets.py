#!/usr/bin/env python
"""Fetch public original datasets; Hugging Face snapshots resume via its cache."""
import argparse
import concurrent.futures
import json
import os
import time
from pathlib import Path
import urllib.request
import zipfile

os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT','120')
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
ROOT = Path(__file__).resolve().parents[1]
HF = {
    'MEGA-CDP': 'nchennnn/MEGA-CDP',
    'MedGUIDE': 'MedGUIDE/MedGUIDE-MCQA-8K',
    'MedCalc-Bench': 'ncbi/MedCalc-Bench',
    'MedPIC-Bench': 'TIM0927/MedPIC-Bench',
    'FaithEval-inconsistent': 'Salesforce/FaithEval-inconsistent-v1.0',
    'FaithEval-counterfactual': 'Salesforce/FaithEval-counterfactual-v1.0',
    'FaithEval-unanswerable': 'Salesforce/FaithEval-unanswerable-v1.0',
}
SOURCES = {
    'MedGUIDE': 'https://arxiv.org/html/2505.11613v1',
    'FaithEval': 'https://github.com/SalesforceAIResearch/FaithEval',
    'ConFiQA': 'https://github.com/byronBBL/Context-DPO',
    'NLI4CT': 'https://github.com/ai-systems/nli4ct',
    'MedNLI': 'https://physionet.org/content/mednli/1.0.0/',
}

def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    tmp = path.with_suffix(path.suffix + '.partial')
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'TRIAGE-deployment'}), timeout=180) as r, tmp.open('wb') as f:
        import shutil
        shutil.copyfileobj(r, f)
    tmp.replace(path)

def download_hf(name, repo, full_mega=False):
    from huggingface_hub import snapshot_download, HfApi
    entry = {'name': name, 'source': f'https://huggingface.co/datasets/{repo}', 'repo_id': repo, 'format': 'original'}
    try:
        for attempt in range(4):
            try:
                info=HfApi().dataset_info(repo)
                break
            except Exception:
                if attempt==3:
                    raise
                time.sleep(30*(attempt+1))
        entry.update(endpoint=os.environ.get('HF_ENDPOINT','https://huggingface.co'), revision=info.sha)
        patterns=None if name != 'MEGA-CDP' or full_mega else ['*.jsonl','*.md','.gitattributes']
        entry['download_scope']='all original files' if patterns is None else 'all original JSONL and Markdown; guideline images omitted for rapid text deployment'
        path = snapshot_download(repo_id=repo, repo_type='dataset', revision=info.sha, allow_patterns=patterns, cache_dir=ROOT / 'data' / '.hf_cache', max_workers=4)
        link = ROOT / 'data' / name
        if not link.exists():
            link.symlink_to(path, target_is_directory=True)
        entry.update(status='downloaded', path=str(Path(path).resolve()), project_path=str(link), complete=True, contains_all_original_files=patterns is None)
    except Exception as exc:
        entry.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
    return name, entry

def download_aux(name):
    path = ROOT / 'data' / name
    entry = {'name': name, 'source': SOURCES[name], 'path': str(path), 'format': 'original'}
    try:
        if name == 'NLI4CT':
            archive = path / 'Complete_dataset.zip'
            fetch('https://raw.githubusercontent.com/ai-systems/nli4ct/main/Complete_dataset.zip', archive)
            extracted = path / 'original'
            if not (path / '.extracted').exists():
                with zipfile.ZipFile(archive) as z:
                    z.extractall(extracted)
                (path / '.extracted').touch()
            entry['path'] = str(extracted)
        else:
            entry.update(fetch_github_original('byronBBL/Context-DPO','master',path,prefix='ConFiQA/ConFiQA-'))
            entry['download_scope']='official QA/MR/MC benchmark splits; DPO training files omitted'
        entry['status']='downloaded'
        entry['complete']=True
    except Exception as exc:
        entry.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
    return name, entry

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--primary-only', action='store_true')
    parser.add_argument('--full-mega',action='store_true',help='Include original guideline images; substantially longer download')
    args=parser.parse_args()
    config_path=ROOT/'configs'/'datasets.json'
    registry=json.loads(config_path.read_text()) if config_path.exists() else {}
    work=list(HF.items())[:4] if args.primary_only else list(HF.items())
    # Limited parallelism avoids overwhelming the network shared with model downloads.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures=[]
        if registry.get('NLI4CT-2024',{}).get('status')!='downloaded':
            futures.append(executor.submit(download_nli4ct2024))
        futures += [executor.submit(download_hf,*item,args.full_mega) for item in work if (args.full_mega and item[0]=='MEGA-CDP') or registry.get(item[0],{}).get('status') != 'downloaded' or not registry.get(item[0],{}).get('complete',False)]
        if not args.primary_only:
            futures += [executor.submit(download_aux,'ConFiQA')]
        for future in concurrent.futures.as_completed(futures):
            name, entry=future.result()
            current=json.loads(config_path.read_text()) if config_path.exists() else {}
            registry={**current,**registry,name:entry}
            config_path.write_text(json.dumps(registry,ensure_ascii=False,indent=2))
            print(name, entry['status'], flush=True)
    current=json.loads(config_path.read_text()) if config_path.exists() else {}
    registry={**registry,**current}
    registry['MedNLI']={'name':'MedNLI','source':SOURCES['MedNLI'],'status':'skipped','reason':'PhysioNet credentialed access, required training and data use agreement; no additional authorization requested.'}
    config_path.write_text(json.dumps(registry,ensure_ascii=False,indent=2))



def fetch_github_original(repo,branch,path,names=None,prefix=None):
    import base64
    import hashlib
    import requests
    session=requests.Session()
    session.headers['User-Agent']='TRIAGE-deployment'
    api=f'https://api.github.com/repos/{repo}'
    response=session.get(api+f'/git/trees/{branch}?recursive=1',timeout=(15,60))
    response.raise_for_status()
    tree=response.json()
    entries={item['path']:item for item in tree['tree'] if item['type']=='blob' and
             ((names is not None and item['path'] in names) or
              (prefix is not None and item['path'].startswith(prefix)))}
    if names is not None and set(names)-set(entries):
        raise FileNotFoundError(f'Official repository missing {set(names)-set(entries)}')
    for filename,item in entries.items():
        target=path/filename
        if target.exists() and target.stat().st_size>0:
            continue
        response=session.get(api+'/git/blobs/'+item['sha'],timeout=(15,120))
        response.raise_for_status()
        payload=response.json()
        data=base64.b64decode(payload['content'])
        if hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()!=item['sha']:
            raise ValueError(f'Incomplete original file: {filename}')
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix(target.suffix+'.api.partial')
        temporary.write_bytes(data)
        temporary.replace(target)
    return {'git_tree_sha':tree['sha'],'source_blob_sha':{k:v['sha'] for k,v in entries.items()}}


def download_nli4ct2024():
    """Fetch exact official 2024 files; verify the recorded official Git blob hashes."""
    import hashlib
    import requests
    path=ROOT/'data'/'NLI4CT-2024'
    expected={'training_data.zip':'ac3899e3cd920f75f87b209acd95300828443c54',
              'test.json':'dccbc2b7bd50ed6cfcbf32f062f1e7ae3818575f',
              'practice_test.json':'20800c2799f07d649f5c2a13f48c7ecfac5dfa68',
              'gold_test.json':'38de30f1e7b83fc9668a086a18ea0e05a38959a8',
              'gold_practice_test.json':'e075f02ec180555eda3ebae9999b0257319dfd53'}
    entry={'name':'NLI4CT-2024','year':2024,'source':'https://github.com/ai-systems/Task-2-SemEval-2024',
           'official_website':'https://sites.google.com/view/nli4ct/semeval-2024/get-data-and-starting-kit',
           'format':'original','path':str(path),'source_blob_sha':expected,
           'test_labels':'test.json has no Label; separate official gold_test.json retained and joined by original id'}
    path.mkdir(parents=True,exist_ok=True)
    try:
        for name,sha in expected.items():
            target=path/name
            if not target.exists():
                original='https://raw.githubusercontent.com/ai-systems/Task-2-SemEval-2024/main/'+name
                # Direct public mirror is transport only; the official hashes define the data.
                endpoint='https://ghfast.top/'+original
                entry['transport_mirror']='https://ghfast.top/'
                response=requests.get(endpoint,timeout=(15,120),stream=True)
                response.raise_for_status()
                temporary=target.with_suffix(target.suffix+'.mirror.partial')
                with temporary.open('wb') as handle:
                    for chunk in response.iter_content(65536):
                        handle.write(chunk)
                data=temporary.read_bytes()
                if hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()!=sha:
                    raise ValueError(f'Official Git blob hash mismatch: {name}')
                temporary.replace(target)
        if not (path/'.extracted').exists():
            with zipfile.ZipFile(path/'training_data.zip') as archive:
                archive.extractall(path/'original')
            (path/'.extracted').touch()
        entry.update(status='downloaded',complete=True,available_splits=['train','dev','test','practice_test'],
                     download_scope='all original 2024 splits, clinical trial JSONs and official gold files')
    except Exception as exc:
        entry.update(status='failed',reason=f'{type(exc).__name__}: {exc}',complete=False)
        # Keep any valid already-downloaded official split usable after a failed optional transfer.
        if (path/'training_data.zip').exists():
            with zipfile.ZipFile(path/'training_data.zip') as archive:
                archive.extractall(path/'original')
            entry.update(status='downloaded',available_splits=['train','dev'],download_scope='official training ZIP complete; standalone files partially downloaded')
    return 'NLI4CT-2024',entry


if __name__=="__main__":
    main()
