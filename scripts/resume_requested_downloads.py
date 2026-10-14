#!/usr/bin/env python3
"""Resume requested TRIAGE resources without changing the environment.
All content is pinned by official Git blob SHA or LFS SHA256. Logs stay in results/.
"""
import argparse, base64, concurrent.futures as cf, fcntl, hashlib, json, os, threading, time
from pathlib import Path
from urllib.parse import quote
import requests
from download_network import install_resolution_cache
install_resolution_cache()
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'
TLS=threading.local()
LOCKS={}
LOCK_GUARD=threading.Lock()

def session():
    if not hasattr(TLS,'session'):
        TLS.session=requests.Session()
        TLS.session.headers.update({'User-Agent':'TRIAGE-resource-resume','Accept-Encoding':'identity'})
    return TLS.session

def atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    tmp.replace(path)

def register(name,entry,model=False):
    config=ROOT/'configs'/('models.json' if model else 'datasets.json')
    with config.with_suffix('.resume.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        d=json.loads(config.read_text())
        bucket=d.setdefault('models',{}) if model else d
        bucket.setdefault(name,{}).update(entry)
        if entry.get('status')=='downloaded':
            bucket[name].pop('reason',None)
        atomic(config,d)

def metadata(repo,model=False,revision=None):
    file=RESULTS/('source_'+('medical_nli' if model else repo.replace('/','__'))+'.json')
    if file.exists():
        d=json.loads(file.read_text())
        if not revision or d['sha']==revision:return d
    api='models' if model else 'datasets'
    url=f'https://hf-mirror.com/api/{api}/{repo}'
    if revision:url+='/revision/'+revision
    r=session().get(url,params={'blobs':'true'},timeout=(10,40));r.raise_for_status()
    d=r.json();atomic(file,d);return d

def valid(path,item):
    if not path.is_file() or path.stat().st_size!=item['size']:return False
    lfs=item.get('lfs')
    digest=hashlib.sha256() if lfs else hashlib.sha1(f"blob {item['size']}\0".encode())
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()==(lfs['sha256'] if lfs else item.get('blobId',item.get('sha')))

def fetch(url,path,item,refresh=False,headers=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    part=path.with_name(path.name+'.incomplete')
    if path.exists():
        if valid(path,item):return 'reused'
        path.rename(path.with_name(path.name+'.invalid-'+str(time.time_ns())))
    for attempt in range(8):
        try:
            offset=part.stat().st_size if part.exists() else 0
            if offset==item['size']:
                if valid(part,item):part.replace(path);return 'resumed'
                part.rename(part.with_name(part.name+'.invalid-'+str(time.time_ns())));offset=0
            if offset>item['size']:
                part.rename(part.with_name(part.name+'.invalid-'+str(time.time_ns())));offset=0
            h=dict(headers or {})
            if offset:h['Range']=f'bytes={offset}-'
            params={'triage_refresh':time.time_ns()} if refresh else None
            with session().get(url,params=params,headers=h,stream=True,timeout=(10,30)) as response:
                response.raise_for_status()
                if offset and response.status_code==206:
                    if not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                        raise ValueError('invalid range offset')
                    mode='ab'
                elif response.status_code==200:
                    mode='wb'
                else:raise ValueError(f'unexpected response {response.status_code}')
                with part.open(mode) as f:
                    for chunk in response.iter_content(256*1024):
                        if chunk:f.write(chunk)
            if not valid(part,item):raise ValueError('file size or official digest mismatch')
            part.replace(path);return 'downloaded'
        except Exception as e:
            if attempt==7:raise
            print(json.dumps({'retry':path.name,'attempt':attempt+1,'error':type(e).__name__}),flush=True)
            time.sleep(min(2*(attempt+1),15))

def hf_paths(repo,d,model=False):
    if model:return ROOT/'models'/repo.split('/')[-1],None
    cache=ROOT/'data'/'.hf_cache'/('datasets--'+repo.replace('/','--'))
    return cache/'snapshots'/d['sha'],cache

def hf_one(repo,d,item,model=False,direct=None):
    dest,cache=hf_paths(repo,d,model)
    name=item['rfilename'];target=dest/name
    digest=item.get('lfs',{}).get('sha256',item['blobId'])
    key=str(cache or dest)+digest
    with LOCK_GUARD:lock=LOCKS.setdefault(key,threading.Lock())
    with lock:
        if valid(target,item):return 'reused'
        payload=(cache/'blobs'/digest) if cache else target
        if model and not payload.exists() and not payload.with_name(payload.name+'.incomplete').exists():
            candidates=list((dest/'.cache'/'huggingface'/'download').glob('*.'+digest+'.incomplete'))
            if candidates:
                max(candidates,key=lambda p:p.stat().st_size).rename(payload.with_name(payload.name+'.incomplete'))
        prefix='' if model else 'datasets/'
        url=f"https://hf-mirror.com/{prefix}{repo}/resolve/{d['sha']}/{quote(name,safe='/')}"
        result=fetch(direct['href'] if direct else url,payload,item,refresh=bool(item.get('lfs')) and not direct,headers=direct.get('header') if direct else None)
        if cache:
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.is_symlink() or target.exists():target.unlink()
            target.symlink_to(os.path.relpath(payload,target.parent))
        return result

def hf_download(name,repo,selector=None,model=False,workers=6):
    d=metadata(repo,model,revision=None)
    dest,cache=hf_paths(repo,d,model)
    items=[i for i in d['siblings'] if selector is None or selector(i)]
    source=f"https://huggingface.co/{'' if model else 'datasets/'}{repo}"
    entry={'source':source,'repo_id':repo,'revision':d['sha'],'path':str(dest),'endpoint':'https://hf-mirror.com','format':'original','status':'downloading'}
    register(name,entry,model)
    counts={}
    errors=[]
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures={ex.submit(hf_one,repo,d,i,model):i for i in items}
        for future in cf.as_completed(futures):
            item=futures[future]
            try:
                state=future.result();counts[state]=counts.get(state,0)+1
            except Exception as e:errors.append({'file':item['rfilename'],'error':str(e)})
    if errors:
        atomic(RESULTS/(name+'_resume_errors.json'),errors)
        register(name,{'status':'failed','reason':f'{len(errors)} files failed; see results/{name}_resume_errors.json'},model)
        raise RuntimeError(f'{name}: {len(errors)} files failed')
    link=ROOT/'data'/name
    if not model and not link.exists():link.symlink_to(dest,target_is_directory=True)
    entry.update(status='downloaded',complete=True,download_scope='all original files' if selector is None else 'selected official files')
    if model:
        entry.update(task='nli',role='module1_medical_nli_baseline',weight_bytes=sum(i['size'] for i in items if i['rfilename'].endswith('.safetensors')),id2label=json.loads((dest/'config.json').read_text())['id2label'])
    elif name=='MEGA-CDP':
        entry.update(complete=True,available_splits=['test','train_val'],download_scope='all original train_val/test JSONL and Markdown verified; images excluded by user request',contains_all_original_files=False,images_requested=False,markdown_files_downloaded=2275,markdown_files_expected=2275)
    elif name=='MedCalc-Bench':entry.update(available_splits=['train','test'])
    elif name=='FaithEval-unanswerable':entry.update(available_splits=['test'])
    elif name=='ConflictMedQA':entry.update(data_files=['ConflictMed.jsonl','ConflictMed_v2.jsonl'],auxiliary_files=['advice_pairs.jsonl'],available_splits=['original','v2'],split_files={'original':['ConflictMed.jsonl'],'v2':['ConflictMed_v2.jsonl']})
    register(name,entry,model)
    print(json.dumps({'complete':name,'counts':counts}),flush=True)
    return entry

def git_download(name,repo,branch,names,metadata_file=None):
    meta=RESULTS/(metadata_file or 'source_'+repo.replace('/','__')+'.json')
    d=json.loads(meta.read_text())
    if d.get('truncated'):raise ValueError('truncated official tree')
    selected=[i for i in d['tree'] if i['type']=='blob' and i['path'] in names]
    if len(selected)!=len(names):raise ValueError('missing requested official files')
    dest=ROOT/'data'/name
    hashes={}
    for item in selected:
        target=dest/item['path']
        if not valid(target,item):
            raw=f"https://raw.githubusercontent.com/{repo}/{branch}/{quote(item['path'],safe='/')}"
            try:fetch(raw,target,item)
            except Exception:
                response=session().get(item['url'],timeout=(10,90));response.raise_for_status()
                payload=base64.b64decode(response.json()['content'])
                temporary=target.with_name(target.name+'.api.incomplete');temporary.parent.mkdir(parents=True,exist_ok=True);temporary.write_bytes(payload)
                if not valid(temporary,item):raise ValueError('official Git blob mismatch')
                temporary.replace(target)
        hashes[item['path']]=item['sha']
    entry={'name':name,'source':f'https://github.com/{repo}','path':str(dest),'format':'original','status':'downloaded','complete':True,'git_tree_sha':d['sha'],'source_blob_sha':hashes,'transport_mirror':'https://ghfast.top/','download_scope':'requested official data files and README'}
    if name=='ConFiQA':entry.update(available_splits=['QA','MR','MC'])
    elif name=='MedChangeQA':entry.update(data_files=['Datasets/MedChangeQA.csv'],split_files={'test':['Datasets/MedChangeQA.csv']},available_splits=['test'])
    elif name=='ClashEval':entry.update(data_files=['data/dataset/dataset.pqt'],available_splits=['test'],split_files={'test':['data/dataset/dataset.pqt']},revision=branch,download_scope='original dataset.pqt from official repository history; current HF dataset unavailable')
    register(name,entry)
    print(json.dumps({'complete':name,'files':len(selected)}),flush=True)
    return entry

def nli_verify():
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    start=time.time()
    path=ROOT/'models'/'PubMedBERT-MNLI-MedNLI'
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(path,local_files_only=True).to('cuda:0').eval()
    samples=[
       ('The patient reports chest pain.','The patient has chest pain.','entailment'),
       ('The patient has no fever.','The patient is febrile.','contradiction'),
       ('The patient reports chest pain.','The patient has diabetes.','neutral')]
    inputs=tokenizer([x[0] for x in samples],[x[1] for x in samples],padding=True,truncation=True,max_length=128,return_tensors='pt').to('cuda:0')
    with torch.inference_mode():logits=model(**inputs).logits.float()
    assert logits.shape==(3,3) and torch.isfinite(logits).all()
    labels=model.config.id2label
    assert set(labels.values())=={'entailment','contradiction','neutral'}
    predictions=[labels[i] for i in logits.argmax(-1).tolist()]
    report={'status':'passed' if predictions==[x[2] for x in samples] else 'failed','path':str(path),'device':'GPU 0','id2label':labels,'samples':[{'premise':s[0],'hypothesis':s[1],'expected':s[2],'predicted':p,'probabilities':prob} for s,p,prob in zip(samples,predictions,logits.softmax(-1).cpu().tolist())],'max_allocated_mib':round(torch.cuda.max_memory_allocated()/1024**2,2),'seconds':round(time.time()-start,2),'scope':'three short synthetic examples; no training or benchmark evaluation'}
    atomic(RESULTS/'medical_nli_smoke.json',report)
    register('medical_nli',{'verification':report['status'],'verification_file':str(RESULTS/'medical_nli_smoke.json'),'baseline_module':1},True)
    del model,inputs,logits
    torch.cuda.empty_cache()
    print(json.dumps({'medical_nli_validation':report['status'],'predictions':predictions}),flush=True)
    if report['status']!='passed':raise ValueError('three example predictions did not match expected labels')

def core():
    tasks=[
      lambda:git_download('ClashEval','kevinwu23/StanfordClashEval','ba892d17f9b125264494689b7ce5d8ab22ed821d',['data/dataset/dataset.pqt','README.md','data/dataset/croissant.json'],metadata_file='source_ClashEval_history.json'),
      lambda:hf_download('medical_nli','pritamdeka/PubMedBERT-MNLI-MedNLI',lambda i:i['rfilename'] in {'config.json','tokenizer_config.json','tokenizer.json','special_tokens_map.json','vocab.txt','README.md','model.safetensors'},True,2),
      lambda:hf_download('MedCalc-Bench','ncbi/MedCalc-Bench'),
      lambda:hf_download('FaithEval-unanswerable','Salesforce/FaithEval-unanswerable-v1.0'),
      lambda:hf_download('ConflictMedQA','RDBH/DriftMed'),
      lambda:git_download('ConFiQA','byronBBL/Context-DPO','master',['ConFiQA/ConFiQA-QA.json','ConFiQA/ConFiQA-MR.json','ConFiQA/ConFiQA-MC.json','README.md']),
      lambda:git_download('MedChangeQA','jvladika/MedChange','main',['Datasets/MedChangeQA.csv','README.md']),
      lambda:hf_download('MEGA-CDP','nchennnn/MEGA-CDP',lambda i:'/imgs/' not in i['rfilename'],workers=12),
    ]
    summary={'pid':os.getpid(),'started':time.time(),'status':'running','tasks':[]}
    atomic(RESULTS/'resume_download_status.json',summary)
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        for f in cf.as_completed([ex.submit(t) for t in tasks]):
            try:
                entry=f.result();summary['tasks'].append({'name':entry.get('name',entry['repo_id'] if 'repo_id' in entry else entry['source']),'status':'downloaded'})
            except Exception as e:summary['tasks'].append({'status':'failed','reason':str(e)})
            atomic(RESULTS/'resume_download_status.json',summary)
    try:nli_verify()
    except Exception as e:summary['tasks'].append({'name':'medical_nli_validation','status':'failed','reason':str(e)})
    summary['status']='completed' if all(x['status']!='failed' for x in summary['tasks']) else 'failed'
    summary['finished']=time.time()
    atomic(RESULTS/'resume_download_status.json',summary)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['core'],default='core')
    args=parser.parse_args()
    with (RESULTS/('resume_'+args.stage+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        core()

