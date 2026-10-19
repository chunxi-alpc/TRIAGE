import pathlib,json,hashlib,requests,subprocess,datetime
root=pathlib.Path('.')
manifest=json.loads((root/'results/manifest.json').read_text())
s=requests.Session(); s.proxies={'https':'http://127.0.0.1:28089'}
r=s.get('https://huggingface.co/api/models/Qwen/Qwen3-4B-Base',params={'blobs':'true'},timeout=40)
r.raise_for_status(); meta=r.json()
(root/'results/model_source_metadata.json').write_text(json.dumps(meta,indent=2))
expected={x['rfilename']:x.get('lfs',{}).get('sha256') for x in meta['siblings'] if x['rfilename'].endswith('.safetensors')}
hashes={}
for path in (root/'models/Qwen3-4B-Base').glob('*.safetensors'):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''): h.update(b)
    value=h.hexdigest(); assert expected[path.name]==value,(path.name,value,expected[path.name])
    hashes[path.name]={'sha256':value,'bytes':path.stat().st_size,'huggingface_lfs_match':True}
manifest['model_revision']=meta['sha']
manifest['weights']=hashes
manifest['weight_download']='ModelScope mirror; each shard SHA256 independently matched official Hugging Face LFS at recorded revision'
manifest['source_audit']=json.loads((root/'results/source_audit.json').read_text())
manifest['finished_at_beijing']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
manifest['artifact_hashes']={}
for folder in ['scripts','tests','configs','data/derived','figures','results/obs1','results/obs2','results/obs3','results/projections']:
    for path in sorted((root/folder).rglob('*')):
        if path.is_file() and '__pycache__' not in str(path):
            manifest['artifact_hashes'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
(root/'results/manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
(root/'results/logs/environment.txt').write_text(subprocess.check_output(['python','-m','pip','freeze'],text=True))
summary=root/'results/observations_summary.md'
text=summary.read_text()
# Report identity from independently checked finalized manifest.
start=text.index('模型 revision: '); end=text.index('\nSEKA commit:',start)
text=text[:start]+f"模型 revision: {meta['sha']}（全部三份权重SHA-256与官方Hugging Face LFS逐一匹配）。"+text[end:]
summary.write_text(text)
print('Model identity and artifact checksums verified.')
