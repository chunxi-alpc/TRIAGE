"""Resume identical Phi-4 weight shards from ModelScope, verifying SHA-256.

The ModelScope file listing supplies SHA-256; partial HF files are reused only
when their filename already contains exactly that digest. No model substitution.
"""
from __future__ import annotations
import concurrent.futures, hashlib, json, os, pathlib, time
import requests

ROOT = pathlib.Path(os.environ.get('MODEL_DIR', '/home/gcx/models/care_module1_v5')) / 'phi-4'
REPO = 'LLM-Research/phi-4'
API = f'https://modelscope.cn/api/v1/models/{REPO}/repo/files?Revision=master&Recursive=true'

def session():
    s = requests.Session(); s.trust_env = False
    return s

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(16*1024*1024), b''): h.update(b)
    return h.hexdigest()

def download(item):
    name, size, expected = item['Name'], int(item['Size']), item['Sha256']
    target = ROOT / name
    if target.is_file() and target.stat().st_size == size and digest(target) == expected:
        return {'name': name, 'bytes': size, 'sha256': expected, 'status': 'reused_complete'}
    partials = list((ROOT / '.cache/huggingface/download').glob(f'*.{expected}.incomplete'))
    partial = max(partials, key=lambda p:p.stat().st_size) if partials else ROOT / (name + '.modelscope.partial')
    partial.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 6):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == size: break
        if offset > size: raise RuntimeError(f'partial is larger than expected: {name}')
        print(json.dumps({'name': name, 'resume_bytes': offset, 'expected': size, 'attempt':attempt}), flush=True)
        try:
            with session().get(f'https://modelscope.cn/models/{REPO}/resolve/master/{name}',
                    headers={'Range': f'bytes={offset}-'}, stream=True, timeout=(15, 90)) as response:
                response.raise_for_status()
                if offset and (response.status_code != 206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')):
                    raise RuntimeError(f'Unexpected resume response for {name}: {response.status_code} {response.headers.get("Content-Range")}')
                if not offset and response.status_code not in (200,206):
                    raise RuntimeError(f'Unexpected response for {name}: {response.status_code}')
                with partial.open('ab') as out:
                    for chunk in response.iter_content(4*1024*1024):
                        out.write(chunk)
                    out.flush(); os.fsync(out.fileno())
        except Exception as error:
            print(json.dumps({'name':name,'retry_error':f'{type(error).__name__}: {error}'}),flush=True)
            if attempt == 5: raise
            time.sleep(3)
    if partial.stat().st_size != size or digest(partial) != expected:
        raise RuntimeError(f'SHA-256 or size mismatch for {name}; preserved partial for inspection')
    partial.replace(target)
    print(json.dumps({'name':name,'status':'verified','bytes':size}),flush=True)
    return {'name':name,'bytes':size,'sha256':expected,'status':'resumed_and_verified'}

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    r=session().get(API,timeout=(15,60)); r.raise_for_status(); metadata=r.json()
    if metadata.get('Code') != 200: raise RuntimeError(metadata.get('Message'))
    items=[x for x in metadata['Data']['Files'] if x['Name'].endswith('.safetensors')]
    if len(items)!=6: raise RuntimeError(f'Expected six Phi-4 shards; got {len(items)}')
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        files=list(pool.map(download,items))
    result={'source':'https://modelscope.cn/models/LLM-Research/phi-4','model':'microsoft/phi-4',
            'identity':'each shard SHA-256 verified, matches reused Hugging Face object digest', 'files':files}
    (ROOT/'modelscope_download_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'completed','files':len(files)}),flush=True)

if __name__=='__main__': main()
