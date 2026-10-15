import requests,pathlib,json,hashlib,time,concurrent.futures
base='https://modelscope.cn/api/v1/models/Qwen/Qwen3-4B-Base/repo/files'
files=requests.get(base,params={'Revision':'master','Recursive':'true'},timeout=30).json()['Data']['Files']
pathlib.Path('results/modelscope_files.json').write_text(json.dumps(files,indent=2))
p=pathlib.Path('models/Qwen3-4B-Base')
def download(f):
    name=f['Name']
    if not name.endswith('.safetensors'): return
    dest=p/name; tmp=p/(name+'.partial')
    u='https://modelscope.cn/api/v1/models/Qwen/Qwen3-4B-Base/repo'
    t=time.time()
    with requests.get(u,params={'Revision':'master','FilePath':name},stream=True,timeout=(30,120)) as r:
        r.raise_for_status()
        h=hashlib.sha256(); total=0
        with tmp.open('wb') as out:
            for b in r.iter_content(4*1024*1024):
                out.write(b); h.update(b); total+=len(b)
    assert total==f['Size'],(total,f['Size'])
    assert h.hexdigest()==f['Sha256']
    tmp.rename(dest)
    print(name,total,'sha256 verified',time.time()-t,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
    list(ex.map(download,files))
print('COMPLETE',flush=True)
