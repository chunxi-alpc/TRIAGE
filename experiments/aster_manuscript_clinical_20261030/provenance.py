"""Download an explicitly pinned ModelScope snapshot; no local file hashes."""
import argparse,concurrent.futures,json,time,urllib.request,urllib.parse
from pathlib import Path
def getjson(url):
 with urllib.request.urlopen(url,timeout=60) as r:return json.load(r)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
 root=Path(a.output);root.mkdir(exist_ok=False,parents=True);model='Qwen/Qwen3-8B';revision='26028140be3ee69b82b1d1450179ab71bb1121b9'
 api=f'https://modelscope.cn/api/v1/models/{model}/repo/files?'+urllib.parse.urlencode({'Revision':revision,'Recursive':'true'})
 meta=getjson(api);files=meta['Data']['Files'];(root/'source_files.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
 needed=[f for f in files if f['Type']=='blob' and (f['Path'].endswith(('.json','.txt','.safetensors')) or f['Path'] in ['README.md','LICENSE'])]
 start=time.time();completed=[];failures=[]
 def download(f):
  target=root/f['Path'];target.parent.mkdir(parents=True,exist_ok=True);part=Path(str(target)+'.partial')
  url=f'https://modelscope.cn/api/v1/models/{model}/repo?'+urllib.parse.urlencode({'Revision':revision,'FilePath':f['Path']})
  last=None
  for attempt in range(3):
   try:
    with urllib.request.urlopen(url,timeout=120) as response,part.open('wb') as out:
     while True:
      chunk=response.read(4*1024*1024)
      if not chunk:break
      out.write(chunk)
    if part.stat().st_size!=f['Size']:raise ValueError('incomplete HTTP transfer')
    part.rename(target)
    return {'path':f['Path'],'bytes':f['Size'],'file_source_revision':f['Revision']}
   except Exception as e:last=e
  raise last
 with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
  futures={pool.submit(download,f):f for f in needed}
  for future in concurrent.futures.as_completed(futures):
   f=futures[future]
   try:completed.append(future.result())
   except Exception as e:failures.append({'file':f['Path'],'error':str(e)})
   (root/'download_status.json').write_text(json.dumps({'status':'running','completed':completed,'failures':failures},indent=2))
 record={'status':'complete' if not failures else 'failed','model_id':model,'snapshot_revision':revision,'source_api':api,'downloaded_from_pinned_revision':True,
  'completed':completed,'failures':failures,'elapsed_seconds':time.time()-start,'local_hashes_computed':False,'previous_snapshot_untouched':True}
 (root/'download_status.json').write_text(json.dumps(record,indent=2))
 if failures:raise RuntimeError('pinned download failed; see status')
if __name__=='__main__':main()

