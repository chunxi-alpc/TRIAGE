"""Private UTF-8 CSV forms, independent blinded reviewer sheets, Cohen kappa."""
import argparse,collections,csv,hashlib,json
from pathlib import Path
import pyarrow.parquet as pq
P=Path('/home/gcx/aster_e0_e8_20261024/private')
FIELDS=['来源一致','条件适用','构造有效','答案正确']
def build():
 rows=list(map(json.loads,(P/'cases.jsonl').read_text().splitlines()));source=pq.read_table('/home/gcx/aster_medguide_20261022/private/train.parquet').to_pylist();paths=json.loads(Path('/home/gcx/aster_medguide_20261022/private/all_paths.json').read_text());selected=[];used=set()
 for v,n in [('CS',24),('CS-null',24),('NC',26),('AC',26),('IC',25),('SW',25)]:
  pool=sorted([r for r in rows if r['seed']==0 and r['variant']==v],key=lambda r:hashlib.sha256((r['id']+v).encode()).hexdigest())
  for r in pool:
   if r['id'] in used:continue
   selected.append(r);used.add(r['id'])
   if sum(x['variant']==v for x in selected)>=n:break
 assert len(selected)==150 and len(used)==150
 # Reviewers receive source and constructed task but no model predictions.
 sheets=[]
 for r in selected:
  i=int(r['id'].split('_')[-1]);raw=source[i]
  sheets.append({'编号':r['id'],'卷子':r['variant'],'树':r['tree'],'病历':r['patient'],'原始路径':json.dumps(raw['path'],ensure_ascii=False),'原始全部路径':json.dumps(paths[r['tree']],ensure_ascii=False),'原始答案':raw.get('answer_text',raw['path'][-1]),'实验规则':'\n'.join(f"R{x['rid']+1}: {x['condition']} => {x['action']}" for x in r['rules']),'实验选项':'\n'.join(f'{chr(65+j)}. {x}' for j,x in enumerate(r['options'])),'拟定适用规则':f"R{r['gold_rule']+1}",'拟定答案':chr(65+r['gold_option']),'阈值证据':json.dumps(r['threshold_proof'],ensure_ascii=False),**{f:'' for f in FIELDS},'建议适用规则':'','建议答案':'','问题说明':''})
 for reviewer in ['A','B']:
  with (P/f'annotation_{reviewer}.csv').open('w',encoding='utf-8-sig',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(sheets[0]));w.writeheader();w.writerows(sheets)
 (P/'annotation_manifest.json').write_text(json.dumps({'rows':150,'distinct_patients':150,'variants':dict(collections.Counter(r['variant'] for r in selected)),'ids':[r['id']+':'+r['variant'] for r in selected]},indent=2))
def score(a,b,out):
 def read(p):
  with open(p,encoding='utf-8-sig') as f:return {(r['编号'],r['卷子']):r for r in csv.DictReader(f)}
 aa,bb=read(a),read(b);assert aa.keys()==bb.keys() and len(aa)>=150
 result={};disagreements=[]
 for field in FIELDS:
  pairs=[(aa[k][field].strip(),bb[k][field].strip()) for k in aa];assert all(x in ['是','否','无法判断'] and y in ['是','否','无法判断'] for x,y in pairs),'Incomplete or invalid ratings'
  n=len(pairs);observed=sum(x==y for x,y in pairs)/n;ca=collections.Counter(x for x,y in pairs);cb=collections.Counter(y for x,y in pairs);chance=sum(ca[t]*cb[t] for t in set(ca)|set(cb))/n**2;kappa=(observed-chance)/(1-chance) if chance<1 else None
  result[field]={'n':n,'agreement':observed,'kappa':kappa,'pass':kappa is not None and kappa>=.8}
  disagreements.extend({'id':k[0],'variant':k[1],'field':field,'A':aa[k][field],'B':bb[k][field]} for k in aa if aa[k][field]!=bb[k][field] or aa[k][field]!='是')
 Path(out).write_text(json.dumps({'metrics':result,'adjudication_required':disagreements,'note':'Do not edit independent ratings; adjudicate separately. Degenerate kappa is undefined, never force to 1.'},ensure_ascii=False,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--a');p.add_argument('--b');p.add_argument('--out');a=p.parse_args()
 if a.a:score(a.a,a.b,a.out)
 else:build()
