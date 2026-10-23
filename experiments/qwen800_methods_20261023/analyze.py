"""Complete-run K/U reports and explicitly unpaired comparisons to original 200."""
import collections,gzip,hashlib,importlib.util,json,math
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parents[2];EXP=Path(__file__).parent;OUT=EXP/'results';PRIVATE=Path('/home/gcx/qwen800_methods_20261023/private')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return list(map(json.loads,Path(p).read_text().splitlines()))
def ci(rows,values=None):
 groups=collections.defaultdict(list)
 for j,r in enumerate(rows):groups[r['disease']].append(float(r['correct']) if values is None else values[j])
 if len(groups)<2:return None
 vals=list(groups.values());s=np.array([sum(v) for v in vals]);n=np.array([len(v) for v in vals]);ix=np.random.default_rng(20261023).integers(len(vals),size=(2000,len(vals)));return np.quantile(s[ix].sum(1)/n[ix].sum(1),[.025,.975]).tolist()
def cell(rs):
 n=len(rs);return {'n':n,'correct':sum(r['correct'] for r in rs),'accuracy':sum(r['correct'] for r in rs)/n,'disease_ci':ci(rs),'parse_failures':sum(r.get('parse_failure',r['prediction'] is None) for r in rs),'original_action_rate':sum(r.get('original_action_selected',False) for r in rs)/n,'branch_permitted_rate':sum(r.get('branch_permitted_action',False) for r in rs)/n}
def paired(a,b):
 bb={r['id']:r for r in b};aa=[r for r in a if r['id'] in bb];d=[int(bb[r['id']]['correct'])-int(r['correct']) for r in aa];wins=d.count(1);loss=d.count(-1)
 return {'n':len(aa),'delta':sum(d)/len(d) if d else None,'disease_ci':ci(aa,d),'wins':wins,'losses':loss,'p':float(binomtest(wins,wins+loss,.5).pvalue) if wins+loss else 1.}
def main():
 config=json.loads((OUT/'dataset_manifest.json').read_text());cases={c['id']:c for c in read(PRIVATE/'cases.jsonl')};rows=[]
 for k in range(2):
  p=PRIVATE/f'predictions{k}.jsonl';m=json.loads(p.with_suffix('.manifest.json').read_text());assert m['complete'] and m['input_sha256']==sha(PRIVATE/f'shard{k}.jsonl') and m['output_sha256']==sha(p);rows+=read(p)
 keys=[(r['id'],r['variant'],r['method']) for r in rows];expected={(c['id'],v,m) for c in cases.values() for v in c['variant_rules'] for m in config['methods']+config['branch_methods']};assert len(keys)==len(set(keys)) and set(keys)==expected and len(keys)==config['expected_rows']
 buckets=collections.defaultdict(list)
 for r in rows:
  assert r['stratum']==cases[r['id']]['stratum']
  scopes=['all']+(['PR23_strict'] if r['pair_protocol']=='PR23_strict' else [])+(['PR23_strict_single'] if r['pair_protocol']=='PR23_strict' and r['single_action_gold_branch'] else [])
  for scope in scopes:
   for st in ['all',r['stratum']]:buckets[scope,r['variant'],r['method'],st].append(r)
 cells={'/'.join(k):cell(v) for k,v in buckets.items()};comparisons={}
 for scope in ['all','PR23_strict','PR23_strict_single']:
  for v in ['A','B','C','D','CS']:
   for st in ['all','K','U']:
    for ref,method in [('existing_baseline','hidden_prune'),('visible_prune','hidden_prune'),('CAD128','CAD_filtered128')]:
     a=buckets.get((scope,v,ref,st),[]);b=buckets.get((scope,v,method,st),[])
     if a and b:comparisons['/'.join([scope,v,st,ref+'->'+method])]=paired(a,b)
 tests=sorted(comparisons.values(),key=lambda r:r['p']);prev=0
 for j,r in enumerate(tests):prev=max(prev,min(1,r['p']*(len(tests)-j)));r['p_holm']=prev
 olddir=ROOT/'experiments/aster_validation_20261022/mechanism_20261023/results';oldrows=read(olddir/'priority200/qwen/full_parser_v2/scores.jsonl');cadrows=read(olddir/'cad200_final/cad_scores.jsonl')
 for r in cadrows:
  if r['method']=='cad' and not r.get('assigned_wrong') and r['revision'] in ['CAD8','CAD128','CAD_filtered_original']:
   oldrows.append({**r,'method':{'CAD_filtered_original':'CAD_filtered128'}.get(r['revision'],r['revision']),'variant':r['variant'][0],'kind':'action','parse_failure':r['prediction'] is None})
 oldb=collections.defaultdict(list)
 for r in oldrows:
  for st in ['all',r['stratum']]:oldb[r['variant'],r['method'],st].append(r)
 oldcells={'/'.join(k):cell(v) for k,v in oldb.items()};between={};overlap={}
 for scope in ['all','PR23_strict','PR23_strict_single']:
  for (v,m,st),oo in oldb.items():
   nn=buckets.get((scope,v,m,st),[])
   if nn:between['/'.join([scope,v,m,st])]={'new_n':len(nn),'old_n':len(oo),'new_accuracy':sum(r['correct'] for r in nn)/len(nn),'old_accuracy':sum(r['correct'] for r in oo)/len(oo),'difference':sum(r['correct'] for r in nn)/len(nn)-sum(r['correct'] for r in oo)/len(oo),'comparison':'descriptive unpaired cohort comparison; different sampling and K/U runs; CS threshold protocol differs'}
 for (v,m,st),oo in oldb.items():
  if st!='all' or v=='CS':continue
  nn=buckets.get(('PR23_strict',v,m,'all'),[]);common={r['id'] for r in oo}&{r['id'] for r in nn}
  if common:
   aa=[r for r in oo if r['id'] in common];bb=[r for r in nn if r['id'] in common];overlap[v+'/'+m]={'paired':paired(aa,bb),'old_vs_new_K_disagreements':sum(r['stratum']!=cases[r['id']]['stratum'] for r in aa),'note':'identical original IDs and strict construction; old K/U versus PR24 frozen K/U recorded separately'}
 harmonic={};lookup={(r['id'],r['variant'],r['method']):r for r in rows}
 for st in ['K','U']:
  for m in config['methods']:
   for vs in [('B','C','D'),('B','C','D','CS')]:
    ids=[i for i,c in cases.items() if c['stratum']==st and all((i,v,m) in lookup for v in vs)]
    acc=[sum(lookup[i,v,m]['correct'] for i in ids)/len(ids) for v in vs] if ids else []
    harmonic['/'.join([st,m,''.join(vs)])]={'n_common':len(ids),'value':len(acc)/sum(1/x for x in acc) if acc and min(acc)>0 else 0 if acc else None,'tree_count':len({cases[i]['disease'] for i in ids}),'descriptive':True}
 summary={'complete':True,'cases':len(cases),'K':sum(c['stratum']=='K' for c in cases.values()),'U':sum(c['stratum']=='U' for c in cases.values()),'rows':len(rows),'variant_counts':config['variants'],'cells':cells,'paired_comparisons':comparisons,'comparison_family':'all scopes/variants/K-U comparisons in paired_comparisons jointly Holm-corrected','old200_cells':oldcells,'old200_descriptive_comparison':between,'overlap200_paired_reproduction':overlap,'harmonic':harmonic,'scope_notes':['A/B/C 800; D 797 with 3 structurally unavailable; CS 84.','K/U frozen from PR24; no resampling based on results.','Strict paired subset and strict single-action subset separated from extended construction.','CS source-path conditioned, one tree only; no multi-disease CI.','Original 200 comparison is descriptive; no causal conclusion from cohort differences.','No option rotations, gain sweeps, random wrong-branch/noise sweeps or attention-blocking experiment in this method comparison.']}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));safe=[{k:v for k,v in r.items() if k not in ('completion','option_logits','ranked')} for r in rows];(OUT/'scores.jsonl.gz').write_bytes(gzip.compress(''.join(json.dumps(r)+'\n' for r in safe).encode(),mtime=0))
 names={'existing_baseline':'普通基线','existing_prune':'金标单规则裁剪','existing_app':'金标注意力增强','existing_signed':'金标带符号引导','gold_branch_prune':'金标分支组裁剪','visible_prune':'可见预测＋裁剪','hidden_prune':'隐藏预测＋裁剪','hidden_top2':'隐藏预测＋Top-2','hidden_signed':'隐藏预测＋带符号引导','hidden_unsigned':'隐藏预测＋无符号引导','CAD8':'CAD8','CAD128':'CAD128','CAD_filtered128':'过滤CAD128'}
 lines=['# Qwen 800 例方法对比','全部推理与统计完成。固定 PR24 的 800 例、K426/U374；A/B/C 各 800，D797，CS84。D 缺失的 3 例原因见 case_manifest.jsonl。','## 构造与可比性','304 例满足 PR23 原构造；其余采用同树最近共同前缀的其他条件分支，二选一题 C 改为原动作。原协议、原协议且单动作分支子集独立统计，避免把扩展构造混作原协议复现。K/U 沿用 PR24；本轮推理恢复 PR23 eager，增益 .5，主方法 8 token、CAD128、过滤 beta=.1。','原 200 例本身是单动作分支子集，与当前抽样分布不同；跨队列差值只作描述。重叠 19 例另列配对复现。CS 当前按患者值选择阈值，原实验固定 6→12，不能把 84 例与 2 例视作相同协议的独立效果检验。']
 def table(scope,st):
  lines.extend([f'## {scope} / {st}','|方法|A|B|C|D|CS|','|---|---:|---:|---:|---:|---:|'])
  for m in config['methods']:
   vals=[]
   for v in ['A','B','C','D','CS']:
    x=cells.get('/'.join([scope,v,m,st]));vals.append(f"{x['accuracy']*100:.1f}% ({x['correct']}/{x['n']})" if x else 'NA')
   lines.append('|'+names[m]+'|'+'|'.join(vals)+'|')
 for st in ['all','K','U']:table('all',st)
 for scope in ['PR23_strict','PR23_strict_single']:
  for st in ['K','U']:table(scope,st)
 lines+=['## 与原 200 例比较','表中为当前全样本准确率减去原 200 例准确率（百分点），不是同分布效果估计。','|方法|组别|A|B|C|D|CS|','|---|---|---:|---:|---:|---:|---:|']
 for st in ['all','K','U']:
  for m in config['methods']:
   vals=[]
   for v in ['A','B','C','D','CS']:
    x=between.get('/'.join(['all',v,m,st]));vals.append(f"{100*x['difference']:+.1f}" if x else 'NA')
   lines.append('|'+names[m]+'|'+st+'|'+'|'.join(vals)+'|')
 lines+=['## 统计与文件','解析失败计错，逐题保留；方法比较使用同一病例，报告精确 McNemar、按树重采样 2000 次区间、全比较族 Holm 校正。CS 只有一棵树，不生成多疾病区间。K/U 的 BCD 和 BCDCS 共同病例调和平均见 summary.json。','summary.json 包含原 200 例各 K/U 单元、全样本和严格子集描述性差值、重叠病例配对复现、分支判断准确率及解析失败。scores.jsonl.gz 保存全部逐题统计；完整生成与 logits 在服务器 private/predictions0.jsonl、predictions1.jsonl。','本轮为原选项顺序的方法对比，未重跑选项轮转、增益搜索、噪声及错误适用性分配、注意力阻断。']
 (OUT/'REPORT.zh.md').write_text('\n\n'.join(lines[:4])+'\n'+'\n'.join(lines[4:])+'\n');(OUT/'STATUS.json').write_text(json.dumps({'status':'complete','cases':800,'rows':len(rows),'report':'REPORT.zh.md'},indent=2));(OUT/'artifact_hashes.json').write_text(json.dumps({p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='artifact_hashes.json'},indent=2));print(json.dumps({'complete':True,'rows':len(rows)}))
if __name__=='__main__':main()
