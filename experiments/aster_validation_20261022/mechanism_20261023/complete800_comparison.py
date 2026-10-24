"""CPU summary of completed frozen 800 versus original 200; no new inference."""
import argparse
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path
from step0_continuation import cancer
from step0_reanalysis import cluster_ci

HERE=Path(__file__).resolve().parent
NAMES={'existing_baseline':'普通基线','existing_prune':'金标单规则裁剪','existing_app':'金标注意力增强',
 'existing_signed':'金标带符号引导','gold_branch_prune':'金标分支组裁剪','visible_prune':'可见预测＋裁剪',
 'hidden_prune':'隐藏预测＋裁剪','hidden_top2':'隐藏预测＋Top-2','hidden_signed':'隐藏预测＋带符号引导',
 'hidden_unsigned':'隐藏预测＋无符号引导','CAD8':'CAD8','CAD128':'CAD128','CAD_filtered128':'过滤CAD128'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jsonl(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]

def paired(rows,left,right):
 a={r['id']:r for r in rows if r['method']==left};b={r['id']:r for r in rows if r['method']==right}
 if a.keys()!=b.keys():raise ValueError('paired methods must contain identical IDs')
 ids=sorted(a)
 if not ids:return {'n':0,'delta':None}
 ds=[int(b[i]['correct'])-int(a[i]['correct']) for i in ids]
 return dict(n=len(ids),left_correct=sum(a[i]['correct'] for i in ids),right_correct=sum(b[i]['correct'] for i in ids),
  delta=sum(ds)/len(ds),wins=ds.count(1),losses=ds.count(-1),
  tree_ci_descriptive=cluster_ci(ds,[a[i]['disease'] for i in ids]))

def analyse(summary,rows,cases,old200):
 if not summary['complete'] or summary['rows']!=52496 or len(rows)!=52496:raise ValueError('completed 52496 rows required')
 ids={r['id'] for r in cases};old_ids={r['id'] for r in old200}
 if len(ids)!=800 or len(cases)!=800 or len(old_ids)!=200:raise ValueError('cohort sizes differ')
 old_trees={r['disease'] for r in old200};old_cancers={cancer(x) for x in old_trees}
 methods=list(NAMES);keys=[(r['id'],r['variant'],r['method']) for r in rows]
 expected={(c['id'],v,m) for c in cases for v in c['variants'] for m in methods+['branch_hidden','branch_visible','branch_hide_inapp']}
 if len(set(keys))!=len(keys) or set(keys)!=expected:raise ValueError('score coverage/uniqueness mismatch')
 scopes={'all':ids,'strict304':{c['id'] for c in cases if c['pair_protocol']=='PR23_strict'},
  'strict_single220':{c['id'] for c in cases if c['pair_protocol']=='PR23_strict' and c['single_action_gold_branch']},
  'no_original_id':ids-old_ids,'new_data_tree':{c['id'] for c in cases if c['disease'] not in old_trees},
  'new_cancer_type':{c['id'] for c in cases if cancer(c['disease']) not in old_cancers}}
 comparisons=[]
 for st in ('all','K','U'):
  for variant in ('A','B','C','D','CS'):
   for method in methods:
    key=f'all/{variant}/{method}/{st}';x=summary['cells'].get(key);old=summary['old200_cells'].get(f'{variant}/{method}/{st}')
    if not x:continue
    comparisons.append(dict(stratum=st,variant=variant,method=method,new_n=x['n'],new_correct=x['correct'],new_accuracy=x['accuracy'],
     old_n=old['n'] if old else None,old_correct=old['correct'] if old else None,old_accuracy=old['accuracy'] if old else None,
     difference=x['accuracy']-old['accuracy'] if old else None,
     comparison='descriptive across different cohorts; not an independent paired confirmation'))
 scoped={}
 for scope,chosen in scopes.items():
  scoped[scope]={'unique_cases':len(chosen),'cells':{},'paired_descriptive':{}}
  chosen_rows=[r for r in rows if r['id'] in chosen]
  buckets=collections.defaultdict(list)
  for r in chosen_rows:
   for st in ('all',r['stratum']):buckets[r['variant'],st].append(r)
  for (v,st),rs in buckets.items():
   for m in methods+['branch_hidden','branch_visible']:
    selected=[r for r in rs if r['method']==m]
    if selected:scoped[scope]['cells'][f'{v}/{m}/{st}']={'n':len(selected),'correct':sum(r['correct'] for r in selected),'accuracy':sum(r['correct'] for r in selected)/len(selected),'parse_failures':sum(r['parse_failure'] for r in selected)}
   for left,right in [('visible_prune','hidden_prune'),('CAD128','CAD_filtered128')]:
    scoped[scope]['paired_descriptive'][f'{v}/{st}/{left}->{right}']=paired(rs,left,right)
 # Existing publication-level paired statistics are reused, not reassigned a smaller Holm family.
 return dict(complete=True,cases=800,rows=len(rows),old_cases=200,old_K=76,old_U=124,new_K=426,new_U=374,
  overlap_ids=len(ids&old_ids),cross_cohort=comparisons,scopes=scoped,
  published_paired_statistics=summary['paired_comparisons'],comparison_family=summary['comparison_family'],
  published_comparison_count=len(summary['paired_comparisons']),
  overlap_reuse_consistency=summary['overlap200_reuse_consistency'],
  limitations=['Different sampling and construction; cross-cohort changes are descriptive.',
   'Old K/U uses its original eager closed-book run; PR24 K/U uses SDPA. Group membership is frozen, not redefined.',
   'Strict-single subset matches construction more closely; it is post-hoc and shares source trees.',
   'New ID/tree/cancer subsets do not establish guideline-version or publisher independence.',
   'CS has different threshold protocols and denominator 2 versus 84; all 84 new CS are nscl_3.',
   'Original 200 has no reported D CAD rows in the frozen old200 table; absence is not zero.',
   'Additional subset differences and tree intervals are exploratory; no extra significance claim.',
   'Original 200 Holm family and the completed 800 135-comparison Holm family stay separate.'])

def write_report(d,out):
 by={(r['stratum'],r['variant'],r['method']):r for r in d['cross_cohort']}
 def value(row,period):
  if row is None or row[period+'_accuracy'] is None:return 'NA'
  return f"{100*row[period+'_accuracy']:.2f}% ({row[period+'_correct']}/{row[period+'_n']})"
 lines=['# 固定800例与原200例比较','800例推理及统计全部完成，52496条无重复且覆盖全部预期键。A/B/C各800、D797、CS84，K426/U374；原200为K76/U124。原结果未修改，本报告只读取已有输出。',
 '## 全样本结果','单元格为“原200 → 新800”；NA表示原冻结表未提供该方法/条件，不能记作0。CS的2例与84例阈值协议不同，仅并列。',
 '|方法|A|B|C|D|CS|','|---|---|---|---|---|---|']
 for m,name in NAMES.items():lines.append('|'+name+'|'+'|'.join(value(by.get(('all',v,m)),'old')+' → '+value(by.get(('all',v,m)),'new') for v in ('A','B','C','D','CS'))+'|')
 lines+=['## 隐藏判断＋裁剪相对可见判断＋裁剪','原200的A/B/C/D差值分别−2.50/+8.50/−4.50/+16.50个百分点。以下为新800的同病例配对比较；区间按数据树重采样，Holm沿用新800全部135项比较，不能解释为独立指南级确认。',
 '|条件|隐藏−可见（百分点）|获益/损失病例|数据树95%描述区间（百分点）|800比较族Holm p|','|---|---:|---:|---|---:|']
 for v in ('A','B','C','D','CS'):
  x=d['published_paired_statistics'][f'all/{v}/all/visible_prune->hidden_prune'];ci=x['disease_ci'];txt='NA（单树）' if ci is None else f"[{100*ci[0]:.2f}, {100*ci[1]:.2f}]"
  lines.append(f"|{v}|{100*x['delta']:+.2f}|{x['wins']}/{x['losses']}|{txt}|{x['p_holm']:.5g}|")
 lines+=['新800延续B/D收益与A/C代价方向。D收益从原200的16.50降到9.16个百分点，但跨队列差异不能解释为样本量导致效果衰减；病例构造、来源分布和分组方式不同。A/C未证明非劣效。',
 '## 协议与来源分层','以下仅为新800内部的描述性敏感性，不能按最有利子集替代全样本。数据树和癌种的新颖性均不等于新的独立指南版本。',
 '|子集|病例数|D可见→隐藏|D差值（百分点）|B差值（百分点）|','|---|---:|---|---:|---:|']
 for scope,label in [('all','全部'),('strict304','原构造'),('strict_single220','原构造且金标单动作'),('no_original_id','去掉19个重叠ID'),('new_data_tree','原200未含的数据树'),('new_cancer_type','原200未含的癌种')]:
  x=d['scopes'][scope];z=x['paired_descriptive']['D/all/visible_prune->hidden_prune'];b=x['paired_descriptive']['B/all/visible_prune->hidden_prune']
  lines.append(f"|{label}|{x['unique_cases']}|{100*z['left_correct']/z['n']:.2f}% → {100*z['right_correct']/z['n']:.2f}% (n={z['n']})|{100*z['delta']:+.2f}|{100*b['delta']:+.2f}|")
 lines+=['新800有304例符合原构造、其中220例金标分支单动作；其余为明确标记的扩展构造，D有3例无法构造。两队列重叠19个ID且相关历史推理复用；397例共享原数据树、702例共享癌种，全部共享NCCN发布机构。',
 '## K/U分组','新800的K/U沿用PR24闭卷SDPA；原200为既有eager闭卷结果。不得按本次方法输出重新选K/U，跨队列K/U差异只作描述。全部方法/条件的分组结果与旧值见accuracy_comparison.csv。',
 '|组别|条件|可见裁剪|隐藏裁剪|隐藏−可见（百分点）|','|---|---|---:|---:|---:|']
 for st in ('K','U'):
  for v in 'ABCD':
   a=by[st,v,'visible_prune'];b=by[st,v,'hidden_prune'];lines.append(f"|{st}|{v}|{100*a['new_accuracy']:.2f}%|{100*b['new_accuracy']:.2f}%|{100*(b['new_accuracy']-a['new_accuracy']):+.2f}|")
 lines+=['## CAD与执行瓶颈','过滤CAD在新800 B为44.75%，旧CAD49.625%，下降4.875个百分点（47例失分、8例获益，800比较族Holm p=9.52e−6）；原200对应45.0%与52.0%，方向一致。A/C过滤版较高，但在800完整比较族下未通过Holm门槛。CAD8与CAD128在本800准确率相同，不意味着完整文本或逐词元轨迹相同。',
 '金标单规则裁剪在新800 B/D仍分别为80.75%/86.20%；说明即使按构造金标给出规则，仍有执行错误。该金标属于合成干预，不能由此认定临床错误或参数记忆覆盖。',
 '## 文件与证据','cohort_comparison.json保留全部方法、新旧分组、来源与协议子集及原800配对统计；accuracy_comparison.csv便于查表。输入和程序SHA256见provenance，旧200和新800正式文件只读。18例CAD真实轨迹结论另见CAD18_TRACE.zh.md。']
 (out/'COHORT800_COMPARISON.zh.md').write_text('\n'.join(lines)+'\n')
 with (out/'accuracy_comparison.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(d['cross_cohort'][0]),lineterminator='\n');w.writeheader();w.writerows(d['cross_cohort'])

def main():
 p=argparse.ArgumentParser();p.add_argument('--results800',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 r=a.results800;summary=json.loads((r/'summary.json').read_text());status=json.loads((r/'STATUS.json').read_text())
 if status['status']!='complete':raise ValueError('800 queue not complete')
 with gzip.open(r/'scores.jsonl.gz','rt') as f:rows=[json.loads(line) for line in f]
 oldpath=HERE/'results/priority200/qwen/full_parser_v2/scores.jsonl';cases=jsonl(r/'case_manifest.jsonl')
 d=analyse(summary,rows,cases,jsonl(oldpath));d['provenance']={str(x):sha(x) for x in [r/'summary.json',r/'STATUS.json',r/'scores.jsonl.gz',r/'case_manifest.jsonl',oldpath,Path(__file__)]}
 a.output.mkdir(parents=True,exist_ok=False);(a.output/'cohort_comparison.json').write_text(json.dumps(d,indent=2)+'\n');write_report(d,a.output)
 print(json.dumps({'complete':True,'rows':len(rows),'subsets':{k:v['unique_cases'] for k,v in d['scopes'].items()}}))
if __name__=='__main__':main()
