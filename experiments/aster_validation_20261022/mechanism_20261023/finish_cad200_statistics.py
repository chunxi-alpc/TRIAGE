"""CPU-only paired option-order analysis after all selected-200 inference finishes."""
from pathlib import Path
import json,subprocess,collections
from analyze import bootstrap,paired
r=Path('/home/gcx/aster_validation_20261022/private/run500');p=r/'cad200';repo=Path('/home/gcx/TRIAGE-worktrees/aster-validation');exp=repo/'experiments/aster_validation_20261022';out=exp/'mechanism_20261023/results/cad200_option_order';ids=set(json.loads((r/'mechanism/priority200/selection.json').read_text())['ids']);v=p/'option_analysis_view';(v/'option_order').mkdir(parents=True,exist_ok=True)
original=r/'mechanism/priority200/qwen_existing_complete.jsonl';strata={x['id']:x['stratum'] for x in map(json.loads,original.read_text().splitlines()) if x['method']=='closed_book'};assert set(strata)==ids
design=json.loads((r/'option_order/design.json').read_text());design['ids']=[i for i in design['ids'] if i in ids];assert len(design['ids'])==200
for rot in design['rotations']:design['inverse_label_maps'][rot]={i:mp for i,mp in design['inverse_label_maps'][rot].items() if i in ids}
(v/'option_order/design.json').write_text(json.dumps(design,indent=2))
for src,dest in [(original,v/'qwen_complete.jsonl'),(p/'left_complete.jsonl',v/'option_order/qwen_left_complete.jsonl'),(p/'right_complete.jsonl',v/'option_order/qwen_right_complete.jsonl')]:
 if not dest.exists():dest.symlink_to(src)
subprocess.run(['python',str(exp/'option_order.py'),'analyze','--root',str(v),'--model','qwen','--out',str(out)],check=True)
report=json.loads((out/'summary.json').read_text());pairs=list(map(json.loads,(out/'paired_scores.jsonl').read_text().splitlines()));groups=collections.defaultdict(list)
for x in pairs:
 for st in ['all',strata[x['id']]]:groups['/'.join(map(str,[x['rotation'],x['method'],x['variant'],x['assigned_wrong'],st]))].append(x)
for key,rows in groups.items():report['comparisons'][key]['disease_bootstrap_ci']=bootstrap([int(x['reordered_correct'])-int(x['original_correct']) for x in rows],[x['disease'] for x in rows])
# Filtered CAD order comparisons also use only these same 200 IDs.
def load_filtered(rot):
 rows=list(map(json.loads,(p/f'{rot}_filtered.jsonl').read_text().splitlines()));assert len(rows)==1800 and {x['id'] for x in rows}==ids;return {(x['id'],x['variant'],x['method'],x.get('assigned_wrong',False)):x for x in rows}
base=load_filtered('original');report['filtered_comparisons']={}
for rot in ['left','right']:
 rr=load_filtered(rot);buckets=collections.defaultdict(list)
 for key,x in rr.items():
  b=base[key]
  for st in ['all',strata[x['id']]]:buckets[(x['method'],x['variant'],x.get('assigned_wrong',False),st)].append((b,x))
 for key,xy in buckets.items():
  aa=[z[0] for z in xy];bb=[z[1] for z in xy];value=paired(aa,bb);valid=[(a,b) for a,b in xy if a['prediction'] is not None and b['prediction'] is not None];value['valid_pairs']=len(valid);value['action_agreement_valid']=sum(a['prediction']==design['inverse_label_maps'][rot][b['id']].get(b['prediction']) for a,b in valid)/len(valid) if valid else None;report['filtered_comparisons']['/'.join(map(str,[rot,*key]))]=value
ordered=sorted(report['filtered_comparisons'].values(),key=lambda x:x['p']);prev=0
for j,x in enumerate(ordered):prev=max(prev,min(1,x['p']*(len(ordered)-j)));x['p_holm']=prev
report['notes'].append('Disease-cluster bootstrap 2000 replicates; K/U fixed from original closed-book. Filtered comparisons have their own explicitly reported Holm family.')
(out/'summary.json').write_text(json.dumps(report,indent=2));(out/'REPORT.zh.md').write_text('# 选定200题选项顺序检验\n\n原顺序、左移、右移使用完全相同200病例。按选项文本对应回原动作，不能以字母相同判断一致。summary.json同时给出原方法与过滤CAD的准确率差、配对检验、疾病簇区间及动作一致率；K/U固定为原顺序分组。重排不作为新增独立样本。逐题数值见paired_scores.jsonl。\n');print('selected200 option-order statistics complete')
