import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .data import read_rows,save_json
from .run import ROOT,METHODS,SEED

def report(root=ROOT):
    root=Path(root);rs=[]
    for path in sorted((root/'test').glob('predictions_*.jsonl')):rs+=read_rows(path)
    lookup={(r['case_id'],r['method']):r for r in rs}
    expected=read_rows(root/'data/test.jsonl')
    complete=all((c['case_id'],m) in lookup for c in expected for m in METHODS)
    if not complete:
        save_json(root/'report_status.json',dict(status='incomplete',records=len(lookup),expected=len(expected)*len(METHODS)))
        return False
    sources=sorted(set(c['source_id'] for c in expected));rng=np.random.default_rng(SEED)
    bysource={s:[c['case_id'] for c in expected if c['source_id']==s] for s in sources}
    sizes=np.array([len(bysource[s]) for s in sources]);draws=rng.integers(0,len(sources),(2000,len(sources)))
    table={};vectors={}
    for method in METHODS:
        counts=np.array([sum(lookup[(cid,method)]['correct'] for cid in bysource[s]) for s in sources])
        vectors[method]=counts
        boot=counts[draws].sum(1)/sizes[draws].sum(1)
        fixed=sum(not lookup[(c['case_id'],'full')]['correct'] and lookup[(c['case_id'],method)]['correct'] for c in expected)
        harmed=sum(lookup[(c['case_id'],'full')]['correct'] and not lookup[(c['case_id'],method)]['correct'] for c in expected)
        table[method]=dict(cases=len(expected),correct=int(counts.sum()),path_accuracy=float(counts.sum()/len(expected)),
            source_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist(),corrected_vs_full=fixed,degraded_vs_full=harmed)
    comparisons={}
    for a,b in [('learned_reference','aster'),('learned_reference','learned_self')]:
        differences=vectors[a]-vectors[b]
        boot=differences[draws].sum(1)/sizes[draws].sum(1)
        comparisons[a+' minus '+b]=dict(accuracy_difference=float(differences.sum()/len(expected)),
            source_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist(),
            corrected=sum(not lookup[(c['case_id'],b)]['correct'] and lookup[(c['case_id'],a)]['correct'] for c in expected),
            degraded=sum(lookup[(c['case_id'],b)]['correct'] and not lookup[(c['case_id'],a)]['correct'] for c in expected))
    save_json(root/'comparison.json',dict(methods=table,comparisons=comparisons,
        bootstrap=dict(unit='source',resamples=2000,seed=SEED),historical_accuracy=.8921))
    lines=['# ASTER：仅改造 M3','',f'原版测试病例 {len(expected)} 例；各方法共用当前 ASTER 模板与自主路径执行。',
        '', '|方法|完整路径正确率|来源重采样 95% 区间|纠正原模型|退化原模型|','|---|---:|---:|---:|---:|']
    for m,r in table.items():
        lo,hi=r['source_bootstrap_95ci'];lines.append(f"|{m}|{100*r['path_accuracy']:.2f}%|{100*lo:.2f}–{100*hi:.2f}%|{r['corrected_vs_full']}|{r['degraded_vs_full']}|")
    lines+=['','关键比较：']
    for k,r in comparisons.items():
        lo,hi=r['source_bootstrap_95ci'];lines.append(f"- {k}：{100*r['accuracy_difference']:+.2f} 个百分点，95% 区间 [{100*lo:+.2f}, {100*hi:+.2f}]；纠正 {r['corrected']}，退化 {r['degraded']}。")
    lines+=['','旧版 89.21% 仅列为历史结果，其模板和实验设置不与本轮直接合并。',
        'M1 固定 51 个头；M2 沿用 U_N、S 及原秩。仅学习 M3 映射与门控，不加载旧编辑器权重。',
        '参考仅包含原病历、当前条件和不含分支判断的原祖先条件。S 用于记录信号方向上的改变量，不保证零改动。',
        '全部数据、配置、检查、逐轮开发预测、模型及测试逐例预测保存在本目录。']
    (root/'REPORT.zh.md').write_text('\n'.join(lines)+'\n')
    save_json(root/'report_status.json',dict(status='complete',records=len(lookup)))
    return True
if __name__=='__main__':report()
