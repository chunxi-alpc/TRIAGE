"""Export source-cluster statistics, per-example paths, and figures for Experiment B."""
import csv,gzip,json,shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
from .fit_controlled import OUT,NAMES,csvwrite,dump,BOOT
from .cached_fit_helpers import rows
WORK=Path('/home/gcx/aster_cross_model_mega_20261111')

def binary_summary(rr,method,sourcekey='source_id'):
    bysource=defaultdict(lambda:np.zeros(3))
    for r in rr:
        bysource[r[sourcekey]]+=np.array([1,int(r['correct']),int(r['raw_correct'])])
    ids=sorted(bysource);a=np.array([bysource[s] for s in ids])
    rng=np.random.default_rng(20261111);v=a[rng.integers(len(ids),size=(BOOT,len(ids)))].sum(1)
    ci=np.percentile((v[:,1]-v[:,2])/v[:,0],[2.5,97.5]);total=a.sum(0)
    summary=dict(method=method,n=int(total[0]),correct=int(total[1]),accuracy=float(total[1]/total[0]),
      raw_correct=int(total[2]),delta_correct=int(total[1]-total[2]),delta_accuracy=float((total[1]-total[2])/total[0]),
      paired_source_cluster_ci_low=float(ci[0]),paired_source_cluster_ci_high=float(ci[1]),sources=len(ids))
    src=[dict(source_id=s,n=int(bysource[s][0]),correct=int(bysource[s][1]),raw_correct=int(bysource[s][2])) for s in ids]
    return summary,src

def summarize_one(model):
    out=OUT/model;work=WORK/model
    records={};naturals={}
    for d in sorted(work.glob('shard*')):
        for r in rows(d/'path_results.jsonl'):records[(r['case_id'],r['method'])]=r
        for r in rows(d/'natural_nodes.jsonl'):naturals[r['id']]=r
    allr=list(records.values());summary=[];sources=[]
    for group in ('controlled_dev','controlled_test','official_test'):
        raw={r['case_id']:r for r in allr if r['evaluation_group']==group and r['method']=='raw'}
        for method in NAMES:
            rr=[dict(r,raw_correct=raw[r['case_id']]['correct']) for r in allr if r['evaluation_group']==group and r['method']==method]
            if not rr:continue
            stat,src=binary_summary(rr,method);stat.update(model=model,evaluation_group=group,metric='dynamic_path',
              failed=sum(r.get('error') is not None for r in rr))
            summary.append(stat)
            sources.extend(dict(r,model=model,method=method,evaluation_group=group,metric='dynamic_path') for r in src)
    for method in NAMES:
        rr=[]
        for r in naturals.values():
            score=r.get('scores');gold=bool(r['gold_label'])
            rr.append(dict(source_id=r['source_id'],correct=bool(score is not None and (score[method]>0)==gold),
              raw_correct=bool(score is not None and (score['raw']>0)==gold)))
        if rr:
            stat,src=binary_summary(rr,method);stat.update(model=model,evaluation_group='official_test',metric='natural_gold_path_node',
               failed=sum(r.get('scores') is None for r in naturals.values()))
            summary.append(stat);sources.extend(dict(r,model=model,method=method,evaluation_group='official_test',metric='natural_gold_path_node') for r in src)
    csvwrite(out/'path_summary.csv',summary);csvwrite(out/'path_source_metrics.csv',sources)
    for filename,records in [('path_predictions.jsonl.gz',allr),('official_natural_nodes.jsonl.gz',list(naturals.values()))]:
        with gzip.open(out/filename,'wt',encoding='utf8') as f:
            for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    for shard in work.glob('shard*'):
        for name in ('done.json','status.json'):
            if (shard/name).exists():shutil.copy2(shard/name,out/(shard.name+'_'+name))
    dump(out/'complete.json',dict(state='complete' if all((work/d.name/'done.json').exists() for d in work.glob('shard*')) else 'partial',
       dynamic_path_records=len(allr),official_natural_nodes=len(naturals),methods=NAMES))
    return summary

def summarize_all():
    summary=[];controlled=[];complete={}
    for model in ('llama31','gemma2','gemma3','medgemma'):
        out=OUT/model
        if (WORK/model).exists():summary+=summarize_one(model)
        if (out/'controlled_summary.csv').exists():
            with (out/'controlled_summary.csv').open() as f:controlled+=list(csv.DictReader(f))
        complete[model]=json.loads((out/'complete.json').read_text()) if (out/'complete.json').exists() else {'state':'pending'}
    csvwrite(OUT/'path_summary.csv',summary);csvwrite(OUT/'controlled_summary.csv',controlled)
    dump(OUT/'status.json',complete)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['Llama-3.1-8B','Gemma-2-9B','Gemma-3-27B','MedGemma-27B']
    models=['llama31','gemma2','gemma3','medgemma'];colors=['#366599','#db9438','#4e9677']
    fig,ax=plt.subplots(figsize=(8.0,4.5))
    for j,(method,color) in enumerate(zip(NAMES[1:],colors)):
        selected={r['model']:r for r in summary if r['method']==method and r['evaluation_group']=='official_test' and r['metric']=='dynamic_path'}
        xs=[];ys=[];los=[];his=[]
        for i,model in enumerate(models):
            if model not in selected:continue
            r=selected[model];xs.append(i+(j-1)*.23);ys.append(r['delta_accuracy']*100)
            los.append((r['delta_accuracy']-r['paired_source_cluster_ci_low'])*100);his.append((r['paired_source_cluster_ci_high']-r['delta_accuracy'])*100)
        if xs:ax.errorbar(xs,ys,yerr=[los,his],fmt='o',capsize=3,label=method,color=color)
    ax.axhline(0,color='gray',lw=.8);ax.set_xticks(range(4),labels,rotation=12)
    ax.set_ylabel('Official path accuracy change (percentage points)')
    ax.legend(frameon=False);fig.tight_layout()
    fig.savefig(OUT/'official_path_changes.png',dpi=220);fig.savefig(OUT/'official_path_changes.svg');plt.close(fig)
    lines=['# 跨模型 MEGA 核心复现','',
       '四模型分别复用各自已完成的真实激活、独立选头、S/N、标准化和线性参数；只新拟合同维随机控制，并补齐官方路径推理。',
       'fit80/dev20/test40 来源划分保持原样。固定 alpha=0.01、tau=0.1；截距仅用自然 fit 标签。随机16维种子20261111，未根据测试结果调整。',
       'AP prediction_r2 是受 C/F 预算约束的部署修正对 AP 分数差的解释量；不是另训无约束探针的解释量。来源簇bootstrap 2000次，区间为配对准确率变化95%区间。',
       '','| 模型 | 方法 | 官方路径正确数 | 相对原模型 | 差值95% CI（百分点） |','|---|---|---:|---:|---:|']
    for r in summary:
        if r['evaluation_group']=='official_test' and r['metric']=='dynamic_path':
            lines.append(f"| {r['model']} | {r['method']} | {r['correct']}/{r['n']} | {r['delta_correct']:+d} | [{r['paired_source_cluster_ci_low']*100:.3f}, {r['paired_source_cluster_ci_high']*100:.3f}] |")
    lines+=['','逐例输出 path_predictions.jsonl.gz / controlled_predictions.jsonl.gz / controlled_pairs.jsonl.gz；来源指标 *_source_metrics.csv；受控配对所有指标与区间 controlled_summary.csv。','',
       '权重的上游不可变revision未在本地下载元数据记录，config.json明确标记 unrecorded 并保留实际checkpoint路径。未编造revision。']
    (OUT/'REPORT.zh.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
if __name__=='__main__':summarize_all()
