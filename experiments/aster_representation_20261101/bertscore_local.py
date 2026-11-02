"""Single-turn BERTScore F1 pathway evaluation on all frozen predictions. No remote judge."""
import argparse
import collections
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from . import official, semantic

ROOT = official.OUT / 'bertscore_local'
INDEX = official.OUT / 'codex_judging/all_methods_pathways/inputs.jsonl'
METHODS = ('zero','fusion','ordinary','context','blind_readout','no_js')
LABELS = dict(zip(METHODS, ('零干预','标签概率融合','普通低秩干预','概率上下文','盲表示分类器','处置盲内部干预（本文）')))
MODELS = {'en':('roberta-large',17), 'zh':('bert-base-chinese',8)}
PARAMS = dict(release='0.3.13',component='F1',idf=False,rescale_with_baseline=True,
              clip=[0,1],use_fast_tokenizer=False,token_limit=512,position_weight=.1,
              lnds_threshold=.2,ot_reg=.2,ot_tau=1.,aggregation='MEGA-CDP paper eqs 3-7',
              public_script_commit='15a3a12e73a9169a870bff2ac431ec2e9968b8c7')
EXPECTED = {'en':4205, 'zh':3253}
METRICS = ('DTW_PCC','OT_PCC','LNDS_PCS')

def read(path):
    return json.loads(path.read_text())

class Inventory:
    def __init__(self, lang):
        self.cases = []
        self.pairs = []
        lookup = {}
        with INDEX.open() as handle:
            for line in handle:
                r=json.loads(line)
                if r['language']!=lang:
                    continue
                layout={}
                for owner,steps in r['steps'].items():
                    matrix=[]
                    for ref in r['reference_steps']:
                        row=[]
                        for pred in steps:
                            key=(ref,pred)
                            idx=lookup.get(key)
                            if idx is None:
                                idx=len(self.pairs)
                                self.pairs.append(key)
                                lookup[key]=idx
                            row.append(idx)
                        matrix.append(row)
                    layout[owner]=np.array(matrix,dtype=np.int32)
                self.cases.append((r['case_id'],r['source'],r['owners'],layout))
        if len(self.cases)!=EXPECTED[lang]:
            raise ValueError('Unexpected registered case count')
        self.lang=lang

def values_cache(inv,folder):
    protocol={**PARAMS,'lang':inv.lang,'encoder':MODELS[inv.lang][0],'layer':MODELS[inv.lang][1],
              'pair_count':len(inv.pairs),'case_count':len(inv.cases)}
    folder.mkdir(exist_ok=True,parents=True)
    config=folder/'protocol.json'
    if config.exists() and read(config)!=protocol:
        raise ValueError('Existing cache protocol differs')
    official.write(config,protocol)
    file=folder/'rescaled_f1.npy'
    if file.exists():
        arr=np.load(file,mmap_mode='r+')
        if arr.shape!=(len(inv.pairs),) or arr.dtype!=np.float32:
            raise ValueError('Incompatible text-pair cache')
        return arr
    arr=np.lib.format.open_memmap(file,mode='w+',dtype=np.float32,shape=(len(inv.pairs),))
    arr[:]=np.nan
    arr.flush()
    return arr

def make_scorer(lang):
    import bert_score
    from bert_score import BERTScorer
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model,layer=MODELS[lang]
    if importlib.metadata.version('bert-score')!=PARAMS['release']:
        raise ValueError('Incorrect BERTScore release')
    base=Path(bert_score.__file__).parent/'rescale_baseline'/lang/(model+'.tsv')
    instance=BERTScorer(model_type=str(ROOT/'models'/model),lang=lang,num_layers=layer,
                        idf=False,rescale_with_baseline=True,baseline_path=str(base),
                        use_fast_tokenizer=False,device='cuda:0')
    instance._tokenizer.model_max_length=512
    return instance

def encode(inv,folder,arr,pilot):
    import torch
    batch=64
    block=1024
    ids=np.arange(len(arr),dtype=np.int32)
    if pilot:
        starts=(0,len(arr)//2,len(arr)-block)
        ids=np.concatenate([ids[s:s+block] for s in starts])
    ids=ids[np.isnan(arr[ids])]
    bench=[]
    start=time.monotonic()
    if len(ids):
        scorer=make_scorer(inv.lang)
        for pos in range(0,len(ids),block):
            chosen=ids[pos:pos+block]
            refs,candidates=zip(*(inv.pairs[i] for i in chosen))
            torch.cuda.reset_peak_memory_stats()
            t=time.monotonic()
            while True:
                try:
                    _,_,f1=scorer.score(list(candidates),list(refs),batch_size=batch)
                    break
                except torch.cuda.OutOfMemoryError:
                    if batch==1:raise
                    batch=max(1,batch//2)
                    torch.cuda.empty_cache()
            scores=f1.detach().float().cpu().numpy()
            if scores.shape!=(len(chosen),) or not np.all(np.isfinite(scores)):
                raise ValueError('Invalid BERTScore output')
            arr[chosen]=scores
            arr.flush()
            bench.append(dict(count=len(chosen),seconds=time.monotonic()-t,
                peak_gb=torch.cuda.max_memory_allocated()/1e9,batch_size=batch))
            official.write(folder/'progress.json',dict(mode='pilot' if pilot else 'full',
                cached=int(np.isfinite(arr).sum()),registered=len(arr),seconds=time.monotonic()-start))
    if pilot:
        if bench:
            official.write(folder/'pilot.json',dict(blocks=bench,seconds=time.monotonic()-start,
                measured_pairs=sum(x['count'] for x in bench),peak_gb=max(x['peak_gb'] for x in bench)))
        elif not (folder/'pilot.json').exists():
            raise ValueError('No pilot timings available')

def aggregate_language(inv,folder,arr):
    from ot.unbalanced import sinkhorn_knopp_unbalanced
    public=Path('/home/gcx/aster_manuscript_20261030/clinical01/official_mega/mega_cdp/scripts/evaluate')
    sys.path.insert(0,str(public))
    from pathway_metrics import dynamic_time_warping
    if not np.isfinite(arr).all():
        raise ValueError('Incomplete BERTScore cache')
    output=folder/'case_metrics.jsonl'
    temp=output.with_suffix('.tmp')
    with temp.open('w') as handle:
        for cid,source,owners,layout in inv.cases:
            values={}
            for owner,index in layout.items():
                c=np.clip(np.asarray(arr[index],dtype=np.float64),0.,1.)
                m,n=c.shape
                diff=1.-c
                _,dtw,_=dynamic_time_warping(diff.tolist())
                loc=np.abs((np.arange(m)[:,None]+.5)/m-(np.arange(n)[None,:]+.5)/n)
                cost=(diff+.1*loc)/1.1
                plan=sinkhorn_knopp_unbalanced(np.full(m,1./m),np.full(n,1./n),cost,
                    PARAMS['ot_reg'],PARAMS['ot_tau'],numItermax=3000,stopThr=1e-9)
                mass=float(plan.sum())
                if not np.isfinite(plan).all() or mass<=0:raise ValueError('Invalid transport plan')
                best=c.argmax(axis=1)
                seq=[int(j) for i,j in enumerate(best) if c[i,j]>=.2]
                values[owner]=dict(DTW_PCC=float(dtw),OT_PCC=float((plan*cost).sum()/mass),
                                   LNDS_PCS=semantic.lnds_length(seq)/m)
            if set(owners)!=set(METHODS) or any(o not in values for o in owners.values()):
                raise ValueError('Missing valid prediction')
            handle.write(json.dumps(dict(case_id=cid,language=inv.lang,source=source,
                                         methods={k:values[o] for k,o in owners.items()}))+'\n')
    temp.replace(output)
    official.write(folder/'done.json',dict(n=len(inv.cases),pairs=len(arr)))

def paired_differences(rows):
    with INDEX.open() as handle:
        differs = {r['case_id'] for r in (json.loads(l) for l in handle)
                   if r['owners']['no_js'] != r['owners']['blind_readout']}
    paired = {}
    for key, group in [('all',rows),('different_paths',[r for r in rows if r['case_id'] in differs])]:
        x=np.array([[r['methods'][a]['DTW_PCC'] for a in ['no_js','blind_readout']] for r in group])
        by_source=collections.defaultdict(list)
        for r,v in zip(group,x[:,0]-x[:,1]):by_source[r['source']].append(v)
        blocks=np.array([[sum(v),len(v)] for v in by_source.values()])
        resampled=blocks[np.random.default_rng(20261102).integers(len(blocks),size=(2000,len(blocks)))].sum(axis=1)
        paired[key]=dict(n=len(group),main_percent=100*float(x[:,0].mean()),blind_percent=100*float(x[:,1].mean()),
                         delta_pp=100*float((x[:,0]-x[:,1]).mean()),
                         source_ci95_pp=np.quantile(100*resampled[:,0]/resampled[:,1],[.025,.975]).tolist())
    return paired

def render_manuscript(txt,summary):
    import re
    numbers=summary['means_percent']
    ref=read(Path(__file__).with_name('external_bertscore_singleturn.json'))['percent']
    arm_label=dict(LABELS);arm_label['zero']='Qwen3-8B：零干预'
    arm_label['no_js']='**处置盲内部干预（本文）**'
    by_label={label:k for k,label in arm_label.items()}
    start='<!-- ASTER_FINAL_MAIN_START -->';end='<!-- ASTER_FINAL_MAIN_END -->'
    prefix,rest=txt.split(start,1);table,suffix=rest.split(end,1)
    lines=table.split('\n');changed=0
    for i,line in enumerate(lines):
        if not line.startswith('|'):continue
        cells=line.split('|')
        k=cells[1].strip()
        if k in ref:
            cells[3]=' '+ref[k][0]+' '
        elif k in by_label:
            val=f"{numbers[by_label[k]]['DTW_PCC']:.2f}"
            cells[3]=' **'+val+'** ' if by_label[k]=='no_js' else ' '+val+' '
        else:continue
        changed+=1;lines[i]='|'.join(cells)
    if changed!=22:raise ValueError('Unexpected main table rows')
    txt=prefix+start+'\n'.join(lines)+end+suffix
    txt=re.sub(r'^\*\*表1　.*$', '**表1　MEGA-CDP 单轮测试结果（%）。** DTW-PCC 列采用 BERTScore 步骤相似度，外部数值引用原论文表5。外部 Acc 引自原论文表1的 GPT-5.4 Mini 裁判结果。本文六种方法的 PathAcc 和 DTW-PCC 均覆盖7,458例，生成式 Acc 另标有效例数。',txt,flags=re.M)
    txt=re.sub(r'^\| \*\*DTW-PCC\*\*.*$',lambda _: '| **DTW-PCC** | 对步骤语义距离 $1-C_{ij}$ 作有序动态对齐，将最小累计代价除以对应对齐长度，再对病例取均值 | ↓ |',txt,flags=re.M)
    protocol=('表1的 DTW-PCC 和表2a的三项路径指标使用 BERTScore F1（bert-score 0.3.13）。英语使用 RoBERTa-large 第17层，中文使用 BERT-base-Chinese 第8层。两种语言均关闭 IDF 加权，使用软件包对应语言和编码器的基线重缩放，将 F1 截断至 [0,1] 作为步骤相似度。'
        'LNDS 匹配阈值为0.2，OT 成本按原论文式（4）计算为 $(D_{\\rm semantic}+0.1D_{\\rm position})/1.1$，熵正则0.2、边际松弛系数1取作者公开汇总脚本的默认值。'
        '编码器、层和基线重缩放为本文记录的实现设置，外部表格所用的完整编码配置尚未核实。'
        ' [BERTScore 实现](https://github.com/Tiiiger/bert_score) [MEGA-CDP 附录B](https://arxiv.org/html/2608.26592v1#A2)\n\n'
        '表2b保留使用 gpt-6.1-sol 的生成式路径评分。该组对每个输入独立采样五次，平均步骤评分后归一化，使用预先固定的 LNDS 阈值0.9、OT 位置权重0.2、熵正则0.05及边际松弛系数1。')
    if '表1的 DTW-PCC 和表2a的三项路径指标使用 BERTScore F1' not in txt:
        txt=re.sub(r'^语义评价采用以下固定协议[^\n]*',lambda _:protocol,txt,flags=re.M)
    txt=txt.replace('本文表2a引用原论文附录的 BERTScore 路径指标，表2b列出本文五次生成式裁判评分的结果。','表2a列出本文与原论文附录的 BERTScore 路径指标，表2b报告已有生成式评分。')
    txt=txt.replace('每项语义指标按取得五次有效评分的病例汇总，并报告对应例数。','BERTScore 路径指标使用完整测试集，生成式评分另报取得五次有效评分的例数。')
    first=txt.index('### 5.2 ');second=txt.index('**表2b　',first)
    head=['### 5.2 路径语义评价','',
          '**表2a　BERTScore 单轮路径指标（%）。** 前16行引用 MEGA-CDP 原论文表5、表6和表7的 BERTScore 列，末6行为本文在全部7,458例上的结果。外部模型使用原始指南自由生成，本文方法使用结构化树和受约束路径解码。评分配置见 §4.3。[原论文附录D](https://arxiv.org/html/2608.26592v1#A4)','',
          '| 模型／方法 | DTW-PCC↓ | OT-PCC↓ | LNDS-PCS↑ |','|---|---:|---:|---:|']
    head.extend('| '+label+' | '+' | '.join(v)+' |' for label,v in ref.items())
    for k in METHODS:
        head.append('| '+arm_label[k]+' | '+' | '.join(f"{numbers[k][m]:.2f}" for m in METRICS)+' |')
    if 'paired' in summary:
        v=summary['paired']['different_paths'];lo,hi=v['source_ci95_pp']
        head+=['',f"在主方法与盲表示分类器路径不同的{v['n']:,}例中，DTW-PCC 分别为{v['main_percent']:.2f}%和{v['blind_percent']:.2f}%。按主方法减分类器计算，配对差为{v['delta_pp']:+.2f}个百分点（来源簇重采样95%区间 {lo:+.2f} 至 {hi:+.2f}）。"]
    txt=txt[:first]+'\n'.join(head)+'\n\n'+txt[second:]
    txt=re.sub(r'^\*\*表2b　.*$', '**表2b　已保存的生成式裁判结果（%，gpt-6.1-sol）。** 每行 n 为取得五次有效步骤评分的病例数，三项指标共用相同的评分矩阵。',txt,flags=re.M)
    return txt

def final():
    rows=[]
    for lang in EXPECTED:
        p=ROOT/lang/'case_metrics.jsonl'
        if not (ROOT/lang/'done.json').exists():return
        with p.open() as handle:rows.extend(json.loads(l) for l in handle)
    if len({r['case_id'] for r in rows})!=7458 or len(rows)!=7458:
        raise ValueError('Incomplete test split')
    means={k:{metric:100*np.mean([r['methods'][k][metric] for r in rows]) for metric in METRICS} for k in METHODS}
    summary=dict(status='complete',n=len(rows),means_percent=means,protocol=PARAMS,encoders=MODELS)
    summary['paired']=paired_differences(rows)
    official.write(ROOT/'summary.json',summary)
    lines=['| 模型／方法 | DTW-PCC↓ | OT-PCC↓ | LNDS-PCS↑ |','|---|---:|---:|---:|']
    for k in METHODS:lines.append('| '+LABELS[k]+' | '+' | '.join(f'{means[k][m]:.2f}' for m in METRICS)+' |')
    report='**本文 BERTScore 结果（%，每种方法 7,458 例）**\n\n'+'\n'.join(lines)+'\n'
    (ROOT/'REPORT.zh.md').write_text(report)
    path=semantic.MANUSCRIPT
    txt=path.read_text()
    updated=render_manuscript(txt,summary)
    if txt!=updated:
        temp=path.with_name(path.name+'.bert.tmp');temp.write_text(updated);temp.replace(path)
    return summary

def main():
    import fcntl
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['pilot','run','summary'])
    p.add_argument('--lang',choices=['en','zh'])
    a=p.parse_args()
    if a.mode=='summary':
        d=final()
        if d:print(json.dumps(d,ensure_ascii=False))
        return
    if not a.lang:p.error('--lang required')
    folder=ROOT/a.lang;folder.mkdir(exist_ok=True,parents=True)
    with (folder/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        inv=Inventory(a.lang);arr=values_cache(inv,folder)
        encode(inv,folder,arr,a.mode=='pilot')
        if a.mode=='run':
            aggregate_language(inv,folder,arr)
            # Parallel workers may finish together.
            with (ROOT/'final.lock').open('a') as endlock:
                fcntl.flock(endlock,fcntl.LOCK_EX);final()
if __name__=='__main__':main()
