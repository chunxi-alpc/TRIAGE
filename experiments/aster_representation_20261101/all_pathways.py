"""Score every registered method's single-turn pathways using identical judge settings."""
import argparse
import collections
import concurrent.futures as futures
import fcntl
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from . import api_judge as judge
from . import official, paired_judge as paired, semantic

ROOT = judge.BASE / 'all_methods_pathways'
INDEX = ROOT / 'inputs.jsonl'
METHODS = ('no_js', 'blind_readout', 'zero', 'fusion', 'ordinary', 'context')
DISPLAY = ('zero', 'fusion', 'ordinary', 'context', 'blind_readout', 'no_js')
LABELS = dict(zip(DISPLAY, ('Qwen3-8B：零干预', '标签概率融合', '普通低秩干预', '概率上下文',
                            '盲表示分类器', '**处置盲内部干预（本文）**')))
METRICS = ('DTW_PCC', 'OT_PCC', 'LNDS_PCS')
PROTOCOL = {'model': judge.MODEL, 'draws': 5, 'lnds_threshold': semantic.CONFIG['lnds_threshold'],
            'ot_reg': semantic.CONFIG['ot_reg'], 'ot_tau': semantic.CONFIG['ot_tau'],
            'ot_beta': semantic.CONFIG['ot_beta'], 'ot_cost': semantic.CONFIG['ot_cost']}
def read_lines(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]

def prepare():
    ROOT.mkdir(exist_ok=True)
    if INDEX.exists():
        return
    cases = {r['case_id']: r for r in official.all_cases()}
    paired_inputs = {r['case_id']: r for r in read_lines(paired.INDEX)}
    records = {}
    for file in sorted((official.OUT/'results').glob('*.jsonl')):
        for row in read_lines(file):
            kind, cid = row['kind'], row['case_id']
            if kind not in METHODS:
                continue
            prompt_steps = None
            if row['status'] == 'complete' and not row['score'].get('invalid',False):
                case = cases[cid]
                text = semantic.canonical(case, row['result']['text'])
                parsed = judge.official.parse_record(dict(id=cid, case_path_id=case['guideline_id'],
                    path_index=case['path_id'], filename=case['filename'], language=case['language'], llm_output=text))
                prompt_steps = judge.official.pcc.get_candidate_steps(parsed)
            records.setdefault(cid,{})[kind] = prompt_steps
    if len(cases) != 7458 or set(cases) != set(records):
        raise ValueError('Incomplete official prediction coverage')
    tmp = INDEX.with_suffix('.tmp')
    with tmp.open('w') as handle:
        for cid, case in cases.items():
            per_method = records[cid]
            if set(per_method) != set(METHODS):
                raise ValueError('Missing method for '+cid)
            old = paired_inputs[cid]
            for kind, key in [('no_js','main'),('blind_readout','blind')]:
                if per_method[kind] != old[key]['candidate_steps']:
                    raise ValueError('Previously registered input differs for '+cid)
            owners, steps = {}, {}
            for kind in METHODS:
                sequence = per_method[kind]
                if sequence is None:
                    owners[kind] = None
                    continue
                owner = next((k for k,v in steps.items() if v == sequence), kind)
                owners[kind] = owner
                steps[owner] = sequence
            rec = dict(case_id=cid, source=old['source'], language=case['language'],
                       reference_steps=old['main']['reference_steps'], owners=owners, steps=steps)
            handle.write(json.dumps(rec,ensure_ascii=False)+'\n')
    tmp.replace(INDEX)

def owner_root(owner):
    if owner == 'no_js':
        return judge.OUT
    if owner == 'blind_readout':
        return paired.ROOT
    return ROOT/'extra'/owner

def score_path(rec,owner,allow_requests):
    cid = rec['case_id']
    steps = rec['steps'][owner]
    refs = rec['reference_steps']
    root = owner_root(owner)
    folder = paired.folder(root,cid)
    prompt = judge.official.pcc.build_prompt(rec['language'],refs,steps)
    if allow_requests:
        case = dict(case_id=cid,language=rec['language'],reference_steps=refs,candidate_steps=steps)
        result = judge.score(case,output_root=root,path_only=True)
    else:
        result = {}
    draws, blocked = [], False
    for i in range(1,6):
        saved = judge.read(folder/f'path_{i}.json')
        if saved and saved.get('status') == 'complete':
            if saved['model'] != judge.MODEL or saved['prompt'] != prompt:
                raise ValueError('Judge or pathway prompt differs for '+cid)
            draws.append(saved['score'])
        else:
            blocked |= any(folder.glob(f'path_{i}_invalid_*.json'))
    if len(draws) != 5:
        status = 'unscorable' if blocked else 'failed' if result.get('status')=='failed' else 'pending'
        return dict(status=status,draws=len(draws))
    if np.shape(draws) != (5,len(refs),len(steps)):
        raise ValueError('Unexpected matrix dimensions for '+cid)
    values = semantic.compute_path_metrics(draws)
    return dict(status='complete',draws=5,**{k:float(values[k]) for k in METRICS})

def evaluate(rec,allow_requests):
    path = ROOT/'cases'/rec['case_id'].replace(':','_')/'path_metrics.json'
    prior = judge.read(path) or {}
    values = prior.get('values',{}) if prior.get('protocol')==PROTOCOL else {}
    for owner in rec['steps']:
        previous_status = values.get(owner,{}).get('status')
        if previous_status == 'complete':
            continue
        # A former job may have saved a valid matrix after our snapshot. Check it locally
        # even for a blocked stage; do not send this case/prompt again after refusal.
        values[owner] = score_path(rec,owner,allow_requests and previous_status != 'unscorable')
    row = dict(case_id=rec['case_id'],source=rec['source'],owners=rec['owners'],values=values,protocol=PROTOCOL)
    official.write(path,row)
    return row

def stats(values):
    valid = [v for v in values if v.get('status')=='complete']
    return dict(n=len(valid),status_counts=dict(collections.Counter(v['status'] for v in values)),
                means={k:100*sum(v[k] for v in valid)/len(valid) if valid else None for k in METRICS})

def summarize(states):
    if len(states)!=7458:
        raise ValueError('Summary requires all registered case records')
    per_method = {k:[] for k in METHODS}
    common = {k:[] for k in METHODS}
    unique = collections.Counter()
    filled_draws = additional = all_missing = 0
    for rec in states.values():
        values = rec['values']
        group = {k: values.get(o,{'status':'invalid_prediction'}) for k,o in rec['owners'].items()}
        all_ready = all(v['status']=='complete' for v in group.values())
        for k,v in group.items():
            per_method[k].append(v)
            if all_ready:
                common[k].append(v)
        for owner,v in values.items():
            unique[v['status']]+=1
            filled_draws+=v['draws']
            if v['status'] in ('pending','failed'):
                all_missing+=5-v['draws']
                if owner not in ('no_js','blind_readout'):
                    additional+=5-v['draws']
    d=dict(protocol=PROTOCOL,registered_cases=len(states),registered_method_cases=6*len(states),
           unique_paths=sum(unique.values()),unique_status=dict(unique),valid_draws=filled_draws,
           remaining_draws_excluding_known_blocked=all_missing,extra_draws_beyond_paired_queue=additional,
           methods={k:stats(per_method[k]) for k in DISPLAY},common_cases={k:stats(common[k]) for k in DISPLAY})
    official.write(ROOT/'summary.json',d)
    report=['# 六种方法的路径语义评分','',
            f"每种方法登记 {len(states):,} 例。使用 {judge.MODEL}，每份路径取五次有效步骤评分矩阵。",'']
    for title,key in [('各方法已有有效评分','methods'),('六种方法共同有有效评分的病例','common_cases')]:
        report.extend([f'## {title}','','| 方法 | 有效例数 | DTW-PCC↓ | OT-PCC↓ | LNDS-PCS↑ |','|---|---:|---:|---:|---:|'])
        for method in DISPLAY:
            stat=d[key][method]
            numbers=['未取得有效评分' if stat['means'][k] is None else f"{stat['means'][k]:.2f}" for k in METRICS]
            report.append('| '+LABELS[method]+f" | {stat['n']:,} | "+' | '.join(numbers)+' |')
        report.append('')
    (ROOT/'REPORT.zh.md').write_text('\n'.join(report)+'\n')
    return d

def render(text,d):
    # The external papers' reported cells and the measured Acc column stay intact.
    h='<!-- ASTER_FINAL_PATH_START -->'
    e='<!-- ASTER_FINAL_PATH_END -->'
    before,rest=text.split(h,1)
    table,after=rest.split(e,1)
    def row_name(line):
        return line.split('|')[1].strip().split('（n=')[0] if line.startswith('|') else ''
    external=[l for l in table.strip().splitlines() if row_name(l) not in LABELS.values()]
    for kind in DISPLAY:
        row=d['methods'][kind]
        cells=['未取得有效评分' if row['means'][k] is None else f"{row['means'][k]:.2f}" for k in METRICS]
        external.append('| '+LABELS[kind]+f"（n={row['n']:,}） | "+' | '.join(cells)+' |')
    text=before+h+'\n\n'+'\n'.join(external)+'\n\n'+e+after
    text=text.replace('主方法的语义评分仅在路径和结论均取得五次有效评分后纳入阶段均值，另报覆盖率。','每项语义指标按取得五次有效评分的病例汇总，并报告对应例数。')
    text=text.replace('表1的主方法使用gpt-6.1-sol裁判','本文六种方法的步骤语义评分使用gpt-6.1-sol裁判')
    text=text.replace('LNDS阈值固定为0.9。','本文对六种方法使用预先固定的 LNDS 匹配阈值 0.9。')
    # Table 1 and Table 2 should show the same DTW and effective denominator.
    h='<!-- ASTER_FINAL_MAIN_START -->';e='<!-- ASTER_FINAL_MAIN_END -->'
    pre,rest=text.split(h,1);table,post=rest.split(e,1)
    old=re.search(r'阶段性语义评分（†）[^\n]*?([\d,]+)/7,458',text)
    by_label={v:k for k,v in LABELS.items()}
    lines=table.split('\n')
    for i,line in enumerate(lines):
        k=by_label.get(row_name(line))
        if not k:continue
        cells=line.split('|');v=d['methods'][k]
        if v['n']:
            cells[3]=f" {v['means']['DTW_PCC']:.2f}（n={v['n']:,}） "
        if k=='no_js' and old and '†' in cells[4]:
            cells[4]=cells[4].replace('†','').rstrip()+f"（n={old[1]}） "
        lines[i]='|'.join(cells)
    text=pre+h+'\n'.join(lines)+e+post
    text=re.sub(r'^> 阶段性语义评分（†）[^\n]*\n?', '',text,flags=re.M)
    text=text.replace('‡ 为输出一致子集的均值。','‡ 为结论语义评分（Acc）在输出一致子集上的均值。').replace('在主方法已评分的','在当次主方法已评分的')
    return text

def write_server_table(d):
    path=semantic.MANUSCRIPT
    old=path.read_text();updated=render(old,d)
    if updated!=old:
        tmp=path.with_name(path.name+'.all-paths.tmp');tmp.write_text(updated);tmp.replace(path)

def run(mode):
    prepare()
    records=read_lines(INDEX)
    if mode=='snapshot':
        states={r['case_id']:evaluate(r,False) for r in records}
        d=summarize(states)
        print(json.dumps(d),flush=True)
        return
    if mode=='render':
        write_server_table(judge.read(ROOT/'summary.json'))
        return
    with (ROOT/'all.lock').open('a') as own_lock,(judge.OUT/'runner.lock').open('a') as shared_lock:
        fcntl.flock(own_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        info=dict(status='queued',pid=os.getpid(),workers=32,registered=7458,methods=DISPLAY,started=time.time())
        official.write(ROOT/'launch.json',info)
        # Preserve the 32-request API ceiling and in-flight draws of the existing paired experiment.
        fcntl.flock(shared_lock,fcntl.LOCK_EX)
        info.update(status='running',api_started=time.time());official.write(ROOT/'launch.json',info)
        states={r['case_id']:judge.read(ROOT/'cases'/r['case_id'].replace(':','_')/'path_metrics.json') for r in records}
        with futures.ThreadPoolExecutor(32) as pool:
            jobs={pool.submit(evaluate,r,True):r for r in records}
            for i,job in enumerate(futures.as_completed(jobs),1):
                row=job.result();states[row['case_id']]=row
                if i%100==0:summarize(states)
            for _ in range(2):
                retry=[r for r in records if any(v['status']=='failed' for v in states[r['case_id']]['values'].values())]
                if not retry:break
                for row in pool.map(lambda r:evaluate(r,True),retry):
                    states[row['case_id']]=row
        d=summarize(states);write_server_table(d)
        info.update(status='completed' if d['unique_status'].get('complete',0)==d['unique_paths'] else 'finished_with_missing',
                    finished=time.time(),unique_status=d['unique_status'])
        official.write(ROOT/'launch.json',info)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['snapshot','run','render'])
    run(parser.parse_args().mode)
