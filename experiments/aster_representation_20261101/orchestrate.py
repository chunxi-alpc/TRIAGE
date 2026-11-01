import concurrent.futures
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from .data import OUT, read, write
from .train import selection_key

GPUS = [0, 1]


def run_job(name, args, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
    started = time.time()
    with (OUT/(name+'.log')).open('a') as log:
        child = subprocess.Popen([sys.executable, '-m', 'experiments.aster_representation_20261101.train', *args],
                                 env=env, stdout=log, stderr=log)
        write(OUT/(name+'_job.json'), {'status': 'running', 'pid': child.pid, 'gpu': gpu, 'args': args})
        code = child.wait()
    write(OUT/(name+'_job.json'), {'status': 'complete' if code == 0 else 'failed',
                                 'exit_code': code, 'gpu': gpu, 'elapsed': time.time()-started})
    return name, code, gpu


def finish_report():
    import numpy as np
    text = ['# 处置盲表示引导的内部干预实验', '',
            '新增训练200例／20来源，原年龄最小对继续监督条件翻转；开发100例选配置，新100例／10来源确认。',
            '骨干冻结。所有方法使用相同合法路径与终点处置读取。原120例不作为本轮独立确认。', '']
    selected = {kind: json.loads((OUT/f'selected_{kind}.json').read_text()) for kind in ['full','ordinary','no_js']}
    write(OUT/'final_selected.json', selected)
    paired = {}
    for split in ['development', 'confirmation']:
        text += ['## '+split, '', '| 方法 | 原指南联合正确 | 交换处置联合正确 | 交换前后路径一致率 |', '|---|---:|---:|---:|']
        all_rows = {}
        for kind in ['zero', 'static', 'fusion', 'ordinary', 'no_js', 'full']:
            if split == 'development' and kind in selected:
                chosen = selected[kind]
                folder = Path(chosen['checkpoint']).parent
                path = folder/f'development_epoch{chosen["epoch"]}.jsonl'
            else:
                path = OUT/f'{split}_{kind}.jsonl'
            stats = json.loads(path.with_suffix('.summary.json').read_text())
            text.append(f"| {kind} | {stats['original']['joint']}/{stats['original']['n']} | {stats['swap']['joint']}/{stats['swap']['n']} | {stats['swap_path_agreement']:.3f} |")
            all_rows[kind] = {(r['case_id'], r['view']): r for r in read(path)}
        for control in ['zero', 'static', 'fusion', 'ordinary', 'no_js']:
            for view in ['original', 'swap']:
                groups = defaultdict(list); gains = losses = 0
                for key, row in all_rows['full'].items():
                    if key[1] != view: continue
                    base = all_rows[control][key]
                    a = bool(row['score']['exact_path'] and row['score']['leaf_correct'])
                    b = bool(base['score']['exact_path'] and base['score']['leaf_correct'])
                    gains += a and not b; losses += b and not a
                    groups[row['source_id']].append(int(a)-int(b))
                effects = np.array([np.mean(v) for v in groups.values()])
                rng = np.random.default_rng(20261101)
                interval = np.quantile(effects[rng.integers(len(effects), size=(2000,len(effects)))].mean(1), [.025,.975]).tolist()
                paired[f'{split}/{view}/full_vs_{control}'] = {'corrected': gains, 'regressed': losses,
                    'source_equal_delta': float(effects.mean()), 'source_bootstrap_95': interval}
    text += ['', '## 条件翻转', '', '训练年龄对与已暴露确认年龄对分别报告；后者只作机制诊断，不称新的独立确认。', '',
             '| 方法 | 数据 | 原指南两态均正确 | 处置交换两态均正确 |', '|---|---|---:|---:|']
    for kind in ['zero','static','fusion','full','ordinary','no_js']:
        for split in ['age_train','age_diagnostic']:
            path = (Path(selected[kind]['checkpoint']).parent/(split+'.summary.json') if kind in selected
                    else OUT/f'{split}_{kind}.summary.json')
            if path.exists():
                stats = json.loads(path.read_text())
                text.append(f"| {kind} | {split} | {stats['original']['pairs_both_correct']}/{stats['original']['pairs']} | {stats['swap']['pairs_both_correct']}/{stats['swap']['pairs']} |")
    text += ['', '## 结论规则', '', '比较完整方法与零干预、标签融合、普通低秩干预及无不变性损失对照。',
             '开发或确认若无额外收益，应如实报告未获支持；结构整理收益不算内部编辑收益。',
             '时间包含表示提取及诊断，需单独测量部署成本，不能直接声称提速。']
    write(OUT/'paired_results.json', paired)
    (OUT/'REPORT.zh.md').write_text('\n'.join(text)+'\n')
    repository = Path(__file__).resolve().parents[2]
    (repository/'docs/ASTER/representation-results-20261101.zh.md').write_text('\n'.join(text)+'\n')


def main():
    pending = [(f'train_full_{layer}', ['--train','full','--layer',str(layer)]) for layer in [18,24,30]]
    pending += [(f'dev_{kind}', ['--evaluate','development','--kind',kind]) for kind in ['zero','fusion','static']]
    complete, failed = set(), {}
    full_selected = False; confirmation_queued = False
    free = list(GPUS); running = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        while pending or running:
            while pending and free:
                name, args = pending.pop(0)
                status = OUT/(name+'_job.json')
                if status.exists() and json.loads(status.read_text())['status'] == 'complete':
                    complete.add(name); continue
                gpu = free.pop(0)
                future = executor.submit(run_job, name, args, gpu)
                running[future] = {'name': name, 'gpu': gpu}
            if running:
                finished, _ = concurrent.futures.wait(running, timeout=30,
                    return_when=concurrent.futures.FIRST_COMPLETED)
                for future in finished:
                    name, code, gpu = future.result(); running.pop(future); free.append(gpu)
                    if code: failed[name] = code
                    else: complete.add(name)
            if not full_selected and all('train_full_'+str(l) in complete for l in [18,24,30]):
                choices = [json.loads((OUT/f'full_layer{layer}/selected.json').read_text()) for layer in [18,24,30]]
                chosen = max(choices, key=selection_key); write(OUT/'selected_full.json', chosen)
                pending[0:0] = [(f'train_{variant}', ['--train',variant,'--layer',str(chosen['layer'])])
                                for variant in ['ordinary','no_js']]
                full_selected = True
            if full_selected and not confirmation_queued and all('train_'+v in complete for v in ['ordinary','no_js']):
                layer = json.loads((OUT/'selected_full.json').read_text())['layer']
                for variant in ['ordinary','no_js']:
                    chosen = json.loads((OUT/f'{variant}_layer{layer}/selected.json').read_text())
                    write(OUT/f'selected_{variant}.json', chosen)
                write(OUT/'freeze_before_confirmation.json', {'status': 'frozen', 'selection_data': 'development100',
                     'models': {v:json.loads((OUT/f'selected_{v}.json').read_text()) for v in ['full','ordinary','no_js']}})
                for kind in ['zero','fusion','static','full','ordinary','no_js']:
                    args = ['--evaluate','confirmation','--kind',kind]
                    if kind in ['full','ordinary','no_js']:
                        args += ['--selection',str(OUT/f'selected_{kind}.json')]
                    pending.append(('confirmation_'+kind, args))
                confirmation_queued = True
            write(OUT/'queue_status.json', {'status':'running', 'active':list(running.values()),
                  'pending':[name for name,args in pending], 'complete':sorted(complete), 'failed':failed})
    success = not failed and confirmation_queued
    if success: finish_report()
    write(OUT/'queue_status.json', {'status':'complete' if success else 'failed', 'complete':sorted(complete), 'failed':failed})


if __name__ == '__main__': main()
