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
LEARNED = ['full', 'ordinary', 'no_js', 'context', 'blind_readout']
KINDS = ['zero', 'static', 'fusion', 'blind_lookup', 'ordinary', 'no_js', 'context', 'blind_readout', 'full']
CONFIRMATION_ORDER = ['full', 'blind_readout', 'ordinary', 'no_js', 'zero', 'fusion', 'blind_lookup', 'static', 'context']


def worker_alive(pid):
    root = Path('/proc') / str(pid)
    try:
        return root.joinpath('stat').read_text().split(') ', 1)[1][0] != 'Z' and b'aster_representation_20261101' in root.joinpath('cmdline').read_bytes()
    except FileNotFoundError:
        return False


def result_status(args, module):
    if '--train' in args:
        variant = 'blind_readout' if module == 'comparators' else args[args.index('--train') + 1]
        layer = args[args.index('--layer') + 1]
        return OUT / f'{variant}_layer{layer}/status.json'
    split = args[args.index('--evaluate') + 1]
    kind = args[args.index('--kind') + 1]
    return OUT / f'{split}_{kind}_status.json'


def decided_full_selection(complete):
    """Start dependent work once unfinished candidates cannot win on accuracy."""
    choices = []
    for layer in [18, 24, 30]:
        path = OUT / f'full_layer{layer}/selected.json'
        if path.exists():choices.append(json.loads(path.read_text()))
    if not choices:return None
    chosen = max(choices, key=selection_key)
    if all('train_full_'+str(layer) in complete for layer in [18,24,30]):return chosen
    total = len(read(OUT/'development.jsonl'))
    bounds = []
    for layer in [18,24,30]:
        if 'train_full_'+str(layer) in complete:continue
        for epoch in [1,2,3]:
            path = OUT/f'full_layer{layer}/development_epoch{epoch}.jsonl'
            rows = {}
            if path.exists():
                for line in path.read_text().splitlines():
                    try:row=json.loads(line)
                    except json.JSONDecodeError:continue  # writer may be completing its final line
                    if row.get('status')=='complete' and row.get('view')=='original':rows[row['case_id']]=row
            correct=sum(r['score']['exact_path'] and r['score']['leaf_correct'] for r in rows.values())
            upper=correct+total-len(rows)
            bounds.append({'layer':layer,'epoch':epoch,'evaluated':len(rows),'correct':correct,'maximum_possible':upper})
            # A possible tie still needs the original stability/edit-size tie breaks.
            if upper >= chosen['summary']['original']['joint']:return None
    write(OUT/'layer_selection_evidence.json',{'selected_layer':chosen['layer'],'selected_epoch':chosen['epoch'],
          'selected_joint':chosen['summary']['original']['joint'],'unfinished_candidate_bounds':bounds,
          'reason':'No unfinished checkpoint can tie the selected original joint count; evaluation continues.'})
    return chosen


def run_job(name, args, gpu, module='train', adopted=None):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
    started = time.time()
    if adopted is not None:
        while worker_alive(adopted['pid']):
            time.sleep(5)
        status = result_status(args, module)
        code = 0 if status.exists() and json.loads(status.read_text()).get('status') == 'complete' else 1
    else:
        with (OUT/(name+'.log')).open('a') as log:
            child = subprocess.Popen([sys.executable, '-m', 'experiments.aster_representation_20261101.'+module, *args],
                                     env=env, stdout=log, stderr=log)
            write(OUT/(name+'_job.json'), {'status': 'running', 'pid': child.pid, 'gpu': gpu, 'args': args, 'module':module})
            code = child.wait()
    write(OUT/(name+'_job.json'), {'status': 'complete' if code == 0 else 'failed',
                                 'exit_code': code, 'gpu': gpu, 'elapsed': time.time()-started,
                                 'args':args,'module':module,'adopted_existing_worker':adopted is not None})
    return name, code, gpu


def finish_report():
    import numpy as np
    text = ['# 处置盲表示引导的内部干预实验', '',
            '新增训练200例／20来源，原年龄最小对继续监督条件翻转；开发100例选配置，新100例／10来源确认。',
            '骨干冻结。所有方法使用相同合法路径与终点处置读取。原120例不作为本轮独立确认。', '']
    selected = {kind: json.loads((OUT/f'selected_{kind}.json').read_text()) for kind in LEARNED}
    write(OUT/'final_selected.json', selected)
    paired = {}
    for split in ['development', 'confirmation']:
        text += ['## '+split, '', '| 方法 | 原指南联合正确 | 交换处置联合正确 | 交换前后路径一致率 |', '|---|---:|---:|---:|']
        all_rows = {}
        for kind in KINDS:
            if split == 'development' and kind in selected:
                chosen = selected[kind]
                folder = Path(chosen['checkpoint']).parent
                path = folder/f'development_epoch{chosen["epoch"]}.jsonl'
            else:
                path = OUT/f'{split}_{kind}.jsonl'
            stats = json.loads(path.with_suffix('.summary.json').read_text())
            text.append(f"| {kind} | {stats['original']['joint']}/{stats['original']['n']} | {stats['swap']['joint']}/{stats['swap']['n']} | {stats['swap_path_agreement']:.3f} |")
            all_rows[kind] = {(r['case_id'], r['view']): r for r in read(path)}
        text += ['', '| 完整方法相对对照 | 视图 | 纠正 | 退化 | 来源等权差值及95%区间 |', '|---|---|---:|---:|---|']
        for control in [k for k in KINDS if k != 'full']:
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
                text.append(f'| {control} | {view} | {gains} | {losses} | {effects.mean():.3f} [{interval[0]:.3f}, {interval[1]:.3f}] |')
        text += ['', '| 原指南：盲判查表／完整上下文零干预 | 病例数 | 完整方法正确数 |', '|---|---:|---:|']
        complement = defaultdict(lambda: [0,0])
        for key,row in all_rows['full'].items():
            if key[1] != 'original':continue
            def correct(kind):
                score=all_rows[kind][key]['score'];return bool(score['exact_path'] and score['leaf_correct'])
            label=('正确' if correct('blind_lookup') else '错误')+'／'+('正确' if correct('zero') else '错误')
            complement[label][0]+=1;complement[label][1]+=int(correct('full'))
        for label,(n,correct_n) in sorted(complement.items()):text.append(f'| {label} | {n} | {correct_n} |')
        paired[split+'/complementarity']={k:{'n':v[0],'full_correct':v[1]} for k,v in complement.items()}
    text += ['', '## 条件翻转', '', '训练年龄对与已暴露确认年龄对分别报告；后者只作机制诊断，不称新的独立确认。', '',
             '| 方法 | 数据 | 原指南两态均正确 | 处置交换两态均正确 |', '|---|---|---:|---:|']
    for kind in KINDS:
        for split in ['age_train','age_diagnostic']:
            path = (Path(selected[kind]['checkpoint']).parent/(split+'.summary.json') if kind in selected
                    else OUT/f'{split}_{kind}.summary.json')
            if path.exists():
                stats = json.loads(path.read_text())
                text.append(f"| {kind} | {split} | {stats['original']['pairs_both_correct']}/{stats['original']['pairs']} | {stats['swap']['pairs_both_correct']}/{stats['swap']['pairs']} |")
    text += ['', '## 结论规则', '', '比较完整方法与零干预、直接盲判查表、标签融合、普通低秩、同样训练的上下文条件信息、盲表示分类器及无不变性损失对照。',
             '直接盲判与盲表示分类器在每个节点均作Y/N二选一，Unknown概率只作诊断，避免用拒答人为削弱对照。',
             '上下文对照把当前节点的T/F/U概率写入问题，并训练同参数量的普通低秩模块；盲表示分类器参数量匹配，不读取完整指南表示。',
             '互补表按完整路径计数，两侧路径都错时仍可能存在节点层面的互补，不能据此直接认定新推理能力。',
             '开发或确认若无额外收益，应如实报告未获支持；结构整理收益不算内部编辑收益。',
             '时间包含表示提取及诊断，需单独测量部署成本，不能直接声称提速。']
    write(OUT/'paired_results.json', paired)
    (OUT/'REPORT.zh.md').write_text('\n'.join(text)+'\n')
    repository = Path(__file__).resolve().parents[2]
    (repository/'docs/ASTER/representation-results-20261101.zh.md').write_text('\n'.join(text)+'\n')


def ready_confirmation_jobs(complete, scheduled):
    """Freeze each finished method once and queue only its missing confirmation."""
    full_path = OUT/'selected_full.json'
    if not full_path.exists():
        return []
    layer = json.loads(full_path.read_text())['layer']
    jobs = []
    for kind in CONFIRMATION_ORDER:
        name = 'confirmation_' + kind
        if name in scheduled:
            continue
        args = ['--evaluate', 'confirmation', '--kind', kind]
        frozen_path = OUT/f'freeze_confirmation_{kind}.json'
        if kind in LEARNED:
            prerequisite = f'train_full_{layer}' if kind == 'full' else 'train_' + kind
            if prerequisite not in complete:
                continue
            selected_path = OUT/f'selected_{kind}.json'
            if not selected_path.exists():
                write(selected_path, json.loads((OUT/f'{kind}_layer{layer}/selected.json').read_text()))
            chosen = json.loads(selected_path.read_text())
            if not frozen_path.exists():
                write(frozen_path, {'status': 'frozen', 'selection_data': 'development100',
                                   'frozen_at_unix': time.time(), 'model': chosen})
            args += ['--selection', str(selected_path)]
        else:
            if 'dev_' + kind not in complete:
                continue
            if not frozen_path.exists():
                write(frozen_path, {'status': 'frozen', 'kind': kind,
                                   'frozen_at_unix': time.time(), 'configuration': 'unchanged_from_development'})
        jobs.append((name, args, 'comparators' if kind in ['blind_lookup', 'blind_readout'] else 'train'))
    if jobs:
        frozen = {v: json.loads((OUT/f'freeze_confirmation_{v}.json').read_text())['model']
                  for v in LEARNED if (OUT/f'freeze_confirmation_{v}.json').exists()}
        write(OUT/'freeze_before_confirmation.json', {
            'status': 'frozen' if len(frozen) == len(LEARNED) else 'partially_frozen',
            'selection_data': 'development100', 'models': frozen,
            'pending_models': [v for v in LEARNED if v not in frozen],
            'policy': 'Each method is frozen before its own confirmation; other training may continue.'})
    return jobs


def main():
    pending = [(f'train_full_{layer}', ['--train','full','--layer',str(layer)], 'train') for layer in [18,24,30]]
    pending += [(f'dev_{kind}', ['--evaluate','development','--kind',kind], 'train') for kind in ['zero','fusion','static']]
    pending += [('dev_blind_lookup', ['--evaluate','development','--kind','blind_lookup'], 'comparators')]
    complete, failed = set(), {}
    full_selected = False
    free = list(GPUS); running = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        adopted_names = set()
        for path in OUT.glob('*_job.json'):
            name = path.name[:-len('_job.json')]
            if name == 'context_pilot':continue
            saved = json.loads(path.read_text())
            if saved.get('status') == 'complete':
                complete.add(name)
            elif saved.get('status') == 'running':
                args=saved['args'];module=saved.get('module','train')
                if worker_alive(saved['pid']):
                    gpu=saved['gpu'];free.remove(gpu)
                    future=executor.submit(run_job,name,args,gpu,module,saved)
                    running[future]={'name':name,'gpu':gpu};adopted_names.add(name)
                else:
                    status=result_status(args,module)
                    if status.exists() and json.loads(status.read_text()).get('status')=='complete':complete.add(name)
        pending=[job for job in pending if job[0] not in complete|adopted_names]
        # Keep one iteration even when every initial task is already complete;
        # dependent controls and confirmation still need to be reconstructed.
        first=True
        while first or pending or running:
            first=False
            while pending and free:
                name, args, module = pending.pop(0)
                if name in complete or name in adopted_names:continue
                status = OUT/(name+'_job.json')
                if status.exists() and json.loads(status.read_text())['status'] == 'complete':
                    complete.add(name); continue
                gpu = free.pop(0)
                future = executor.submit(run_job, name, args, gpu, module)
                running[future] = {'name': name, 'gpu': gpu}
            if running:
                finished, _ = concurrent.futures.wait(running, timeout=30,
                    return_when=concurrent.futures.FIRST_COMPLETED)
                for future in finished:
                    name, code, gpu = future.result(); running.pop(future); free.append(gpu)
                    if code: failed[name] = code
                    else: complete.add(name)
            chosen = decided_full_selection(complete) if not full_selected else None
            if chosen is not None:
                write(OUT/'selected_full.json', chosen)
                pending[0:0] = [(f'train_{variant}', ['--train',variant,'--layer',str(chosen['layer'])], 'train')
                                for variant in ['context','ordinary','no_js']]
                pending.append(('train_blind_readout',['--train','--layer',str(chosen['layer'])],'comparators'))
                full_selected = True
            if full_selected:
                scheduled = complete | adopted_names | set(failed) | {job[0] for job in pending}
                scheduled |= {job['name'] for job in running.values()}
                pending.extend(ready_confirmation_jobs(complete, scheduled))
            write(OUT/'queue_status.json', {'status':'running', 'active':list(running.values()),
                  'pending':[name for name,args,module in pending], 'complete':sorted(complete), 'failed':failed})
    success = not failed and all('confirmation_'+kind in complete for kind in KINDS)
    if success: finish_report()
    write(OUT/'queue_status.json', {'status':'complete' if success else 'failed', 'complete':sorted(complete), 'failed':failed})


if __name__ == '__main__': main()
