"""Development-only structure x key-suppression experiment; persistent GPU workers."""
import argparse, copy, fcntl, json, os, subprocess, sys, time, traceback
from collections import defaultdict
from pathlib import Path
from .run import OUT, read, write
from .module_one_data import M1

DEST = OUT / 'structure_followup_20261101'
GPUS = [0, 1]


def prepare():
    DEST.mkdir(exist_ok=True)
    cases = read(M1 / 'development100.jsonl')
    frozen = json.loads((M1 / 'frozen.json').read_text())['selected']
    mean = {k: v for k, v in frozen.items() if k not in ['name', 'deeper', 'probe_only']}
    specs = [{'name': 'zero_structured', 'kind': 'zero', 'structured': True}]
    for name in ['mean', 'suppress', 'clip']:
        for structured in [False, True]:
            spec = {'kind': 'clip'} if name == 'clip' else copy.deepcopy(mean)
            if name == 'suppress':
                spec['suppress_only'] = True
            spec.update(name=name + ('_structured' if structured else '_free'), structured=structured)
            specs.append(spec)
    baseline = {r['case_id']: r for i in range(4) for r in read(M1 / f'baseline_{i}.jsonl')}
    ordered = sorted(cases, key=lambda c: baseline[c['case_id']]['result']['input_tokens'])
    pilot = [ordered[len(ordered)//2]['case_id'], ordered[-1]['case_id']]
    cfg = {'dataset': 'development100 only', 'case_ids': [c['case_id'] for c in cases],
           'pilot_case_ids': pilot, 'specs': specs, 'gpus': GPUS, 'route_variant': 'direct:0.0',
           'frozen_mean_gain': mean['gain'], 'direction_heads': len(mean['direction_heads']),
           'logical_cells': 800, 'reused_baseline_cells': 100, 'new_cells_including_pilot': 700,
           'primary': 'exact_path AND leaf_correct',
           'decoder': 'full original tree grammar for every arm; terminal action lookup',
           'validation_policy': 'No validation120 execution or selection in this stage',
           'next_stage': 'Review development results before adding other operators'}
    file = DEST / 'config.json'
    if file.exists() and json.loads(file.read_text()) != cfg:
        raise ValueError('Existing configuration differs')
    write(file, cfg)
    reused = []
    for case in cases:
        row = copy.deepcopy(baseline[case['case_id']])
        row.update(arm='zero_free', stage='structure_followup',
                   reused_from='module_one/baseline', reuse_reason='identical no-edit free generation')
        reused.append(row)
    (DEST / 'reused.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in reused))
    return cfg


def records():
    rows = {}
    for file in sorted(DEST.glob('*.jsonl')):
        for row in read(file):
            if row.get('status') == 'complete' and 'arm' in row:
                rows[row['case_id'], row['arm']] = row
    return list(rows.values())


def claim(mode):
    with (DEST / (mode + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        file = DEST / (mode + '_queue.json')
        queue = json.loads(file.read_text())
        if not queue:
            return None
        cid = queue.pop(0)
        write(file, queue)
        return cid


def worker(mode, gpu):
    import torch
    from .module_one import make_engine
    from .runtime import execute
    from .structured import terminal_paths
    from experiments.aster_manuscript_clinical_20261030.native_memory import query_chunked_attention
    from experiments.aster_abc_diagnostics_20261030.c.audit import strict_score
    cfg = json.loads((DEST / 'config.json').read_text())
    cases = {c['case_id']: c for c in read(M1 / 'development100.jsonl')}
    routing = {r['case_id']: r['result']['outcomes'][cfg['route_variant']]['path']
               for i in range(4) for r in read(OUT / f'routing_{i}.jsonl') if r['status'] == 'complete'}
    done = {(r['case_id'], r['arm']) for r in records()}
    started = time.time()
    engine = make_engine()
    output = DEST / f'{mode}_gpu{gpu}.jsonl'
    n = 0
    with query_chunked_attention(engine.model, 256), output.open('a') as handle:
        while True:
            cid = claim(mode)
            if cid is None:
                break
            case = cases[cid]
            legal = {tuple(path): leaf for path, leaf in terminal_paths(case['guideline'])}
            for spec in cfg['specs']:
                if (cid, spec['name']) in done:
                    continue
                write(DEST / f'{mode}_gpu{gpu}_status.json',
                      {'status': 'running', 'case_id': cid, 'arm': spec['name'], 'records': n})
                row = {'case_id': cid, 'source_id': case['source_id'], 'arm': spec['name'],
                       'spec': spec, 'applied_route': routing[cid], 'gold_route_diagnostic': False,
                       'stage': 'development100_structure_followup'}
                try:
                    result = execute(engine, case, spec, route=routing[cid])
                    score = strict_score(case, result['text'])
                    score['joint_correct'] = score['exact_path'] and score['leaf_correct']
                    score['graph_consistent'] = legal.get(tuple(score['path'])) == score['leaf'] and score['leaf'] is not None
                    if spec.get('structured') and not result['output_limit'] and (not score['graph_consistent'] or score['invalid']):
                        raise RuntimeError('Structured decoder violated the original tree')
                    row.update(status='complete', result=result, score=score)
                except Exception as exc:
                    traceback.print_exc()
                    row.update(status='failed', error=str(exc))
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                handle.flush()
                if row['status'] != 'complete':
                    raise RuntimeError(row['error'])
                done.add((cid, spec['name']))
                n += 1
    write(DEST / f'{mode}_gpu{gpu}_status.json', {'status': 'complete', 'records': n,
          'elapsed': time.time()-started, 'peak_gb': torch.cuda.max_memory_allocated()/1e9})


def summarize(mode):
    cfg = json.loads((DEST / 'config.json').read_text())
    selected = set(cfg['pilot_case_ids'] if mode == 'pilot' else cfg['case_ids'])
    groups = defaultdict(list)
    for row in records():
        if row['case_id'] in selected:
            groups[row['arm']].append(row)
    stats = {}
    for arm, rows in groups.items():
        stats[arm] = {'n': len(rows),
          'path': sum(r['score']['exact_path'] for r in rows),
          'leaf': sum(r['score']['leaf_correct'] for r in rows),
          'joint': sum(r['score']['exact_path'] and r['score']['leaf_correct'] for r in rows),
          'invalid': sum(r['score']['invalid'] for r in rows),
          'mean_seconds': sum(r['result']['seconds'] for r in rows)/len(rows)}
    comparisons = {}
    pairs = [('zero_structured', 'zero_free'), ('mean_structured', 'mean_free'),
             ('suppress_structured', 'suppress_free'), ('clip_structured', 'clip_free'),
             ('suppress_free', 'mean_free'), ('suppress_structured', 'mean_structured'),
             ('mean_structured', 'zero_structured'), ('suppress_structured', 'zero_structured'),
             ('mean_structured', 'clip_structured'), ('suppress_structured', 'clip_structured')]
    for first, second in pairs:
        a = {r['case_id']: r for r in groups[first]}
        b = {r['case_id']: r for r in groups[second]}
        common = sorted(a.keys() & b.keys())
        changes = []
        source_changes = defaultdict(list)
        for cid in common:
            sa, sb = a[cid]['score'], b[cid]['score']
            x = int(sa['exact_path'] and sa['leaf_correct'])
            y = int(sb['exact_path'] and sb['leaf_correct'])
            changes.append((x,y))
            source_changes[a[cid]['source_id']].append(x-y)
        comparisons[first + ' vs ' + second] = {
            'paired_cases': len(common), 'corrected': sum(x and not y for x,y in changes),
            'regressed': sum(y and not x for x,y in changes),
            'net': sum(x-y for x,y in changes),
            'source_equal_delta': sum(sum(v)/len(v) for v in source_changes.values())/len(source_changes) if source_changes else None}
    write(DEST / (mode + '_paired.json'), comparisons)
    write(DEST / (mode + '_summary.json'), stats)
    lines = ['# 结构约束与只抑制编辑：开发集实验', '',
             '仅开发100例；复用模块二direct:0.0，均值方向、280个头及增益5.989389629154409固定。',
             '全部结构化条件使用完整原始规则树约束贪心生成，不强制遵循模块二路径；终点处置由规则读取。',
             '8个条件×100例，零编辑自由生成100例复用旧结果；另700个条件单元包含小批次。',
             '主要指标为路径与处置同时正确。当前不是独立验证结果。', '',
             '| 条件 | 已完成 | 路径 | 处置 | 两者同时正确 | 无效 | 平均秒/例 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for arm, s in stats.items():
        lines.append(f"| {arm} | {s['n']} | {s['path']} | {s['leaf']} | {s['joint']} | {s['invalid']} | {s['mean_seconds']:.2f} |")
    lines += ['', '所有对照均以预测路径执行；不改变原120例报告。自由生成耗时包含旧诊断前向，不能据此声称部署速度优势。', '',
              '| 配对比较（联合指标） | 配对例数 | 纠正 | 退化 | 净变化 |', '|---|---:|---:|---:|---:|']
    for label, result in comparisons.items():
        lines.append(f"| {label} | {result['paired_cases']} | {result['corrected']} | {result['regressed']} | {result['net']} |")
    (DEST / ('PILOT.zh.md' if mode == 'pilot' else 'REPORT.zh.md')).write_text('\n'.join(lines) + '\n')
    return stats


def launch(mode):
    cfg = prepare()
    wanted = cfg['pilot_case_ids'] if mode == 'pilot' else cfg['case_ids']
    done = {(r['case_id'], r['arm']) for r in records()}
    queue = [cid for cid in wanted if any((cid, s['name']) not in done for s in cfg['specs'])]
    write(DEST / (mode + '_queue.json'), queue)
    children = []
    write(DEST / (mode + '_status.json'), {'status': 'running', 'gpus': GPUS, 'pending_cases': len(queue)})
    for gpu in GPUS:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
        with (DEST / f'{mode}_gpu{gpu}.log').open('a') as log:
            child = subprocess.Popen([sys.executable, '-m', __package__ + '.structure_followup',
                                      '--worker', mode, '--gpu', str(gpu)], env=env, stdout=log, stderr=log)
            children.append(child)
    write(DEST / (mode + '_launch.json'), {'pid': os.getpid(), 'workers': [p.pid for p in children], 'gpus': GPUS})
    codes = [child.wait() for child in children]
    summary = summarize(mode)
    complete = not any(codes) and all(s['n'] == len(wanted) for s in summary.values()) and len(summary) == 8
    write(DEST / (mode + '_status.json'), {'status': 'complete' if complete else 'failed', 'gpus': GPUS,
          'exit_codes': codes, 'completed_records': sum(s['n'] for s in summary.values())})
    if not complete:
        raise RuntimeError('Follow-up worker failed or records missing')
    if mode == 'pilot':
        per_case = sum(s['mean_seconds'] for arm, s in summary.items() if arm != 'zero_free')
        peak = max(json.loads((DEST/f'pilot_gpu{gpu}_status.json').read_text())['peak_gb'] for gpu in GPUS)
        write(DEST/'pilot_estimate.json', {'pilot_cases': len(wanted), 'new_pilot_cells': len(wanted)*7,
              'peak_gb': peak, 'mean_seconds_per_case_seven_arms': per_case,
              'estimated_two_gpu_minutes': per_case*100/2/60,
              'scope': 'median-length and longest development inputs; no outcome-based sampling'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['pilot', 'formal'])
    parser.add_argument('--worker', choices=['pilot', 'formal'])
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.worker:
        worker(args.worker, args.gpu)
    else:
        launch(args.mode)
