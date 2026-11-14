"""Give GPU 1 a fifth of each remaining partition, with serial model loading."""
import signal
import subprocess
import sys
import time
from common import *
from repartition_queue import alive, stop, official


def main():
    history = OUT / 'provenance/queue_gpu1_20261114'
    if history.exists():
        raise RuntimeError('GPU 1 handoff already has a record; inspect before repeating')
    busy = subprocess.run(['nvidia-smi', '-i', '1', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                          capture_output=True, text=True, check=True)
    if busy.stdout.strip():
        raise RuntimeError('GPU 1 is occupied')
    queue = read(OUT / 'queue_status.json')
    manifest = read(OUT / 'queue_manifest.json')
    keys = ['G03', 'G03_HELPER', 'G04', 'G04_HELPER']
    if queue['jobs']['E5_REPRESENTATION']['state'] != 'complete':
        raise RuntimeError('E5 has not completed')
    for key in keys:
        if queue['jobs'][key]['state'] != 'running' or not alive(queue['jobs'][key]['pid']):
            raise RuntimeError('Worker state changed: ' + key)
    cases = official.rows(official.OFFICIAL_CASES)
    by_case = {case['case_id']: case for case in cases}
    if len(cases) != 7458 or len(by_case) != 7458:
        raise ValueError('Unexpected official inventory')
    history.mkdir(parents=True)
    write(history / 'queue_status.before.json', queue)
    write(history / 'queue_manifest.before.json', manifest)
    for model in ['gemma3', 'medgemma']:
        directory = OUT / 'e6' / model / 'evaluation/mega_official'
        for name in ['assignment_shard0.json', 'assignment_shard1.json', 'partition_plan.json']:
            write(history / model / name, read(directory / name))
    write(history / 'handoff.json', dict(state='stopping_for_resume', gpu=1, reason='User identified idle GPU 1',
          requested_share=.2, started=time.time(), new_training=False, model_downloads=False))
    stop(queue['pid'], str(CODE / 'unified_queue.py'), signal.SIGTERM)
    limit = time.time() + 15
    while alive(queue['pid']) and time.time() < limit:
        time.sleep(.25)
    if alive(queue['pid']):
        raise RuntimeError('Previous scheduler has not stopped')
    for key in keys:
        stop(queue['jobs'][key]['pid'], str(CODE / 'official_partition_worker.py'), signal.SIGINT)
    limit = time.time() + 45
    while any(alive(queue['jobs'][key]['pid']) for key in keys) and time.time() < limit:
        time.sleep(.5)
    if any(alive(queue['jobs'][key]['pid']) for key in keys):
        raise RuntimeError('A partition worker has not stopped')
    queue = read(OUT / 'queue_status.json')
    jobs = {job['id']: job for job in manifest['jobs']}
    report = {}
    for model, base in [('gemma3', 'G03'), ('medgemma', 'G04')]:
        directory = OUT / 'e6' / model / 'evaluation/mega_official'
        complete, remaining_all, helper_ids = set(), set(), []
        result = dict(model=model, total_paths=7458, work_indices=[0, 1, 2], workers=[])
        global_pairs = set()
        for index, key in [(0, base), (1, base + '_HELPER')]:
            work = directory / ('shard' + str(index))
            predictions = rows(work / 'path_predictions.jsonl')
            pairs = {(r['case_id'], r['method']) for r in predictions}
            if len(pairs) != len(predictions) or pairs & global_pairs:
                raise ValueError('Pre-existing duplicate predictions')
            global_pairs.update(pairs)
            done_here = {r['case_id'] for r in predictions
                         if all((r['case_id'], m) in pairs for m in official.METHODS)}
            complete.update(done_here)
            previous = read(directory / ('assignment_shard' + str(index) + '.json'))
            remaining = [key for key in previous['case_ids'] if key not in done_here]
            if remaining_all & set(remaining):
                raise ValueError('Remaining assignments overlap')
            remaining_all.update(remaining)
            cached = rows(work / 'nodes.jsonl') + rows(work / 'natural_nodes.jsonl') + predictions
            touched = {r['case_id'] for r in cached}
            target_nodes = round(sum(len(by_case[key]['gold_path']) for key in remaining) * .2)
            moved, moved_nodes = set(), 0
            for case_id in reversed(remaining):
                if case_id in touched or moved_nodes >= target_nodes:
                    continue
                moved.add(case_id)
                moved_nodes += len(by_case[case_id]['gold_path'])
            retained = [key for key in remaining if key not in moved]
            helper_ids += [key for key in remaining if key in moved]
            assignment = dict(previous, case_ids=retained, paths=len(retained),
                              gold_path_nodes=sum(len(by_case[key]['gold_path']) for key in retained),
                              execution_round=2, share_to_gpu1=.2, no_parameter_changes=True)
            write(directory / ('assignment_shard' + str(index) + '.json'), assignment)
            old = queue['jobs'][key]
            queue['jobs'][key] = dict(state='pending', gpu=old['gpu'], units=[base],
                                     history=[dict(old, state='interrupted_for_partition', planned=True)])
            result['workers'].append(dict(work_index=index, gpu=old['gpu'], assigned_paths=len(retained),
                                         gold_nodes=assignment['gold_path_nodes'], moved_to_gpu1=len(moved),
                                         partial_cases_reused=len(set(retained) & touched)))
        # Original completed cases and both active partitions must cover the same population.
        if complete & remaining_all or complete | remaining_all != set(by_case):
            raise ValueError('Completed and unfinished cases do not cover official MEGA')
        selected = set(helper_ids)
        if len(selected) != len(helper_ids):
            raise ValueError('Duplicate GPU 1 assignments')
        helper_ids = [case['case_id'] for case in cases if case['case_id'] in selected]
        helper_assignment = directory / 'assignment_shard2.json'
        helper_nodes = sum(len(by_case[key]['gold_path']) for key in helper_ids)
        write(helper_assignment, dict(model=model, work_index=2, gpu=1, case_ids=helper_ids,
              paths=len(helper_ids), gold_path_nodes=helper_nodes, execution_round=2,
              source_cases=str(official.OFFICIAL_CASES), no_parameter_changes=True,
              selection='untouched remaining case IDs from each original partition; 20 percent of gold nodes'))
        new_key = base + '_HELPER2'
        job = dict(id=new_key, units=[base], script='official_partition_worker.py',
                   args=['--model', model, '--work-index', '2', '--assignment', str(helper_assignment)],
                   preferred_gpu=1, completion_file=str(directory / 'shard2/done.json'))
        if model == 'medgemma':
            job['dependencies'] = ['G03_HELPER2']
        manifest['jobs'].append(job)
        jobs[base + '_MERGE']['dependencies'].append(new_key)
        result.update(completed_paths_preserved=len(complete), remaining_paths=len(remaining_all),
                      gpu1_paths=len(helper_ids), gpu1_gold_nodes=helper_nodes,
                      overlap=0, helper_sequence='Gemma3 then MedGemma', existing_cache_preserved=True)
        result['workers'].append(dict(work_index=2, gpu=1, assigned_paths=len(helper_ids), gold_nodes=helper_nodes))
        if sum(w['assigned_paths'] for w in result['workers']) != len(remaining_all):
            raise ValueError('New assignments do not cover the remaining population')
        report[model] = result
        write(directory / 'partition_plan.json', result)
    manifest['gpu1_handoff'] = str(history / 'handoff.json')
    queue.update(state='restarting', active=0, updated=time.time())
    write(OUT / 'queue_manifest.json', manifest)
    write(OUT / 'queue_status.json', queue)
    write(history / 'handoff.json', dict(state='prepared', models=report, gpu=1,
          new_training=False, model_downloads=False, existing_outputs_preserved=True,
          helper_memory_gib=[55, 65], expected_remaining_hours=[1.5, 2.5], prepared_at=time.time()))
    with (OUT / 'scheduler.log').open('a') as log:
        process = subprocess.Popen([sys.executable, str(CODE / 'unified_queue.py')], cwd=PROJECT,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONUNBUFFERED='1'),
                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (OUT / 'scheduler.pid').write_text(str(process.pid) + '\n')
    print(json.dumps(dict(scheduler_pid=process.pid, models=report)))


if __name__ == '__main__':
    main()
