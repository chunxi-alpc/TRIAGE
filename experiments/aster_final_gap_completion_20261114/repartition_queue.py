"""Move only unfinished official cases to GPU pairs 5/0 and 6/4."""
import signal
import subprocess
import time
import sys
from common import *

sys.path.insert(0, str(REPO6))
from experiments.aster_e678 import official_mega as official


def alive(pid):
    try:
        return (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def stop(pid, expected, sig):
    command = (Path('/proc') / str(pid) / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    if expected not in command:
        raise RuntimeError('Process identity changed before planned handoff')
    os.kill(pid, sig)


def main():
    history = OUT / 'provenance/queue_repartition_20261114'
    if history.exists():
        raise RuntimeError('This handoff already has a record; inspect it before repeating')
    queue = read(OUT / 'queue_status.json')
    manifest = read(OUT / 'queue_manifest.json')
    for gpu in [0, 4]:
        r = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, check=True)
        if r.stdout.strip():
            raise RuntimeError('Requested helper GPU is occupied: ' + str(gpu))
    cases = official.rows(official.OFFICIAL_CASES)
    if len(cases) != 7458 or len({c['case_id'] for c in cases}) != 7458:
        raise ValueError('Unexpected official inventory')
    for key in ['G03', 'G04']:
        if queue['jobs'][key]['state'] != 'running' or not alive(queue['jobs'][key]['pid']):
            raise RuntimeError('Primary worker state changed: ' + key)
    history.mkdir(parents=True)
    write(history / 'queue_status.before.json', queue)
    write(history / 'queue_manifest.before.json', manifest)
    write(history / 'handoff.json', dict(state='stopping_for_resume', reason='User requested GPUs 0 and 4',
          started=time.time(), existing_outputs_preserved=True, new_training=False, model_downloads=False))
    # Terminating the scheduler leaves its separately running model workers alive.
    stop(queue['pid'], str(CODE / 'unified_queue.py'), signal.SIGTERM)
    limit = time.time() + 15
    while alive(queue['pid']) and time.time() < limit:
        time.sleep(.25)
    if alive(queue['pid']):
        raise RuntimeError('Previous scheduler has not stopped')
    for key in ['G03', 'G04']:
        stop(queue['jobs'][key]['pid'], str(CODE / 'official_worker.py'), signal.SIGINT)
    limit = time.time() + 45
    while any(alive(queue['jobs'][key]['pid']) for key in ['G03', 'G04']) and time.time() < limit:
        time.sleep(.5)
    if any(alive(queue['jobs'][key]['pid']) for key in ['G03', 'G04']):
        raise RuntimeError('Official worker has not reached its interruption point')
    # Reload the scheduler's final snapshot after it relinquishes the single queue lock.
    queue = read(OUT / 'queue_status.json')
    by_job = {job['id']: job for job in manifest['jobs']}
    report = {}
    for model, key, primary_gpu, helper_gpu in [('gemma3', 'G03', 5, 0), ('medgemma', 'G04', 6, 4)]:
        directory = OUT / 'e6' / model / 'evaluation/mega_official'
        work = directory / 'shard0'
        paths = rows(work / 'path_predictions.jsonl')
        pairs = {(r['case_id'], r['method']) for r in paths}
        if len(pairs) != len(paths):
            raise ValueError('Duplicate pre-existing predictions')
        complete = {case['case_id'] for case in cases if all((case['case_id'], m) in pairs for m in official.METHODS)}
        remaining = [case for case in cases if case['case_id'] not in complete]
        nodes = rows(work / 'nodes.jsonl')
        naturals = rows(work / 'natural_nodes.jsonl')
        touched = {r['case_id'] for r in nodes + naturals + paths}
        assignments = [[], []]
        weights = [0, 0]
        for case in remaining:
            if case['case_id'] in touched:
                assignments[0].append(case['case_id'])
                weights[0] += len(case['gold_path'])
        for case in remaining:
            if case['case_id'] in touched:
                continue
            slot = 0 if weights[0] <= weights[1] else 1
            assignments[slot].append(case['case_id'])
            weights[slot] += len(case['gold_path'])
        left, right = map(set, assignments)
        if left & right or left | right != {case['case_id'] for case in remaining} or (left | right) & complete:
            raise ValueError('Partition overlap or missing cases')
        helper_key = key + '_HELPER'
        merge_key = key + '_MERGE'
        for index, gpu, job_id in [(0, primary_gpu, key), (1, helper_gpu, helper_key)]:
            assignment_file = directory / ('assignment_shard' + str(index) + '.json')
            assignment = dict(model=model, work_index=index, gpu=gpu, case_ids=assignments[index],
                              paths=len(assignments[index]), gold_path_nodes=weights[index],
                              selection='remaining case IDs, balanced by gold-path node count',
                              source_cases=str(official.OFFICIAL_CASES), no_parameter_changes=True)
            write(assignment_file, assignment)
            job = dict(id=job_id, units=[key], script='official_partition_worker.py',
                       args=['--model', model, '--work-index', str(index), '--assignment', str(assignment_file)],
                       preferred_gpu=gpu, completion_file=str(directory / ('shard' + str(index)) / 'done.json'))
            if index == 0:
                by_job[key].update(job)
                previous = queue['jobs'][key]
                queue['jobs'][key] = dict(state='pending', gpu=gpu, units=[key],
                                         history=[dict(previous, state='interrupted_for_partition', planned=True)],
                                         reason='Resume saved outputs with disjoint remaining IDs')
            else:
                manifest['jobs'].append(job)
        manifest['jobs'].append(dict(id=merge_key, units=[key], script='official_worker.py',
               args=['--model', model, '--merge-only'], preferred_gpu=helper_gpu,
               dependencies=[key, helper_key], completion_file=str(directory / 'completed.json'),
               computation='CPU statistics and shard merge; no model loading'))
        report[model] = dict(completed_paths_preserved=len(complete), total_paths=7458,
                            remaining_paths=len(remaining), preserved_node_records=len(nodes),
                            preserved_natural_nodes=len(naturals), partial_cases_reused=len((left | right) & touched),
                            primary_gpu=primary_gpu, helper_gpu=helper_gpu,
                            paths_per_worker=[len(v) for v in assignments], gold_nodes_per_worker=weights,
                            overlap=0, model_reload=True, new_fitting=False)
        write(directory / 'partition_plan.json', report[model])
    # The E5 worker continues on GPU 1 and is adopted using its durable done marker.
    by_job['E5_REPRESENTATION']['completion_file'] = str(OUT / 'e5/representation_worker_done.json')
    manifest.update(allowed_gpus=[0, 1, 4, 5, 6], repartition_report=str(history / 'handoff.json'))
    queue.update(state='restarting', active=1, allowed_gpus=[0, 1, 4, 5, 6], updated=time.time())
    write(OUT / 'queue_manifest.json', manifest)
    write(OUT / 'queue_status.json', queue)
    write(history / 'handoff.json', dict(state='prepared', reason='User requested GPUs 0 and 4',
          models=report, existing_outputs_preserved=True, new_training=False, model_downloads=False,
          e5_worker_retained=True, prepared_at=time.time(), expected_remaining_hours=[3, 5],
          helper_memory_gib=[60, 65]))
    log = (OUT / 'scheduler.log').open('a')
    process = subprocess.Popen([sys.executable, str(CODE / 'unified_queue.py')], cwd=PROJECT,
                               env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONUNBUFFERED='1'),
                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (OUT / 'scheduler.pid').write_text(str(process.pid) + '\n')
    log.close()
    print(json.dumps(dict(scheduler_pid=process.pid, models=report)))


if __name__ == '__main__':
    main()
