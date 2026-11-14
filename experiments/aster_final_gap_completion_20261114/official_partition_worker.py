"""Evaluate disjoint remaining case IDs with the existing frozen E1 scorer."""
import argparse
import fcntl
import sys
import time
from common import *

sys.path.insert(0, str(REPO6))
from experiments.aster_e678 import official_mega as official


def run(model, work_index, assignment_file):
    root = OUT / 'e6' / model
    work = root / 'evaluation/mega_official' / ('shard' + str(work_index))
    work.mkdir(parents=True, exist_ok=True)
    (work / 'features').mkdir(exist_ok=True)
    lock = (work / '.execution.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assignment = read(assignment_file)
    if assignment['model'] != model or assignment['work_index'] != work_index:
        raise ValueError('Assignment belongs to another worker')
    wanted = assignment['case_ids']
    if len(wanted) != len(set(wanted)):
        raise ValueError('Duplicate assigned case IDs')
    inventory = {case['case_id']: case for case in official.rows(official.OFFICIAL_CASES)}
    cases = [inventory[key] for key in wanted]
    frozen = official.Frozen(root)
    config = dict(model=model, parameter_directory=frozen.parameter_root,
                  cases_file=str(official.OFFICIAL_CASES), selected_heads=frozen.heads.tolist(),
                  assignment_file=str(assignment_file), work_index=work_index,
                  methods=list(official.METHODS), physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  source_protocol='E1-balanced-2540', target_labels_used_for_fitting=False,
                  execution='disjoint remaining case IDs; existing outputs retained')
    for previous in ['configuration.json', 'partition_configuration.json']:
        prior = read(work / previous)
        if prior:
            for key in ['model', 'parameter_directory', 'cases_file', 'selected_heads']:
                if prior[key] != config[key]:
                    raise ValueError('Frozen configuration mismatch: ' + key)
            if previous == 'partition_configuration.json' and prior['assignment_file'] != str(assignment_file):
                raise ValueError('Worker assignment changed')
    write(work / 'partition_configuration.json', config)
    own = {record['id']: record for record in iterrows(work / 'nodes.jsonl')}
    completed = {(record['case_id'], record['method']) for record in iterrows(work / 'path_predictions.jsonl')}
    naturals = {record['id'] for record in iterrows(work / 'natural_nodes.jsonl')}
    pending = [case for case in cases if any((case['case_id'], method) not in completed for method in official.METHODS)]
    started, forwards, reused, engine = time.monotonic(), 0, 0, None
    for case_index, case in enumerate(pending):
        case_cache = {}

        def node(gid):
            nonlocal engine, forwards, reused
            if gid in case_cache:
                return case_cache[gid]
            key = case['case_id'] + ':' + gid
            cached = own.get(key)
            if cached is not None and cached.get('status') == 'ok' and Path(cached['feature']).exists():
                with np.load(cached['feature'], allow_pickle=False) as values:
                    if not np.array_equal(values['head_ids'], frozen.heads):
                        raise ValueError('Cached heads differ from the frozen model')
                    scores = frozen.scores(cached['ell'], values['selected_heads'])
                reused += 1
            else:
                if engine is None:
                    from experiments.aster_e678.engine import Engine
                    metadata = root / ('official_model_metadata.partition' + str(work_index) + '.json')
                    engine = Engine(model, metadata_path=metadata)
                text = official.prompt(case['guideline'], case.get('definitions', []), case['patient'], gid)
                record = dict(id=key, case_id=case['case_id'], source_id=case['source_id'], gid=gid)
                try:
                    result = official.score_selected(engine, text, frozen.heads)
                    feature = work / 'features' / ('node_' + str(len(own)).zfill(8) + '.npz')
                    np.savez_compressed(feature, selected_heads=result.pop('selected_heads'), head_ids=frozen.heads)
                    record.update(result, feature=str(feature), status='ok')
                    with np.load(feature, allow_pickle=False) as values:
                        scores = frozen.scores(record['ell'], values['selected_heads'])
                except ValueError as exc:
                    record.update(status='unavailable', error=str(exc))
                    scores = None
                append(work / 'nodes.jsonl', record)
                own[key] = record
                forwards += 1
            case_cache[gid] = scores
            return scores

        for gold in case['gold_path']:
            key = case['case_id'] + ':' + gold['gid']
            if key in naturals:
                continue
            scores = node(gold['gid'])
            append(work / 'natural_nodes.jsonl', dict(id=key, case_id=case['case_id'], source_id=case['source_id'],
                   gid=gold['gid'], gold_label=gold['label'], scores=scores,
                   status='ok' if scores is not None else 'unavailable'))
            naturals.add(key)
        for record in official.traverse(case, node):
            if (record['case_id'], record['method']) not in completed:
                record.update(model=model, evaluation_group='official_test',
                              usage_metadata={key: case.get(key) for key in
                                              ('current_source_role', 'history_group', 'known_earlier_input_use')})
                append(work / 'path_predictions.jsonl', record)
                completed.add((record['case_id'], record['method']))
        if case_index < 3 or (case_index + 1) % 20 == 0:
            write(work / 'status.json', dict(state='running', model=model, work_index=work_index,
                  cases_completed=case_index + 1, pending_at_start=len(pending), assigned_paths=len(cases),
                  new_forwards=forwards, reused_nodes=reused, seconds=time.monotonic() - started,
                  last_case=case['case_id']))
    if any((key, method) not in completed for key in wanted for method in official.METHODS):
        raise ValueError('Assigned predictions are incomplete')
    peak = None
    if engine is not None:
        import torch
        peak = torch.cuda.max_memory_allocated(engine.device) / 2**30
    result = dict(state='complete', complete=True, model=model, work_index=work_index,
                  assigned_paths=len(cases), cases_completed=len(pending), pending_at_start=len(pending),
                  new_forwards=forwards, reused_nodes=reused, seconds=time.monotonic() - started,
                  peak_gib=peak, assignment_file=str(assignment_file),
                  path_records_including_preserved=len(completed), natural_nodes_including_preserved=len(naturals))
    write(work / 'status.json', result)
    write(work / 'done.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['gemma3', 'medgemma'])
    parser.add_argument('--work-index', type=int, required=True, choices=[0, 1, 2])
    parser.add_argument('--assignment', type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    run(args.model, args.work_index, args.assignment)
