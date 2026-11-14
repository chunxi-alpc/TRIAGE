"""Resume official MEGA with immutable E1 parameters and existing shards."""
import sys
import shutil
import time
from common import *
sys.path.insert(0, str(REPO6))
from experiments.aster_e678 import official_mega as official

def prepare(model):
    source = Q if model == 'qwen' else R / 'models' / model
    target = OUT / 'e6' / model
    params = target / 'models/E1/full'
    params.mkdir(parents=True, exist_ok=True)
    for name in ['model.json', 'm1.json', 'm2.npz', 'm3.json', 'm3.npz']:
        if not (params / name).exists():
            shutil.copy2(source / 'models/E1/full' / name, params / name)
    if not (params / 'calibration.json').exists():
        if model == 'qwen':
            cal = read(D / 'configs/independent_intercepts_reused.json')['full']
            write(params / 'calibration.json', dict(b=cal['b0'], nodes=cal['nodes'],
                  split='source fit', target_labels_used=False, source_record=str(D / 'configs/independent_intercepts_reused.json')))
        else:
            shutil.copy2(source / 'models/E1/full/calibration.json', params / 'calibration.json')
    if not (target / 'frozen.json').exists():
        write(target / 'frozen.json', dict(frozen=True, source=str(source / 'frozen.json'),
              parameter_source=str(source / 'models/E1/full'), source_protocol='E1-balanced-2540',
              new_fitting=False, target_labels_used_for_fit=False))
    if model == 'gemma2':
        work = target / 'evaluation/mega_official/shard0'
        marker = work / 'resume_import.json'
        if not marker.exists():
            work.mkdir(parents=True, exist_ok=True)
            old = source / 'evaluation/mega_official/shard0'
            for name in ['nodes.jsonl', 'natural_nodes.jsonl', 'path_predictions.jsonl']:
                if (work / name).exists():
                    raise RuntimeError('Unmarked resume directory already has outputs')
                shutil.copy2(old / name, work / name)
            shutil.copy2(old / 'configuration.json', work / 'source_configuration.json')
            write(marker, dict(source=str(old), preserved_paths_per_method=2379, pending_paths=5079,
                  reused_node_count=15363, old_features_read_only=True))
    return target

def summarize(target):
    directory = target / 'evaluation/mega_official'
    paths = [r for shard in directory.glob('shard*') if shard.is_dir() for r in iterrows(shard / 'path_predictions.jsonl')]
    naturals = [r for shard in directory.glob('shard*') if shard.is_dir() for r in iterrows(shard / 'natural_nodes.jsonl')]
    raw = [r for r in paths if r['method'] == 'raw']
    summary = {}
    for method in official.METHODS:
        rr = [r for r in paths if r['method'] == method]
        if len(rr) != 7458 or len({r['source_id'] for r in rr}) != 418:
            raise ValueError('Official path denominator is incomplete')
        summary[method] = paired_binary(rr, raw, lambda r: r['case_id'], seed=20261111)
    if len(naturals) != 45627 or len({r['id'] for r in naturals}) != 45627:
        raise ValueError('Official natural-node denominator is incomplete')
    base = [dict(r, correct=r['scores'] is not None and (r['scores']['raw'] > 0) == bool(r['gold_label'])) for r in naturals]
    natural_stats = {}
    for method in official.METHODS:
        rr = [dict(r, correct=r['scores'] is not None and (r['scores'][method] > 0) == bool(r['gold_label'])) for r in naturals]
        natural_stats[method] = paired_binary(rr, base, lambda r: r['id'], seed=20261111)
    write(directory / 'paired_statistics.json', dict(complete=True, paths=summary, natural_nodes=natural_stats))
    lines(directory / 'path_predictions.jsonl', paths)
    lines(directory / 'natural_nodes.jsonl', naturals)
    write(directory / 'completed.json', dict(complete=True, paths_per_method=7458, sources=418,
          natural_nodes=45627, duplicate_path_ids=0, missing_path_ids=0, time=time.time()))

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    a = p.parse_args()
    target = prepare(a.model)
    if (target / 'evaluation/mega_official/completed.json').exists():
        return
    import torch
    torch.set_num_threads(4)
    result = official.run(a.model, target)
    if result['state'] != 'complete':
        raise RuntimeError('Official evaluation ended with a partial population')
    summarize(target)

if __name__ == '__main__':
    main()
