"""Build a keyed index of archived V3-relevant rows. Never calculate predictions."""
import argparse
import csv
import gzip
import json
import time
from collections import Counter
from pathlib import Path


def read_rows(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream) if path.suffix == '.csv' else map(json.loads, stream)
        yield from enumerate(reader, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, default=Path.home())
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[2]/'reports/aster_v3_selective_20261124')
    args = parser.parse_args()
    started = time.process_time()
    base = args.archive
    e1 = base/'aster-cr-diagnostics-e3cv-20261111/reports/aster_cr_diagnostics_e3cv_20261111'
    sj = base/'aster-m3-joint-20261115/reports/aster_m3_joint_study_20261115'
    pr87 = base/'aster-m3-missing-controls-20261123/reports/aster_m3_missing_controls_20261123'
    pr85 = base/'aster-table2-20261123/reports/aster_table2_20261123'
    path_error = base/'aster-path-error-20261123/reports/aster_path_error_20261123/results/first_errors.csv'
    files = {}
    entries = {}
    keys = ('unit','archive','score_version','split_or_fold','source','uid','variant','gid')
    fields = (*keys,'view_uid','structure','record_refs')
    j_folds = {}
    for path in sorted((sj/'selection').glob('fold*/split.json')):
        for source in json.loads(path.read_text())['outer_validation_sources']:
            j_folds[source] = path.parent.name

    def rows(path):
        file_id = f'a{len(files)+1:02}'
        files[file_id] = str(path.relative_to(base))
        for i, row in read_rows(path):
            yield (file_id, i), row

    def add(unit, archive, method, fold, source, uid, ref, variant='', gid='', view='', structure=''):
        # Protocol + unit + source + fold distinguishes numerically similar scores.
        key = (unit,archive,method,fold,source,uid,variant,gid)
        record = entries.setdefault(key, ['', '', {}])
        if view:
            record[0] = view
        if structure:
            record[1] = structure
        # Repeated comparisons can point to the same first error within a file.
        record[2].setdefault(ref[0], ref[1])

    def pair_table(path, archive, fold=None):
        for ref, r in rows(path):
            if r['kind'] not in {'C','F','F0'}:
                continue
            if archive.startswith('E1') and r['method'] not in {'raw','independent_intercept'} and not r['method'].startswith('E1_'):
                continue
            prefix = fold or j_folds[r['source_id']]
            pair = r['pair_id'].split('::')[-1]
            add('CF_pair', archive, r['method'], prefix, r['source_id'], pair, ref, variant=r['kind'])

    def path_table(path, archive, fold=None):
        for ref, r in rows(path):
            if not r['first_error']:
                continue
            if archive.startswith('E1') and r['method'] not in {'raw','independent_intercept'} and not r['method'].startswith('E1_'):
                continue
            add('first_error', archive, r['method'], fold or j_folds[r['source_id']], r['source_id'],
                r['case_id'], ref, variant=r['variant'], gid=r['first_error'])

    for fold in [f'cv_fold{i}_common' for i in range(1,6)] + ['dev','test','new60']:
        root = e1/'evaluation'/fold
        split = fold.removeprefix('cv_').removesuffix('_common')
        pair_table(root/'pair_predictions.jsonl.gz','E1',split)
        path_table(root/'path_predictions.jsonl.gz','E1',split)
    for i in range(1,6):
        hist = e1/'evaluation'/f'cv_fold{i}_historical'/'pair_predictions.jsonl.gz'
        pair_table(hist, 'E1_historical', f'fold{i}')
    # The separate first-error archive also contains the pair-plus-b0 score.
    for ref, r in rows(path_error):
        add('first_error','E1_path_diagnostic',r['method'],r['fold'],r['source_id'],r['case_id'],ref,
            variant=r['variant'],gid=r['first_error_gid'])
    for archive, path in [('SJ', sj/'evaluation/path_predictions.jsonl.gz'),('PR87_B',pr87/'b_paths.jsonl.gz')]:
        path_table(path,archive)
    for archive, path in [('SJ', sj/'evaluation/pair_predictions.jsonl.gz'),('PR87_B',pr87/'b_pairs.jsonl.gz')]:
        pair_table(path, archive)
    for ref, r in rows(sj/'evaluation/natural_first_error.jsonl.gz'):
        for pred, err, view in [('rhs','before_first_error',r['first_error_view_id']),('lhs','after_first_error',None)]:
            if r[err]:
                add('first_error','SJ',r[pred],j_folds[r['source_id']],r['source_id'],r['case_id'],ref,
                    variant='original',gid=r[err],view=view or '')
    for ref, r in rows(sj/'diagnostics/correction_bounds.jsonl'):
        add('LP_bound','SJ','R1_shared_b0',r['fold'],r['source_id'],r['case_id'],ref,
            variant='original',view=r['fold'] + '::' + r['view_id'])
    for ref, r in rows(pr87/'A_first_error_boundaries.jsonl.gz'):
        for pred, err in [('rhs','before_first_error'),('lhs','after_first_error')]:
            node = r[err]
            if node:
                add('first_error','PR87_A',r[pred],j_folds[r['source_id']],r['source_id'],r['case_id'],ref,
                    variant=r['variant'],gid=node['gid'],view=node.get('view_id') or '')
    for ref, r in rows(pr87/'b_cf_budgets.jsonl.gz'):
        prefix, uid = r['pair_id'].split('::', 1)
        add('CF_budget','PR87_B',r['method'],prefix,r['source_id'],uid,ref)
    for split in ('dev','test'):
        for ref, r in rows(pr85/(split+'_pair_metrics.csv')):
            add('structure','PR85','adaptive_M1_M2',split,r['source_id'],r['pair_id'],ref,
                variant=r['kind'],structure=r['node_type'])

    count = Counter()
    args.output.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output/'UID_COVERAGE.csv.gz','wt',newline='',encoding='utf-8') as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(fields)
        for key, (view, structure, refs) in sorted(entries.items()):
            count[key[:2]] += 1
            writer.writerow([*key, view, structure, '|'.join(f'{k}:{v}' for k,v in refs.items())])
    counts = {f'{unit}/{archive}':n for (unit,archive),n in sorted(count.items())}
    report = dict(row_count=sum(count.values()), rows_by_unit=counts, files=files,
                  pointer='file ID:1-based data record (CSV header excluded); joined with | across archives',
                  key=keys, costs=dict(cpu_seconds=time.process_time()-started,new_model_calls=0,new_replays=0))
    (args.output/'UID_COVERAGE_FILES.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
