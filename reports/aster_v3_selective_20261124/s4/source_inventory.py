"""Index the historical source registries without loading labels or scoring models."""
import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path


def records(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def read(path):
    return json.loads(path.read_text())


def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, lineterminator="\n", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, default=Path.home())
    parser.add_argument('--output', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    t0 = time.monotonic()
    cpu0 = time.process_time()
    out, base = args.output, args.archive
    repo = Path(__file__).resolve().parents[3]
    out.mkdir(parents=True, exist_ok=True)
    families = read(base/'aster_manuscript_20261030/clinical01/source_clusters.json')['records']
    counts = Counter()
    inputs = {}
    templates = {}

    def add(cohort, split, cohort_key, family, template, path, units=1):
        counts[(cohort, split, cohort_key, family)] += units
        templates[cohort] = template
        inputs.setdefault((cohort, split), set()).add(str(path))

    def mega(cohort, split, key, template, path, units=1):
        add(cohort, split, key, families[key]['cluster'], template, path, units)

    frozen = Path('reports/aster_codex03_external_20261112/prepared/source_splits.json')
    for split, keys in read(repo/frozen).items():
        for key in keys:
            mega('E1', split, key, 'MEGA generated path and paired views', frozen)
    official = Path('aster_next_20261109/inventory/sources.jsonl')
    for row in records(base/official):
        add('MEGA_official', 'test', row['source_id'], row['source_id'], 'MEGA generated path', official, row['cases'])

    prior = Path('aster_cr_repair_v2_run_20261108/data/history_usage.json')
    for key in read(base/prior)['excluded']:
        add('MEGA_prior_registry', 'excluded', key, key, 'Historical exclusions across 31 prior files and official source groups', prior)

    train_dir = Path('aster_next_20261109/primary_results/mega_train_val')
    for path in sorted((base/train_dir).glob('shard*/results.jsonl')):
        for row in records(path):
            if row['method'] == 'raw' and row['status'] == 'ok':
                add('MEGA_prior_predictions', 'train_val_observed', row['source_key'], row['source_id'],
                    'MEGA generated path', path.relative_to(base))

    prior_dev = Path('aster_e0_e8_20261024/private/cases.jsonl')
    for row in records(base/prior_dev):
        if row['split'] == 'dev':
            tree = row['tree']
            add('P3_development', 'development', tree, 'MedGUIDE:' + tree.rsplit('_', 1)[0],
                'MedGUIDE seeded action selection', prior_dev)

    for study, directory in [('Qwen_path', 'aster_representation_20261101'), ('Gemma_path', 'gemma3_27b_head_pilot_20261103/data')]:
        for split in ('train', 'development', 'confirmation'):
            path = Path(directory, split + '.jsonl')
            for row in records(base/path):
                mega(study, split, row['source_id'], 'MEGA path; original / swapped guideline', path)

    pilot = Path('gemma3_27b_fourway_development_20261104/data_manifest.json')
    for key, n in read(base/pilot)['groups_per_source'].items():
        mega('Gemma_four_view_pilot', 'development', key, 'MEGA four constructed fact views', pilot, n)
    expanded = Path('gemma3_27b_fourway_expanded_20261104/data_manifest.json')
    for key, item in read(base/expanded)['sources'].items():
        mega('Gemma_PR52', item['split'], key, 'MEGA four constructed fact views', expanded, len(item['target_nodes']))

    for model in ('qwen', 'llama'):
        path = Path('aster_p0_p3_20261025/test', model, 'predictions.jsonl')
        for row in records(base/path):
            tree = row['tree']
            family = 'MedGUIDE:' + tree.rsplit('_', 1)[0]
            add('P3_' + model, 'test', tree, family, 'MedGUIDE numbered NCCN pages; seeded NC/AC/IC/SW/CS variants', path)

    nli = Path('aster_next_20261109/inventory/nli4ct_all_cases.jsonl')
    for row in records(base/nli):
        add('NLI4CT', row['split'], row['source_key'], row['family_cluster_id'], 'NLI4CT trial statement', nli)
    medpic = Path('aster_cross_task_transfer_20261109/medpic/split.json')
    layout = read(base/medpic)
    assignment = {k: s for s in ['fit', 'development', 'test'] for k in layout[s + '_item_ids']}
    for i, (_, members) in enumerate(layout['groups'].items(), 1):
        # Use the original group's position. The private grouping keys may contain vignette text.
        family = f'MedPIC:group-{i:04}'
        for key in members:
            add('MedPIC', assignment[key], family, family, 'MedPIC multi-select, options grouped by case', medpic)

    rows = [dict(cohort=c, split=s, source=k, family=f, units=n, template=templates[c]) for (c,s,k,f), n in sorted(counts.items())]
    write_csv(out/'prior_sources.csv', list(rows[0]), rows)
    by_family = defaultdict(set)
    for r in rows:
        if r['cohort'].startswith(('MEGA_', 'Gemma_', 'Qwen_')) or r['cohort'] == 'E1':
            by_family[r['family']].add((r['cohort'], r['split']))
    summary = []
    for (cohort, split), paths in sorted(inputs.items()):
        subset = [r for r in rows if r['cohort'] == cohort and r['split'] == split]
        fam = {r['family'] for r in subset}
        summary.append(dict(cohort=cohort, split=split, source_ids=len({r['source'] for r in subset}), families=len(fam), units=sum(r['units'] for r in subset), template=templates[cohort], evidence='|'.join(sorted(paths)), status='previously_used'))
    write_csv(out/'prior_source_summary.csv', list(summary[0]), summary)
    overlaps = []
    for a in sorted(inputs):
        for b in sorted(inputs):
            shared = sum(a in groups and b in groups for groups in by_family.values())
            if a < b and shared:
                overlaps.append(dict(cohort_a='/'.join(a), cohort_b='/'.join(b), shared_mega_clusters=shared))
    write_csv(out/'mega_source_overlap.csv', ['cohort_a','cohort_b','shared_mega_clusters'], overlaps)
    sample_groups = {r['cohort'] for r in rows}
    (out/'status.json').write_text(json.dumps(dict(source_audit='complete', confirmation='blocked', historical_cohorts=len(sample_groups), historical_cohort_splits=len(summary), source_memberships=len(rows), qualified_new_sources=0, new_independent_evaluations=0, model_calls=0, gpu_hours=0, cpu_seconds=time.process_time()-cpu0, elapsed_seconds=time.monotonic()-t0), indent=2)+'\n')


if __name__ == '__main__':
    main()
