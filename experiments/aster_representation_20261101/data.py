import copy
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from experiments.aster_bottleneck_20261031.prepare import ROOT, BASE, OUT as OLD, read, tree_guideline, category, build

OUT = Path('/home/gcx/aster_representation_20261101')
SEED = 20261101
CATEGORIES = ['age', 'laboratory', 'stage_diagnosis', 'time_treatment', 'symptom_other']


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows))


def swap_actions(case):
    result = copy.deepcopy(case)
    terminals = []
    for group in result['guideline']['groups']:
        for rule in group['rules']:
            match = re.fullmatch(r'(L\d+[：:])(.*)', rule['action'], re.S)
            if match:
                terminals.append((rule, match[1], match[2]))
    if len(terminals) > 1:
        for i, (rule, prefix, _) in enumerate(terminals):
            rule['action'] = prefix + terminals[(i + 1) % len(terminals)][2]
    return result


def deployment(case):
    return {k: case[k] for k in ['case_id', 'source_id', 'patient', 'guideline', 'question']}


def nodes(case):
    if 'target_rule' in case:
        return [{'gid': 'G' + case['target_rule'][1:-1], 'history': [], 'target': case['target_rule']}]
    path = case['gold_rules']
    indices = sorted({0, len(path) // 2, len(path) - 1})
    return [{'gid': 'G' + path[i][1:-1], 'history': path[:i], 'target': path[i]} for i in indices]


def age_cases(pairs, split):
    # The historical Y/N target reversals contain identical patient states.
    # Preserve pair provenance, but do not repeat identical model executions.
    records = {}
    for pair in pairs:
        if pair['split'] != split:
            continue
        for state in ['positive', 'negative']:
            rid = pair['verification'][state + '_rule_id']
            key = (pair['source_id'], pair['states'][state], rid)
            if key not in records:
                records[key] = {'case_id': 'age:' + pair['pair_id'] + ':' + state,
                    'source_id': pair['source_id'], 'patient': pair['states'][state],
                    'guideline': pair['guideline'], 'question': pair['question'],
                    'target_rule': rid, 'pair_ids': [], 'age_state': state}
            records[key]['pair_ids'].append(pair['pair_id'])
    return list(records.values())


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'manifest.json').exists():
        return json.loads((OUT / 'manifest.json').read_text())
    previous = json.loads((OLD / 'module_one/manifest.json').read_text())
    excluded = set(previous['excluded_sources']) | {x['source_id'] for x in previous['sources']}
    clusters = json.loads((BASE / 'source_clusters.json').read_text())
    counts = Counter(int(r['guideline_id']) for r in read(ROOT / 'train_val/vignette_cases.jsonl'))
    buckets = defaultdict(list)
    for item in read(ROOT / 'train_val/decision_trees.jsonl'):
        gid = int(item['guideline_id'])
        source = clusters['records']['train_val:' + str(gid)]['cluster']
        if source in excluded or counts[gid] < 10:
            continue
        try:
            tree = json.loads(item['text'])['decision_tree']
            tree_guideline(tree)
        except (ValueError, KeyError, TypeError):
            continue
        buckets[category(str(tree.get('description', '')))].append((gid, source))
    rng = random.Random(SEED)
    used, selected = set(), []
    for cat in CATEGORIES:
        choices = sorted(buckets[cat]); rng.shuffle(choices)
        chosen = []
        for gid, source in choices:
            if source not in used:
                chosen.append({'category': cat, 'guideline_id': gid, 'source_id': source,
                               'split': 'train' if len(chosen) < 4 else 'confirmation'})
                used.add(source)
            if len(chosen) == 6:
                break
        if len(chosen) != 6:
            raise ValueError('Insufficient unused sources: ' + cat)
        selected += chosen
    full = build('train_val', {r['guideline_id']: r['split'] for r in selected}, clusters['records'])
    grouped = defaultdict(list)
    for case in full:
        if 'source_failure' not in case:
            grouped[case['source_id']].append(case)
    splits = defaultdict(list)
    for source in selected:
        bins = defaultdict(list)
        for case in grouped[source['source_id']]:
            depth = len(case['gold_rules'])
            bins[(case['gold_rules'][0][-1], 0 if depth <= 3 else 1 if depth <= 7 else 2)].append(case)
        for values in bins.values():
            rng.shuffle(values)
        picked = []
        while len(picked) < 10 and any(bins.values()):
            for key in sorted(bins):
                if bins[key] and len(picked) < 10:
                    picked.append(bins[key].pop())
        if len(picked) != 10:
            raise ValueError('Insufficient cases for ' + source['source_id'])
        splits[source['split']] += picked
    pairs = read(BASE / 'review/accepted_pairs.jsonl')
    splits['age_train'] = age_cases(pairs, 'fit')
    splits['age_diagnostic'] = age_cases(pairs, 'confirm')
    splits['development'] = read(OLD / 'module_one/development100.jsonl')
    for split, values in splits.items():
        jsonl(OUT / (split + '.jsonl'), values)
    manifest = {'seed': SEED, 'sources': selected, 'excluded_sources': sorted(excluded),
        'counts': {s: len(v) for s, v in splits.items()},
        'train_nodes': sum(len(nodes(c)) for c in splits['train'] + splits['age_train']),
        'train_source_count': 20, 'confirmation_source_count': 10,
        'original_fit_pairs': sum(p['split'] == 'fit' for p in pairs),
        'age_diagnostic_scope': 'Previously exposed confirm minimal pairs; not independent validation',
        'views': ['original', 'terminal_action_description_cyclic_swap'],
        'layers': [18, 24, 30], 'rank': 4, 'epochs': 3, 'lr': 0.001,
        'accumulation': 8, 'gradient_clip': 1.0, 'preservation_weight': 0.1, 'edit_weight': 0.001,
        'loss_weighting': 'node losses averaged within each patient case; one case per microbatch',
        'selection': 'development original joint correctness, then action-swap path agreement, then smaller edit',
        'labels_in_deployment': False, 'backbone_frozen': True,
        'confirmation_policy': 'Freeze full/control checkpoints before reading new confirmation outcomes'}
    write(OUT / 'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    print(json.dumps(prepare(), ensure_ascii=False))
