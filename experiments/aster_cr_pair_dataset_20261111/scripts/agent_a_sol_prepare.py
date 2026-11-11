"""Prepare fit-only reading tasks for the assigned Sol/high Agent A.
This module never invents edits, calls a model, or reads reviewer reasoning.
"""
from runtime_paths import ROOT, DATA, RAW, REPO, SCRIPTS, SCHEMA, source_document_path
import argparse
import collections
import csv
import json
import math
import pathlib
import re

KINDS = ('P', 'C', 'F', 'A')
MODEL = 'gpt-6.1-sol'
EFFORT = 'high'


def read_rows(path):
    with pathlib.Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def fit_rows(name):
    return [r for r in read_rows(DATA / name) if r.get('split') == 'fit']


def strip_decision(condition):
    return re.sub(r'\uff08\u5224\u65ad\u4e3a[\u662f\u5426]\uff09$', '', condition)


def split_prompt(prompt):
    guide, tail = prompt.split('\n\n\u6307\u5357\u539f\u6587\u5b9a\u4e49\u4e0e\u8bf4\u660e\uff1a\n', 1)
    definitions, tail = tail.split('\n\n\u75c5\u5386\uff1a\n', 1)
    patient, tail = tail.rsplit('\n\n\u5f53\u524d\u8282\u70b9\uff1a', 1)
    gid, tail = tail.split('\n\u5f53\u524d\u6761\u4ef6\uff1a', 1)
    condition = tail.rsplit('\n\u8bf7\u5224\u65ad\u5f53\u524d\u6761\u4ef6\u3002', 1)[0]
    return dict(guide=guide, definitions=definitions, patient=patient, node_id=gid, condition=condition)


def render(side):
    return (side['guide'] + '\n\n\u6307\u5357\u539f\u6587\u5b9a\u4e49\u4e0e\u8bf4\u660e\uff1a\n' + side['definitions']
            + '\n\n\u75c5\u5386\uff1a\n' + side['patient'] + '\n\n\u5f53\u524d\u8282\u70b9\uff1a' + side['node_id']
            + '\n\u5f53\u524d\u6761\u4ef6\uff1a' + side['condition'] + '\n\u8bf7\u5224\u65ad\u5f53\u524d\u6761\u4ef6\u3002\u53ea\u8f93\u51fa\u201c\u662f\u201d\u6216\u201c\u5426\u201d\u3002\n\u7b54\u6848\uff1a')


def size_bucket(size):
    return 'small' if size <= 15 else 'medium' if size <= 30 else 'large'


def raw_index_map(tree):
    mapping = {}
    def visit(node):
        internal = any(isinstance(node.get(key), dict) for key in ('yes', 'no'))
        mapping[('G' if internal else 'L') + str(len(mapping))] = node['index']
        if internal:
            for key in ('yes', 'no'):
                if isinstance(node.get(key), dict):
                    visit(node[key])
    visit(tree)
    return mapping


class FitCatalog:
    """All original fit cases and gold-path targets, including unexpanded nodes."""
    def __init__(self):
        self.cases = {r['case_id']: r for r in fit_rows('path_cases.jsonl')}
        self.nodes = fit_rows('nodes.jsonl')
        self.existing_nodes = {(n['case_id'], n['gid']): n for n in self.nodes}
        self.nodes_by_group = {n['group_id']: n for n in self.nodes}
        self.languages = {n['guideline_id']: n['language'] for n in self.nodes}
        self.groups = {cid: {g['group_id']: g for g in c['guideline']['groups']} for cid, c in self.cases.items()}
        self.references = {}
        self.views = {}
        for view in fit_rows('views.jsonl'):
            if view.get('variant') != 'full':
                continue
            n = self.nodes_by_group[view['group_id']]
            self.views[(n['case_id'], n['gid'])] = view
            self.references.setdefault(n['guideline_id'], split_prompt(view['prompt']))
        ids = {c['guideline_id'] for c in self.cases.values()}
        self.raw_indices = {}
        for row in read_rows(RAW / 'decision_trees.jsonl'):
            if row['guideline_id'] in ids:
                self.raw_indices[row['guideline_id']] = raw_index_map(json.loads(row['text'])['decision_tree'])
                self.languages[row['guideline_id']] = row['language']
        self.rules = {}
        for rule in fit_rows('rules.jsonl'):
            n = self.nodes_by_group.get(rule['group_id'])
            if n:
                self.rules[(n['case_id'], n['gid'])] = rule

    def original(self, case_id, node_id):
        case = self.cases[case_id]
        target = next(g for g in case['gold_path'] if g['gid'] == node_id)
        condition = strip_decision(self.groups[case_id][node_id]['rules'][0]['condition'])
        ref = self.references[case['guideline_id']]
        side = dict(patient=case['patient'], condition=condition, guide=ref['guide'], definitions=ref['definitions'], node_id=node_id, label=target['label'])
        side['prompt'] = render(side)
        old = self.views.get((case_id, node_id))
        side['view_id'] = old['view_id'] if old else 'sol-original:' + case_id + ':' + node_id
        if old and side['prompt'] != old['prompt']:
            raise ValueError('Reconstructed original differs from released full view: ' + side['view_id'])
        return side

    def metadata(self, case_id, node_id):
        c = self.cases[case_id]
        old = self.existing_nodes.get((case_id, node_id))
        return dict(source_id=c['source_id'], split='fit', case_id=case_id,
                    group_id=old['group_id'] if old else 'sol-node:' + case_id + ':' + node_id,
                    guideline_id=c['guideline_id'], source_filename=c['source_filename'], node_id=node_id,
                    raw_node_index=self.raw_indices[c['guideline_id']][node_id], language=self.languages[c['guideline_id']],
                    tree_complexity=len(c['guideline']['groups']), source_document=c['source_document'])

    def targets(self):
        for cid, case in self.cases.items():
            for target in case['gold_path']:
                task = self.metadata(cid, target['gid'])
                task['condition'] = strip_decision(self.groups[cid][target['gid']]['rules'][0]['condition'])
                task['original_label'] = target['label']
                yield task

    def packet(self, case_id, node_id):
        c = self.cases[case_id]
        path = c['gold_path']
        index = next(i for i, g in enumerate(path) if g['gid'] == node_id)
        return dict(metadata=self.metadata(case_id, node_id), original=self.original(case_id, node_id),
                    source_guideline=source_document_path(c['source_document']).read_text(encoding='utf-8'),
                    guideline=c['guideline'], definitions=c['definitions'],
                    ancestors=[dict(node_id=p['gid'], label=p['label'], condition=strip_decision(self.groups[case_id][p['gid']]['rules'][0]['condition'])) for p in path[:index]],
                    program_rule=self.rules.get((case_id, node_id)))


def accepted_metadata(path):
    """Read only acceptance counts, coverage, original target, and actual edit signatures."""
    if not pathlib.Path(path).exists():
        return []
    return [dict(kind=r['kind'], case_id=r['case_id'], node_id=r['node_id'], source_id=r['source_id'],
                 language=r['language'], tree_complexity=r['tree_complexity'],
                 edits=[{k:e[k] for k in ('component','old','new')} for e in r['edits']])
            for r in read_rows(path)]


def choose_tasks(targets, accepted, counts, coverage=None, target=500, multiplier=1.25):
    """Choose one finite reading batch, with no semantic edit creation or retries."""
    if target < 0 or not math.isfinite(multiplier) or multiplier < 1:
        raise ValueError('Invalid target or candidate multiplier')
    all_targets = list(targets)
    coverage = coverage or {}
    selected = []
    plans = {}
    globally_read_cases = set()
    for kind in KINDS:
        gap = max(0, target - int(counts.get(kind, 0)))
        budget = math.ceil(gap * multiplier)
        if not budget:
            plans[kind] = dict(accepted=int(counts.get(kind,0)), gap=0, budget=0, selected=0)
            continue
        old = [r for r in accepted if r['kind'] == kind]
        old_cases = {r['case_id'] for r in old}
        old_nodes = {(r['case_id'],r['node_id']) for r in old}
        src = collections.Counter(r['source_id'] for r in old)
        for sid, n in coverage.get(kind, {}).items():
            src[sid] = n
        lang = collections.Counter(r['language'] for r in old)
        size = collections.Counter(size_bucket(r['tree_complexity']) for r in old)
        buckets = collections.defaultdict(list)
        for item in all_targets:
            buckets[(item['source_id'],item['language'],size_bucket(item['tree_complexity']))].append(item)
        for bucket in buckets.values():
            bucket.sort(key=lambda r: ((r['case_id'],r['node_id']) in old_nodes, r['case_id'] in old_cases, r['case_id'], int(r['node_id'][1:])))
        picked_cases = set()
        picked = []
        while len(picked) < budget and any(buckets.values()):
            # Existing class/source coverage is used directly; language and size break ties.
            keys = [key for key, bucket in buckets.items() if bucket]
            key = min(keys,key=lambda k:(src[k[0]],lang[k[1]],size[k[2]],k))
            bucket = buckets[key]
            j = min(range(len(bucket)),key=lambda i:(bucket[i]['case_id'] in picked_cases, (bucket[i]['case_id'],bucket[i]['node_id']) in old_nodes, bucket[i]['case_id'] in old_cases, bucket[i]['case_id'] in globally_read_cases, i))
            item = bucket.pop(j)
            picked.append(dict(kind=kind, **item))
            picked_cases.add(item['case_id'])
            globally_read_cases.add(item['case_id'])
            src[key[0]] += 1
            lang[key[1]] += 1
            size[key[2]] += 1
        selected.extend(picked)
        plans[kind] = dict(accepted=int(counts.get(kind,0)), gap=gap, budget=budget, selected=len(picked), cases=len(picked_cases), sources=len({r['source_id'] for r in picked}))
    return selected, plans


def prepare(output, plan_output, target=500, multiplier=1.25):
    completion = ROOT / 'reports/completion_historical.json'
    if not completion.is_file():
        raise RuntimeError('Historical review is not complete; no reading tasks may be selected')
    counts = json.loads(completion.read_text(encoding='utf-8'))['counts']
    accepted = accepted_metadata(ROOT / 'accepted_historical_pairs.jsonl')
    coverage = collections.defaultdict(dict)
    with (ROOT / 'source_coverage_historical.csv').open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            coverage[row['kind']][row['source_id']] = int(row['accepted'])
    catalog = FitCatalog()
    tasks, plans = choose_tasks(catalog.targets(),accepted,counts,coverage,target,multiplier)
    output, plan_output = pathlib.Path(output), pathlib.Path(plan_output)
    if output.exists() or plan_output.exists():
        raise FileExistsError('Existing task batch/plan is preserved; choose a new explicit path')
    with output.open('x',encoding='utf-8') as stream:
        for task in tasks:
            stream.write(json.dumps(task,ensure_ascii=False)+'\n')
    with plan_output.open('x',encoding='utf-8') as stream:
        json.dump(dict(configured_model=MODEL, configured_reasoning_effort=EFFORT, underlying_snapshot=None,
                       target_per_class=target, one_batch=True, automatic_retry=False, semantic_candidates_generated=False,
                       counts=plans, fit_cases=len(catalog.cases), fit_gold_nodes=sum(len(c['gold_path']) for c in catalog.cases.values())),stream,ensure_ascii=False,indent=2)
    return plans


def self_test():
    assert strip_decision('test\uff08\u5224\u65ad\u4e3a\u662f\uff09') == 'test'
    assert raw_index_map(dict(index=1,yes=dict(index=2),no=dict(index=3))) == {'G0':1,'L1':2,'L2':3}
    tasks=[dict(source_id='s',language='zh',tree_complexity=2,case_id='c',node_id='G0'),dict(source_id='s',language='zh',tree_complexity=2,case_id='c',node_id='G1')]
    old=[dict(kind='P',source_id='s',language='zh',tree_complexity=2,case_id='c',node_id='G0')]
    chosen,plan=choose_tasks(tasks,old,{'P':1,'C':2,'F':2,'A':2},target=2,multiplier=1)
    assert len(chosen)==1 and chosen[0]['node_id']=='G1' and plan['A']['selected']==0
    assert choose_tasks(tasks,[],dict.fromkeys(KINDS,2),target=2)[0] == []
    print('prepare pure-function tests passed')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--show',nargs=2,metavar=('CASE_ID','NODE_ID'))
    parser.add_argument('--output',default=str(ROOT/'sol_generation_tasks.jsonl'))
    parser.add_argument('--plan-output',default=str(ROOT/'sol_generation_plan.json'))
    parser.add_argument('--target',type=int,default=500)
    parser.add_argument('--multiplier',type=float,default=1.25)
    args=parser.parse_args()
    if args.self_test:
        self_test()
    elif args.show:
        print(json.dumps(FitCatalog().packet(*args.show),ensure_ascii=False))
    else:
        print(json.dumps(prepare(args.output,args.plan_output,args.target,args.multiplier),ensure_ascii=False))
