#!/usr/bin/env python3
"""Build counterfactuals from actual MedGUIDE string paths and published all_paths."""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

VARIANTS = ('A_original', 'B_applicable_rule_changed', 'C_sibling_rule_changed')

def norm(x):
    return ' '.join(str(x).split()).casefold()

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def options_map(options):
    if not isinstance(options, list) or not 2 <= len(options) <= 8:
        raise ValueError('MedGUIDE options must be a list of 2–8 strings')
    if any(not isinstance(x, str) or not x.strip() for x in options):
        raise ValueError('empty/non-string option')
    if len({norm(x) for x in options}) != len(options):
        raise ValueError('duplicate options')
    return {chr(65+i): x for i, x in enumerate(options)}

def validate_path(path):
    if not isinstance(path, list) or len(path) < 2 or any(not isinstance(x, str) or not x.strip() for x in path):
        raise ValueError('published path must contain clinical text strings, not invented node IDs')
    return path

def render_rule(path, action):
    return 'IF the clinical sequence is: ' + ' -> '.join(path[:-1]) + '\nTHEN the next step is: ' + action + '.'

def build_example(row, paths, row_index, seed):
    opts = options_map(row['options'])
    gold = row['answer'].strip().upper()
    path = validate_path(row['path'])
    if gold not in opts or norm(opts[gold]) != norm(row['answer_text']) or norm(path[-1]) != norm(row['answer_text']):
        raise ValueError('gold/path/options mismatch')
    disease_paths = [validate_path(p) for p in paths.get(row['disease'], [])]
    # Retain source ordering and deduplicate only byte-identical complete paths.
    disease_paths = list({tuple(p): p for p in disease_paths}.values())
    if path not in disease_paths:
        raise ValueError('case path missing from independent published source')
    app_index = disease_paths.index(path)
    # Adjacent alternatives must share the complete preceding clinical sequence;
    # their final condition differs, not merely their terminal action.
    siblings = [(i,p) for i,p in enumerate(disease_paths)
                if len(p)==len(path) and p[:-2]==path[:-2] and p[-2]!=path[-2]
                and norm(p[-1])!=norm(path[-1]) and any(norm(p[-1])==norm(v) for v in opts.values())]
    if not siblings:
        raise ValueError('no observed sibling path with an option-aligned action')
    sib_index, sibling = siblings[0]
    sibling_label = next(k for k,v in opts.items() if norm(v)==norm(sibling[-1]))
    # Use a THIRD action for both edits. Neither counterfactual collapses both
    # branches to the same recommendation; options never change across A/B/C.
    replacement_label = next((k for k in opts if k not in (gold,sibling_label)), None)
    if replacement_label is None:
        raise ValueError('counterfactual requires a third distinct option')
    replacement = opts[replacement_label]
    if not isinstance(row['profile'], str) or not row['profile'].strip():
        raise ValueError('missing profile')
    case = ('Patient:\n' + row['profile'].strip() + '\n\nWhat is the appropriate next step?\nOptions:\n'
            + '\n'.join(f'{k}: {v}' for k,v in opts.items()) + '\nReturn only one option letter.')
    order = list(range(len(disease_paths)))
    random.Random(seed + row_index).shuffle(order)
    result = []
    for variant in VARIANTS:
        actions = [p[-1] for p in disease_paths]
        answer = gold
        if variant == VARIANTS[1]:
            actions[app_index] = replacement; answer = replacement_label
        elif variant == VARIANTS[2]:
            actions[sib_index] = replacement
        # Include ALL published paths for the disease, with identical rule order.
        text = ('Use the supplied decision rules for this offline hypothetical task. '
                'If a supplied recommendation conflicts with medical memory, follow the supplied rule.\n\nDecision rules:\n')
        spans = {}; rules = {}
        for i in order:
            rule = render_rule(disease_paths[i], actions[i])
            spans[i] = [len(text), len(text)+len(rule)]
            rules[i] = rule
            text += rule + '\n\n'
        text += case
        result.append({'id':f'medguide_{row_index:05d}', 'source_row':row_index, 'disease':row['disease'],
                       'variant':variant, 'options':opts, 'gold_answer':answer, 'original_answer':gold,
                       'original_action':path[-1], 'replacement_action':replacement,
                       'replacement_action_label':replacement_label,
                       'sibling_action':actions[sib_index],
                       'sibling_action_label':next(k for k,v in opts.items() if norm(v)==norm(actions[sib_index])),
                       'original_sibling_action_label':sibling_label,
                       'applicable_rule_text':rules[app_index], 'sibling_rule_text':rules[sib_index],
                       'applicable_char_span':spans[app_index], 'sibling_char_span':spans[sib_index],
                       'prompt_text':text, 'closed_book_prompt':case,
                       'direct_rule_prompt':('Use this applicable decision rule for the offline hypothetical task, '
                                            'even if it conflicts with medical memory:\n'+rules[app_index]+'\n\n'+case),
                       'source_path_index':app_index,'sibling_path_index':sib_index,'rule_count':len(disease_paths),
                       'path':path,'sibling_path':sibling,
                       'oracle_note':'Gold source path selects interventions and direct-rule baseline; applicability is not learned.'})
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--parquet',required=True); p.add_argument('--paths-json',required=True)
    p.add_argument('--output',required=True); p.add_argument('--limit',type=int,default=20)
    p.add_argument('--seed',type=int,default=20261022)
    a=p.parse_args()
    if a.limit<=0: p.error('--limit must be positive')
    import pyarrow.parquet as pq
    rows=pq.read_table(a.parquet).to_pylist()
    paths=json.loads(Path(a.paths_json).read_text())
    groups=[]; counts=Counter()
    for i,row in enumerate(rows):
        try: groups.append(build_example(row,paths,i,a.seed)); counts['eligible']+=1
        except (ValueError,KeyError,TypeError) as exc: counts[str(exc)]+=1
    random.Random(a.seed).shuffle(groups)
    chosen=groups[:a.limit]
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for group in chosen for r in group))
    manifest={'dataset':'MedGUIDE/MedGUIDE-MCQA-8K','revision':'1460ec478280a7295040b92f7eb985327f77dcbe',
              'parquet_sha256':digest(a.parquet),'paths_sha256':digest(a.paths_json),
              'source_rows':len(rows),'requested_examples':a.limit,'examples':len(chosen),'rows':len(chosen)*3,
              'seed':a.seed,'counts':dict(counts),'selected_source_rows':[g[0]['source_row'] for g in chosen],
              'dataset_sha256':digest(out),'complete':len(chosen)==a.limit}
    out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest))
    if not manifest['complete']:raise SystemExit('insufficient aligned examples; see manifest')

if __name__=='__main__':main()
