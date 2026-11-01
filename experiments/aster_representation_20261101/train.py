import argparse
import copy
import gc
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

from experiments.aster_manuscript_clinical_20261030.native_memory import query_chunked_attention
from experiments.aster_abc_diagnostics_20261030.c.audit import strict_score
from .data import OUT, OLD, SEED, read, write, nodes, swap_actions, deployment
from .intervention import ConditionalIntervention, objective
from .engine import Engine, fill_action


def completed(path):
    return {(r['case_id'], r['view']): r for r in read(path) if r['status'] == 'complete'} if path.exists() else {}


def load_module(engine, checkpoint):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    module = ConditionalIntervention(engine.width, variant=saved['variant']).to(engine.device)
    module.load_state_dict(saved['model']); module.eval()
    return module, saved['layer']


def save_checkpoint(path, module, optimizer, epoch, layer, variant, stats):
    torch.save({'model': module.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                'layer': layer, 'variant': variant, 'rank': 4, 'stats': stats}, path)


def train_case(engine, case, module, layer, backward=True, accumulation=8):
    sample_nodes = nodes(case)
    inputs = [deployment(case), deployment(swap_actions(case))]
    contexts = [engine.context(c) for c in inputs]
    totals = defaultdict(float)
    for node in sample_nodes:
        state = None if module.variant == 'ordinary' else engine.blind(inputs[0], node['gid'])['states'][layer]
        logits, baseline, magnitudes = [], [], []
        for context in contexts:
            edited, clean, diag = engine.training_decision(context, node['history'], node['gid'], layer,
                                                         module, state, training=backward)
            logits.append(edited); baseline.append(clean); magnitudes.append(diag['magnitude'])
        target = torch.tensor(0 if node['target'].endswith('Y') else 1, device=engine.device)
        loss, terms = objective(logits, baseline, target, magnitudes, use_js=module.variant != 'no_js')
        if backward:
            (loss / len(sample_nodes) / accumulation).backward()
        totals['loss'] += float(loss.detach()) / len(sample_nodes)
        for key, value in terms.items():
            totals[key] += value / len(sample_nodes)
        del logits, baseline, magnitudes, loss, edited, clean, diag
    del contexts
    return dict(totals)


def static_route(engine, case):
    gid, path = 'G0', []
    groups = {g['group_id']: g for g in case['guideline']['groups']}
    while gid is not None:
        probabilities = engine.probabilities(case, gid)
        choice = int(probabilities.argmax())
        if choice == 2:
            break
        rid = 'R' + gid[1:] + ('Y' if choice == 0 else 'N')
        path.append(rid)
        gid = next((g['group_id'] for g in groups.values() if g.get('parent_rule_id') == rid), None)
    return path


def evaluate(engine, cases, path, module=None, layer=None, kind='zero', reuse_old=False):
    done = completed(path)
    old = {}
    if reuse_old:
        directory = OLD / 'structure_followup_20261101'
        for file in directory.glob('*.jsonl'):
            for row in read(file):
                name = 'zero_structured' if kind == 'zero' else 'mean_structured'
                if row.get('arm') == name and row.get('status') == 'complete':
                    old[row['case_id']] = row
    with path.open('a') as handle:
        for case in cases:
            for view in ['original', 'swap']:
                key = case['case_id'], view
                if key in done:
                    continue
                row = {'case_id': case['case_id'], 'source_id': case['source_id'], 'view': view,
                       'status': 'complete', 'kind': kind}
                if view == 'original' and case['case_id'] in old:
                    prev = old[case['case_id']]
                    row.update(result=prev['result'], score=prev['score'],
                               reused_from='structure_followup_20261101/' + prev['arm'])
                else:
                    current = case if view == 'original' else swap_actions(case)
                    public = deployment(current)
                    if 'target_rule' in case and kind == 'static':
                        from experiments.aster_bottleneck_20261031.runtime import execute
                        spec = json.loads((OLD/'module_one/frozen.json').read_text())['selected'].copy()
                        spec.pop('deeper', None); spec['structured'] = True
                        result = execute(engine.old, public, spec, route=static_route(engine, public))
                        import re
                        match = re.search(r'PATH:\s*(R0[YN])', result['text'])
                        rid = match[1] if match else None
                        row.update(result=result, score={'branch_correct': rid == case['target_rule']},
                                   pair_ids=case['pair_ids'], target=case['target_rule'])
                    elif 'target_rule' in case:
                        # Historical age pairs provide a local root truth, not a full-path truth.
                        gid = 'G0'
                        context = engine.context(public)
                        prefix = 'PATH: R' + gid[1:]
                        tokens = engine.builder.tokenizer.encode(prefix, add_special_tokens=False)
                        from .engine import copy_cache
                        from experiments.aster_bottleneck_20261031.runtime import attention_control
                        with torch.no_grad(), attention_control(engine.back):
                            clean = engine.back._forward(torch.tensor([tokens]), cache=copy_cache(context['result'].past_key_values),
                                start=context['length'], use_cache=True)
                        logits = clean.logits[0, -1].float()
                        if module is not None:
                            blind = None if module.variant == 'ordinary' else engine.blind(public, gid)['states'][layer]
                            logits, _ = engine.edit_query(clean, tokens[-1], context['length']+len(tokens)-1, layer, module, blind)
                        probs = logits[[engine.yes, engine.no]].softmax(-1)
                        if kind == 'fusion':
                            pb = engine.probabilities(public, gid).to(probs.device)
                            w = 0.5*(1-pb[2]); probs = (1-w)*probs + w*pb[:2]/pb[:2].sum().clamp_min(1e-12)
                        rid = 'R'+gid[1:]+('Y' if int(probs.argmax()) == 0 else 'N')
                        row.update(result={'prediction': rid}, score={'branch_correct': rid == case['target_rule']},
                                   pair_ids=case['pair_ids'], target=case['target_rule'])
                    elif kind == 'static':
                        from experiments.aster_bottleneck_20261031.runtime import execute
                        spec = json.loads((OLD/'module_one/frozen.json').read_text())['selected'].copy()
                        spec.pop('deeper', None); spec['structured'] = True
                        # The legacy executor expects gold fields for its root diagnostic
                        # only on free-generation paths; structured returns before that code.
                        result = execute(engine.old, public, spec, route=static_route(engine, public))
                        row.update(result=result, score=strict_score(case, result['text']))
                    else:
                        result = engine.generate(public, module, layer, fusion=kind == 'fusion')
                        row.update(result=result, score=strict_score(case, result['text']))
                if 'text' in row['result']:
                    fill_action(row['result'], (case if view == 'original' else swap_actions(case))['guideline'])
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'); handle.flush()
                done[key] = row
    return summarize(path)


def summarize(path):
    rows = list(completed(path).values())
    stats = {}
    for view in ['original', 'swap']:
        group = [r for r in rows if r['view'] == view]
        if not group:
            continue
        if 'branch_correct' in group[0]['score']:
            pairs = defaultdict(list)
            for r in group:
                for pair in r['pair_ids']:
                    pairs[pair.rsplit(':', 1)[0]].append(r['score']['branch_correct'])
            stats[view] = {'n': len(group), 'branch_correct': sum(r['score']['branch_correct'] for r in group),
                           'pairs_both_correct': sum(all(v) for v in pairs.values()), 'pairs': len(pairs)}
        else:
            edits = [x['magnitude'] for r in group for x in r['result'].get('diagnostics', [])]
            stats[view] = {'n': len(group), 'joint': sum(r['score']['exact_path'] and r['score']['leaf_correct'] for r in group),
                           'path': sum(r['score']['exact_path'] for r in group),
                           'leaf': sum(r['score']['leaf_correct'] for r in group),
                           'invalid': sum(r['score']['invalid'] for r in group),
                           'mean_edit': sum(edits)/len(edits) if edits else 0.0}
    original = {r['case_id']: r for r in rows if r['view'] == 'original'}
    swapped = {r['case_id']: r for r in rows if r['view'] == 'swap'}
    common = original.keys() & swapped.keys()
    if common and 'exact_path' in next(iter(original.values()))['score']:
        stats['swap_path_agreement'] = sum(original[k]['score']['path'] == swapped[k]['score']['path'] for k in common) / len(common)
        stats['swap_pairs'] = len(common)
    write(path.with_suffix('.summary.json'), stats)
    return stats


def train(variant, layer):
    torch.manual_seed(SEED); torch.set_num_threads(4)
    folder = OUT / f'{variant}_layer{layer}'; folder.mkdir(exist_ok=True)
    cases = read(OUT/'train.jsonl') + read(OUT/'age_train.jsonl')
    dev = read(OUT/'development.jsonl')
    engine = Engine()
    torch.manual_seed(SEED)
    module = ConditionalIntervention(engine.width, variant=variant).to(engine.device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.001, weight_decay=0.01)
    started = time.time()
    with query_chunked_attention(engine.model, 256):
        for epoch in [1, 2, 3]:
            checkpoint = folder / f'epoch{epoch}.pt'
            if checkpoint.exists():
                state = torch.load(checkpoint, map_location=engine.device, weights_only=False)
                module.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
            else:
                module.train(); optimizer.zero_grad(set_to_none=True)
                order = list(cases); random.Random(SEED+epoch).shuffle(order)
                totals = defaultdict(float)
                for i, case in enumerate(order):
                    write(folder/'status.json', {'status': 'training', 'epoch': epoch, 'completed_cases': i,
                                                'total_cases': len(order), 'elapsed': time.time()-started})
                    values = train_case(engine, case, module, layer)
                    for key, value in values.items():
                        totals[key] += value / len(order)
                    if (i+1) % 8 == 0 or i+1 == len(order):
                        if (i+1) == len(order) and len(order) % 8:
                            for param in module.parameters():
                                if param.grad is not None: param.grad.mul_(8/(len(order)%8))
                        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0, error_if_nonfinite=True)
                        optimizer.step(); optimizer.zero_grad(set_to_none=True)
                save_checkpoint(checkpoint, module, optimizer, epoch, layer, variant, dict(totals))
            module.eval()
            write(folder/'status.json', {'status': 'evaluating_development', 'epoch': epoch})
            evaluate(engine, dev, folder/f'development_epoch{epoch}.jsonl', module, layer, kind=variant)
        candidates = []
        for epoch in [1, 2, 3]:
            stats = json.loads((folder/f'development_epoch{epoch}.summary.json').read_text())
            candidates.append({'epoch': epoch, 'layer': layer, 'variant': variant,
                               'checkpoint': str(folder/f'epoch{epoch}.pt'), 'summary': stats})
        best = max(candidates, key=selection_key)
        write(folder/'selected.json', best)
        for split in ['age_train', 'age_diagnostic']:
            module, _ = load_module(engine, best['checkpoint'])
            evaluate(engine, read(OUT/(split+'.jsonl')), folder/(split+'.jsonl'), module, layer, kind=variant)
    write(folder/'status.json', {'status': 'complete', 'elapsed': time.time()-started,
          'peak_gb': torch.cuda.max_memory_allocated()/1e9, 'parameters': sum(p.numel() for p in module.parameters())})


def selection_key(candidate):
    s = candidate['summary']
    return s['original']['joint'], s['swap_path_agreement'], -s['original']['mean_edit'], -candidate['epoch'], -candidate['layer']


def evaluation_job(split, kind, selection=None):
    torch.manual_seed(SEED); torch.set_num_threads(4)
    engine = Engine(); module = None; layer = None
    if selection:
        selected = json.loads(Path(selection).read_text())
        module, layer = load_module(engine, selected['checkpoint'])
    started = time.time()
    with query_chunked_attention(engine.model, 256):
        evaluate(engine, read(OUT/(split+'.jsonl')), OUT/f'{split}_{kind}.jsonl', module, layer,
                 kind=kind, reuse_old=split == 'development' and kind in ['zero', 'static'])
        if kind in ['zero', 'fusion', 'static'] and split == 'development':
            for age in ['age_train', 'age_diagnostic']:
                evaluate(engine, read(OUT/(age+'.jsonl')), OUT/f'{age}_{kind}.jsonl', kind=kind)
    write(OUT/f'{split}_{kind}_status.json', {'status': 'complete', 'elapsed': time.time()-started,
                                           'peak_gb': torch.cuda.max_memory_allocated()/1e9})


def pilot():
    torch.manual_seed(SEED); torch.set_num_threads(4)
    engine = Engine(); module = ConditionalIntervention(engine.width).to(engine.device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.001)
    cases = read(OUT/'train.jsonl')
    lengths = []
    for case in cases:
        prompt = engine.builder.prepare(case['guideline'], case['patient'], case['question'], online=True)
        lengths.append((prompt.input_ids.shape[1], case))
    lengths.sort(key=lambda x: x[0]); chosen = [lengths[len(lengths)//2], lengths[-1]]
    records = []
    with query_chunked_attention(engine.model, 256):
        for length, case in chosen:
            torch.cuda.reset_peak_memory_stats(); started = time.time()
            module.train(); optimizer.zero_grad(set_to_none=True)
            values = train_case(engine, case, module, 18, accumulation=1)
            grad = float(torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0))
            if not torch.isfinite(torch.tensor(grad)) or grad <= 0:
                raise RuntimeError('Pilot has no finite nonzero intervention gradient')
            if any(p.grad is not None for p in engine.model.parameters()):
                raise RuntimeError('Backbone received gradients')
            optimizer.step()
            training_seconds = time.time()-started
            module.eval(); started = time.time()
            answers = [engine.generate(deployment(c), module, 18) for c in [case, swap_actions(case)]]
            generation_seconds = time.time()-started
            records.append({'case_id': case['case_id'], 'input_tokens': length, 'nodes': len(nodes(case)),
                'train_seconds_two_views': training_seconds, 'generation_seconds_two_views': generation_seconds,
                'peak_gb': torch.cuda.max_memory_allocated()/1e9, 'gradient_norm': grad, 'loss': values,
                'generated_steps': [len(r['path']) for r in answers]})
    write(OUT/'pilot.json', {'status': 'complete', 'cases': records,
          'parameters': sum(p.numel() for p in module.parameters()), 'backbone_gradients': False,
          'pilot_weights_discarded': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', choices=['full', 'ordinary', 'no_js'])
    parser.add_argument('--layer', type=int)
    parser.add_argument('--evaluate', choices=['development', 'confirmation'])
    parser.add_argument('--kind', default='zero')
    parser.add_argument('--selection')
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    if args.pilot: pilot()
    elif args.train: train(args.train, args.layer)
    else: evaluation_job(args.evaluate, args.kind, args.selection)
