"""Controls for hard branch selection and supervised use of blind states."""
import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

from experiments.aster_manuscript_clinical_20261030.native_memory import query_chunked_attention
from experiments.aster_abc_diagnostics_20261030.c.audit import strict_score
from .data import OUT, SEED, read, write, deployment, swap_actions, nodes
from .engine import Engine, fill_action
from .intervention import objective
from .train import completed, summarize, selection_key


class BlindReadout(nn.Module):
    """A matched-size supervised classifier; no full-guide representation input."""
    def __init__(self, width):
        super().__init__()
        self.features = nn.Linear(width, 8, bias=False)
        self.output = nn.Linear(8, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, state, base):
        shift = self.output(torch.tanh(self.features(state.float()))).squeeze(-1)
        delta = torch.stack([shift, -shift], dim=-1)
        magnitude = delta.square().sum() / base.detach().square().sum().clamp_min(1e-8)
        return base.float() + delta, magnitude


def branch(engine, public, gid, module=None, layer=None):
    if module is None:
        probs = engine.probabilities(public, gid)
        return int(probs[:2].argmax()), {'group': gid, 'probabilities': probs.tolist(), 'magnitude': 0.0}
    record = engine.blind(public, gid)
    with torch.no_grad():
        logits, magnitude = module(record['states'][layer].to(engine.device), record['logits'][:2].to(engine.device))
    return int(logits.argmax()), {'group': gid, 'magnitude': float(magnitude),
                                 'base_logits': record['logits'][:2].tolist(), 'edited_logits': logits.cpu().tolist()}


def lookup(engine, public, module=None, layer=None):
    groups = {g['group_id']: g for g in public['guideline']['groups']}
    gid, path, diagnostics, leaf = 'G0', [], [], None
    for _ in range(len(groups)):
        index, detail = branch(engine, public, gid, module, layer)
        rid = 'R' + gid[1:] + ('Y' if index == 0 else 'N')
        rule = next(r for r in groups[gid]['rules'] if r['rule_id'] == rid)
        path.append(rid); diagnostics.append(detail)
        if len(engine.builder.tokenizer.encode('PATH: ' + ', '.join(path), add_special_tokens=False)) > engine.old.freeze['max_new_tokens']:
            break
        terminal = re.match(r'(L\d+)[：:]', rule['action'])
        if terminal:
            leaf = terminal[1]; break
        gid = re.search(r'规则组 (G\d+)', rule['action'])[1]
    text = 'PATH: ' + ', '.join(path) + '\nFINAL: ' + (leaf or 'UNKNOWN')
    return fill_action({'text': text, 'path': path, 'leaf': leaf, 'diagnostics': diagnostics,
                        'output_limit': leaf is None, 'decoder': 'blind_branch_then_lookup'}, public['guideline'])


def evaluate_lookup(engine, cases, path, module=None, layer=None):
    done = completed(path)
    with path.open('a') as handle:
        for case in cases:
            original_key = case['case_id'], 'original'
            if original_key in done:
                original_result = done[original_key]['result']
            else:
                public = deployment(case)
                if 'target_rule' in case:
                    index, detail = branch(engine, public, 'G0', module, layer)
                    original_result = {'prediction': 'R0' + ('Y' if index == 0 else 'N'), 'diagnostics': [detail]}
                else:
                    original_result = lookup(engine, public, module, layer)
            for view in ['original', 'swap']:
                key = case['case_id'], view
                if key in done:
                    continue
                # Blind inputs are identical under terminal-action swaps: reuse the
                # selected path and look up the action in the current input guide.
                result = dict(original_result)
                current = case if view == 'original' else swap_actions(case)
                row = {'case_id': case['case_id'], 'source_id': case['source_id'], 'view': view,
                       'status': 'complete', 'kind': 'blind_readout' if module is not None else 'blind_lookup',
                       'result': result, 'swapped_path_reused': view == 'swap'}
                if 'target_rule' in case:
                    row.update(score={'branch_correct': result['prediction'] == case['target_rule']},
                               pair_ids=case['pair_ids'], target=case['target_rule'])
                else:
                    fill_action(result, current['guideline'])
                    row['score'] = strict_score(case, result['text'])
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'); handle.flush()
                done[key] = row
    return summarize(path)


def train_readout(layer):
    torch.set_num_threads(4)
    engine = Engine(); torch.manual_seed(SEED)
    module = BlindReadout(engine.width).to(engine.device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.001, weight_decay=0.01)
    cases = read(OUT/'train.jsonl') + read(OUT/'age_train.jsonl')
    folder = OUT/f'blind_readout_layer{layer}'; folder.mkdir(exist_ok=True)
    started = time.time()
    with query_chunked_attention(engine.model, 256):
        for epoch in [1, 2, 3]:
            checkpoint = folder/f'epoch{epoch}.pt'
            if checkpoint.exists():
                saved = torch.load(checkpoint, map_location=engine.device, weights_only=False)
                module.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
            else:
                module.train(); optimizer.zero_grad(set_to_none=True)
                order = list(cases); random.Random(SEED+epoch).shuffle(order)
                total = defaultdict(float)
                for i, case in enumerate(order):
                    write(folder/'status.json', {'status':'training', 'epoch':epoch, 'completed_cases':i, 'total_cases':len(order)})
                    public = deployment(case); selected_nodes = nodes(case)
                    for node in selected_nodes:
                        record = engine.blind(public, node['gid'])
                        base = record['logits'][:2].to(engine.device)
                        logits, magnitude = module(record['states'][layer].to(engine.device), base)
                        target = torch.tensor(0 if node['target'].endswith('Y') else 1, device=engine.device)
                        # JS is exactly zero: the blind representation is unchanged
                        # by action swaps. Regularize the normalized logit correction.
                        loss, terms = objective([logits,logits], [base,base], target, [magnitude,magnitude])
                        (loss/len(selected_nodes)/8).backward()
                        total['loss'] += float(loss.detach())/len(selected_nodes)/len(order)
                    if (i+1)%8 == 0 or i+1 == len(order):
                        if i+1 == len(order) and len(order)%8:
                            for p in module.parameters():
                                if p.grad is not None: p.grad.mul_(8/(len(order)%8))
                        torch.nn.utils.clip_grad_norm_(module.parameters(),1.0,error_if_nonfinite=True)
                        optimizer.step(); optimizer.zero_grad(set_to_none=True)
                torch.save({'model':module.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,
                            'layer':layer,'variant':'blind_readout','stats':dict(total)},checkpoint)
            module.eval(); write(folder/'status.json', {'status':'evaluating_development','epoch':epoch})
            evaluate_lookup(engine, read(OUT/'development.jsonl'), folder/f'development_epoch{epoch}.jsonl', module, layer)
        choices = [{'epoch':ep,'layer':layer,'variant':'blind_readout','checkpoint':str(folder/f'epoch{ep}.pt'),
                    'summary':json.loads((folder/f'development_epoch{ep}.summary.json').read_text())} for ep in [1,2,3]]
        selected = max(choices,key=selection_key); write(folder/'selected.json',selected)
        saved = torch.load(selected['checkpoint'], map_location=engine.device, weights_only=False)
        module.load_state_dict(saved['model'])
        for split in ['age_train','age_diagnostic']:
            evaluate_lookup(engine,read(OUT/(split+'.jsonl')),folder/(split+'.jsonl'),module,layer)
    write(folder/'status.json',{'status':'complete','elapsed':time.time()-started,
          'parameters':sum(p.numel() for p in module.parameters()),'peak_gb':torch.cuda.max_memory_allocated()/1e9})


def evaluation_job(split,kind,selection=None):
    torch.set_num_threads(4);engine=Engine();module=None;layer=None;started=time.time()
    if selection:
        selected=json.loads(Path(selection).read_text());layer=selected['layer']
        saved=torch.load(selected['checkpoint'],map_location=engine.device,weights_only=False)
        module=BlindReadout(engine.width).to(engine.device);module.load_state_dict(saved['model']);module.eval()
    with query_chunked_attention(engine.model,256):
        evaluate_lookup(engine,read(OUT/(split+'.jsonl')),OUT/f'{split}_{kind}.jsonl',module,layer)
        if split=='development':
            for age in ['age_train','age_diagnostic']:
                evaluate_lookup(engine,read(OUT/(age+'.jsonl')),OUT/f'{age}_{kind}.jsonl',module,layer)
    write(OUT/f'{split}_{kind}_status.json',{'status':'complete','elapsed':time.time()-started,
                                         'peak_gb':torch.cuda.max_memory_allocated()/1e9})


def pilot():
    from .intervention import ConditionalIntervention
    from .train import train_case
    torch.set_num_threads(4);engine=Engine();torch.manual_seed(SEED)
    module=ConditionalIntervention(engine.width,variant='context').to(engine.device)
    optimizer=torch.optim.AdamW(module.parameters(),lr=.001)
    ids={r['case_id'] for r in json.loads((OUT/'pilot.json').read_text())['cases']}
    records=[]
    with query_chunked_attention(engine.model,256):
        for case in read(OUT/'train.jsonl'):
            if case['case_id'] not in ids:continue
            torch.cuda.reset_peak_memory_stats();optimizer.zero_grad(set_to_none=True);start=time.time()
            train_case(engine,case,module,18,accumulation=1)
            grad=float(torch.nn.utils.clip_grad_norm_(module.parameters(),1.,error_if_nonfinite=True));optimizer.step()
            training_seconds=time.time()-start;start=time.time()
            for view in [case,swap_actions(case)]:engine.generate(deployment(view),module,18)
            records.append({'case_id':case['case_id'],'nodes':len(nodes(case)),
                            'train_seconds_two_views':training_seconds,'generation_seconds_two_views':time.time()-start,
                            'gradient_norm':grad,'peak_gb':torch.cuda.max_memory_allocated()/1e9})
    write(OUT/'context_pilot.json',{'status':'complete','cases':records,'weights_discarded':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--train',action='store_true')
    parser.add_argument('--layer',type=int);parser.add_argument('--evaluate',choices=['development','confirmation'])
    parser.add_argument('--kind',default='blind_lookup');parser.add_argument('--selection');parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args()
    if args.pilot:pilot()
    elif args.train:train_readout(args.layer)
    else:evaluation_job(args.evaluate,args.kind,args.selection)
