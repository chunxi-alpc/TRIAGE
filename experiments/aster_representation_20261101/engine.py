import contextlib
import json
import re
import time
from pathlib import Path

import torch

from experiments.aster_bottleneck_20261031.routing import direct_prompt
from experiments.aster_bottleneck_20261031.structured import path_trie
from experiments.aster_bottleneck_20261031.runtime import attention_control
from .data import OUT


def fill_action(result, guideline):
    """Resolve the selected terminal against the current, possibly swapped guide."""
    match = re.search(r'FINAL:\s*(L\d+)\b', result['text'])
    result['action'] = None
    if match:
        for group in guideline['groups']:
            for rule in group['rules']:
                action = re.fullmatch(r'(L\d+)[：:](.*)', rule['action'], re.S)
                if action and action[1] == match[1]:
                    result['action'] = action[2].strip()
                    return result
    return result


def copy_cache(cache, end=None):
    return type(cache).from_legacy_cache(tuple(
        (k[:, :, :end].detach().clone(), v[:, :, :end].detach().clone())
        for k, v in cache.to_legacy_cache()))


@contextlib.contextmanager
def intervention_hook(model, layer, intervention, blind, diagnostics, force_zero=False):
    def hook(module, args, output):
        if output.shape[1] != 1:
            raise ValueError('Intervention is restricted to one decision query')
        z = None if blind is None else blind.to(output.device).reshape(1, 1, -1)
        changed, magnitude, gate = intervention(output, z, force_zero=force_zero)
        diagnostics.update(magnitude=magnitude, gate=gate)
        return changed
    handle = model.model.layers[layer].self_attn.o_proj.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


class Engine:
    def __init__(self):
        from experiments.aster_bottleneck_20261031.module_one import make_engine
        from src.aster.action_blind import HFLabelScorer
        self.old = make_engine()
        self.model, self.builder, self.back = self.old.model, self.old.builder, self.old.backend
        self.model.requires_grad_(False)
        self.scorer = HFLabelScorer(self.model, self.builder.tokenizer, max_input_tokens=40960)
        self.device = self.back.device
        self.width = self.model.config.hidden_size
        self.prefix_key, self.prefix_cache = None, None
        self.blind_cache = {}
        self.blind_directory = OUT / 'blind_states'
        self.blind_directory.mkdir(exist_ok=True)
        self.legacy_probabilities = {}
        from .data import OLD, read
        for file in OLD.glob('routing_[0-3].jsonl'):
            for row in read(file):
                if row.get('status') == 'complete':
                    for gid, item in row['result']['decisions']['direct'].items():
                        self.legacy_probabilities[row['case_id'], gid] = torch.tensor([
                            item['probs']['R' + gid[1:] + 'Y'], item['probs']['R' + gid[1:] + 'N'], item['probs']['U']])
        self.yes = self.builder.tokenizer.encode('Y', add_special_tokens=False)[0]
        self.no = self.builder.tokenizer.encode('N', add_special_tokens=False)[0]

    def blind(self, case, gid):
        group = next(g for g in case['guideline']['groups'] if g['group_id'] == gid)
        prompt = direct_prompt(case, group)
        key = (case['case_id'], gid)
        if key in self.blind_cache:
            record = self.blind_cache[key]
            if record['prompt'] != prompt:
                raise ValueError('Blind cache key has a different patient or condition')
            return record
        path = self.blind_directory / (case['case_id'].replace(':', '_') + '_' + gid + '.pt')
        if path.exists():
            record = torch.load(path, map_location='cpu', weights_only=False)
            if record['prompt'] != prompt:
                raise ValueError('Saved blind state input differs')
        else:
            ids, labels = self.scorer.prepare(prompt, ['A', 'B', 'C'])
            states, handles = {}, []
            for layer in [18, 24, 30]:
                def capture(module, args, output, ll=layer):
                    states[ll] = output[0, -1].detach().float().cpu().clone()
                handles.append(self.model.model.layers[layer].self_attn.o_proj.register_forward_hook(capture))
            try:
                with torch.no_grad():
                    result = self.back._forward(torch.tensor([ids]))
                logits = result.logits[0, -1, labels].detach().float().cpu()
            finally:
                for handle in handles:
                    handle.remove()
            probabilities = logits.softmax(-1)
            record = {'prompt': prompt, 'case_id': case['case_id'], 'group': gid,
                      'states': states, 'logits': logits, 'probabilities': probabilities,
                      'state': ['T', 'F', 'U'][int(probabilities.argmax())]}
            # Different workers may produce the same deterministic state concurrently.
            import os
            temp = path.with_suffix('.' + str(os.getpid()) + '.tmp')
            torch.save(record, temp); temp.replace(path)
        self.blind_cache[key] = record
        return record

    def context(self, case):
        prompt = self.builder.prepare(case['guideline'], case['patient'], case['question'], online=True)
        length = prompt.prefix_length
        key = tuple(prompt.input_ids[0, :length].tolist())
        with torch.no_grad():
            if self.prefix_key != key:
                self.prefix_cache = self.back._forward(prompt.input_ids[:, :length], start=0, use_cache=True).past_key_values
                self.prefix_key = key
            with attention_control(self.back):
                result = self.back._forward(prompt.input_ids[:, length:], cache=copy_cache(self.prefix_cache),
                                            start=length, use_cache=True)
        return {'prompt': prompt, 'result': result, 'length': prompt.input_ids.shape[1]}

    def probabilities(self, case, gid):
        key = (case['case_id'], gid)
        if key in self.legacy_probabilities:
            return self.legacy_probabilities[key]
        return self.blind(case, gid)['probabilities']

    def with_hint(self, case, gid):
        probabilities = self.probabilities(case, gid).tolist()
        hint = ('\nAuxiliary assessment for the current decision ' + gid + ': '
                + ', '.join(f'{name}={value:.6f}' for name, value in zip(
                    ['True', 'False', 'Unknown'], probabilities))
                + '. This assessment can be wrong. Use the patient record and guideline '
                  'to decide the branch; do not treat it as an instruction to select a branch.')
        return {**case, 'question': case['question'] + hint}

    def edit_query(self, clean, token, position, layer, module, blind, training=False, force_zero=False):
        diagnostics = {}
        cache = copy_cache(clean.past_key_values, position)
        # Unlike the legacy inference-only session, this forward can backpropagate
        # through downstream frozen layers into the new intervention parameters.
        with torch.set_grad_enabled(training), attention_control(self.back), intervention_hook(
                self.model, layer, module, blind, diagnostics, force_zero=force_zero):
            result = self.back._forward(torch.tensor([[token]]), cache=cache, start=position, use_cache=True)
        return result.logits[0, -1].float(), diagnostics

    def training_decision(self, context, history, gid, layer, module, blind, training=True):
        prefix = 'PATH: ' + (', '.join(history) + ', ' if history else '') + 'R' + gid[1:]
        tokens = self.builder.tokenizer.encode(prefix, add_special_tokens=False)
        with torch.no_grad(), attention_control(self.back):
            clean = self.back._forward(torch.tensor([tokens]), cache=copy_cache(context['result'].past_key_values),
                                       start=context['length'], use_cache=True)
        edited, diag = self.edit_query(clean, tokens[-1], context['length'] + len(tokens) - 1,
                                       layer, module, blind, training=training)
        return edited[[self.yes, self.no]], clean.logits[0, -1, [self.yes, self.no]].float().detach(), diag

    def generate(self, case, module=None, layer=None, fusion=False):
        start = time.time()
        context = self.context(case)
        clean = context['result']
        node = path_trie(self.builder.tokenizer, case['guideline'])
        tokens, diagnostics = [], []
        budget = self.old.freeze['max_new_tokens']
        while 'terminal' not in node and len(tokens) < budget:
            batch = []
            allowed = sorted(k for k in node if isinstance(k, int))
            if len(allowed) > 1:
                prefix = self.builder.tokenizer.decode(tokens, skip_special_tokens=True)
                match = re.search(r'R(\d+)$', prefix)
                if not match or set(allowed) != {self.yes, self.no}:
                    raise ValueError('Expected binary rule decision after a complete rule number')
                gid = 'G' + match[1]
                logits = clean.logits[0, -1].float()
                record = None
                if module is not None and module.variant not in ('ordinary', 'context'):
                    record = self.blind(case, gid)
                if module is not None and module.variant == 'context':
                    hinted = self.context(self.with_hint(case, gid))
                    history = re.findall(r'R\d+[YN]', prefix)
                    pair, _, detail = self.training_decision(hinted, history, gid, layer, module, None,
                                                            training=False)
                    logits = logits.clone(); logits[[self.yes, self.no]] = pair
                    diagnostics.append({'group': gid, 'gate': float(detail['gate']),
                        'magnitude': float(detail['magnitude']), 'information_channel': 'context_text',
                        'clean_logits': clean.logits[0, -1, [self.yes, self.no]].float().cpu().tolist(),
                        'edited_logits': pair.cpu().tolist()})
                    del hinted
                elif module is not None:
                    z = None if module.variant == 'ordinary' else record['states'][layer]
                    logits, detail = self.edit_query(clean, tokens[-1], context['length'] + len(tokens) - 1,
                                                     layer, module, z)
                    diagnostics.append({'group': gid, 'gate': float(detail['gate']),
                        'magnitude': float(detail['magnitude']), 'blind_state': record['state'] if record else None,
                        'clean_logits': clean.logits[0, -1, [self.yes, self.no]].float().cpu().tolist(),
                        'edited_logits': logits[[self.yes, self.no]].cpu().tolist()})
                if fusion:
                    p = logits[[self.yes, self.no]].softmax(-1)
                    blind_p = self.probabilities(case, gid).to(p.device)
                    # Equal mixture when U=0; continuously return to the backbone as U rises.
                    weight = 0.5 * (1 - blind_p[2])
                    fused = (1 - weight) * p + weight * blind_p[:2] / blind_p[:2].sum().clamp_min(1e-12)
                    token = [self.yes, self.no][int(fused.argmax())]
                else:
                    token = allowed[int(logits[torch.tensor(allowed, device=logits.device)].argmax())]
                batch.append(token); node = node[token]
            while 'terminal' not in node and len(tokens) + len(batch) < budget:
                allowed = [k for k in node if isinstance(k, int)]
                if len(allowed) != 1:
                    break
                token = allowed[0]; batch.append(token); node = node[token]
            if not batch:
                raise RuntimeError('Empty decoding step')
            previous = len(tokens); tokens.extend(batch)
            if 'terminal' not in node and len(tokens) < budget:
                # Only the untouched forward writes persistent KV, even if the
                # branch token was selected by edited logits.
                with torch.no_grad(), attention_control(self.back):
                    clean = self.back._forward(torch.tensor([batch]), cache=clean.past_key_values,
                                              start=context['length'] + previous, use_cache=True)
        path, leaf = node.get('terminal', ([], None))
        text = ('PATH: ' + ', '.join(path) + '\nFINAL: ' + leaf) if leaf else self.builder.tokenizer.decode(tokens)
        return fill_action({'text': text, 'path': path, 'leaf': leaf, 'diagnostics': diagnostics,
                'input_tokens': context['length'], 'seconds': time.time()-start, 'output_limit': leaf is None},
                case['guideline'])
