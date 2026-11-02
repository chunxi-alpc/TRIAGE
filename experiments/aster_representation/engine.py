from typing import Any
import contextlib
import re
import time

import torch

from .inputs import direct_prompt, PromptBuilder
from .inputs import path_trie
from .attention import attention_control
from src.aster.action_blind import HFLabelScorer


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


class Backbone:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.blocks = model.model.layers

    def _forward(self, ids: torch.Tensor, *, cache: Any = None, start: int = 0, use_cache: bool = False):
        ids = ids.to(self.device)
        n = ids.shape[1]
        position = torch.arange(start, start + n, device=self.device)
        return self.model(input_ids=ids, attention_mask=torch.ones((1, start + n), device=self.device, dtype=torch.long),
                          position_ids=position[None, :], cache_position=position, past_key_values=cache,
                          use_cache=use_cache, return_dict=True, logits_to_keep=1)

class Engine:
    def __init__(self, model, tokenizer):
        self.model = model.eval().requires_grad_(False)
        self.builder = PromptBuilder(tokenizer, chat_template_kwargs={'enable_thinking': False})
        self.back = Backbone(model)
        self.scorer = HFLabelScorer(model, tokenizer, max_input_tokens=40960)
        self.device = self.back.device
        self.width = model.config.hidden_size
        self.prefix_key, self.prefix_cache = None, None
        self.blind_cache = {}
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
        ids, labels = self.scorer.prepare(prompt, ['A', 'B', 'C'])
        states, handles = {}, []
        for layer in [18]:
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

    def generate(self, case, module, layer=18):
        start = time.time()
        context = self.context(case)
        clean = context['result']
        node = path_trie(self.builder.tokenizer, case['guideline'])
        tokens, diagnostics = [], []
        budget = 512
        while 'terminal' not in node and len(tokens) < budget:
            batch = []
            allowed = sorted(k for k in node if isinstance(k, int))
            if len(allowed) > 1:
                prefix = self.builder.tokenizer.decode(tokens, skip_special_tokens=True)
                match = re.search(r'R(\d+)$', prefix)
                if not match or set(allowed) != {self.yes, self.no}:
                    raise ValueError('Expected binary rule decision after a complete rule number')
                gid = 'G' + match[1]
                record = self.blind(case, gid)
                logits, detail = self.edit_query(clean, tokens[-1], context['length'] + len(tokens) - 1,
                                                 layer, module, record['states'][layer])
                diagnostics.append({'group': gid, 'gate': float(detail['gate']),
                                    'magnitude': float(detail['magnitude'])})
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
