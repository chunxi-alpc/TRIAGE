"""Limited causal query-attention-head localization for Qwen3 GQA.

The o_proj input is the concatenated output of QUERY heads. Scaling its head_dim
slices changes one query head without confusing it with shared K/V groups.
Only last-token logits and scalar head norms are retained; no activation cache.
"""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
import math
import json
import time
import torch


@contextmanager
def scale_query_heads(model, heads, scale=0.0, stats=None, token_positions=None):
    config = model.config
    n_heads = int(config.num_attention_heads)
    head_dim = int(getattr(config, 'head_dim', config.hidden_size // n_heads))
    handles = []
    by_layer = {}
    for layer, head in heads:
        if not (0 <= layer < config.num_hidden_layers and 0 <= head < n_heads):
            raise ValueError(f'Invalid query head {layer}:{head}')
        by_layer.setdefault(layer, []).append(head)
    def make_hook(indices):
        def hook(module, args):
            activation = args[0]
            if activation.shape[-1] != n_heads * head_dim:
                raise ValueError('o_proj input does not match query-head concatenation')
            changed = activation.clone()
            for head in indices:
                sl = slice(head * head_dim, (head + 1) * head_dim)
                if stats is not None:
                    stats['calls'] = stats.get('calls', 0) + 1
                    stats['changed_activation_l1'] = stats.get('changed_activation_l1', 0.0) + float(
                        ((activation[..., sl] if token_positions is None else activation[:, [i for i in token_positions if 0 <= i < activation.shape[1]], sl]).float() * (1.0-scale)).abs().sum().item())
                if token_positions is None:
                    changed[..., sl] *= scale
                else:
                    positions = [i for i in token_positions if 0 <= i < changed.shape[1]]
                    changed[:, positions, sl] *= scale
            return (changed, *args[1:])
        return hook
    try:
        for layer, indices in by_layer.items():
            handles.append(model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(make_hook(indices)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _encode_probe(tokenizer, rule, patient, module1_context=None):
    messages = [{'role': 'system', 'content': 'Evaluate the supplied clinical rule using only explicit patient findings. Answer with exactly one letter: A if the recommendation applies, B if it does not apply. Every required condition must hold.'},
                {'role': 'user', 'content': f'Rule:\n{rule}\n\n{patient}\n\nShould the recommendation be applied? A or B:'}]
    if module1_context is not None:
        messages[1]['content'] += '\nClinical question: ' + module1_context['question']
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    encoded = tokenizer(prompt, return_tensors='pt', add_special_tokens=False, return_offsets_mapping=module1_context is not None)
    if module1_context is not None:
        offsets = encoded.pop('offset_mapping')[0].tolist()
        patient_start = prompt.find(patient)
        positions = {encoded['input_ids'].shape[1]-1}
        weights = []
        for relation in module1_context['relations']:
            weight = float(relation.get('binding_gate', relation.get('weight', 0.)))
            if weight <= 0:
                continue
            weights.append(weight)
            for evidence in relation.get('evidence', []):
                text = evidence['text']
                start = prompt.find(text, patient_start, patient_start+len(patient))
                if start < 0:
                    raise ValueError('Cached evidence text missing from probe patient')
                end = start+len(text)
                positions.update(i for i,(a,b) in enumerate(offsets) if a < end and b > start and b>a)
        encoded['_causal_token_positions'] = sorted(positions)
        encoded['_cached_binding_weight'] = sum(weights)/len(weights) if weights else 0.
    return encoded


def _label_id(tokenizer, label):
    ids = tokenizer.encode(label, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f'Label {label} is not a single token')
    return ids[0]


def locate_heads(model, tokenizer, pairs, candidates=None, max_selected=4):
    """Score baseline-minus-ablation applicable execution and inapplicable misuse.

    Effect uses log P(A)-log P(B). A beneficial head must positively contribute to
    applicable execution AND contribute more to applicable than inapplicable A.
    Scores on two synthetic rule contrasts are screening evidence, not a benchmark.
    """
    if max_selected < 1 or not pairs:
        raise ValueError('Need pairs and positive max_selected')
    config = model.config
    n_heads, n_kv = int(config.num_attention_heads), int(config.num_key_value_heads)
    if n_heads % n_kv:
        raise ValueError('Unsupported unequal GQA grouping')
    if candidates is None:
        layers = sorted(set([config.num_hidden_layers//3, 2*config.num_hidden_layers//3, config.num_hidden_layers-1]))
        heads = sorted(set([0, n_heads//4, n_heads//2, 3*n_heads//4]))
        candidates = [(int(layer), int(head)) for layer in layers for head in heads]
    device = model.get_input_embeddings().weight.device
    label_a, label_b = _label_id(tokenizer, 'A'), _label_id(tokenizer, 'B')
    probes = []
    for pair in pairs:
        for applicability in ('applicable', 'inapplicable'):
            encoded = _encode_probe(tokenizer, pair['rule'], pair[applicability + '_patient'], pair.get(applicability + '_module1_context'))
            if encoded['input_ids'].shape[1] > 2048:
                raise ValueError('Calibration prompt exceeds 2048 tokens; select another rule')
            positions = encoded.pop('_causal_token_positions', None)
            binding_weight = encoded.pop('_cached_binding_weight', None)
            probes.append({'pair_id': pair['pair_id'], 'applicability': applicability,
                           'causal_token_positions': positions, 'cached_binding_weight': binding_weight,
                           'encoded': {k: v.to(device) for k, v in encoded.items()}})
    def margin(encoded):
        with torch.inference_mode():
            logits = model(**encoded, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            # log-softmax normalization cancels exactly in this two-label margin.
            value = float((logits[label_a]-logits[label_b]).item())
        if not math.isfinite(value):
            raise RuntimeError('Nonfinite decision margin')
        return value
    started = time.time()
    baselines = [margin(p['encoded']) for p in probes]
    rows = []
    for layer, head in candidates:
        stats, effects = {}, []
        changed = []
        for p in probes:
            with scale_query_heads(model, [(layer, head)], 0.0, stats, token_positions=p['causal_token_positions']):
                changed.append(margin(p['encoded']))
        for p, base, ablated in zip(probes, baselines, changed):
            effects.append({'pair_id': p['pair_id'], 'applicability': p['applicability'],
                            'baseline_margin': base, 'ablated_margin': ablated, 'effect': base-ablated})
        app = sum(e['effect'] for e in effects if e['applicability']=='applicable') / len(pairs)
        inapp = sum(e['effect'] for e in effects if e['applicability']=='inapplicable') / len(pairs)
        score = app-inapp
        beneficial = bool(app > 0 and score > 0)
        rows.append({'layer': int(layer), 'head': int(head), 'kv_head': int(head)//(n_heads//n_kv),
                     'causal_score': score, 'applicable_effect': app, 'inapplicable_effect': inapp,
                     'beneficial': beneficial, 'active': False, 'effects': effects,
                     'hook_calls': stats.get('calls',0), 'changed_activation_l1': stats.get('changed_activation_l1',0.)})
    rows.sort(key=lambda r: r['causal_score'], reverse=True)
    selected = [dict(r, active=True) for r in rows if r['beneficial']][:max_selected]
    selected_ids = {(r['layer'], r['head']) for r in selected}
    for row in rows:
        row['active'] = (row['layer'], row['head']) in selected_ids
    return {'schema_version': 'module2.v1', 'created_at': datetime.now(timezone.utc).isoformat(),
        'model': {'path': str(getattr(config,'_name_or_path','')), 'model_type': config.model_type,
                  'num_hidden_layers': int(config.num_hidden_layers), 'num_attention_heads': n_heads,
                  'num_key_value_heads': n_kv, 'head_dim': int(getattr(config,'head_dim',config.hidden_size//n_heads))},
        'selected_heads': selected, 'candidate_heads': rows,
        'baseline_probes': [{'pair_id': p['pair_id'], 'applicability': p['applicability'],
                             'margin': m, 'decision': 'A' if m>0 else 'B',
                             'expected_decision': 'A' if p['applicability']=='applicable' else 'B',
                             'prompt_tokens': p['encoded']['input_ids'].shape[1],
                             'causal_token_positions': p['causal_token_positions'],
                             'cached_binding_weight': p['cached_binding_weight']} for p,m in zip(probes,baselines)],
        'metadata': {'method': 'query-head output ablation through o_proj forward_pre_hook',
            'decision_target': 'binary rule applicability A/B; not full treatment/action quality',
            'baseline_correct': sum((m > 0) == (p["applicability"] == "applicable") for p,m in zip(probes,baselines)),
            'baseline_total': len(probes),
            'effect': 'baseline - ablated log P(A)/P(B)',
            'score': 'mean applicable effect - mean inapplicable misuse effect',
            'selection': 'applicable_effect > 0 and causal_score > 0; top score; no fallback',
            'max_selected': max_selected, 'pair_count': len(pairs), 'candidate_count': len(rows),
            'forward_count': len(probes)*(len(rows)+1), 'ablation_scale': 0.,
            'elapsed_seconds': round(time.time()-started,3), 'split': 'train_val',
            'synthetic_probes': all(p.get('provenance',{}).get('synthetic_patient',True) for p in pairs),
            'module1_cache_drives_target_selection': all('applicable_module1_context' in p for p in pairs),
            'module1_calls_during_localization': 0,
            'labels_or_action_gates_in_prompt': False,
            'cached_signal_strategy': 'cached rules, decisive evidence, applicability labels, relation weights select pairs and positive-binding evidence hook targets; action/status labels excluded from prompts',
            'cache_hook_scope': 'positive-binding cached patient evidence tokens and final decision token; rule/patient/question only in prompt',
            'calibration_provenance': [p.get('provenance',{}) for p in pairs], 'trained_main_model': False, 'test_data_used': False,
            'activation_storage': 'only scalar norms; no full-model activation cache',
            'limitation': 'tiny cached-rule applicability probes with automatic evidence counterfactuals; positive score is provisional, not general clinical efficacy',
            'selection_status': 'positive_effect_candidates' if selected else 'no_positive_effect_candidates'}}
