"""Greedy decoding constrained by the complete guideline tree, not a gate route."""
import re
import torch


def terminal_paths(guideline):
    groups = {g['group_id']: g for g in guideline['groups']}
    def walk(gid, path, ancestors):
        if gid in ancestors:
            raise ValueError('Guideline contains a cycle')
        for rule in groups[gid]['rules']:
            current = path + [rule['rule_id']]
            leaf = re.match(r'(L\d+)[：:]', rule['action'])
            if leaf:
                yield current, leaf[1]
            else:
                child = re.search(r'规则组 (G\d+)', rule['action'])
                if not child:
                    raise ValueError('Cannot read rule transition: ' + rule['rule_id'])
                yield from walk(child[1], current, ancestors | {gid})
    return list(walk('G0', [], set()))


def path_trie(tokenizer, guideline):
    root = {}
    for path, leaf in terminal_paths(guideline):
        node = root
        for token in tokenizer.encode('PATH: ' + ', '.join(path), add_special_tokens=False):
            node = node.setdefault(token, {})
        node['terminal'] = (path, leaf)
    return root


def decode(engine, guideline, result, start):
    """Force only graph/syntax tokens, choose remaining tokens with model logits.

    FINAL is looked up from the selected terminal rule. All arms get the same
    full-tree grammar, including text clipping. No gold labels or M2 path enter.
    """
    tokenizer = engine.builder.tokenizer
    node = path_trie(tokenizer, guideline)
    tokens = []
    choices = []
    budget = engine.freeze['max_new_tokens']
    forced = 0
    while 'terminal' not in node and len(tokens) < budget:
        batch = []
        allowed = sorted(k for k in node if isinstance(k, int))
        if len(allowed) > 1:
            logits = result.logits[0, -1].float()
            idx = torch.tensor(allowed, device=logits.device)
            token = allowed[int(logits[idx].argmax())]
            choices.append({'position': len(tokens), 'allowed': allowed, 'chosen': token})
            batch.append(token)
            node = node[token]
        while 'terminal' not in node and len(tokens) + len(batch) < budget:
            allowed = [k for k in node if isinstance(k, int)]
            if len(allowed) != 1:
                break
            token = allowed[0]
            batch.append(token)
            forced += 1
            node = node[token]
        if not batch:
            raise RuntimeError('Empty structured decoder step')
        previous_length = len(tokens)
        tokens.extend(batch)
        if 'terminal' not in node and len(tokens) < budget:
            result = engine.backend._forward(torch.tensor([batch]),
                cache=result.past_key_values, start=start + previous_length, use_cache=True)
    if 'terminal' in node:
        path, leaf = node['terminal']
        text = 'PATH: ' + ', '.join(path) + '\nFINAL: ' + leaf
    else:
        path, leaf = [], None
        text = tokenizer.decode(tokens, skip_special_tokens=True)
    return {'text': text, 'token_ids': tokenizer.encode(text, add_special_tokens=False),
            'path_token_ids': tokens, 'selected_path': path, 'selected_leaf': leaf,
            'output_limit': leaf is None, 'decoder': 'full_tree_constrained_greedy',
            'forced_path_tokens': forced, 'branch_choices': choices,
            'final_from_rule_lookup': leaf is not None}
