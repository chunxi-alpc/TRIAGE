"""Single-token answer continuations, checked at the actual prompt boundary."""
import math
import torch
from .data import digest

SCORING_CONTRACT = 'answer_space_label_v3_min_token_id_tie'

def candidate_contract(tokenizer, prompt, options):
    if not prompt.endswith('Answer:'):
        raise ValueError('candidate prompt must end with Answer:')
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    result = {}
    for label in options:
        if len(label) != 1 or label not in 'ABCDEFGH':
            raise ValueError('unsupported option label: ' + label)
        continuation = ' ' + label
        joined = tokenizer.encode(prompt + continuation, add_special_tokens=False)
        if joined[:len(prefix)] != prefix or len(joined) != len(prefix) + 1:
            raise ValueError('candidate is not one token after the unchanged prompt: ' + label)
        token_id = joined[-1]
        if tokenizer.decode([token_id]) != continuation:
            raise ValueError('candidate token does not decode to the exact continuation')
        result[label] = {'continuation': continuation, 'token_id': token_id}
    if not result or len({v['token_id'] for v in result.values()}) != len(result):
        raise ValueError('empty or duplicate candidate tokens')
    return result

def score_key(modelhash, prompt, options, spec, candidates):
    return digest([SCORING_CONTRACT, modelhash, prompt, options, spec, candidates])

def score_candidates(logits, candidates, tokenizer):
    scores = {k: float(logits[c['token_id']]) for k, c in candidates.items()}
    if not all(math.isfinite(x) for x in scores.values()):
        raise ValueError('nonfinite candidate logits')
    # Same tie convention as torch.argmax over the vocabulary. This also permits
    # exact winner recovery from a saved global argmax when it is a candidate.
    pred = min(scores, key=lambda k: (-scores[k], candidates[k]['token_id']))
    order = sorted(scores.values(), reverse=True)
    labels = list(scores)
    probs = torch.softmax(torch.tensor([scores[k] for k in labels], dtype=torch.float64), 0)
    top = int(logits.argmax())
    return {'prediction': pred, 'candidate_logits': scores,
            'candidate_probabilities': dict(zip(labels, probs.tolist())),
            'candidate_margin': order[0] - order[1] if len(order) > 1 else None,
            'candidate_ties': [k for k in scores if scores[k] == scores[pred]],
            'candidate_tokens': candidates, 'scoring_contract': SCORING_CONTRACT,
            'top_unconstrained_token_id': top,
            'top_unconstrained_token': tokenizer.decode([top]),
            'unconstrained_is_candidate': top in {c['token_id'] for c in candidates.values()},
            'score_availability': 'complete_candidate_logits'}

def recover_legacy_top(payload, candidates, unique_decodings):
    """Recover only a provable winner; never reinterpret the old bare logits.

    unique_decodings maps decoded text to its sole vocabulary ID. Ambiguous or
    noncandidate global winners cannot recover restricted-candidate predictions.
    """
    token = payload['top_unconstrained_token']
    token_id = unique_decodings.get(token)
    label = next((k for k, c in candidates.items()
                  if token_id is not None and c['token_id'] == token_id
                  and c['continuation'] == token), None)
    if label is None:
        return None
    return {**{k: v for k, v in payload.items() if k not in ('prediction', 'candidate_logits', 'output_sha256')},
            'prediction': label, 'candidate_logits': None,
            'legacy_bare_candidate_logits': payload['candidate_logits'],
            'candidate_tokens': candidates, 'scoring_contract': SCORING_CONTRACT,
            'top_unconstrained_token_id': token_id, 'unconstrained_is_candidate': True,
            'candidate_probabilities': None, 'candidate_margin': None, 'candidate_ties': None,
            'score_availability': 'exact_winner_only_from_legacy_global_argmax',
            'legacy_output_sha256': payload['output_sha256'],
            'output_sha256': digest({'prediction': label, 'candidate_tokens': candidates}),
            'forward_count': 0, 'cache_hit': True}
