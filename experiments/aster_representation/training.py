from collections import defaultdict

import torch

from .inputs import nodes, deployment, swap_actions
from .intervention import objective

def train_case(engine, case, module, layer, backward=True, accumulation=8):
    sample_nodes = nodes(case)
    inputs = [deployment(case), deployment(swap_actions(case))]
    contexts = [engine.context(c) for c in inputs]
    totals = defaultdict(float)
    for node in sample_nodes:
        state = engine.blind(inputs[0], node['gid'])['states'][layer]
        logits, baseline, magnitudes = [], [], []
        for context in contexts:
            edited, clean, diag = engine.training_decision(context, node['history'], node['gid'], layer,
                                                         module, state, training=backward)
            logits.append(edited); baseline.append(clean); magnitudes.append(diag['magnitude'])
        target = torch.tensor(0 if node['target'].endswith('Y') else 1, device=engine.device)
        loss, terms = objective(logits, baseline, target, magnitudes, use_js=False)
        if backward:
            (loss / len(sample_nodes) / accumulation).backward()
        totals['loss'] += float(loss.detach()) / len(sample_nodes)
        for key, value in terms.items():
            totals[key] += value / len(sample_nodes)
        del logits, baseline, magnitudes, loss, edited, clean, diag
    del contexts
    return dict(totals)
