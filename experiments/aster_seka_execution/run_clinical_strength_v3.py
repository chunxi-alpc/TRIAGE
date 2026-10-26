"""Fixed development cohort, grounded A0 rules, small SEKA strength comparison."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

import torch
from aster_seka_adapter import Control, PhasePolicy, StageSEKAAdapter
from clinical_core import chat_prompt, parse_quoted_rules_individually, sha
from grounded_clinical_prompt import grounded_prompts
from mega_trainval_reader import TEMPLATES, iter_method_tasks
from native_generation_baselines import FusionConfig, generate_native
from run_mega_additional_baselines import load_selected_primary
from run_mega_development import file_sha, write_json

METHODS = ['no_hook', 'gain_0', 'gain_0_2', 'gain_0_6',
           'text_emphasis', 'fusion_static', 'fusion_dynamic']
GAINS = {'gain_0': 0., 'gain_0_2': .2, 'gain_0_6': .6}


def cli():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    return p.parse_args()


def prepared_cases(plan, tokenizer):
    primary = Path(plan['primary_run'])
    cases, _, _ = load_selected_primary(primary, 'A0')
    frozen = json.loads(Path(plan['input_manifest']).read_text())
    for identity, task in iter_method_tasks(Path(plan['dataset_root']), frozen):
        artifact = cases[identity.sample_id]
        cache = primary / 'private_guideline_extractions' / (artifact['cache_key'] + '.json')
        if file_sha(cache) != plan['extraction_sha256'][identity.sample_id]:
            raise ValueError('Frozen extraction changed')
        record = json.loads(cache.read_text())
        if record['guideline_sha256'] != sha(task.guideline):
            raise ValueError('Extraction guideline mismatch')
        rules = parse_quoted_rules_individually(record['generation']['text'], task.guideline)[0] \
            if record['validation_status'] in {'quotes_validated', 'quotes_partially_validated'} else ()
        tfu = [artifact['A0'][rule.rule_id] for rule in rules]
        template = (Path(plan['official_root']) / TEMPLATES['zh' if task.language == 'zh' else 'en']).read_text()
        base, support, mapping = grounded_prompts(template, task.guideline, task.vignette, rules, tfu)
        base, support = chat_prompt(base, tokenizer), chat_prompt(support, tokenizer)
        for prompt in [base, support]:
            length = len(tokenizer.encode(prompt.text, add_special_tokens=True))
            if length + plan['max_new_tokens'] > plan['max_context_tokens']:
                raise ValueError('Grounded prompt exceeds frozen context budget')
        yield identity, task, base, support, mapping, artifact


def main(args):
    plan = json.loads(args.plan.read_text())
    if plan['methods'] != METHODS or plan['seed'] != 0 or plan['condition_variant'] != 'A0':
        raise ValueError('Unexpected fixed development plan')
    for path, expected in plan['frozen_files'].items():
        if file_sha(path) != expected:
            raise ValueError('Frozen source or artifact changed: ' + path)
    os.umask(0o077)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if torch.cuda.device_count() != 1:
        raise ValueError('Expose exactly one assigned GPU')
    sys.path.insert(0, plan['seka_root'])
    from src.model import SEKALLM
    operator = SEKALLM(plan['model'], device='cuda', pos_pt=plan['pos'], neg_pt=plan['neg'],
                       layers='last10', amplify_pos=0., amplify_neg=0.,
                       torch_dtype='auto', attn_implementation='sdpa',
                       rope_scaling=plan['rope_scaling'])
    adapter = StageSEKAAdapter(operator, kind='seka', source_commit=plan['seka_commit'],
                               projection_paths=[plan['pos'], plan['neg']])
    cfg = FusionConfig(**plan['fusion_parameters'])
    cfg.validate()
    protocol = dict(plan, status='running', plan_sha256=file_sha(args.plan),
                    model_dtype=str(next(operator.model.parameters()).dtype),
                    projection_manifest=adapter.projections, completed_rows=0)
    write_json(args.output_dir / 'protocol.json', protocol)
    handles = {m: (args.output_dir / (m + '.predictions.private.jsonl')).open('x') for m in METHODS}
    counts = Counter()
    start = time.monotonic()
    try:
        for identity, task, base, support, mapping, artifact in prepared_cases(plan, operator.tok):
            zero_reference = None
            for method in METHODS:
                calls = [0]
                hook = operator.model.register_forward_hook(lambda *a: calls.__setitem__(0, calls[0] + 1))
                torch.cuda.reset_peak_memory_stats(operator.device)
                before = time.monotonic()
                try:
                    if method == 'no_hook' or method in GAINS:
                        operator.amplify_pos = GAINS.get(method, 0.)
                        control = Control.NO_HOOK if method == 'no_hook' else \
                            Control.MATCHED_ZERO if method == 'gain_0' else Control.NORMAL
                        result = adapter.generate(base, PhasePolicy(False, method != 'no_hook'),
                                                  max_new_tokens=plan['max_new_tokens'], control=control,
                                                  generation_seed=0, capture_first_token_scores=True)
                        scores = result.pop('first_token_scores')
                        if method == 'no_hook':
                            zero_reference = (scores, result['text'])
                        if method == 'gain_0' and zero_reference is not None:
                            result['numerical_control'] = {
                                'max_abs_first_token_score_difference': float((scores - zero_reference[0]).abs().max()),
                                'same_full_answer': result['text'] == zero_reference[1],
                                'reference': 'no_hook'}
                    else:
                        operator.remove_projection()
                        prompt = support.text if method == 'text_emphasis' else base.text
                        kind = {'text_emphasis': 'plain', 'fusion_static': 'static', 'fusion_dynamic': 'dynamic'}[method]
                        result = generate_native(operator.model, operator.tok, prompt,
                                                 support.text if kind != 'plain' else None,
                                                 kind=kind, config=cfg, max_new_tokens=plan['max_new_tokens'],
                                                 max_context_tokens=plan['max_context_tokens'])
                        result.pop('first_log_probs'); result.pop('first_raw_base')
                    result['status'] = 'ok'
                except Exception as error:
                    result = {'status': 'error', 'text': '', 'error_type': type(error).__name__,
                              'error': str(error), **getattr(error, 'partial_generation', {})}
                finally:
                    hook.remove()
                    operator.remove_projection()
                result.update(actual_forward_calls=calls[0], wall_seconds=time.monotonic() - before,
                              peak_allocated_gib=torch.cuda.max_memory_allocated(operator.device) / 2**30,
                              peak_reserved_gib=torch.cuda.max_memory_reserved(operator.device) / 2**30)
                row = {**asdict(identity), 'method': method, 'seed': 0, 'language': task.language,
                       'llm_output': result.get('text', ''), 'status': result['status'], 'generation': result,
                       'condition_variant': 'A0', 'condition_rule_count': len(mapping),
                       'selected_span_count': len(base.evidence.spans),
                       'base_prompt_sha256': sha(base.text), 'support_prompt_sha256': sha(support.text),
                       'mapping_sha256': sha(json.dumps(mapping, ensure_ascii=False, sort_keys=True)),
                       'max_new_tokens': plan['max_new_tokens'], 'model': plan['model'],
                       'cost_references': {'primary_stage_sha256': plan['primary_stage_sha256'],
                                           'extraction_cache_key': artifact['cache_key'],
                                           'condition_costs': artifact['condition_stage_costs'],
                                           'include_shared_preparation_cost': True}}
                handles[method].write(json.dumps(row, ensure_ascii=False) + '\n'); handles[method].flush()
                counts[method + ':' + result['status']] += 1
                protocol.update(completed_rows=sum(counts.values()), outcome_counts=dict(counts))
                write_json(args.output_dir / 'protocol.json', protocol)
        if protocol['completed_rows'] != 16 * len(METHODS):
            raise ValueError('Development denominator changed')
        protocol['status'] = 'finished_pending_evaluation'
    except BaseException as error:
        protocol.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        for handle in handles.values():
            handle.close()
        protocol['elapsed_seconds'] = time.monotonic() - start
        write_json(args.output_dir / 'protocol.json', protocol)


if __name__ == '__main__':
    main(cli())
