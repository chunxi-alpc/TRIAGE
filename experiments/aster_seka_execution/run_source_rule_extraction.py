"""Server-only, seed0 frozen development extraction; no clinical answer generation."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    cli = argparse.ArgumentParser(__doc__)
    for key in ['dataset-root', 'input-manifest', 'seka-root', 'model', 'pos', 'neg', 'output-dir']:
        cli.add_argument('--' + key, type=Path, required=True)
    cli.add_argument('--max-new-tokens', type=int, default=4096)
    cli.add_argument('--max-context-tokens', type=int, default=65536)
    a = cli.parse_args()
    sys.path[:0] = [str(Path(__file__).resolve().parent), str(a.seka_root)]
    import torch
    from aster_seka_adapter import Control, PhasePolicy, PreparedPrompt, StageSEKAAdapter
    from clinical_core import chat_prompt, empty_evidence, sha
    from mega_trainval_reader import iter_method_tasks
    from source_anchored_rules import parse_source_anchored_rules, VERSION
    from source_rule_prompt import INSTRUCTION
    from src.model import SEKALLM
    freeze = json.loads(a.input_manifest.read_text())
    if freeze['split'] != 'train_val' or freeze['config']['method_development_seed'] != 0:
        raise ValueError('Use the existing frozen seed0 development cases only')
    source = subprocess.check_output(['git', '-C', str(a.seka_root), 'rev-parse', 'HEAD'], text=True).strip()
    if source != 'aab9cfed80dc012fb9e83f8bdefb8675c33f40e7' or torch.cuda.device_count() != 1:
        raise ValueError('Pinned author source and exactly one assigned GPU required')
    tasks = list(iter_method_tasks(a.dataset_root, freeze))
    a.output_dir.mkdir(parents=True, exist_ok=False)
    protocol = {'status':'running','scope':'Frozen development extraction only, not clinical scores',
                'seed':0,'expected_cases':len(tasks),'max_new_tokens':a.max_new_tokens,
                'max_context_tokens':a.max_context_tokens,'prompt_sha256':sha(INSTRUCTION),
                'parser_version':VERSION,'test_or_reference_labels_read':False,
                'active_original_runs_modified':False,'semantic_correctness':'not established'}
    write = lambda name, value: (a.output_dir/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    write('protocol.json',protocol)
    torch.manual_seed(0)
    operator = SEKALLM(str(a.model),device='cuda',pos_pt=str(a.pos),neg_pt=str(a.neg),layers='last10',
                       amplify_pos=2.4,amplify_neg=0.,torch_dtype='auto',attn_implementation='sdpa')
    adapter = StageSEKAAdapter(operator,kind='seka',source_commit=source,projection_paths=[a.pos,a.neg])
    records=[]; calls=[0]; began=time.perf_counter(); seen={}
    handle=operator.model.register_forward_hook(lambda *unused: calls.__setitem__(0,calls[0]+1))
    try:
        for identity,task in tasks:
            key=sha(task.guideline)
            if key in seen:
                records.append({'sample_id':identity.sample_id,'guideline_sha256':key,'reused':True,
                                'rule_count':seen[key],'new_forward_calls':0})
                continue
            started=time.perf_counter();before=calls[0]; result={}; rules=(); audit=[]; error=None
            try:
                prepared=chat_prompt(PreparedPrompt(INSTRUCTION+task.guideline,{},empty_evidence(),'execution','rules'),operator.tok)
                tokens,_,_=adapter._encode(prepared,())
                if tokens.numel()+a.max_new_tokens>a.max_context_tokens:
                    raise ValueError('Input and output exceed unchanged context budget; no truncation')
                result=adapter.generate(prepared,PhasePolicy(False,False),max_new_tokens=a.max_new_tokens,
                                        control=Control.NO_HOOK,generation_seed=0,capture_first_token_scores=False)
                result.pop('first_token_scores',None)
                rules,audit=parse_source_anchored_rules(result['text'],task.guideline)
            except Exception as exc:
                error=type(exc).__name__+': '+str(exc)
            finally:
                operator.remove_projection()
            seen[key]=len(rules)
            row={'sample_id':identity.sample_id,'guideline_sha256':key,'reused':False,'rule_count':len(rules),
                 'new_forward_calls':calls[0]-before,'wall_seconds':time.perf_counter()-started,'error':error,
                 'audit_status_counts':dict(Counter(x['status'] for x in audit))}
            records.append(row)
            write(key+'.private.json',{'record':row,'generation':result,'rules':[asdict(x) for x in rules],'item_audit':audit})
            with (a.output_dir/'case_records.private.jsonl').open('a') as stream:
                stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    finally:
        handle.remove()
    protocol.update(status='finished_pending_rule_review',completed_cases=len(records),
                    cases_with_rules=sum(x['rule_count']>0 for x in records),
                    generation_errors=sum(bool(x.get('error')) for x in records),
                    forward_calls=calls[0],elapsed_seconds=time.perf_counter()-began,
                    peak_allocated_gib=torch.cuda.max_memory_allocated(operator.device)/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved(operator.device)/2**30)
    write('protocol.json',protocol)
    return int(bool(protocol['generation_errors']))


if __name__=='__main__':
    raise SystemExit(main())
