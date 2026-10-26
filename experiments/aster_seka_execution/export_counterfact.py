"""Independently recompute official CounterFact metrics and export hashes only.

Server CPU only. The raw input remains private and unchanged. Efficacy is a
first-token probability comparison, not text-answer accuracy.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import traceback


def require(condition,message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def text_hash(value):
    require(isinstance(value,str),'Expected an original string for hashing')
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def metric(values):
    return {'mean':math.fsum(values)/len(values),'std':statistics.pstdev(values),'values':None}


def compare_metrics(actual,expected,origin):
    for name in ('score','magnitude'):
        for field in ('mean','std'):
            old=float(expected[name][field])
            require(math.isfinite(old),origin+' has a non-finite metric')
            require(math.isclose(actual[name][field],old,rel_tol=1e-10,abs_tol=1e-12),
                    f'{origin} metric mismatch: {name}.{field}: recomputed={actual[name][field]}, saved={old}')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--metrics',type=Path)
    parser.add_argument('--run-manifest',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--expected-rows',type=int,required=True,help='Expected source cases, not flattened paraphrase prompts')
    parser.add_argument('--benchmark',choices=['efficacy','paraphrase'],default='efficacy')
    args=parser.parse_args()
    args.output_dir=args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    report=dict(status='running',benchmark=args.benchmark,new_gpu_forwards=0,
                verified_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                metric_definition='Per case mean of exp(target_logprob) > exp(comparator_logprob); magnitude is mean probability difference. Then average across cases, population std (ddof=0).',
                metric_tolerance={'absolute':1e-12,'relative':1e-10},
                clinical_accuracy_claim=False,
                generation_token_count_policy='null when original token IDs are absent; never infer token count by retokenizing decoded text',
                raw_text_published=False)
    code=0
    try:
        report['raw_file_sha256']=digest(args.raw)
        report['run_manifest_sha256']=digest(args.run_manifest)
        raw=json.loads(args.raw.read_text())
        run=json.loads(args.run_manifest.read_text())
        require(run.get('process_state')=='exited' and run.get('exit_code')==0,'Evaluation run did not complete successfully')
        samples=raw['samples']
        require(len(samples)==args.expected_rows,'Unexpected source-case count')
        require(len({str(row['id']) for row in samples})==len(samples),'Duplicate source-case IDs')
        require(samples,'No samples')
        exported=[]
        per_case=[]
        scores=[]
        magnitudes=[]
        for sample in samples:
            prompts=sample['prompts'] if args.benchmark=='paraphrase' else [sample]
            require(prompts,'Source case has no prompts')
            wins=[]
            margins=[]
            for index,row in enumerate(prompts):
                target=float(row['target_score'])
                comparator=float(row['comparator_score'])
                require(math.isfinite(target) and math.isfinite(comparator),'Non-finite log probability')
                require(target<=1e-6 and comparator<=1e-6,'Expected log probabilities <= 0')
                p_target=math.exp(target)
                p_comparator=math.exp(comparator)
                win=int(p_target>p_comparator)
                margin=p_target-p_comparator
                wins.append(win)
                margins.append(margin)
                token_ids=row.get('generation_token_ids')
                token_count=len(token_ids) if isinstance(token_ids,list) else None
                exported.append(dict(id=sample['id'],prompt_index=index,
                    prompt_sha256=text_hash(row['prompt']),generation_sha256=text_hash(row['generation']),
                    generation_token_count=token_count,generation_is_empty=row['generation']=='',
                    target_score=target,comparator_score=comparator,score_units='log_probability',
                    target_probability=p_target,comparator_probability=p_comparator,
                    target_wins=bool(win),probability_tie=p_target==p_comparator,probability_margin=margin))
            case_score=math.fsum(wins)/len(wins)
            case_magnitude=math.fsum(margins)/len(margins)
            if args.benchmark=='paraphrase':
                require(math.isclose(case_score,float(sample['efficacy_score']),abs_tol=1e-12,rel_tol=1e-10),'Saved per-case paraphrase score differs')
                require(math.isclose(case_magnitude,float(sample['efficacy_magnitude']),abs_tol=1e-12,rel_tol=1e-10),'Saved per-case paraphrase magnitude differs')
            per_case.append(dict(id=sample['id'],prompt_count=len(prompts),score=case_score,magnitude=case_magnitude))
            scores.append(case_score)
            magnitudes.append(case_magnitude)
        result={'score':metric(scores),'magnitude':metric(magnitudes)}
        report.update(source_cases=len(samples),scored_prompts=len(exported),recomputed_metrics=result)
        # Write the complete anonymous values before checking reported aggregate
        # metrics, so a discrepancy remains independently inspectable.
        export_path=args.output_dir/'predictions.jsonl'
        with export_path.open('x') as handle:
            for row in exported:
                handle.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
        cases_path=args.output_dir/'case_metrics.jsonl'
        with cases_path.open('x') as handle:
            for row in per_case:
                handle.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
        report['anonymous_predictions_sha256']=digest(export_path)
        report['case_metrics_sha256']=digest(cases_path)
        compare_metrics(result,raw['metrics'],'raw embedded metrics')
        if args.metrics:
            report['published_metric_file_sha256']=digest(args.metrics)
            compare_metrics(result,json.loads(args.metrics.read_text()),'separate metrics file')
        report['cost_from_run_manifest']={key:run.get(key) for key in [
            'forward_attempts','forward_completed','elapsed_seconds','gpu_lock_wall_seconds',
            'peak_allocated_gib','peak_reserved_gib','forward_input_hashing_seconds']}
        report['source_commit']=run.get('source_commit')
        report['model_files_sha256']=run.get('model_files_sha256',{})
        report['input_files_sha256']=run.get('input_files_sha256',{})
        report['run_scope_note']='The evaluated subset and source checkpoint determine scope; passing this audit does not turn a 16-case smoke into full standard reproduction.'
        report['status']='complete_verified'
    except BaseException as exc:
        code=1
        report['status']='failed'
        report['error']=type(exc).__name__+': '+str(exc)
        traceback.print_exc()
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        report['checker_sha256']=digest(Path(__file__).resolve())
        with (args.output_dir/'audit.json').open('x') as handle:
            json.dump(report,handle,ensure_ascii=False,indent=2)
            handle.write('\n')
    print(json.dumps({k:report[k] for k in ('status','benchmark','new_gpu_forwards','raw_text_published')}))
    return code


if __name__=='__main__':
    raise SystemExit(main())
