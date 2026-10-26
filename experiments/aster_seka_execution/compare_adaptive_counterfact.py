"""Read-only Ada positive/zero and FP32 no-hook efficacy comparison.

Run on the research server alongside compare_standard.py. Official Ada uses
one routing forward followed by one one-token generation forward per case.
Only saved target/comparator log probabilities can be recomputed here; full
vocabulary logits and generation token IDs were not exported by that wrapper.
"""
import argparse
import json
import math
from pathlib import Path
import time
import traceback

from compare_standard import load_export, read_trace, require, sha, tensor_key


def option(arguments,name,default=None):
    for index,value in enumerate(arguments):
        if value==name and index+1<len(arguments):
            return arguments[index+1]
        if value.startswith(name+'='):
            return value.split('=',1)[1]
    return default


def load_method(name,folder,manifest_path,trace_override,expected_cases,adaptive):
    data=load_export(folder,'efficacy',expected_cases)
    export_audit=json.loads((folder/'audit.json').read_text())
    require(export_audit['run_manifest_sha256']==sha(manifest_path),name+': export refers to a different run manifest')
    manifest=json.loads(manifest_path.read_text())
    require(manifest.get('process_state')=='exited' and manifest.get('exit_code')==0,name+': unsuccessful process')
    require(manifest.get('first_forward_parameter_dtype')=='torch.float32',name+': FP32 matching requirement not met')
    arguments=manifest['module_args']
    require(int(option(arguments,'--batch_size','16'))==1,name+': official Ada comparison requires batch_size=1')
    require('--adaptive-seka' in arguments if adaptive else '--adaptive-seka' not in arguments,
            name+': method flag differs')
    if name=='zero':
        require(float(option(arguments,'--adaptive_amplify_factor','1.0'))==0.,'Zero comparator has nonzero amplitude')
    if name=='positive':
        require(float(option(arguments,'--adaptive_amplify_factor','1.0'))!=0.,'Positive method has zero amplitude')
    if name=='nohook':
        require(not any(flag in arguments for flag in ['--seka','--pasta','--anchor']), 'No-hook run enables another intervention')
    benchmarks=option(arguments,'--benchmarks')
    require(benchmarks=='efficacy',name+': this entrypoint compares efficacy-only runs')
    calls=expected_cases*(2 if adaptive else 1)
    require(manifest['forward_attempts']==manifest['forward_completed']==calls,name+': unexpected attempted/completed forwards')
    declared_data=option(arguments,'--data_path')
    require(declared_data is not None,name+': CounterFact data path absent')
    data_root=Path(declared_data).expanduser()
    if not data_root.is_absolute():
        source_root=option(manifest['command'],'--source')
        require(source_root is not None,name+': cannot resolve relative data path without source root')
        data_root=Path(source_root)/data_root
    data_file=(data_root/'counterfact.jsonl').resolve()
    data_hash=manifest.get('input_files_sha256',{}).get(str(data_file))
    require(data_hash and sha(data_file)==data_hash,name+': actual CounterFact data fingerprint unavailable or changed')
    trace,info=read_trace(str(manifest_path),str(trace_override) if trace_override else None)
    order=list(data['cases'])
    predictions={str(row['id']):row for row in data['prompts'].values()}
    routing={}
    generation={}
    pattern=dict(status='blocked',reason='Trace unavailable')
    if trace is not None:
        require(len(trace)==calls,name+': trace length differs')
        for index,case in enumerate(order):
            if adaptive:
                routing[case]=trace[2*index]
                generation[case]=trace[2*index+1]
            else:
                generation[case]=trace[index]
        issues=[]
        missing=[]
        for case in order:
            generated=generation[case]
            if tensor_key(generated) is None:
                missing.append(dict(id=case,call='generation',reason='Missing field or unavailable tensor fingerprint'))
            if adaptive:
                routed=routing[case]
                if tensor_key(routed) is None:
                    missing.append(dict(id=case,call='routing',reason='Missing field or unavailable tensor fingerprint'))
                elif routed['attention_mask'] is not None:
                    issues.append(dict(id=case,reason='Expected official routing attention_mask=None'))
                if 'attention_mask' in generated and generated['attention_mask'] is None:
                    issues.append(dict(id=case,reason='Expected official generation attention-mask tensor'))
                if tensor_key(routed) is not None and tensor_key(generated) is not None:
                    if routed['input_ids']!=generated['input_ids']:
                        issues.append(dict(id=case,reason='Routing and generation input_ids fingerprints differ'))
        pattern=dict(status='failed' if issues else 'blocked' if missing else 'complete_verified',
                     issues=issues,missing_evidence=missing,
                     assignment='Exported case order maps to routing calls 1,3,... and generation calls 2,4,...' if adaptive else 'One generation call per exported case')
    costs={key:manifest.get(key) for key in ['forward_attempts','forward_completed','elapsed_seconds','gpu_lock_wall_seconds',
                                          'peak_allocated_gib','peak_reserved_gib','forward_input_hashing_seconds']}
    costs.update(routing_forwards=expected_cases if adaptive else 0,generation_forwards=expected_cases,
                 time_partition='Only total process wall time is recorded; route/generation durations are not inferred from counts')
    return dict(data=data,predictions=predictions,manifest=manifest,routing=routing,generation=generation,
                trace_info=info,pattern=pattern,costs=costs,data_file=str(data_file),data_sha256=data_hash)


def input_match(left,right):
    if left is None or right is None:
        return dict(status='blocked',reason='A call is not available')
    a,b=tensor_key(left),tensor_key(right)
    if a is None or b is None:
        return dict(status='blocked',reason='Missing field or unavailable tensor hash; explicit None attention masks are permitted')
    return dict(status='complete_verified' if a==b else 'failed',
                input_ids_and_attention_mask_equal=a==b,
                candidate_call_index=left['completed_call_index'],reference_call_index=right['completed_call_index'],
                candidate_input_ids_sha256=left['input_ids']['sha256'],reference_input_ids_sha256=right['input_ids']['sha256'],
                candidate_attention_mask=left['attention_mask'],reference_attention_mask=right['attention_mask'])


def pair(candidate,reference,name,output_dir,include_routing):
    require(set(candidate['predictions'])==set(reference['predictions']),name+': source-case coverage differs')
    rows=[]
    for case in candidate['data']['cases']:
        a,b=candidate['predictions'][case],reference['predictions'][case]
        target_delta=float(a['target_score'])-float(b['target_score'])
        comparator_delta=float(a['comparator_score'])-float(b['comparator_score'])
        require(math.isfinite(target_delta) and math.isfinite(comparator_delta),'Nonfinite saved score difference')
        row=dict(id=case,target_logp_difference=target_delta,comparator_logp_difference=comparator_delta,
                 available_logp_scores_exactly_equal=target_delta==0. and comparator_delta==0.,
                 candidate_target_wins=bool(a['target_wins']),reference_target_wins=bool(b['target_wins']),
                 corrected=bool(a['target_wins']) and not bool(b['target_wins']),
                 regressed=not bool(a['target_wins']) and bool(b['target_wins']),
                 probability_margin_difference=float(a['probability_margin'])-float(b['probability_margin']),
                 raw_prompt_hash_equal=a['prompt_sha256']==b['prompt_sha256'],
                 generation_input=input_match(candidate['generation'].get(case),reference['generation'].get(case)))
        if include_routing:
            row['routing_input']=input_match(candidate['routing'].get(case),reference['routing'].get(case))
        rows.append(row)
    path=output_dir/(name+'.jsonl')
    with path.open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row,sort_keys=True)+'\n')
    states=[row['generation_input']['status'] for row in rows]
    if include_routing:
        states += [row['routing_input']['status'] for row in rows]
    matching='failed' if 'failed' in states else 'blocked' if 'blocked' in states else 'complete_verified'
    return dict(source_cases=len(rows),per_case_sha256=sha(path),input_matching_status=matching,
                generated_call_pairs_matched=sum(row['generation_input']['status']=='complete_verified' for row in rows),
                routing_call_pairs_matched=sum(row.get('routing_input',{}).get('status')=='complete_verified' for row in rows) if include_routing else None,
                target_logp_max_abs_difference=max(abs(row['target_logp_difference']) for row in rows),
                comparator_logp_max_abs_difference=max(abs(row['comparator_logp_difference']) for row in rows),
                cases_with_available_scores_exactly_equal=sum(row['available_logp_scores_exactly_equal'] for row in rows),
                corrected=sum(row['corrected'] for row in rows),regressed=sum(row['regressed'] for row in rows),
                score_effect_percentage_points=100*sum(int(row['candidate_target_wins'])-int(row['reference_target_wins']) for row in rows)/len(rows),
                probability_margin_effect=math.fsum(row['probability_margin_difference'] for row in rows)/len(rows),
                full_vocabulary_logits_independently_recomputed=False,
                score_scope='Only exported target/comparator log probabilities; no full-vocabulary drift or argmax claim',
                prompt_scope='Raw markers may make prompt hashes differ; actual model inputs use the per-call token/mask fingerprints')


def main():
    parser=argparse.ArgumentParser()
    for method in ['positive','zero','nohook']:
        parser.add_argument('--'+method+'-export',type=Path,required=True)
        parser.add_argument('--'+method+'-manifest',type=Path,required=True)
        parser.add_argument('--'+method+'-trace',type=Path)
    parser.add_argument('--expected-cases',type=int,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    for name,value in vars(args).items():
        if isinstance(value,Path):
            setattr(args,name,value.expanduser().resolve())
    args.output_dir.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    report=dict(status='running',new_gpu_forwards=0,clinical_sota_claim=False,negative_results_retained=True,
                scope='FP32 CounterFact efficacy: saved score effects and observed routing/generation input equality',
                full_vocabulary_logits_independently_recomputed=False,
                attention_mask_none_semantics='An explicit None value equals None; a missing key is missing evidence; None differs from a tensor',
                nohook_call_selection='Every second Ada call (2,4,...) aligns with each no-hook call; odd Ada calls are routing cost',
                methods={},comparisons={})
    code=0
    try:
        require(args.expected_cases>0,'Expected cases must be positive')
        loaded={name:load_method(name,getattr(args,name+'_export'),getattr(args,name+'_manifest'),
                    getattr(args,name+'_trace'),args.expected_cases,name!='nohook') for name in ['positive','zero','nohook']}
        base=loaded['zero']['manifest']
        require(base.get('model_files_sha256'),'Model fingerprints missing')
        for name,value in loaded.items():
            require(value['manifest'].get('model_files_sha256')==base['model_files_sha256'],name+': checkpoint fingerprints differ')
            require(value['manifest'].get('source_commit')==base.get('source_commit'),name+': source revision differs')
            require(value['manifest'].get('source_files_sha256')==base.get('source_files_sha256')
                    and bool(base.get('source_files_sha256')),name+': recorded upstream source fingerprints differ or are absent')
            require(value['data_sha256']==loaded['zero']['data_sha256'],name+': CounterFact source data differs')
            report['methods'][name]=dict(export_summary=value['data']['summary'],dtype='torch.float32',
                trace=value['trace_info'],call_assignment=value['pattern'],costs=value['costs'],
                data_file=value['data_file'],data_sha256=value['data_sha256'])
        for candidate,reference in [('positive','zero'),('zero','nohook'),('positive','nohook')]:
            name=candidate+'_vs_'+reference
            report['comparisons'][name]=pair(loaded[candidate],loaded[reference],name,args.output_dir,reference=='zero')
        states=[value['pattern']['status'] for value in loaded.values()]
        states += [value['input_matching_status'] for value in report['comparisons'].values()]
        report['status']='failed' if 'failed' in states else 'blocked' if 'blocked' in states else 'complete_verified'
        code=0 if report['status']=='complete_verified' else 1
        report['configuration_scope']='All observed first-forward dtypes FP32 and checkpoint file maps equal. Generation config files are therefore equal when recorded; no claim of separately captured runtime generation configs for old Ada runs.'
        report['interpretation']='Nonzero saved-score effects or a losing method remain valid audit results; input mismatch/missing evidence has a separate status. Exact agreement of two saved scores does not prove full-vocabulary equivalence.'
    except BaseException as error:
        code=1
        report.update(status='failed',error=type(error).__name__+': '+str(error))
        traceback.print_exc()
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        report['checker_sha256']=sha(Path(__file__).resolve())
        report['comparison_helper_sha256']=sha(Path(__file__).with_name('compare_standard.py'))
        with (args.output_dir/'comparison.json').open('x') as handle:
            json.dump(report,handle,indent=2)
            handle.write('\n')
    print(json.dumps({key:report[key] for key in ['status','new_gpu_forwards','full_vocabulary_logits_independently_recomputed']}))
    return code


if __name__=='__main__':
    raise SystemExit(main())
