"""Read-only comparison of anonymous CounterFact exports and input traces.

No new inference, no prompt text, and no filtering of negative results. Run on
the research server. Metrics are native CounterFact probability comparisons.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
import traceback


def require(condition,message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def mapping(items):
    result={}
    for item in items:
        name,sep,value=item.partition('=')
        require(sep and name and value,'Expected NAME=PATH or NAME=LABEL')
        require(name not in result,'Duplicate method mapping: '+name)
        result[name]=value
    return result


def jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_export(folder,benchmark,expected_rows):
    folder=Path(folder)
    audit=json.loads((folder/'audit.json').read_text())
    require(audit.get('status')=='complete_verified','Anonymous export audit not verified: '+str(folder))
    require(audit.get('benchmark')==benchmark,'Benchmark label differs')
    pred_path=folder/'predictions.jsonl'
    cases_path=folder/'case_metrics.jsonl'
    require(sha(pred_path)==audit['anonymous_predictions_sha256'],'Anonymous prediction hash changed')
    require(sha(cases_path)==audit['case_metrics_sha256'],'Anonymous case metric hash changed')
    predictions=jsonl(pred_path)
    cases=jsonl(cases_path)
    by_id={str(row['id']):row for row in cases}
    require(len(cases)==len(by_id)==expected_rows,'Missing or duplicate source cases')
    require({str(row['id']) for row in predictions}==set(by_id),'Prediction/case ID coverage differs')
    prompt_keys=[(str(row['id']),row['prompt_sha256']) for row in predictions]
    require(len(set(prompt_keys))==len(prompt_keys),'Duplicate case/prompt SHA keys')
    if benchmark=='efficacy':
        require(len(predictions)==expected_rows,'Efficacy must have one prompt per case')
    score=math.fsum(float(row['score']) for row in cases)/len(cases)
    magnitude=math.fsum(float(row['magnitude']) for row in cases)/len(cases)
    require(math.isclose(score,audit['recomputed_metrics']['score']['mean'],abs_tol=1e-12),'Native score mismatch')
    require(math.isclose(magnitude,audit['recomputed_metrics']['magnitude']['mean'],abs_tol=1e-12),'Native magnitude mismatch')
    return {'cases':by_id,'prompts':dict(zip(prompt_keys,predictions)),
            'summary':dict(source_cases=len(cases),scored_prompts=len(predictions),score_mean=score,
                           magnitude_mean=magnitude,raw_file_sha256=audit['raw_file_sha256'],
                           anonymous_predictions_sha256=audit['anonymous_predictions_sha256'],
                           export_audit_sha256=sha(folder/'audit.json'))}


def compare_cases(candidate,reference,benchmark):
    require(set(candidate['cases'])==set(reference['cases']),'Methods have different source-case ID coverage')
    changes=[]
    for sid in sorted(candidate['cases']):
        current=candidate['cases'][sid]
        prior=reference['cases'][sid]
        delta=float(current['score'])-float(prior['score'])
        changes.append(dict(id=current['id'],candidate_score=current['score'],reference_score=prior['score'],
                            score_delta=delta,candidate_magnitude=current['magnitude'],
                            reference_magnitude=prior['magnitude'],
                            change=('improved' if delta>0 else 'degraded' if delta<0 else 'unchanged'),
                            corrected=(delta==1) if benchmark=='efficacy' else None,
                            regressed=(delta==-1) if benchmark=='efficacy' else None,
                            candidate_prompt_count=current['prompt_count'],reference_prompt_count=prior['prompt_count']))
    result=dict(source_case_coverage_identical=True,source_cases=len(changes),
                score_effect_percentage_points=100*math.fsum(r['score_delta'] for r in changes)/len(changes),
                magnitude_effect=math.fsum(r['candidate_magnitude']-r['reference_magnitude'] for r in changes)/len(changes),
                cases_improved=sum(r['score_delta']>0 for r in changes),
                cases_degraded=sum(r['score_delta']<0 for r in changes),
                cases_unchanged=sum(r['score_delta']==0 for r in changes))
    if benchmark=='efficacy':
        require(all(r['candidate_score'] in (0,1) and r['reference_score'] in (0,1) for r in changes),'Efficacy is not binary per case')
        result.update(corrected=sum(r['score_delta']==1 for r in changes),
                      regressed=sum(r['score_delta']==-1 for r in changes),
                      correction_unit='source case, one efficacy prompt')
    else:
        result['correction_unit']='Paraphrase native case score is fractional; cases improved/degraded are not binary accuracy changes'
    ckeys=set(candidate['prompts'])
    rkeys=set(reference['prompts'])
    shared=ckeys&rkeys
    result['prompt_alignment']=dict(candidate_count=len(ckeys),reference_count=len(rkeys),
        shared_exact_case_prompt_hash_count=len(shared),candidate_only_count=len(ckeys-rkeys),reference_only_count=len(rkeys-ckeys),
        status='complete_verified' if ckeys==rkeys else 'blocked',
        note='Raw prompt hashes may differ because official SEKA markers are stripped before model input; use forward trace for actual token-input equality')
    prompt_changes=None
    if ckeys==rkeys:
        prompt_changes=[]
        for key in sorted(shared):
            current=candidate['prompts'][key]
            prior=reference['prompts'][key]
            prompt_changes.append(dict(id=current['id'],prompt_sha256=key[1],
                candidate_wins=bool(current['target_wins']),reference_wins=bool(prior['target_wins']),
                corrected=bool(current['target_wins']) and not bool(prior['target_wins']),
                regressed=not bool(current['target_wins']) and bool(prior['target_wins'])))
        result['prompt_alignment'].update(
            corrected=sum(bool(candidate['prompts'][k]['target_wins']) and not bool(reference['prompts'][k]['target_wins']) for k in shared),
            regressed=sum(not bool(candidate['prompts'][k]['target_wins']) and bool(reference['prompts'][k]['target_wins']) for k in shared))
    return result,changes,prompt_changes


def read_trace(manifest_path,override):
    if not manifest_path:
        return None,{'status':'blocked','reason':'No run manifest path was supplied; trace path is unknown'}
    manifest_path=Path(manifest_path)
    if not manifest_path.is_file():
        return None,{'status':'blocked','reason':'Run manifest is inaccessible','path':str(manifest_path)}
    manifest=json.loads(manifest_path.read_text())
    require(manifest.get('process_state')=='exited' and manifest.get('exit_code')==0,'Compared run is not successfully exited')
    raw_path=override or manifest.get('forward_input_trace')
    info={'manifest_sha256':sha(manifest_path),'python_hash_seed':manifest.get('python_hash_seed'),
          'model_files_sha256':manifest.get('model_files_sha256'),
          'forward_completed':manifest.get('forward_completed'),
          'hashing_seconds':manifest.get('forward_input_hashing_seconds')}
    if not raw_path:
        return None,{**info,'status':'blocked','reason':'Run did not record a forward trace; input equality cannot be reconstructed'}
    trace_path=Path(raw_path)
    if not trace_path.is_absolute():
        trace_path=manifest_path.parent/trace_path
    info['trace_path']=str(trace_path)
    if not trace_path.is_file():
        return None,{**info,'status':'blocked','reason':'Recorded forward trace is inaccessible'}
    info['trace_sha256']=sha(trace_path)
    expected=manifest.get('forward_input_trace_sha256')
    if expected:
        require(info['trace_sha256']==expected,'Forward trace hash differs from run manifest')
    rows=jsonl(trace_path)
    require(len(rows)==manifest.get('forward_input_trace_rows')==manifest.get('forward_completed'),
            'Forward trace rows do not cover all completed forward calls')
    require([r['completed_call_index'] for r in rows]==list(range(1,len(rows)+1)),
            'Forward trace is not a complete ordered call sequence')
    return rows,{**info,'status':'complete_verified','rows':len(rows),
                 'trace_hash_was_recorded_at_run':expected is not None}


def tensor_key(row):
    result=[]
    for name in ('input_ids','attention_mask'):
        value=row.get(name)
        if not isinstance(value,dict) or not value.get('sha256') or 'dtype' not in value or 'shape' not in value:
            return None
        result.append((name,value['sha256'],value['dtype'],tuple(value['shape'])))
    return tuple(result)


def segment_match(first,second,label,start,end):
    left=first[start:end]
    right=second[start:end]
    lkeys=[tensor_key(r) for r in left]
    rkeys=[tensor_key(r) for r in right]
    missing=sum(k is None for k in lkeys)+sum(k is None for k in rkeys)
    mismatch=[start+i+1 for i,(a,b) in enumerate(zip(lkeys,rkeys)) if a is not None and b is not None and a!=b]
    equal_length=len(left)==len(right)
    exact=equal_length and not missing and not mismatch
    return dict(segment=label,first_call_1based=start+1,last_call_1based=(end if end is not None else max(len(first),len(second))),
                left_calls=len(left),right_calls=len(right),missing_tensor_records=missing,
                status=('blocked' if missing else 'complete_verified' if exact else 'failed'),
                ordered_input_ids_and_attention_masks_identical=exact,
                multiset_identical=(Counter(lkeys)==Counter(rkeys)) if not missing else None,
                mismatch_call_indices=mismatch,
                interpretation='Only ordered complete hash equality verifies identical forward inputs; a matching multiset alone is not a matched numerical path')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--efficacy',action='append',default=[],help='NAME=anonymous export directory')
    parser.add_argument('--paraphrase',action='append',default=[],help='NAME=anonymous export directory')
    parser.add_argument('--manifest',action='append',default=[],help='NAME=run manifest path')
    parser.add_argument('--trace',action='append',default=[],help='NAME=explicit trace path override')
    parser.add_argument('--label',action='append',default=[],help='NAME=source/checkpoint adaptation label')
    parser.add_argument('--primary',default='seka')
    parser.add_argument('--comparator',default='matchedzero')
    parser.add_argument('--expected-cases',type=int,default=5000)
    parser.add_argument('--efficacy-calls',type=int,default=313,
                        help='Actual first efficacy segment length; 5000 cases / batch 16 = 313; set 0 for paraphrase-only reruns')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    report={'status':'running','status_scope':'Read-only comparison computation and evidence report; each input-matching condition has its own status',
            'new_gpu_forwards':0,'clinical_sota_claim':False,'all_negative_results_retained':True,
            'source_labels':mapping(args.label),'benchmarks':{},
            'effect_uncertainty':'Descriptive paired differences; no new confidence or significance claim'}
    code=0
    try:
        for benchmark,locations in [('efficacy',mapping(args.efficacy)),('paraphrase',mapping(args.paraphrase))]:
            if not locations:
                continue
            require(args.comparator in locations,'Comparator export missing for '+benchmark)
            loaded={name:load_export(path,benchmark,args.expected_cases) for name,path in locations.items()}
            summaries={name:{**data['summary'],'source_label':report['source_labels'].get(name,'unspecified: do not infer exact paper reproduction')} for name,data in loaded.items()}
            comparisons={}
            for name,data in loaded.items():
                if name==args.comparator:
                    continue
                effect,rows,prompt_rows=compare_cases(data,loaded[args.comparator],benchmark)
                comparisons[name+'_vs_'+args.comparator]=effect
                output=args.output_dir/(benchmark+'_'+name+'_vs_'+args.comparator+'.jsonl')
                with output.open('x') as handle:
                    for row in rows:
                        handle.write(json.dumps(row,sort_keys=True)+'\n')
                effect['per_case_comparison_sha256']=sha(output)
                if prompt_rows is not None:
                    prompt_output=args.output_dir/(benchmark+'_'+name+'_vs_'+args.comparator+'_prompts.jsonl')
                    with prompt_output.open('x') as handle:
                        for row in prompt_rows:
                            handle.write(json.dumps(row,sort_keys=True)+'\n')
                    effect['per_prompt_comparison_sha256']=sha(prompt_output)
            report['benchmarks'][benchmark]={'methods':summaries,'paired_comparisons':comparisons}
        manifests=mapping(args.manifest)
        traces=mapping(args.trace)
        left,linfo=read_trace(manifests.get(args.primary),traces.get(args.primary))
        right,rinfo=read_trace(manifests.get(args.comparator),traces.get(args.comparator))
        evidence={'primary':args.primary,'comparator':args.comparator,'primary_trace':linfo,'comparator_trace':rinfo,
                  'segment_boundary':args.efficacy_calls,
                  'source_note':'Official efficacy wrapper forces max_new_tokens=1; paraphrase calls that same wrapper. Call count is batches, not cases.'}
        if left is None or right is None:
            evidence.update(status='blocked',reason='Trace unknown or inaccessible; no input-match claim')
        else:
            require(args.efficacy_calls>=0 and min(len(left),len(right))>=args.efficacy_calls,'Invalid efficacy segment boundary')
            segments={}
            if args.efficacy_calls:
                segments['efficacy']=segment_match(left,right,'efficacy',0,args.efficacy_calls)
            if len(left)>args.efficacy_calls or len(right)>args.efficacy_calls:
                segments['paraphrase']=segment_match(left,right,'paraphrase',args.efficacy_calls,None)
            evidence['segments']=segments
            evidence['checkpoint_fingerprints_identical']=linfo.get('model_files_sha256')==rinfo.get('model_files_sha256') and bool(linfo.get('model_files_sha256'))
            evidence['status']='complete_verified' if all(s['status']=='complete_verified' for s in segments.values()) else 'blocked' if any(s['status']=='blocked' for s in segments.values()) else 'failed'
            evidence['interpretation']='A paraphrase mismatch does not invalidate a separately verified efficacy segment. Preserve both segment outcomes.'
        report['matched_input_evidence']=evidence
        report['status']='complete_verified'
    except BaseException as exc:
        code=1
        report['status']='failed'
        report['error']=type(exc).__name__+': '+str(exc)
        traceback.print_exc()
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        report['checker_sha256']=sha(Path(__file__).resolve())
        with (args.output_dir/'comparison.json').open('x') as handle:
            json.dump(report,handle,indent=2)
            handle.write('\n')
    print(json.dumps({'status':report['status'],'new_gpu_forwards':0,
                      'matched_input_status':report.get('matched_input_evidence',{}).get('status','not_checked')}))
    return code


if __name__=='__main__':
    raise SystemExit(main())
