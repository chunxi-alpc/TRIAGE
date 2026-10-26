"""Build a read-only review packet from completed runs; CPU only."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from clinical_v3_identity import index_predictions
from mega_development_evaluation import load_official, parse_with_official, read_jsonl, digest
from summarize_all_mega import catalog

P = Path(__file__).resolve().parent
R = P / 'runtime'
OUT = P / 'review_20261027'

def main():
    OUT.mkdir(exist_ok=False)
    plan = json.loads((P/'clinical_v3_plan.json').read_text())
    frozen = json.loads((P/'mega_dev16_input_lock.json').read_text())
    args = SimpleNamespace(freeze_manifest=P/'mega_dev16_input_lock.json',
        primary_run=R/'mega_dev16_seed0_gpu01', additional_a0_run=R/'mega_dev16_additional_gpu01_A0',
        additional_a1_run=R/'mega_dev16_additional_gpu016_A1', attention_run=R/'mega_dev16_attention_gpu01')
    methods = catalog(args)
    official, _ = load_official(Path(plan['official_root'])/'mega_cdp/scripts/evaluate/parse_prediction_output_singleturn.py')
    summaries, records = [], []
    for item in methods:
        predictions = index_predictions(read_jsonl(Path(item['prediction'])), frozen)
        assert len(predictions) == 16
        failures = 0
        for sid, row in predictions.items():
            parsed, _ = parse_with_official(official['parser'], row)
            failures += not parsed['parsed_success']
            measured = row.get('generation') or {}
            records.append({'sample_id':sid, 'method':item['name'],
                'source_group':item['source_group'], 'generation_status':row['status'],
                'parsed_success':parsed['parsed_success'],
                'prediction_sha256':hashlib.sha256(row.get('llm_output','').encode()).hexdigest(),
                'actual_forward_calls':measured.get('actual_forward_calls'),
                'output_tokens':measured.get('output_tokens'),
                'evidence_note':'Includes declared fallbacks and unavailable controls; not all records imply intervention'})
        summaries.append({**item,'completed_rows':16,'parse_failures':failures,
                          'evaluation_note':'Primary 11 have historical scores; remaining methods await consistent keyed judging'})
    assert len(records)==688
    (OUT/'all43_generation_inventory.json').write_text(json.dumps({'status':'complete_verified',
        'scope':'Generation identity/completeness and parsing only, not clinical effectiveness',
        'methods':summaries,'rows':688,'new_gpu_forwards':0,'raw_text_published':False},indent=2)+'\n')
    (OUT/'all43_generation_metadata.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
    v3=P/'clinical_v3_results_verified'
    report=json.loads((v3/'evaluation_report.json').read_text())
    audit=json.loads((v3/'audit.json').read_text())
    fields=['method','cases','pcc_lower_better','answer_score_higher_better','parse_failures',
            'generation_forward_calls','generation_elapsed_seconds','generation_peak_reserved_gib']
    with (OUT/'v3_method_summary.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator="\n");writer.writeheader()
        for method, row in report['methods'].items():
            cost=audit['generation_summary'][method]
            writer.writerow(dict(method=method,cases=row['n_cases'],
                pcc_lower_better=row['PCC_DTW_cost_lower_better'],
                answer_score_higher_better=row['Acc_normalized_1_to_5_higher_better'],
                parse_failures=row['parse_failures'],generation_forward_calls=cost['forward_calls'],
                generation_elapsed_seconds=cost['elapsed_seconds'],generation_peak_reserved_gib=cost['peak_reserved_gib']))
    comparisons=report['comparisons']+json.loads((v3/'direct_strong_baseline_comparisons.json').read_text())['comparisons']
    fields=['method','reference','metric','difference','ci95_low','ci95_high','improved','degraded','unchanged','n_clusters']
    with (OUT/'v3_paired_comparisons.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator="\n");writer.writeheader()
        for row in comparisons:
            writer.writerow(dict(method=row['method'],reference=row['reference'],metric=row['metric'],
                difference=row['difference_method_minus_reference'],ci95_low=row['ci95'][0],ci95_high=row['ci95'][1],
                improved=row['score_improved'],degraded=row['score_degraded'],unchanged=row['score_unchanged'],
                n_clusters=row['n_tree_clusters']))
    costs=[]
    for name in ['mega_dev16_seed0_gpu01_job','mega_dev16_additional_gpu01_A0_job',
                 'mega_dev16_additional_gpu016_A1_job','mega_dev16_attention_gpu01_job','source_rule_extraction_v2_job']:
        source=R/name; m=json.loads((source/'run_manifest.json').read_text())
        assert m['process_state']=='exited' and m['exit_code']==0
        target=OUT/'run_manifests'/name;target.mkdir(parents=True)
        shutil.copyfile(source/'run_manifest.json',target/'run_manifest.json')
        trace=source/'forward_inputs.jsonl'
        costs.append({'name':name,'manifest_sha256':digest(source/'run_manifest.json'),
            'forward_trace_sha256':digest(trace) if trace.exists() else None,
            'forward_trace_bytes':trace.stat().st_size if trace.exists() else None,
            **{k:m.get(k) for k in ['forward_completed','elapsed_seconds','peak_allocated_gib','peak_reserved_gib','gpu_physical']}})
    failed=[]
    paths=[R/'mega_dev16_remaining_evaluation_gpu016']+list((R/'base_adaseka_standard_gpu016').glob('*'))
    for source in paths:
        f=source/'run_manifest.json'
        if not f.is_file():continue
        m=json.loads(f.read_text())
        if m.get('exit_code')==0:continue
        target=OUT/'failed_start_manifests'/source.name;target.mkdir(parents=True)
        shutil.copyfile(f,target/'run_manifest.json')
        failed.append({'name':source.name,'manifest_sha256':digest(f),'exit_code':m.get('exit_code'),
            'forward_completed':m.get('forward_completed'),'reason':'System interpreter selected after symlink resolution; torch unavailable',
            'rerun_started':False})
    (OUT/'cost_and_failure_inventory.json').write_text(json.dumps({'completed':costs,'failed_starts':failed,
        'new_gpu_forwards_in_this_review':0,'new_model_runs_started':False},indent=2)+'\n')
    print(json.dumps({'status':'complete_verified','generation_records':688,'v3_scored_records':112,'failed_start_records':len(failed)}))

if __name__=='__main__':main()
