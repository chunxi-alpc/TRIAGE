import sys
from pathlib import Path
from care_m2_inc.io import read_json,read_jsonl,write_json,sha256_file
from care_m2_v2.data import digest
from care_m2_v2.n3_mean_basis import code_hashes,snapshot
root=Path(sys.argv[1]);report=Path(__file__).parent
frozen=read_json(root/'frozen_execution.json');manifest=read_json(root/'run_manifest.json')
assert code_hashes()==frozen['source_code_sha256']
assert '31 passed' in (report/'cpu_tests_final.log').read_text()
assert read_json(report/'process.json')['exit_code']==0
assert snapshot(frozen['source_run'])==read_json(root/'protected_source_sha256.json')
for name,h in frozen['files'].items():assert sha256_file(root/name)==h
for name,h in manifest['outputs'].items():assert sha256_file(root/name)==h
records=read_jsonl(root/'raw/forward_journal.jsonl')
assert all(r['checksum']==digest(r['payload']) for r in records)
assert sum(r['payload'].get('forward_count',0) for r in records)==manifest['new_forward_count']==48
for file,expected in [('dev_grid_predictions.jsonl',48),('dev_random_predictions.jsonl',16),('dev_comparison_predictions.jsonl',96)]:
    rows=read_jsonl(root/file);assert len(rows)==expected
    assert all(r['split']=='dev' and r['case_id'] in frozen['dev_case_ids'] for r in rows)
selected=read_json(root/'dev_selection.json');assert selected['sha256']==digest({k:v for k,v in selected.items() if k!='sha256'})
assert selected['selected_route_beta']==.5
rows=read_jsonl(root/'dev_comparison_predictions.jsonl')
expected={'baseline':(9,0),'mean':(9,1),'old_route':(9,1),'route':(9,2),'random':(9,0),'text':(9,3)}
for m,counts in expected.items():
    assert tuple(sum(r['correct'] for r in rows if r['method']==m and r['DQ']==s) for s in ('D','Q'))==counts
write_json(root/'FINAL_REVIEW.json',{'status':'passed','CPU_tests':31,'new_dev_forward_count':48,'new_train_forward_count':0,'new_test_forward_count':0,
    'new_N2_forward_count':0,'new_N5_forward_count':0,'protected_source_files_unchanged':len(read_json(root/'protected_source_sha256.json')),
    'all_output_hashes_and_journal_checksums_valid':True,'frozen_code_and_selection_valid':True,'all_predictions_on_frozen_dev_cases':True,
    'gpu':0,'peak_memory_GiB':manifest['peak_memory_bytes']/2**30,'process_exit_code':0})
write_json(root/'final_artifacts_sha256.json',{str(p.relative_to(root)):sha256_file(p) for p in root.rglob('*') if p.is_file() and p.name!='final_artifacts_sha256.json'})
print('Final review passed: 31 CPU tests, 48 dev forwards, protected outputs unchanged')
