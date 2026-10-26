"""Publish reproducibility metadata, hashes and costs, without source data text."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    root = Path(__file__).resolve().parent
    public = root / 'run_records'
    public.mkdir(exist_ok=True)
    records = []
    for path in sorted((root / 'runtime').glob('*/run_manifest.json')):
        m = json.loads(path.read_text())
        out = public / path.parent.name
        out.mkdir(exist_ok=True)
        shutil.copyfile(path, out / 'run_manifest.json')
        trace = m.get('forward_input_trace')
        if trace and m.get('process_state') == 'exited':
            trace = Path(trace)
            if not trace.is_absolute():
                trace = path.parent / trace
            if trace.is_file():
                # These traces contain hashes, shapes and dtypes, never token IDs/text.
                shutil.copyfile(trace, out / 'forward_input_trace.jsonl')
        row = {'run': path.parent.name}
        for key in ['process_state', 'exit_code', 'verification_status', 'gpu_physical',
                    'forward_attempts', 'forward_completed', 'elapsed_seconds',
                    'peak_allocated_gib', 'peak_reserved_gib', 'python_hash_seed']:
            row[key] = m.get(key)
        row['manifest_sha256'] = sha(path)
        row['error'] = m.get('error')
        records.append(row)
    summary = {'snapshot_utc': datetime.now(timezone.utc).isoformat(),
               'running_counts_are_not_final': True,
               'successful_exit_is_not_independent_verification': True,
               'all_runs_including_failures': records,
               'completed_forwards_in_exited_runs': sum(r['forward_completed'] or 0 for r in records if r['process_state'] == 'exited'),
               'source_bundle_sha256': {str(f.relative_to(root)): sha(f) for f in sorted(root.glob('*.py'))}}
    (public / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    if records:
        with (public / 'costs.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows({key: value.rstrip() if isinstance(value, str) else value for key, value in row.items()} for row in records)
    print(json.dumps({'runs': len(records), 'completed_forwards_in_exited_runs': summary['completed_forwards_in_exited_runs']}))


if __name__ == '__main__':
    main()
