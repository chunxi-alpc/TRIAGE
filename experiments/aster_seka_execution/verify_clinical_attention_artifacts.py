"""Server CPU provenance binding for clinical PASTA heads and Ada experts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--heads', type=Path, required=True)
    parser.add_argument('--heads-run', type=Path, required=True)
    parser.add_argument('--adaptive-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Preserve prior artifact reports')
    manifest = json.loads(args.heads_run.read_text())
    require(manifest.get('process_state') == 'exited' and manifest.get('exit_code') == 0,
            'PASTA head construction did not finish successfully')
    require(manifest.get('module') == 'pastalib.profiler.synthetic_qa_pasta_profile_builder',
            'Unexpected PASTA profiler')
    require(manifest.get('source_commit') == 'aab9cfed80dc012fb9e83f8bdefb8675c33f40e7', 'Author source differs')
    require(Path(manifest['model_path']).resolve() == args.model.resolve(), 'PASTA calibration model path differs')
    require(manifest['forward_attempts'] == manifest['forward_completed'] > 0, 'Incomplete profiling forwards')
    actual_arguments = manifest['module_args']
    output_dir = Path(actual_arguments[actual_arguments.index('--output_dir') + 1]).resolve()
    require(args.heads.resolve().parent == output_dir, 'Head file is not the actual profiler output')
    fingerprints = manifest['model_files_sha256']
    require('config.json' in fingerprints and any(k.endswith('.safetensors') for k in fingerprints),
            'Full model fingerprints missing')
    for name, expected in fingerprints.items():
        require(sha(args.model / name) == expected, 'Actual model bytes differ: ' + name)
    for name, expected in manifest['source_files_sha256'].items():
        require(sha(args.source / name) == expected, 'Author source changed: ' + name)
    for name, expected in manifest['input_files_sha256'].items():
        require(sha(name) == expected, 'Calibration data changed')
    config = json.loads((args.model / 'config.json').read_text())
    rope = config.get('rope_scaling', {})
    require(rope.get('rope_type', rope.get('type')) == 'yarn' and rope.get('factor') == 2.0
            and rope.get('original_max_position_embeddings') == 32768, 'Clinical YaRN configuration differs')
    heads = json.loads(args.heads.read_text())
    require(isinstance(heads, dict) and heads, 'Empty PASTA head configuration')
    for layer, selected in heads.items():
        require(0 <= int(layer) < config['num_hidden_layers'], 'Invalid profiled layer')
        require(isinstance(selected, list) and len(selected) == len(set(selected)), 'Invalid or duplicated heads')
        require(all(isinstance(h, int) and 0 <= h < config['num_attention_heads'] for h in selected), 'Invalid profiled head')
    adaptive = json.loads(args.adaptive_audit.read_text())
    require(adaptive['status'] == 'complete_verified' and adaptive['config_ready'], 'Ada experts not independently verified')
    require(set(adaptive['experts']) == {'synthetic', 'counterfact', 'biasbios', 'hotpotqa'}, 'Four expert sources required')
    for name, expert in adaptive['experts'].items():
        require(expert['checkpoint_files_sha256'] == fingerprints, 'Ada/PASTA checkpoint differs: ' + name)
        require(sha(expert['path']) == expert['sha256'], 'Ada projection bytes changed: ' + name)
        require(sha(expert['manifest_path']) == expert['manifest_sha256'], 'Ada construction record changed')
    require(sha(adaptive['expert_config_path']) == adaptive['expert_config_sha256'], 'Ada expert config bytes changed')
    report = dict(status='complete_verified', new_gpu_forwards=0,
                  scope='Artifact/source/model binding only; no clinical performance or paper reproduction claim',
                  created_at_utc=datetime.now(timezone.utc).isoformat(),
                  model_path=str(args.model.resolve()), model_files_sha256=fingerprints,
                  model_fingerprint=hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest(),
                  pasta=dict(path=str(args.heads.resolve()), sha256=sha(args.heads),
                             run_manifest_path=str(args.heads_run.resolve()), run_manifest_sha256=sha(args.heads_run),
                             selected_heads=sum(map(len, heads.values())),
                             selection_scope='Author Qwen port and correlated KV-head selection; not original exhaustive PASTA head search'),
                  adaseka=dict(expert_config_path=adaptive['expert_config_path'],
                               expert_config_sha256=adaptive['expert_config_sha256'],
                               audit_path=str(args.adaptive_audit.resolve()), audit_sha256=sha(args.adaptive_audit)),
                  checker_sha256=sha(__file__))
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'status': report['status'], 'model_fingerprint': report['model_fingerprint']}))


if __name__ == '__main__':
    main()
