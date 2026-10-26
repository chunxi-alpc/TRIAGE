"""Restore SciPy's removed csr_matrix.A alias for the official BiasBios scorer.

Only the API alias is restored: A is exactly toarray(). The benchmark code,
prompts, generations, token ranking, TF-IDF values and metric formulas are kept.
The original SEKA checkout and installed packages are never edited.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('upstream_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.upstream_args[:1] == ['--']:
        args.upstream_args = args.upstream_args[1:]
    if args.report.exists():
        raise FileExistsError('Use a new compatibility report path')
    import scipy
    from scipy.sparse import csr_matrix
    old = csr_matrix.__dict__.get('A')
    added = not hasattr(csr_matrix, 'A')
    report = {'scipy': scipy.__version__, 'added_csr_matrix_A_alias': added,
              'alias_definition': 'property(lambda self: self.toarray())',
              'benchmark_source_modified': False, 'metric_formula_modified': False,
              'upstream_module': 'benchmarks.eval_bias_gen',
              'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    code = 0
    try:
        if added:
            csr_matrix.A = property(lambda self: self.toarray())
        sys.argv = ['benchmarks.eval_bias_gen'] + args.upstream_args
        runpy.run_module('benchmarks.eval_bias_gen', run_name='__main__')
        report['status'] = 'executed_pending_independent_metric_verification'
    except Exception as error:
        code = 1
        report.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        traceback.print_exc()
    finally:
        if added:
            if old is None:
                delattr(csr_matrix, 'A')
            else:
                csr_matrix.A = old
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
