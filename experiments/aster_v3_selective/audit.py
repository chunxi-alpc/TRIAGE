"""Read archived identities and headers without executing an archived evaluator."""
from pathlib import Path
import csv
import gzip
import json
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/aster_v3_selective_20261124'
HOME = Path.home()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


# unit, checkout, report directory, identity-bearing files, unit of reuse, gap
ARCHIVES = [
 ('E1', 'aster-cr-diagnostics-e3cv-20261111', 'aster_cr_diagnostics_e3cv_20261111',
  'REPORT.zh.md configuration.json completion.json folds.json verification.json evaluation/cv_fold*_common/predictions.jsonl.gz evaluation/cv_fold*_common/path_predictions.jsonl.gz evaluation/dev/predictions.jsonl.gz evaluation/dev/path_predictions.jsonl.gz evaluation/dev/pair_predictions.jsonl.gz',
  '5 source folds and development split', 'selective protection'),
 ('E1_params','aster-cr-retrain-paired-v2-20261111','aster_cr_retrain_paired_v2_20261111',
  'REPORT.zh.md configuration.json folds.json data/E1_views.jsonl data/E1_pairs.jsonl data/natural_views.jsonl models/E1/full/model.json models/E1/full/m2.npz models/E1/full/m3.npz runtime/fit_heads.npy runtime/fit_uids.json',
  '80-source frozen fit', 'training-budget adjustment'),
 ('SJ', 'aster-m3-joint-20261115','aster_m3_joint_study_20261115',
  'REPORT.zh.md configuration.json verification.json diagnostics/correction_bounds.jsonl diagnostics/correction_bounds_summary.json evaluation/natural_first_error.jsonl.gz evaluation/pair_predictions.jsonl.gz evaluation/path_predictions.jsonl.gz selection/fold*/split.json models/fold*/projected_cache.npz j2/status.json J2_REPORT.zh.md',
  '166 LP and 5 source folds', 'semantic cause and inner-fold provenance'),
 ('PR87','aster-m3-missing-controls-20261123','aster_m3_missing_controls_20261123',
  'REPORT.zh.md configuration.json A_verification.json A_scores.jsonl.gz A_paths.jsonl.gz A_source_transitions.tsv A_first_error_boundaries.jsonl.gz b_archive_checks.json b_evaluation_checks.json b_train_constraints.csv status.json',
  '27392 verified historical path records and 25 fits', 'J_zero candidate event subset'),
 ('server_path','aster-path-error-20261123','aster_path_error_20261123',
  'REPORT.zh.md configs/analysis.json results/first_errors.csv results/case_evidence.jsonl.gz results/qualitative_cases.csv results/data_conflicts.csv results/path_transitions.csv',
  '6 E1 path score variants and 8 read cases', 'only missing semantic evidence'),
 ('PR71','aster-full-complement-70','aster_full_complement_70','REPORT.zh.md configuration.json cv_summary.json test_summary.json validation.json fold_results.csv','6528-d hard budget, 5 OOF folds and old holdout','archive comparison'),
 ('PR73','aster-head-quadrant-20261116','aster_head_quadrant_20261116','REPORT.zh.md configuration.json head_quadrant_consensus.csv quadrant_summary.json head_quadrant_metrics.csv','1 head quadrant study','archive only'),
 ('PR76','aster-m3-dual-objective-20261116','aster_m3_dual_objective_20261116','REPORT.zh.md configuration.json cache_manifest.json completion.json oof_grid_70.csv oof_candidate_parameters.json.gz','70 grid points','archive comparison'),
 ('PR78','aster-m3-pareto-knee-20261116','aster_m3_pareto_knee_20261116','PARETO_REPORT.zh.md configuration.json knee_selection.json pareto_grid_annotated.csv','same 70 points, secondary analysis','archive comparison'),
 ('PR82','aster-m1-sensitivity-20261117','aster_m1_sensitivity_20261117','REPORT.zh.md configuration.json completion.json execution_manifest.json configuration_counts.csv all_configuration_heads.csv.gz','1 M1 sensitivity study','archive only'),
 ('PR85','aster-table2-20261123','aster_table2_20261123','REPORT.zh.md configuration.json completion.json dev_pair_metrics.csv test_pair_metrics.csv table2.json cache_coverage.json','2 split-specific rule-type tables','untyped A/P and new semantic causes'),
]


def sample_keys(path):
    op = gzip.open if path.suffix == '.gz' else open
    if '.jsonl' in path.name:
        with op(path, 'rt') as stream:
            return list(json.loads(next(stream)).keys())
    if '.csv' in path.name or path.suffix == '.tsv':
        with op(path, 'rt') as stream:
            return next(csv.reader(stream, delimiter='\t' if path.suffix == '.tsv' else ','))
    return []


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    prs = json.loads((ROOT/'configs/aster_v3_selective/archive_prs.json').read_text())
    archives, coverage = [], []
    for unit, checkout, folder, patterns, count, gap in ARCHIVES:
        worktree = HOME/checkout
        base = worktree/'reports'/folder
        entries = []
        for pattern in patterns.split():
            files = sorted(base.glob(pattern))
            if not files:
                raise FileNotFoundError(base/pattern)
            for p in files:
                entries.append(dict(path=str(p.relative_to(worktree)), bytes=p.stat().st_size,
                                    fields=sample_keys(p)))
        state = 'completed_unmerged' if unit.startswith('PR') or unit=='SJ' else 'reuse_completed'
        tracked = set(subprocess.check_output(['git','-C',str(worktree),'ls-files','-z','--','reports/'+folder]).decode().split('\0'))
        for entry in entries:
            entry['in_git_index'] = entry['path'] in tracked
        sha = subprocess.check_output(['git','-C',str(worktree),'rev-parse','HEAD'],text=True).strip()
        archives.append(dict(unit=unit, checkout=str(worktree), checkout_commit=sha,
                             artifacts=entries))
        coverage.append([unit,state,count,0,'read-only',gap,str(base),sha])
    for pr in prs:
        if 'PR'+str(pr['number']) in [x[0] for x in ARCHIVES]:
            continue
        paths = git('ls-tree','-r','--name-only',pr['head_sha']).splitlines()
        changes = git('diff','--name-only',pr['base_sha']+'...'+pr['head_sha']).splitlines()
        reports = [p for p in changes if p in paths and p.endswith(('.md','.txt','.json','.csv','.jsonl','.jsonl.gz')) and
                   any(s in p.lower() for s in ('report','result','summary','config','status','prediction','metric','freeze','timing'))]
        archives.append(dict(unit=f"PR{pr['number']}",scope='related archive',artifacts=reports))
        coverage.append([f"PR{pr['number']}",'related_not_equivalent','1 historical study',0,'read-only',
                         'different model, task, inputs or labels',pr['url'],pr['head_sha']])
    old_root = HOME/'aster_existing_data_20261106'
    old_files = [old_root/'configuration.json', old_root/'REPORT.zh.md', *sorted((old_root/'paths').glob('predictions_*.jsonl'))]
    archives.append(dict(unit='legacy_paths',checkout=str(old_root),artifacts=[dict(path=str(p.relative_to(old_root)),bytes=p.stat().st_size,fields=sample_keys(p)) for p in old_files]))
    coverage.append(['legacy_paths','related_not_equivalent','432 archived paths',0,'read-only','other checkpoint and task population',str(old_root),'reports/aster_existing_data_20261106/ANALYSIS.zh.md'])
    coverage.extend([
        ['appendix_C2_C7','reuse_completed','16/32 dimensions and budget curves',0,'read-only','archive comparison','docs/ASTER/manuscript.appendix.zh.md',git('rev-parse','HEAD:docs/ASTER/manuscript.appendix.zh.md')],
        ['issue79','planned_only','0',0,'skipped','full complement + soft objective outside V3','https://github.com/chunxi-alpc/TRIAGE/issues/79','']])
    with (OUT/'DEDUP_COVERAGE.csv').open('w') as stream:
        writer=csv.writer(stream, lineterminator="\n")
        writer.writerow(['unit','identity_status','reused_quantity','new_historical_runs','operation','gap_or_disposition','source','existing_version'])
        writer.writerows(coverage)
    spec=dict(main_at_start='c8aed82757b3f6b05cc5a7f72c63df14b59575d0',prs=prs,archives=archives,
              seconds=time.perf_counter()-start,gpu_hours=0,backbone_forwards=0,full_replays=0)
    (OUT/'ARTIFACT_MANIFEST.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
