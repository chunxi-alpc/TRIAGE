"""Export common statistics from existing SEA/CAD/temperature outputs; no model calls."""
import csv, gzip, json
from pathlib import Path
import numpy as np

ROOT = Path('/home/gcx/aster-e5-gaps-20261112')
OUT = ROOT / 'reports/aster_e5_gaps_20261112/additional_historical'
OLD = Path('/home/gcx/aster_cr_repair_v2_run_20261108/qwen')
REAL = Path('/home/gcx/aster_cancel_story_20261109/meeting/real_completion')
SEED, DRAWS, BUDGET = 11, 2000, 0.2923349027297149

def rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    meta = json.loads((REAL / 'metadata.json').read_text())
    test = [v for v in meta['views'] if v['split'] == 'test']
    pairs = [p for p in meta['pairs'] if p['split'] == 'test']
    sources = sorted({v['source_id'] for v in test})
    source_index = {s: i for i, s in enumerate(sources)}
    boot = np.random.default_rng(SEED).multinomial(len(sources), np.full(len(sources), 1 / len(sources)), size=DRAWS).astype(float)
    raw_archive = np.load(REAL / 'evaluated_scores.npz', allow_pickle=False)
    raw_real = dict(zip(raw_archive['ids'].tolist(), raw_archive['raw'].tolist()))
    base_paths = {r['case_id']: r for r in rows(REAL / 'path_results.jsonl') if r['method'] == 'raw'}
    summary = json.loads((OLD / 'summary.json').read_text())
    comp = [r for i in range(4) for r in rows(OLD / 'comparators' / f'test_{i}.jsonl')]
    comp_paths = [r for i in range(4) for r in rows(OLD / 'comparators' / f'paths_test_{i}.jsonl')]
    cmpraw = {r['view_id']: r['scores']['full'] for r in comp}
    traw = {r['view_id']: r['ell'] for r in rows(OLD / 'readout_scores/full_test.jsonl')}
    methods = {
        'CAD': ({r['view_id']: r['scores']['CAD'] for r in comp}, cmpraw,
                [dict(case_id=r['case_id'], source_id=r['source_id'], **r['methods']['CAD']) for r in comp_paths], 'CAD'),
        'SEA': ({r['view_id']: r['scores']['sea'] for r in comp}, cmpraw,
                [dict(case_id=r['case_id'], source_id=r['source_id'], **r['methods']['sea']) for r in comp_paths], 'sea'),
        'Temperature Scaling': ({r['view_id']: r['ell'] for r in rows(OLD / 'readout_scores/temperature_test.jsonl')}, traw,
                rows(OLD / 'path_results/temperature_test.jsonl'), 'temperature')
    }
    def sums(values, ss):
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        result = np.zeros((len(sources), values.shape[1]))
        np.add.at(result, [source_index[s] for s in ss], values)
        return result
    def ci(v):
        v = np.asarray(v)
        v = v[np.isfinite(v)]
        return np.quantile(v, [.025, .975]).tolist() if len(v) else None
    def boot_mean(v, ss):
        a = boot @ sums(np.column_stack([np.ones(len(v)), v]), ss)
        with np.errstate(divide='ignore', invalid='ignore'):
            return ci(a[:, 1] / a[:, 0])
    metrics, table, predictions, path_rows = {}, [], [], []
    for name, (scores, base, paths, old_key) in methods.items():
        missing = [v['view_id'] for v in test if v['view_id'] not in scores or v['view_id'] not in base]
        if missing:
            raise ValueError(f'{name}: missing {len(missing)} test input scores; no partial denominator published')
        base_difference = max(abs(base[v['view_id']] - raw_real[v['view_id']]) for v in test)
        archived_raw_paths = ({r['case_id']: r['methods']['full'] for r in comp_paths}
                              if old_key != 'temperature' else
                              {r['case_id']: r for r in rows(OLD / 'path_results/full_test.jsonl')})
        raw_path_mismatches = sum(
            archived_raw_paths[k]['correct'] != v['correct'] or
            archived_raw_paths[k]['path'] != v['path'] or
            archived_raw_paths[k]['leaf'] != v['leaf']
            for k, v in base_paths.items())
        reference = {
            'score_count': len(test), 'max_abs_raw_score_difference_from_real_completion': base_difference,
            'raw_path_mismatches_from_real_completion': raw_path_mismatches,
            'common_reference_confirmed': base_difference == 0 and raw_path_mismatches == 0,
            'scope': 'historically used MEGA 40-source pool; no E1 training-compatibility decision'
        }
        groups = {}
        for group, kinds in [('A', ['A']), ('P', ['P']), ('AP', ['A','P']), ('C', ['C']), ('F', ['F']), ('CF', ['C','F']), ('F0', ['F0'])]:
            pp = [p for p in pairs if p['kind'] in kinds]
            left = np.array([scores[p['left']] for p in pp]); right = np.array([scores[p['right']] for p in pp])
            bl = np.array([base[p['left']] for p in pp]); br = np.array([base[p['right']] for p in pp])
            yl = np.array([p['left_label'] for p in pp]); yr = np.array([p['right_label'] for p in pp])
            raw, dy = br-bl, right-left
            dc = dy-raw
            cor = ((left>0)==yl)&((right>0)==yr)
            basecor = ((bl>0)==yl)&((br>0)==yr)
            ss = [p['source_id'] for p in pp]
            agg = sums(np.column_stack([np.ones(len(pp)),raw,raw**2,dy,dy**2,dc**2,cor,cor.astype(float)-basecor.astype(float)]),ss)
            bb = boot @ agg
            with np.errstate(divide='ignore',invalid='ignore'):
                brms = np.sqrt(bb[:,4]/bb[:,0])
                bcrms = np.sqrt(bb[:,5]/bb[:,0])
                bpa = bb[:,6]/bb[:,0]
                bdiff = bb[:,7]/bb[:,0]
                bvr = 1-(bb[:,4]/bb[:,0]-(bb[:,3]/bb[:,0])**2)/(bb[:,2]/bb[:,0]-(bb[:,1]/bb[:,0])**2)
            mm = dict(n=len(pp),sources=len(set(ss)),raw_difference_rms=float(np.sqrt(np.mean(raw**2))),
                difference_rms=float(np.sqrt(np.mean(dy**2))),difference_rms_ci=ci(brms),
                correction_rms=float(np.sqrt(np.mean(dc**2))),correction_rms_ci=ci(bcrms),
                correction_abs_max=float(np.max(np.abs(dc))),paired_correct=int(cor.sum()),
                paired_accuracy=float(cor.mean()),paired_accuracy_ci=ci(bpa),
                paired_accuracy_difference_vs_raw=float((cor.astype(float)-basecor.astype(float)).mean()),
                paired_accuracy_difference_vs_raw_ci=ci(bdiff),
                corrected_pairs_vs_raw=int((cor&~basecor).sum()),new_error_pairs_vs_raw=int((~cor&basecor).sum()))
            if group in ['A','P','AP','F0']:
                mm.update(variance_reduction=float(1-np.var(dy)/np.var(raw)),variance_reduction_ci=ci(bvr))
            else:
                rs=(yr-yl)*raw;ds=(yr-yl)*dy
                cc=boot@sums(np.column_stack([rs,ds]),ss)
                with np.errstate(divide='ignore',invalid='ignore'):
                    er=cc[:,1]/cc[:,0]
                mm.update(mean_evidence_retention=float(np.mean(ds)/np.mean(rs)),mean_evidence_retention_ci=ci(er),
                    response_direction_degraded=int(((rs>0)&(ds<=0)).sum()),
                    response_direction_improved=int(((rs<=0)&(ds>0)).sum()),
                    exceeds_reference_budget_count=int((np.abs(dc)>BUDGET+1e-7).sum()),
                    exceeds_reference_budget_fraction=float((np.abs(dc)>BUDGET+1e-7).mean()),
                    budget_note='Descriptive comparison to ASTER historical budget; this method did not train under that constraint.')
            groups[group]=mm
            if group not in ['AP','CF']:
                for j,p in enumerate(pp):
                    predictions.append(dict(method=name,pair_id=p['pair_id'],source_id=p['source_id'],kind=p['kind'],
                        left=p['left'],right=p['right'],left_label=p['left_label'],right_label=p['right_label'],
                        left_score=float(left[j]),right_score=float(right[j]),raw_left_score=float(bl[j]),raw_right_score=float(br[j]),
                        original_difference=float(raw[j]),corrected_difference=float(dy[j]),correction_difference=float(dc[j]),paired_correct=bool(cor[j])))
        bycase={r['case_id']:r for r in paths}
        if set(bycase)!=set(base_paths):
            raise ValueError(f'{name}: historical path denominator mismatch')
        ordered=[bycase[k] for k in sorted(base_paths)]
        cor=np.array([r['correct'] for r in ordered]);ss=[r['source_id'] for r in ordered]
        bc=np.array([archived_raw_paths[r['case_id']]['correct'] for r in ordered])
        pm=dict(n=len(ordered),correct=int(cor.sum()),accuracy=float(cor.mean()),accuracy_ci=boot_mean(cor,ss),
                difference_vs_raw=float((cor.astype(float)-bc.astype(float)).mean()),
                difference_vs_raw_ci=summary['paths'][old_key]['difference_from_full']['ci95'],
                difference_ci_source=str(OLD/'summary.json')+'#paths.'+old_key+'.difference_from_full',
                corrected=int((cor&~bc).sum()),new_errors=int((~cor&bc).sum()))
        path_rows.extend(dict(method=name,**r) for r in ordered)
        metrics[name]=dict(reference=reference,groups=groups,path=pm)
        ap,cf=groups['AP'],groups['CF']
        table.append(dict(method=name,result_scope='historical_statistics_completed_no_new_model_run',
            common_raw_reference=reference['common_reference_confirmed'],sources=len(sources),AP_n=ap['n'],CF_n=cf['n'],
            AP_RMS=ap['difference_rms'],AP_RMS_ci_low=ap['difference_rms_ci'][0],AP_RMS_ci_high=ap['difference_rms_ci'][1],
            AP_variance_reduction=ap['variance_reduction'],AP_vr_ci_low=ap['variance_reduction_ci'][0],AP_vr_ci_high=ap['variance_reduction_ci'][1],
            CF_correction_RMS=cf['correction_rms'],CF_correction_RMS_ci_low=cf['correction_rms_ci'][0],CF_correction_RMS_ci_high=cf['correction_rms_ci'][1],
            CF_evidence_retention=cf['mean_evidence_retention'],CF_er_ci_low=cf['mean_evidence_retention_ci'][0],CF_er_ci_high=cf['mean_evidence_retention_ci'][1],
            CF_correct=cf['paired_correct'],CF_accuracy=cf['paired_accuracy'],CF_accuracy_ci_low=cf['paired_accuracy_ci'][0],CF_accuracy_ci_high=cf['paired_accuracy_ci'][1],
            path_correct=pm['correct'],path_n=pm['n'],path_accuracy=pm['accuracy'],path_accuracy_ci_low=pm['accuracy_ci'][0],path_accuracy_ci_high=pm['accuracy_ci'][1],
            path_difference_ci_low=pm['difference_vs_raw_ci'][0],path_difference_ci_high=pm['difference_vs_raw_ci'][1]))
    result=dict(protocol='Existing historical outputs only; not new model experiments; E1 training compatibility unclassified.',
        bootstrap=dict(draws=DRAWS,seed=SEED,unit='guideline source cluster',interval='percentile 95%',joint_sources_across_methods=True,undefined_draws='omitted'),
        sources=dict(metadata=str(REAL/'metadata.json'),raw=str(REAL/'evaluated_scores.npz'),comparators=str(OLD/'comparators'),
                     temperature=str(OLD/'readout_scores/temperature_test.jsonl')),
        counts=dict(sources=len(sources),AP=2293,CF=40,paths=806),methods=metrics)
    (OUT/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (OUT/'table_4_5_historical.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    for filename,rr in [('pair_predictions.jsonl.gz',predictions),('path_predictions.jsonl.gz',path_rows)]:
        with gzip.open(OUT/filename,'wt',encoding='utf-8') as f:
            for row in rr:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(dict(output=str(OUT),table=table),ensure_ascii=False))

if __name__=='__main__':
    main()
