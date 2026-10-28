"""Summarize frozen automatic P6 pilot; missing outputs never become passes."""
import argparse,collections,json,math,random
from pathlib import Path
D=Path(__file__).resolve().parent
def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def mean(xs):return sum(xs)/len(xs) if xs else None
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);a=ap.parse_args();run=Path(a.run_dir)
 manifest=json.loads((run/'manifest.json').read_text())
 assert (run/'DONE.json').exists() and (run/'scoring/DONE.json').exists(),'Generation and scoring must both complete'
 records=read(run/'scoring/metrics.jsonl');assert len(records)==manifest['expected_outputs'],'Missing scoring records'
 idx={(r['pair_id'],r['variant'],r['method']):r for r in records};assert len(idx)==len(records)
 summary={};base={(r['pair_id'],r['variant']):r for r in records if r['method']=='Vanilla'}
 for method in manifest['methods']:
  summary[method]={}
  for variant in ['native','decoy']:
   rs=[r for r in records if r['method']==method and r['variant']==variant]
   z={'n':len(rs)}
   for key in ['PathCost','BFR','DA','parse_failure','scoring_failure','repetition_loop','output_capped','actual_intervention','execution_seconds','component_total_seconds','input_tokens','generated_tokens']:
    vs=[r[key] for r in rs if r.get(key) is not None];z[key]=mean(vs)
   if variant=='native' and base:
    z['native_regression_rate']=mean([bool(base[r['pair_id'],variant]['path_acceptable']) and not r['path_acceptable'] for r in rs])
    z['native_correction_rate']=mean([not base[r['pair_id'],variant]['path_acceptable'] and bool(r['path_acceptable']) for r in rs])
   summary[method][variant]=z
 gate={'phase':manifest['phase'],'status':'DESCRIPTIVE_ONLY','clinical_semantic_validity':'NOT_HUMAN_VERIFIED','paper_superiority_claim_supported':False}
 if manifest['phase']=='smoke':
  stable=[];checks={}
  for method in ['ASTER-1-no-green','ASTER-0.5-no-green','ASTER-2-no-green']:
   rs=[r for r in records if r['method']==method];van=[r for r in records if r['method']=='Vanilla']
   c={'repetition':mean([r['repetition_loop'] for r in rs])<=.03,'cap':mean([r['output_capped'] for r in rs])<=.02,'parse':mean([r['parse_failure'] for r in rs])<=mean([r['parse_failure'] for r in van])+.01,'intervention':mean([r['actual_intervention'] and all(r['target_token_counts']) for r in rs])>=.85}
   checks[method]=c
   if all(c.values()):stable.append(method)
  gate.update(status='READY_FOR_FULL_BASELINES' if stable else 'STOP_GENERATION_DIAGNOSTIC_REQUIRED',stability_checks=checks,selected_method=stable[0] if stable else None,selection_rule='Prefer lambda1, then0.5, then2 among stable no-green configurations; no efficacy claim from smoke')
  outs=read(run/'outputs.jsonl');by={(r['pair_id'],r['variant'],r['method']):r for r in outs}
  pairs={(r['pair_id'],r['variant']) for r in outs}
  gate['matched_zero_output_agreement']=mean([by[p,v,'Vanilla']['output_ids']==by[p,v,'Matched-Zero']['output_ids'] for p,v in pairs])
 elif manifest['phase']=='baselines':
  ast=next(m for m in manifest['methods'] if m.startswith('ASTER-'))
  n=summary[ast]['native'];d=summary[ast]['decoy'];vn=summary['Vanilla']['native'];vd=summary['Vanilla']['decoy']
  best=max(['TS-Note','TS-Prune','TS-SEKA'],key=lambda m:summary[m]['decoy']['BFR'])
  rs=[r for r in records if r['method']==ast];vs=[r for r in records if r['method']=='Vanilla']
  checks={'native_PathCost':n['PathCost']-vn['PathCost']<=.01,'native_BFR':n['BFR']-vn['BFR']>=-.02,'decoy_DA':vd['DA']-d['DA']>=.08,'decoy_BFR':d['BFR']-vd['BFR']>=.05,'best_TS_BFR':d['BFR']-summary[best]['decoy']['BFR']>=.03,'repetition':mean([r['repetition_loop'] for r in rs])<=.03,'cap':mean([r['output_capped'] for r in rs])<=.02,'parse':mean([r['parse_failure'] for r in rs])<=mean([r['parse_failure'] for r in vs])+.01,'intervention':mean([r['actual_intervention'] and all(r['target_token_counts']) for r in rs])>=.85}
  latency=mean([r['component_total_seconds'] for r in rs])/mean([r['execution_seconds'] for r in vs])
  gate.update(status='PILOT_SIGNAL_CHECKS_PASS' if all(checks.values()) else 'PILOT_ONE_OR_MORE_CHECKS_FAIL',checks=checks,best_TS=best,latency_ratio_measured_components=latency,low_cost_gate=latency<=1.5,human_spot_check_before_formal_use=True)
  # Paired guide-cluster bootstrap: sample guides, retain all cases and paired outputs inside each.
  differences={}
  for comparator in ['Vanilla',best]:
   for variant,key in [('decoy','BFR'),('decoy','DA'),('native','PathCost'),('native','BFR')]:
    grouped=collections.defaultdict(list)
    for r in records:
     if r['method']==ast and r['variant']==variant:
      grouped[r['guideline_id']].append(r[key]-idx[r['pair_id'],variant,comparator][key])
    gids=sorted(grouped);rng=random.Random(20261028);values=[]
    for _ in range(2000):
     draws=[rng.choice(gids) for _ in gids];vals=[v for g in draws for v in grouped[g]];values.append(mean(vals))
    values.sort();differences[f'{ast}-{comparator}:{variant}:{key}']={'difference':mean([v for g in gids for v in grouped[g]]),'cluster_bootstrap_95_interval':[values[49],values[1949]],'guidelines':len(gids)}
  dump(run/'paired_intervals.json',differences)
 dump(run/'summary.json',summary);dump(run/'gate_status.json',gate)
 print(json.dumps(gate,ensure_ascii=False))
if __name__=='__main__':main()
