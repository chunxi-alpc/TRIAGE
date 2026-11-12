"""Finalize corrected serialization and complete official NLI evaluation, CPU only."""
import pathlib,time,traceback
import numpy as np
import run as r
import analysis as a
import report
O=r.ROOT
def main():
 started=time.time()
 a.STORE=r.Store()
 a.ITEMS={v['id']:v for v in r.rows(r.ev.TARGET/'medpic/items.jsonl')}
 a.MED=r.read(O/'calibration/medpic.json')
 a.BEFORE={f:(r.fdir('E3',f)/'m3.npz').read_bytes() for f in ['full']+['fold'+str(i) for i in range(1,6)]}
 p=O/'evaluation/nli4ct_official5500/predictions.jsonl'
 pred=r.rows(p)
 assert len(pred)==5500 and sum(v['original'] for v in pred)==500
 pairs=r.rows(O/'evaluation/nli4ct_official5500/pair_predictions.jsonl')
 assert sum(v['group']=='native_preserving' and v['method']=='E3' for v in pairs)==4136
 assert sum(v['group']=='native_changing' and v['method']=='E3' for v in pairs)==864
 t,b,c=a.recompute()
 a.verify();a.figures()
 tests=r.read(O/'status/tests.json')
 tests['tests'] += [dict(name='official_NLI_5500_and_native_pair_denominators',passed=True),
  dict(name='new60_natural_marker_matches_canonical_definition',passed=sum(v['original'] and v['label'] is not None for v in r.rows(O/'evaluation/new60/predictions.jsonl'))==4928)]
 assert all(x['passed'] for x in tests['tests'])
 r.write(O/'status/tests.json',tests)
 report.finish(t,b,c)
 r.write(O/'status/finalization.json',dict(complete=True,seconds=time.time()-started,all_tests=len(tests['tests']),GPU_inference=False,reran_training=False))
 if (O/'status/nli_error.json').exists():
  error=r.read(O/'status/nli_error.json');error['resolved_by']='successful CPU finalization'
  r.write(O/'status/nli_error.json',error)
if __name__=='__main__':
 try:main()
 except Exception as e:
  r.write(O/'status/finalization_error.json',dict(error=str(e),traceback=traceback.format_exc()));raise
