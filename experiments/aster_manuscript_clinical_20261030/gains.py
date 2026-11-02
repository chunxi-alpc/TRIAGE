"""Pool development gains without reading confirmation or official predictions."""
import argparse,json,subprocess
from pathlib import Path
from .data import write
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run);summary=[]
 for gain in [.25,1.,4.]:
  rows=[]
  for shard in [0,1]:
   root=run/('gain'+str(shard));status=json.loads((root/'status.json').read_text());assert status['status']=='complete'
   for path in (root/'units').glob('*/*.json'):
    if path.name=='classification.json':continue
    r=json.loads(path.read_text())
    if r['gain']==gain:rows.append(r)
  if len(rows)!=35:raise ValueError('development denominator changed; expected 35 registered original cases')
  summary.append({'gain':gain,'denominator':len(rows),'correct':sum(r['score']['final_leaf_correct'] for r in rows),'leaf_accuracy':sum(r['score']['final_leaf_correct'] for r in rows)/len(rows),
   'path_f1':sum(r['score']['path_f1'] for r in rows)/len(rows),'technical_failures':sum(r['status']=='failed' for r in rows)})
 gain=max(summary,key=lambda x:(x['leaf_accuracy'],-x['gain']))['gain'];f=json.loads((run/'clinical_freeze_pre_gain.json').read_text())
 f.update(g_plus=gain,g_minus=gain,development_gain_results=summary,frozen_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),freeze_scope='source-isolated original clinical evaluation; no official test tuning')
 write(run/'frozen.json',f);write(run/'gain_selection.json',{'chosen_gain':gain,'results':summary})
 print(json.dumps({'gain':gain,'results':summary}))
if __name__=='__main__':main()

