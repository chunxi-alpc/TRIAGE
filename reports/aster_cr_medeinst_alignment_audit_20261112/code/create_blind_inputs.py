"""Rebuild blinded packets from the frozen sampling manifest into a NEW directory."""
from pathlib import Path
import argparse,json,difflib
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
out=Path(__file__).resolve().parents[1];manifest=json.loads((out/'sampling_manifest.json').read_text())
src=Path('/home/gcx/aster-cr-retrain-paired-v2-20261111/reports/aster_cr_multisource_20261112/balanced_10006/data/medeinst_train_selected.jsonl')
med={r['pair_id']:r for r in map(json.loads,src.open())}
labels={}
for who in [1,2,3]:
 for phase in ['pilot','main']:
  packet=[]
  for m in manifest['samples']:
   if m['reviewer']!=who or m['phase']!=phase:continue
   p=med[m['pair_id']];aa=[x.strip() for x in p['original_input'].splitlines() if x.strip()];bb=[x.strip() for x in p['edited_input'].splitlines() if x.strip()]
   changes=[dict(operation=t,left=aa[i:j],right=bb[k:l]) for t,i,j,k,l in difflib.SequenceMatcher(None,aa,bb,autojunk=False).get_opcodes() if t!='equal']
   packet.append(dict(review_id=m['review_id'],left_text=p['original_input'],right_text=p['edited_input'],candidates=sorted([p['label_before'],p['label_after']]),text_changes=changes))
   labels[m['review_id']]=dict(pair_id=p['pair_id'],label_before=p['label_before'],label_after=p['label_after'])
  (a.output/f'reviewer_{who}_{phase}_blind.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in packet))
(a.output/'labels_private.json').write_text(json.dumps(labels,ensure_ascii=False,indent=2)+'\n')
print('600 blinded records written; labels in separate private file. Do not expose labels before each blind decision is saved.')
