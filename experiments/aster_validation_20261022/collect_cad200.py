from pathlib import Path
import json,datetime,shutil
r=Path('/home/gcx/aster_validation_20261022/private/run500/cad200');out=Path(__file__).resolve().parent/'mechanism_20261023/results/cad200_scope';out.mkdir(parents=True,exist_ok=True);m=json.loads((r/'migration.json').read_text());ids=set(m['selection_ids']);report={'time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'scope':'only the fixed selected 200 cases; all remaining 500-case tasks saved and stopped','cases':200,'seed_CAD_records_reused':m['selected_CAD_records_reused'],'left_main_cases_reused':200,'workers':{p.name:p.read_text().strip() for p in r.glob('*.status')},'outputs':{}}
for name in ['original_filtered','left_main','left_unsigned','left_filtered','right_main','right_unsigned','right_filtered']:
 p=r/f'{name}.jsonl'
 if not p.exists():continue
 rows=[]
 for line in p.read_text().splitlines():
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:pass
 assert {x['id'] for x in rows}<=ids
 keys=[(x['id'],x['variant'],x['method'],x.get('gain'),x.get('assigned_wrong',False)) for x in rows];assert len(keys)==len(set(keys));report['outputs'][name]={'rows':len(rows),'cases_present':len({x['id'] for x in rows}),'duplicate_keys':0,'outside_selected200':0}
shutil.copyfile(r/'migration.json',out/'migration.json');(out/'STATUS.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
