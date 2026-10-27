from pathlib import Path
import hashlib,json
D=Path(__file__).resolve().parent
def check():
 a=json.loads((D/'results/data_audit.json').read_text());v=json.loads((D/'results/repair_validation.json').read_text());f=json.loads((D/'results/freeze.json').read_text())
 assert v['status']=='PASS' and not a['test_tree_intersection']
 for name,expected in f['sha256'].items():assert hashlib.sha256((D/name).read_bytes()).hexdigest()==expected, 'Frozen file changed: '+name
 assert hashlib.sha256((D/'private/cases.jsonl').read_bytes()).hexdigest()==a['cases_sha256']==v['cases_sha256']
 return a
