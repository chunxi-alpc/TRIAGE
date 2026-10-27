"""Reuse only Qwen outputs whose actual token inputs are proven unchanged."""
from pathlib import Path
import argparse,hashlib,json,shutil
D=Path(__file__).resolve().parent;OLD=D.parent/'aster_prelim_repaired_20261027'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',type=int,choices=[1,2],required=True);a=ap.parse_args()
 proof=json.loads((D/'results/tokenization_audit.json').read_text())['qwen'];assert proof['safe_to_reuse_old_outputs']
 if a.stage==2:assert json.loads((D/'results/gate_status.json').read_text())['P1']=='PASS_SW_GATE'
 prefix='qwen' if a.stage==1 else 'p2_qwen';src=OLD/'private'/prefix;out=D/'private'/prefix
 if out.exists():assert (out/'DONE.json').exists();return
 assert (src/'DONE.json').exists() and sha(OLD/'private/cases.jsonl')==sha(D/'private/cases.jsonl')
 weightsfile=D/'results/qwen_reuse_weight_verification.json'
 if not weightsfile.exists():
  oldm=json.loads((OLD/'private/qwen/manifest.json').read_text());root=Path(oldm['model']);checked={}
  assert sha(root/'config.json')==oldm['config_sha256']
  for name,meta in oldm['weights'].items():assert sha(root/name)==meta['sha256'];checked[name]=meta['sha256']
  weightsfile.write_text(json.dumps({'status':'PASS','weights':checked,'config_sha256':oldm['config_sha256']},indent=2)+'\n')
 shutil.copytree(src,out);shutil.copyfile(OLD/'private'/(prefix+'.log'),D/'private'/(prefix+'.log'))
 m=json.loads((out/'manifest.json').read_text());m.update(reused_from_prior_run=True,source_directory=str(src),source_manifest_sha256=sha(src/'manifest.json'),tokenization_audit_sha256=sha(D/'results/tokenization_audit.json'),weights_rechecked=True,reuse_reason='Qwen tokenizer produces identical token IDs under old and corrected encoding for all P1, P2 and closed-book prompts; same frozen data, weights, and decoding. No new Qwen inference.')
 (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 print(json.dumps({'model':'qwen','stage':a.stage,'source':str(src),'reused':True}))
if __name__=='__main__':main()
