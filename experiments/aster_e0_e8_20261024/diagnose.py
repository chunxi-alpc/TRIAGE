"""Development-only leakage/position and official green smoke, no main claims."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from engine import Engine
from green import official_green
P=Path('/home/gcx/aster_e0_e8_20261024/private')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--out',required=True);ap.add_argument('--projection');a=ap.parse_args();out=P/a.out;out.mkdir(exist_ok=False);e=Engine(a.model);start=time.time()
 rows=[r for r in map(json.loads,(P/'cases.jsonl').read_text().splitlines()) if r['split']=='dev' and r['seed']==0 and r['variant']=='AC'];rows=sorted(rows,key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())[:5];results=[]
 for c in rows:
  ids,roles=e.render(c);s=e.tok.encode('Applicable rule:',add_special_tokens=False);ids+=s;roles += [None]*len(s);keep=[i for i,r in enumerate(roles) if r is None]
  blind=e.forward(ids,roles,len(ids),'blind')[-1];deleted=e.forward([ids[i] for i in keep],[None]*len(keep),len(keep),'none',positions=keep)[-1];reindexed=e.forward([ids[i] for i in keep],[None]*len(keep),len(keep),'none')[-1]
  result={'id':c['id'],'tree':c['tree'],'tokens':len(ids),'masked_tokens':len(ids)-len(keep),'preserved_position_max_logit_diff':float((blind-deleted).abs().max()),'preserved_position_argmax_equal':bool(blind.argmax()==deleted.argmax()),'reindexed_max_logit_diff':float((blind-reindexed).abs().max()),'reindexed_argmax_equal':bool(blind.argmax()==reindexed.argmax()),'literal_action_mentions_in_patient':sum(r['action'].lower() in c['patient'].lower() for r in c['rules'])}
  modified=[e.tok.bos_token_id or e.tok.eos_token_id if roles[i] is not None else token for i,token in enumerate(ids)]
  invariant=e.forward(modified,roles,len(ids),'blind')[-1]
  result['same_length_action_change_max_logit_diff']=float((blind-invariant).abs().max())
  if a.projection:
   with official_green(e,roles,c['gold_rule'],a.projection):steered=e.forward(ids,roles,len(ids),'none')[-1]
   base=e.forward(ids,roles,len(ids),'none')[-1];result['green_max_logit_change']=float((steered-base).abs().max());result['green_finite']=bool(torch.isfinite(steered).all())
  results.append(result)
 (out/'summary.json').write_text(json.dumps({'scope':'5 development AC cases; token deletion vs textual deletion are different controls','results':results,'seconds':time.time()-start,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'projection_sha256':hashlib.sha256(Path(a.projection).read_bytes()).hexdigest() if a.projection else None},indent=2))
if __name__=='__main__':main()
