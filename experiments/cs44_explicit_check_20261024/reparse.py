"""Accept explicit inline Branch markers in saved text; never rerun checker."""
import hashlib,json
from pathlib import Path
from trial import P,OUT,read,sha,dump,jsonl,parse_check
import run as prev
cache={r['key']:r['result'] for r in read(P/'action_cache.jsonl')}
for k in range(2):
 for r in read(P/f'actions{k}.jsonl'):cache[r['key']]=r['result']
counts={'explicit_inline_choices_recovered':0,'unresolved_after_parse':0,'checker_calls':0,'new_final_calls':0};prepared=[]
for k in range(2):
 requests={r['cell_key']:r for r in read(P/f'requests{k}.jsonl')};source=P/f'outputs{k}.jsonl';raw=read(source);fixed=[]
 for r in raw:
  choice=parse_check(r['check_completion']);counts['unresolved_after_parse']+=choice is None;counts['explicit_inline_choices_recovered']+=choice!=r['selected_slot'];route=requests[r['cell_key']]['routes'][choice or 'uncertain'];key=route['key']
  if key!=r['final_request_key']:
   assert key in cache,'Missing final request requires inference: '+key
   result=cache[key];r={**r,'selected_slot':choice,'fallback_two_rules':choice is None,'final_request_key':key,'final_reused':True,'final_source':result.get('source','cache'),'prediction':prev.old.parse(result['completion'],requests[r['cell_key']]['options']),'completion':result['completion'],'inline_marker_reparsed':True}
  fixed.append(r)
 prepared.append((k,source,fixed))
# No writes unless every changed route has an existing final result.
for k,source,fixed in prepared:
 backup=P/f'outputs{k}.initial_parse.jsonl';assert not backup.exists();backup.write_bytes(source.read_bytes());jsonl(source,fixed);mp=OUT/f'worker{k}.json';m=json.loads(mp.read_text());dump(OUT/f'worker{k}.initial_parse.json',m);m.update({'original_output_sha256':m['output_sha256'],'output_sha256':sha(source),'inline_markers_reparsed':True,'parser_script_sha256':sha(Path(__file__).with_name('trial.py'))});dump(mp,m)
dump(OUT/'format_reparse.json',{**counts,'policy':'Use last explicit Branch: A/B/uncertain marker regardless of line position; no evaluation labels read; all changed final prompts found in existing cache.','initial_outputs_preserved':True});print(json.dumps(counts))
