"""CPU-only diagnosis of frozen train_val extraction failures; publishes counts/hashes only."""
from pathlib import Path
import json, hashlib, collections, sys
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P))
from mega_trainval_reader import iter_method_tasks
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def norm(s):return " ".join(s.split())
def main():
    root=Path("/home/gcx/TRIAGE/data/.hf_cache/datasets--nchennnn--MEGA-CDP/snapshots/8f058c88ce1ed43693785b435ad0ff99907a2890")
    freeze=json.loads((P/"mega_dev16_input_lock.json").read_text())
    sources={sha(t.guideline):t.guideline for _, t in iter_method_tasks(root,freeze)}
    records=[]; total=collections.Counter()
    for f in sorted((P/"runtime/mega_dev16_seed0_gpu01/private_guideline_extractions").glob("*.json")):
        d=json.loads(f.read_text());g=sources[d["guideline_sha256"]];ng=norm(g);counts=collections.Counter()
        for item in d.get("item_audit",[]):
            counts["items"]+=1
            if item.get("truncated"):counts["truncated"]+=1;continue
            x=item.get("raw_item")
            if not isinstance(x,dict):counts["non_object"]+=1;continue
            c=x.get("condition_quote");a=x.get("action_quote");rr=x.get("rule_quote");es=x.get("exception_quotes",[])
            if not isinstance(es,list) or any(not isinstance(v,str) or not v for v in [c,a,rr,*es]):
                counts["invalid_fields"]+=1;continue
            if c==a:counts["condition_equals_action"]+=1
            elif a in c:counts["condition_contains_action"]+=1
            else:
                anchors=[c,a,*es]
                if all(v in g for v in anchors):
                    counts["individual_anchors_exact_action_separated"]+=1
                    if rr not in g or any(v not in rr for v in anchors):
                        counts["exact_anchors_but_rule_passage_invalid"]+=1
                elif all(norm(v) in ng for v in anchors):
                    counts["anchors_match_only_after_whitespace_normalization"]+=1
                else:counts["at_least_one_anchor_requires_more_than_whitespace_change"]+=1
            if item.get("status")=="quotes_validated":counts["original_valid"]+=1
        total.update(counts)
        records.append({"guideline_sha256":d["guideline_sha256"],"cache_sha256":hashlib.sha256(f.read_bytes()).hexdigest(),"original_valid_rules":len(d["rules"]),"output_tokens":d["generation"].get("output_tokens"),"array_closed":d.get("array_closed"),"counts":dict(counts)})
    report={"status":"complete_verified","scope":"Frozen development extraction diagnosis only; no clinical scoring or method modification","guidelines":len(records),"test_or_reference_labels_read":False,"source_text_published":False,"new_gpu_forwards":0,"counts":dict(total),"records":records,"limitations":["Exact source anchoring does not establish clinical meaning or correct condition/action pairing","Whitespace-only matches and possible original-span reconstruction are diagnostic candidates, not accepted rules","Original failed runs remain unchanged"],"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (P/"rule_extraction_failure_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"guidelines":len(records),"counts":dict(total)}))
if __name__=="__main__":main()
