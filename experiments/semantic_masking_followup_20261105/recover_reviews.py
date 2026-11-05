"""Recover a complete earlier automatic response when a later formatting retry failed."""
import json
import os
import time
from common import DATA, LOGS, RESULTS, read_rows, write_json
from revision import normalize_annotation
from annotate import parse_json
from annotation_queue import locked, queue_path

def run():
    status=json.loads((RESULTS/"run_status.json").read_text())
    if status.get("status")!="annotation":
        return dict(recovered=[],note="Annotation phase already ended; no records changed")
    nodes={r["node_id"]:r for r in read_rows(DATA/"nodes.jsonl")}
    baseline={r["node_id"]:r for r in read_rows(DATA/"annotations_pilot_0.jsonl")}
    recovered=[]
    for shard in (0,1):
        path=DATA/f"annotations_full_{shard}.jsonl"
        latest={r["node_id"]:r for r in read_rows(path)}
        raw=read_rows(LOGS/f"review_raw_{shard}.jsonl")
        for nid,record in latest.items():
            if record.get("review_status")!="failed":
                continue
            valid=[]
            for entry in raw:
                if entry["node_id"]!=nid:
                    continue
                try:
                    value=parse_json(entry["text"])
                    fields=("source_condition_action_mixed","source_semantic_dependency","dataset_definition_completed")
                    if not all(value.get(k) in ("yes","no","unknown") for k in fields):
                        continue
                    if value.get("definition_links") is None and value["source_semantic_dependency"] in ("no","unknown"):
                        value["definition_links"]=[]
                    if isinstance(value.get("definition_links"),list):
                        valid.append((entry["attempt"],value))
                except (ValueError,TypeError):
                    pass
            if not valid:
                continue
            attempt,response=valid[-1]
            value=dict(baseline[nid]["annotation"])
            value.update({k:response.get(k) for k in ("source_condition_action_mixed",
                "source_semantic_dependency","dataset_definition_completed","dependency_reason","definition_links")})
            value["missing_reference_queries"]=response.get("missing_reference_queries") or []
            value["semantic_review_version"]=3
            if response.get("rule_evidence"):
                value["rule_evidence"]=response["rule_evidence"]
            replacement=normalize_annotation(nodes[nid],value)
            replacement.update({k:v for k,v in record.items() if k not in
                ("annotation","blocks","validation","unresolved_references","review_status")})
            replacement.update(review_status="complete",
                review_recovery="Recovered complete earlier automatic JSON; null empty list normalized for no/unknown",
                recovered_attempt=attempt,recovered_at=time.time())
            encoded=(json.dumps(replacement,ensure_ascii=False,allow_nan=False)+"\n").encode()
            handle=os.open(path,os.O_WRONLY|os.O_APPEND)
            try:
                os.write(handle,encoded)
            finally:
                os.close(handle)
            with locked(queue_path("full")):
                queue=json.loads(queue_path("full").read_text())
                queue["finished"][nid]["status"]="recovered_from_valid_response"
                write_json(queue_path("full"),queue)
            recovered.append(dict(node_id=nid,dependency=replacement["annotation"]["source_semantic_dependency"],
                restores=len(replacement["annotation"]["necessary_restores"])))
    result=dict(recovered=recovered,created_at=time.time(),new_inference=0,
        note="Only complete recorded model responses are used; original failures remain in append-only logs")
    write_json(RESULTS/"format_recovery_checkpoint.json",result)
    return result

if __name__=="__main__":
    print(json.dumps(run(),ensure_ascii=False))
