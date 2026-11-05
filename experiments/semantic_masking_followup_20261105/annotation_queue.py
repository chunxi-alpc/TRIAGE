"""Share pending annotation nodes between GPUs using a small locked work queue."""
import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from common import DATA, RESULTS, read_rows, write_json

def pending_ids(nodes, previous):
    return [n["node_id"] for n in nodes if not (
        previous.get(n["node_id"],{}).get("management_extraction_completed") and
        previous.get(n["node_id"],{}).get("semantic_schema")==2)]

def queue_path(stage):
    return RESULTS/f"annotation_queue_{stage}.json"

@contextmanager
def locked(path):
    path=Path(path)
    with path.with_suffix(".lock").open("a") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle,fcntl.LOCK_UN)

def prepare_queue(stage):
    from revision import all_annotations
    nodes=[n for n in read_rows(DATA/"nodes.jsonl") if stage=="full" or n["pilot"]]
    ids=pending_ids(nodes,all_annotations())
    path=queue_path(stage)
    with locked(path):
        write_json(path,dict(stage=stage,prepared_at=time.time(),total=len(ids),
            completed_before=len(nodes)-len(ids),pending=ids,claims={},finished={}))
    return ids

def claim(path, owner):
    path=Path(path)
    with locked(path):
        state=json.loads(path.read_text())
        if not state["pending"]:
            return None
        node_id=state["pending"].pop(0)
        state["claims"][node_id]=dict(owner=owner,started_at=time.time())
        write_json(path,state)
        return node_id

def finish(path, node_id, status):
    path=Path(path)
    with locked(path):
        state=json.loads(path.read_text())
        item=state["claims"].pop(node_id)
        state["finished"][node_id]=dict(item,status=status,finished_at=time.time())
        write_json(path,state)

def remaining(path):
    with locked(path):
        return len(json.loads(Path(path).read_text())["pending"])

def nodes_for(stage, nodes, first=None):
    by_id={n["node_id"]:n for n in nodes}
    node_id=first
    owner=dict(pid=os.getpid(),physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
    while True:
        if node_id is None:
            node_id=claim(queue_path(stage),owner)
        if node_id is None:
            return
        yield by_id[node_id]
        node_id=None
