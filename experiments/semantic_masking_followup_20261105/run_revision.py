"""Twenty development nodes, requested checks, then automatic full continuation."""
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from common import CONFIG, DATA, LOGS, RESULTS, ROOT, read_rows, render, write_json
from predicates import evaluate, patient_for

def update(state, **kwargs):
    write_json(RESULTS/"run_status.json",dict(status=state,pid=os.getpid(),updated_at=time.time(),**kwargs))

def launch(script,args,log_name,gpu=None):
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu),OMP_NUM_THREADS="4",
        HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
    with (LOGS/log_name).open("a") as log:
        return subprocess.Popen([sys.executable,str(ROOT/script)]+args,cwd=ROOT,env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)

def run_pair(script,stage,prefix):
    extra=[]
    if prefix=="annotation":
        from annotation_queue import prepare_queue
        prepare_queue(stage)
        extra=["--shared-queue"]
    workers=[launch(script,["--stage",stage,"--shard",str(shard)]+extra,f"{prefix}_{stage}_{shard}.log",gpu)
             for shard,gpu in enumerate(CONFIG["gpus"])]
    update(prefix,stage=stage,worker_pids=[p.pid for p in workers],physical_gpus=CONFIG["gpus"],task_distribution="shared_queue" if prefix=="annotation" else "fixed_shards")
    codes=[p.wait() for p in workers]
    if any(codes):
        raise RuntimeError(f"{prefix} {stage} exit codes {codes}; details in logs/")

def verify_inputs(stage):
    inputs=read_rows(DATA/f"inputs_{stage}.jsonl")
    edits=read_rows(DATA/f"edits_{stage}.jsonl")
    specs={r["node_id"]:r for r in read_rows(DATA/"construction_specs.jsonl")}
    nodes={r["node_id"]:r for r in read_rows(DATA/"nodes.jsonl")}
    grouped={}
    for row in inputs:
        grouped.setdefault(row["example_id"],{})[row["view"]]=row
        if row["population"]=="constructed":
            spec=specs[row["node_id"]]
            expected,facts=patient_for(spec,row["metric_level"],row["background"])
            if expected!=row["patient"] or facts!=row["facts"] or row["gold"]!=int(evaluate(spec["predicate"],facts)):
                raise ValueError("Constructed input or complete predicate label mismatch: "+row["score_id"])
            if row["ancestors"]!=nodes[row["node_id"]]["ancestors"]:
                raise ValueError("Ancestor input changed")
    for meta in edits:
        change=meta["changes"];blocks=meta["blocks"]
        full=render(blocks);direct=render(blocks,deleted=change["deleted_block_ids"])
        if change["designated_management"] and full==direct:
            raise ValueError("Designated removal did not occur: "+meta["node_id"])
        for span in change["designated_management"]:
            if span["quote"] not in full:
                raise ValueError("Designated management absent from full context")
        for span in change["necessary_restores"]:
            block=next(b for b in blocks if b["block_id"]==span["block_id"])
            if span["quote"] not in block["text"] or span["block_id"] not in change["deleted_block_ids"]:
                raise ValueError("Restoration is not from its deleted source block")
        for group in grouped.values():
            if group["full"]["node_id"]==meta["node_id"]:
                if group["full"]["context"]!=full or group["direct"]["context"]!=direct:
                    raise ValueError("Input did not use designated edit")
    if stage=="pilot":
        pilot=[n for n in nodes.values() if n["pilot"]]
        if len(pilot)!=20 or any(n["split"]!="development" for n in pilot):
            raise ValueError("Pilot must contain exactly 20 development nodes")
        required={"2c44dbfd26d455b456beca2920543d6d:17","2c44dbfd26d455b456beca2920543d6d:74"}
        observed={r["node_id"] for r in inputs if r["population"]=="constructed"}
        if not required<=observed:
            raise ValueError("Age-range or compound-condition pilot repair missing: "+str(required-observed))
    write_json(RESULTS/f"verification_{stage}.json",dict(status="passed",nodes=len(edits),inputs=len(inputs),
        checked=["designated deletion occurs","source restoration positions","complete predicate labels",
                 "only selected indicator and nonclinical background vary","unchanged ancestor text"],
        constructed_nodes=len({r["node_id"] for r in inputs if r["population"]=="constructed"}),
        unchanged_inputs_reported=sum(not e["changes"]["direct_changed"] for e in edits)))

def estimate_after_pilot():
    annotations=[json.loads((RESULTS/f"annotation_pilot_{s}_status.json").read_text()) for s in (0,1)]
    scores=[json.loads((RESULTS/f"scoring_pilot_{s}_status.json").read_text()) for s in (0,1)]
    finished=sum(a.get("successful",0) for a in annotations)
    seconds=max(a.get("elapsed_seconds",0) for a in annotations)
    remaining=len(read_rows(DATA/"nodes.jsonl"))-finished
    from revision import all_annotations
    pilot_ids={n["node_id"] for n in read_rows(DATA/"nodes.jsonl") if n["pilot"]}
    records=[r for nid,r in all_annotations().items() if nid in pilot_ids]
    finished=len(records)
    remaining=len(read_rows(DATA/"nodes.jsonl"))-finished
    average=sum(r.get("annotation_seconds_for_projection",r["seconds"]) for r in records)/len(records) if records else None
    annotation_hours=remaining*average/len(CONFIG["gpus"])/3600 if average is not None else None
    score_total=sum(s.get("successful",0) for s in scores)
    score_seconds=max(s.get("elapsed_seconds",0) for s in scores)
    upper_hours=(CONFIG["score_budget"]/score_total*score_seconds/3600) if score_total else None
    estimate=dict(physical_gpus=CONFIG["gpus"],pilot_nodes=20,remaining_annotation_nodes=remaining,
        medgemma_peak_reserved_gib=[a.get("peak_reserved_gib") for a in annotations],
        qwen_peak_reserved_gib=[s.get("peak_reserved_gib") for s in scores],
        estimated_remaining_annotation_hours=annotation_hours,estimated_upper_scoring_hours=upper_hours,
        estimated_remaining_upper_hours=annotation_hours+upper_hours if annotation_hours is not None and upper_hours is not None else None,
        maximum_scores=CONFIG["score_budget"],initial_estimate_hours=[6,9],
        estimate_note="Pilot-based estimate written before automatic full launch; complexity varies.",
        launch_time=time.time())
    write_json(RESULTS/"runtime_estimate.json",estimate)
    write_json(RESULTS/"formal_launch_resources.json",estimate)

def run(stages=("pilot","full")):
    started=time.time()
    try:
        update("cpu_checks",stage="preparation")
        # Tests run only in the server environment; detailed output remains in a file.
        with (LOGS/"tests_controller.log").open("w") as log:
            subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-v"],
                           cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        from revision import build
        from analyze_revision import analyze, examples
        for stage in stages:
            done=RESULTS/f"run_{stage}_status.json"
            # Rebuild inputs on every restart; only the scoring worker may reuse an exact input.
            previous_stage=json.loads(done.read_text()) if done.exists() else {}
            first_started=previous_stage.get("started_at",time.time())
            write_json(done,dict(status="running",stage=stage,started_at=first_started,resumed_at=time.time()))
            run_pair("annotate_revision.py",stage,"annotation")
            update("building_inputs",stage=stage)
            summary=build(stage)
            verify_inputs(stage)
            run_pair("score.py",stage,"scoring")
            update("analyzing",stage=stage)
            metrics=analyze(stage)
            if stage=="pilot" and metrics["failed_score_count"]:
                raise RuntimeError("Pilot has missing/failed scores; full stage was not started")
            write_json(done,dict(status="complete",stage=stage,started_at=first_started,finished_at=time.time(),elapsed_seconds=time.time()-first_started,
                complete_score_count=metrics["complete_score_count"],failed_score_count=metrics["failed_score_count"]))
            if stage=="pilot":
                estimate_after_pilot()
                update("pilot_passed_starting_full",stage="full",pilot_verification="passed")
        update("complete",stage="full",elapsed_seconds=time.time()-started,
               report=str(RESULTS/"REPORT_full.zh.md"))
    except Exception as exc:
        traceback.print_exc()
        update("failed",error=str(exc),elapsed_seconds=time.time()-started)
        raise

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["all","pilot","full"],default="all")
    args=parser.parse_args()
    run(("pilot","full") if args.stage=="all" else (args.stage,))
