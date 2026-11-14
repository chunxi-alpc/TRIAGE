"""Run only unfinished prerequisites and absent transfer evaluations with existing E1 parameters."""
import argparse
import contextlib
import fcntl
import os
import time
import traceback
from collections import ChainMap
from .prepare import OUT, SOURCE
from .evaluate import DynamicStore, evaluate_model
from . import run as source_run
from experiments.aster_cr_retrain_paired_v2_20261111.common import read, write, rows, load_inputs

STAGES = ("extract-dev", "freeze", "extract-transfer", "evaluate-transfer")
TARGETS = ("nli4ct_official5500", "medpic", "medpic_official467")
CPU_TARGETS = ("nli4ct",) + TARGETS

def plan_for(key):
    policy = read(OUT / "only_missing_policy.json")
    plan = policy.get("missing_transfer_jobs", {}).get(key)
    if not plan or plan.get("scopes") != list(TARGETS) or plan.get("cpu_only_scopes") != ["nli4ct"]:
        raise RuntimeError("No approved missing-transfer entry for this checkpoint")
    return plan

def validated_root(key):
    plan = plan_for(key)
    root = OUT / "models" / key
    configuration = read(root / "configuration.json")
    if configuration.get("source_protocol") != "E1-balanced-2540" or configuration.get("fit_pairs") != 2540:
        raise RuntimeError("Source configuration differs from the fixed E1 protocol")
    if not read(root / "models/E1/full/model.json").get("complete"):
        raise RuntimeError("Existing completed E1 fit required; this worker never fits parameters")
    if not (root / "models/E1/full/calibration.json").is_file():
        raise RuntimeError("Existing source calibration required")
    return root, configuration, plan

def progress(root, stage, **values):
    write(root / "missing_transfer/status.json", dict(
        state="running", stage=stage, pid=os.getpid(),
        physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        time=time.strftime("%Y-%m-%d %H:%M:%S"), **values))

def pending_ids(root, stage):
    store = DynamicStore(root)
    return [uid for uid in read(root / "stage_ids.json")[stage]
            if not source_run.available(store.records.get(uid), False)]

def extract_missing(root, key, stages):
    needed = [stage for stage in stages if pending_ids(root, stage)]
    if not needed:
        return
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if gpu not in ("0", "1", "4", "5", "6"):
        raise RuntimeError("Exactly one authorized physical GPU must be assigned")
    import torch
    torch.set_num_threads(4)
    from .engine import Engine
    extra = {row["uid"]: row for row in rows(read(root / "extra_inputs.json")["path"])}
    inputs = ChainMap(extra, load_inputs(SOURCE))
    progress(root, "loading_model")
    engine = Engine(key, metadata_path=root / "missing_transfer/model_metadata.json")
    source_run.state = progress
    for stage in stages:
        if stage != "dev" and read(root / "evaluation" / stage / "metrics.json").get("complete"):
            continue
        if pending_ids(root, stage):
            source_run.extract(root, stage, inputs, engine)
    del engine

def freeze_existing(root, key, configuration):
    if read(root / "frozen.json").get("frozen"):
        return
    if pending_ids(root, "dev"):
        raise RuntimeError("Source development extraction is incomplete")
    progress(root, "source_development_evaluation")
    development = evaluate_model(root, scopes=("dev",))
    write(root / "frozen.json", dict(
        frozen=True, model_key=key, source_protocol="E1-balanced-2540",
        configuration=configuration, development=development,
        selection_split="source development only",
        selection="fixed E1 head/rank/regularization/budget; independently fitted checkpoint parameters",
        parameter_directory=str(root / "models/E1/full"),
        time=time.strftime("%Y-%m-%d %H:%M:%S"),
        target_test_labels_used_for_fit=False,
        target_labels_used_for_calibration=False,
        completion_mode="existing E1 fit; unfinished development evaluation only"))

def execute(key, stage):
    root, configuration, plan = validated_root(key)
    marker = root / "missing_transfer" / (stage + ".json")
    if read(marker).get("complete"):
        return
    if stage == "extract-dev":
        if not read(root / "frozen.json").get("frozen"):
            extract_missing(root, key, ("dev",))
    elif stage == "freeze":
        freeze_existing(root, key, configuration)
    elif stage in ("extract-transfer", "evaluate-transfer"):
        if not read(root / "frozen.json").get("frozen"):
            raise RuntimeError("Source parameters must be frozen before transfer evaluation")
        requested = CPU_TARGETS if stage == "evaluate-transfer" else tuple(plan["scopes"])
        missing = tuple(scope for scope in requested
                        if not read(root / "evaluation" / scope / "metrics.json").get("complete"))
        if stage == "extract-transfer":
            extract_missing(root, key, missing)
        elif missing:
            if any(pending_ids(root, scope) for scope in missing):
                raise RuntimeError("Missing target features; do not evaluate incomplete populations")
            for scope in missing:
                progress(root, "evaluate_" + scope)
                evaluate_model(root, scopes=(scope,))
    else:
        raise ValueError(stage)
    write(marker, dict(complete=True, model_key=key, stage=stage,
                      protocol="E1-balanced-2540", existing_fit_reused=True,
                      new_training=False, scopes=plan["scopes"], cpu_only_scopes=plan["cpu_only_scopes"],
                      completed_at=time.strftime("%Y-%m-%d %H:%M:%S")))
    write(root / "missing_transfer/status.json", dict(
        state="transfer_complete" if stage == "evaluate-transfer" else "stage_complete",
        stage=stage, pid=os.getpid(), physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        time=time.strftime("%Y-%m-%d %H:%M:%S")))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("gemma3", "medgemma"), required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    args = parser.parse_args()
    root = OUT / "models" / args.model
    work = root / "missing_transfer"
    work.mkdir(exist_ok=True)
    with (root / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (work / (args.stage + ".log")).open("a", buffering=1) as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                try:
                    execute(args.model, args.stage)
                except Exception as error:
                    write(work / "status.json", dict(state="error", stage=args.stage,
                          error=str(error), traceback=traceback.format_exc()))
                    raise
                finally:
                    with (OUT / ".report.lock").open("a") as report_lock:
                        fcntl.flock(report_lock, fcntl.LOCK_EX)
                        from .missing_only_report import generate
                        generate()
if __name__ == "__main__":
    main()
