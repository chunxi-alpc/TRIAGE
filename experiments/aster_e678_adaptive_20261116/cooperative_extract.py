"""Let queue-assigned GPUs share one checkpoint's raw extraction work."""
import argparse
import fcntl
import os
import time
from collections import ChainMap
from pathlib import Path
import numpy as np
from experiments.aster_cr_retrain_paired_v2_20261111.common import read, write, rows, append, load_inputs
from experiments.aster_e678.run import available
from .prepare import OUT, SOURCE, KEYS


def extract(root, stage, inputs, engine, reverse=False, worker="main"):
    root = Path(root)
    if stage not in ("fit", "dev", "test"):
        raise ValueError(stage)
    if stage == "test" and not read(root / "frozen.json").get("frozen"):
        raise ValueError("Test feature extraction before parameter freeze")
    ids = read(root / "stage_ids.json")[stage]
    gradients = set(read(root / "gradient_ids.json", [])) if stage == "fit" else set()
    cache = root / "cache"
    coordination = cache / "cooperative" / stage
    coordination.mkdir(parents=True, exist_ok=True)
    scorefile = cache / ("scores_" + stage + ".jsonl")
    lockfile = coordination / "scores.lock"
    with lockfile.open("a") as scorelock:
        fcntl.flock(scorelock, fcntl.LOCK_EX)
        existing = {r["uid"]: r for r in rows(scorefile)}
    pending = [uid for uid in ids if not available(existing.get(uid), uid in gradients)]
    if reverse:
        pending.reverse()
    computed = skipped = 0
    started = time.monotonic()
    statuspath = root / "performance" / ("cooperative_" + stage + "_" + worker + ".json")
    def status(complete=False):
        write(statuspath, dict(complete=complete, stage=stage, worker=worker, pid=os.getpid(),
              gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), newly_computed=computed,
              reused_at_start=len(ids)-len(pending), completed_by_other_worker=skipped,
              total=len(ids), seconds=time.monotonic()-started))
    status()
    for uid in pending:
        marker = coordination / (uid + ".json")
        with (coordination / (uid + ".lock")).open("a") as claim:
            fcntl.flock(claim, fcntl.LOCK_EX)
            if available(read(marker), uid in gradients):
                skipped += 1
                continue
            context = engine.prepare(inputs[uid]["prompt"])
            result = engine.run(context, capture=True, grad=uid in gradients)
            arrays = {"heads": result.pop("heads").numpy().astype(np.float16)}
            if "grads" in result:
                arrays["grads"] = result.pop("grads").numpy().astype(np.float16)
            target = cache / "features" / (uid + ".npz")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.stem + "." + str(os.getpid()) + ".tmp.npz")
            np.savez_compressed(temporary, **arrays)
            temporary.replace(target)
            record = dict(result, uid=uid, view_id=uid, feature=str(target),
                          status="ok", available_heads="all", has_grads=uid in gradients)
            with lockfile.open("a") as scorelock:
                fcntl.flock(scorelock, fcntl.LOCK_EX)
                append(scorefile, record)
                write(marker, record)
            computed += 1
            del context
        if computed <= 3 or computed % 50 == 0:
            status()
    status(complete=True)
    return read(statuspath)


def main():
    from .run import require_queue, freeze_identity
    require_queue()
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=KEYS, required=True)
    p.add_argument("--stage", choices=("dev",), default="dev")
    p.add_argument("--worker", default="helper1")
    args = p.parse_args()
    root = OUT / "models" / args.model
    inputs = ChainMap({r["uid"]: r for r in rows(root / "data/extra_inputs.jsonl")}, load_inputs(SOURCE))
    from experiments.aster_e678.engine import Engine
    engine = Engine(args.model, metadata_path=root / "performance" / (args.worker + "_model_metadata.json"))
    freeze_identity(root, engine, inputs, read(root / "stage_ids.json")[args.stage])
    result = extract(root, args.stage, inputs, engine, reverse=True, worker=args.worker)
    write(root / "performance" / (args.worker + "_done.json"), result)


if __name__ == "__main__":
    main()
