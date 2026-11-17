"""Checkpoint identities and resumable extraction in private server storage."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from experiments.aster_cr_retrain_paired_v2_20261111.common import read, write, rows, append
from experiments.aster_e678_adaptive_20261116.run import SplitStore
from .prepare import REPO, OUT, PRIVATE
from .revision import head_array

SCORING_FILES=("experiments/aster_e678/engine.py","experiments/aster_cr/engine.py",
               "experiments/m1_diagnostics/engine.py")

def digest_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def checkpoint_identity(key, engine):
    cfg=read(OUT/"models"/key/"configuration.json")
    path=Path(cfg["model"]["path"])
    files=sorted(path.glob("*.safetensors"))
    if not files: raise ValueError("No safetensors checkpoint shards")
    manifest=[dict(name=p.name,bytes=p.stat().st_size,sha256=digest_file(p)) for p in files]
    meta=engine.describe()
    tokens=engine.tokenizer.backend_tokenizer.to_str()
    token_files=[p for p in path.iterdir() if p.is_file() and
                 (p.name.startswith(("tokenizer","special_tokens","added_tokens","vocab","merges")) or p.name=="config.json")]
    tokenizer_manifest=[dict(name=p.name,sha256=digest_file(p)) for p in sorted(token_files)]
    scoring=[dict(path=p,sha256=digest_file(REPO/p)) for p in SCORING_FILES]
    identity=dict(checkpoint_sha256=digest(manifest),weight_manifest=manifest,
                  tokenizer_sha256=digest(dict(serialized=tokens,files=tokenizer_manifest)),
                  tokenizer_files=tokenizer_manifest,scoring_sha256=digest(scoring),scoring_files=scoring,
                  metadata=meta,configuration_sha256=digest(cfg),
                  head_numbering="flattened layer * num_attention_heads + local query head",
                  protocol_sha256=digest_file(REPO/"experiments/aster_m1_revision_20261117/PROTOCOL.zh.md"))
    for rel in ("identity.json",):
        target=OUT/"models"/key/rel
        if target.exists() and read(target)!=identity: raise ValueError("Checkpoint/tokenizer/scoring identity changed")
        if not target.exists():write(target,identity)
    return identity

def input_identity(root, scope, inputs, ids, engine):
    records=[]
    for uid in ids:
        prompt=inputs[uid]["prompt"]
        rendered=engine.render(prompt)
        records.append(dict(uid=uid,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                            rendered_sha256=hashlib.sha256(rendered.encode()).hexdigest()))
    value=dict(scope=scope,count=len(ids),inputs_sha256=digest(records),records=records)
    path=root/("input_identity_"+scope+".json")
    if path.exists() and read(path)!=value:raise ValueError("Input identity changed: "+scope)
    if not path.exists():write(path,value)
    return value["inputs_sha256"]

class FeatureStore(SplitStore):
    def __init__(self, root, stages=("fit",)):
        for stage in stages:
            records=rows(Path(root)/"cache"/("scores_"+stage+".jsonl"))
            ids=[r.get("uid",r.get("view_id")) for r in records]
            if len(set(ids))!=len(ids):raise ValueError("Duplicate extraction records")
        super().__init__(root,stages)
    def validate(self, ids, gradient_ids, total_heads, head_dim):
        for uid in ids:
            record=self.get(uid)
            for field in (("heads","grads") if uid in gradient_ids else ("heads",)):
                if field not in record: raise ValueError("Missing "+field+": "+uid)
                if head_array(record,field).shape!=(total_heads,head_dim):raise ValueError("Feature geometry mismatch")
            if not np.isfinite(record["ell"]):raise ValueError("Nonfinite cached score")
        return dict(complete=True,inputs=len(ids),gradient_inputs=len(set(ids)&set(gradient_ids)),
                    total_heads=total_heads,head_dim=head_dim)

def extract(key, scope, inputs, engine, progress):
    import torch
    private=PRIVATE/key; root=OUT/"models"/key
    stages=read(OUT/"stage_ids.json")
    ids=stages[scope]
    if scope=="test" and not read(root/"frozen.json").get("frozen"):
        raise ValueError("Test extraction before frozen model manifest")
    gradients=set(read(root/"gradient_plan.json")["required_ids"]) if scope=="fit" else set()
    identity=input_identity(root,scope,inputs,ids,engine)
    store=FeatureStore(private,(scope,))
    expected=set(ids)
    if set(store.records)-expected:raise ValueError("Unexpected extraction records")
    pending=[uid for uid in ids if uid not in store.records]
    started=time.monotonic()
    for i,uid in enumerate(pending):
        context=engine.prepare(inputs[uid]["prompt"])
        result=engine.run(context,capture=True,grad=uid in gradients)
        arrays={field:result.pop(field).numpy().astype(np.float32)
                for field in ("heads","grads") if field in result}
        if not np.isfinite(result["ell"]) or not all(np.isfinite(a).all() for a in arrays.values()):
            raise ValueError("Nonfinite model feature: "+uid)
        feature=private/"cache/features"/(uid+".npz")
        feature.parent.mkdir(parents=True,exist_ok=True)
        if feature.exists():
            # A feature without its score is an interrupted write: preserve it.
            feature.rename(feature.with_name(feature.name+".interrupted."+str(time.time_ns())))
        np.savez_compressed(feature,**arrays)
        append(private/"cache"/("scores_"+scope+".jsonl"),
               dict(result,uid=uid,view_id=uid,feature=str(feature),status="ok",
                    has_grads=uid in gradients,available_heads="all"))
        del context,arrays
        if i<3 or (i+1)%100==0:
            progress("extract_"+scope,completed=len(ids)-len(pending)+i+1,total=len(ids),
                     seconds=time.monotonic()-started,peak_gib=torch.cuda.max_memory_allocated()/2**30)
    store=FeatureStore(private,(scope,))
    coverage=store.validate(ids,gradients,engine.num_heads,engine.head_dim)
    perf=dict(coverage,newly_computed=len(pending),reused_current_run=len(ids)-len(pending),
              seconds=time.monotonic()-started,peak_gib=torch.cuda.max_memory_allocated()/2**30,
              inputs_sha256=identity)
    write(root/"performance"/("extract_"+scope+".json"),perf)
    if scope=="fit":
        write(root/"gradient_completion.json",dict(complete=True,required_count=len(gradients),
              completed_count=len(gradients),missing_ids=[],zero_fill=False,feature_dtype="float32",
              data_identity=identity))
    return store,identity
