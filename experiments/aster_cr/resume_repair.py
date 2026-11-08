"""Continue repair-v2 with isolated completion markers and compatible Full caches."""
import argparse,fcntl,time,shutil,traceback
from collections import Counter
import numpy as np
from .common import *
REPAIR=Path("/home/gcx/aster_cr_repair_20261108")
OLD=Path("/home/gcx/aster_cr_20261108")
def prepare():
 if read(ROOT/"preflight.json").get("complete"):return
 if ROOT in (OLD,REPAIR):raise ValueError("Use a separate downstream output directory")
 if any((ROOT/"jobs").glob("*.json")):raise ValueError("Prepare requires a fresh job directory")
 (ROOT/"logs").mkdir(parents=True,exist_ok=True)
 frozen=read(REPAIR/"data/frozen.json");assert frozen["version"]=="repair-v2" and frozen["construction_complete"]
 views=rows(REPAIR/"data/views.jsonl");pairs=rows(REPAIR/"data/pairs.jsonl");nodes=rows(REPAIR/"data/nodes.jsonl")
 assert (len(nodes),len(views),len(pairs))==(2782,11512,8730)
 index={v["view_id"]:v for v in views};assert len(index)==len(views)
 for pair in pairs:
  a,b=index[pair["left"]],index[pair["right"]]
  assert a["split"]==b["split"]==pair["split"]
  assert a["label"]==pair["left_label"] and b["label"]==pair["right_label"]
  assert a["kind"] in TASK_KINDS and b["kind"] in TASK_KINDS
 original=dict(Counter(v["split"] for v in views if v["variant"]=="full" and v["original"]))
 cases=rows(REPAIR/"data/path_cases.jsonl");path_counts=dict(Counter(c["split"] for c in cases))
 assert original==frozen["original_nodes"] and path_counts==frozen["original_paths"]
 assert cases==rows(OLD/"data/path_cases.jsonl")
 # Preserve repaired inputs exactly; interface changes live in downstream consumers.
 shutil.copytree(REPAIR/"data",ROOT/"data",dirs_exist_ok=True)
 shutil.copytree(REPAIR/"external",ROOT/"external",dirs_exist_ok=True)
 for name in ("history_usage.json",):
  if (OLD/"data"/name).exists():shutil.copy2(OLD/"data"/name,ROOT/"data"/name)
 frozen.update(downstream="ready",sources=frozen["source_splits"],nodes=dict(Counter(n["split"] for n in nodes)),data_source=str(REPAIR))
 write(ROOT/"data/frozen.json",frozen)
 if not (ROOT/"vendor").exists():(ROOT/"vendor").symlink_to(OLD/"vendor",target_is_directory=True)
 cfg=read(OLD/"configuration.json");cfg.update(gpus=GPUS,data_version="repair-v2",models=MODELS)
 write(ROOT/"configuration.json",cfg)
 oldviews={v["view_id"]:v for v in rows(OLD/"data/views.jsonl")}
 reusable=set(read(REPAIR/"cache_reuse.json")["reusable_view_ids"])
 compatible={v["view_id"] for v in views if v["view_id"] in reusable and v["variant"]=="full" and all(v.get(k)==oldviews[v["view_id"]].get(k) for k in ("prompt","labels","label","split"))}
 assert len(compatible)==1332
 reuse={};feature_reads=0
 for model in MODELS:
  reuse[model]={}
  for split in ("fit","dev","test"):
   # Test records, if any, are held aside until this run freezes configuration.
   records={r["view_id"]:r for f in (OLD/model/"scores").glob(split+"_*.jsonl") for r in rows(f) if r.get("status")=="ok" and r["view_id"] in compatible}
   accepted=[]
   for r in records.values():
    feature=Path(r["feature"])
    if not feature.exists():continue
    with np.load(feature) as a:
     needed={"heads","residual"}|({"grads"} if split=="fit" else set())
     if not needed<=set(a.files):continue
    feature_reads+=1;accepted.append(dict(r,reused_from_run=str(OLD)))
   groups=list(dict.fromkeys(v["group_id"] for v in views if v["split"]==split))
   shards=len(GPUS) if model=="qwen" else 1
   for shard in range(shards):
    owned=set(groups[shard::shards]);rr=[r for r in accepted if r["group_id"] in owned]
    target=ROOT/model/("pending_test_scores" if split=="test" else "scores")/(split+"_"+str(shard)+".jsonl")
    lines(target,rr)
   reuse[model][split+"_full_scores"]=len(accepted)
   path_records={r["id"]:r for f in (OLD/model/"paths").glob("features_"+split+"_*.jsonl") for r in rows(f) if r.get("status")=="ok" and Path(r["feature"]).exists()}
   selected=[c for c in cases if c["split"]==split];case_ids={c["case_id"] for c in selected}
   assert all(r["case_id"] in case_ids for r in path_records.values())
   for shard in range(shards):
    owned={c["case_id"] for c in selected[shard::shards]}
    lines(ROOT/model/("pending_test_paths" if split=="test" else "paths")/("features_"+split+"_"+str(shard)+".jsonl"),[dict(r,reused_from_run=str(OLD)) for r in path_records.values() if r["case_id"] in owned])
   reuse[model][split+"_path_features"]=len(path_records)
 # No old fitted models, localizations, selected settings, status or job markers are imported.
 write(ROOT/"cache_reuse.json",dict(source=str(OLD),full_inputs_identical=len(compatible),counts=reuse,
  protocol=cfg["score_protocol"],engine="experiments.aster_cr.engine.Engine (unchanged)",models=MODELS,
  recompute=["changed scores and features","localization","P0","H2/H3","readouts","training controls","development selection"],old_markers_copied=False))
 write(ROOT/"preflight.json",dict(complete=True,data_version="repair-v2",nodes=len(nodes),views=len(views),pairs=len(pairs),
  original_nodes=original,original_paths=path_counts,all_pair_endpoints_supported=True,path_inputs_equal=True,feature_files_checked=feature_reads,
  external_training=False,test_requires_new_freeze=True,cache_counts=reuse))
 status("data_interface",state="complete",version="repair-v2")
def release_test_cache(model):
 if not read(ROOT/model/"frozen.json").get("frozen"):raise ValueError("Test configuration is not frozen")
 for pending,target in (("pending_test_scores","scores"),("pending_test_paths","paths")):
  for p in (ROOT/model/pending).glob("*.jsonl"):
   dest=ROOT/model/target/p.name
   dest.parent.mkdir(parents=True,exist_ok=True)
   if not dest.exists():shutil.copy2(p,dest)
def run(resume_training=False):
 from .controller import phase
 from .pipeline import downstream,run_jobs,shard_jobs
 from .report import report
 from .external_score import summarize
 prepare();started=time.time()
 with (ROOT/"controller.lock").open("a") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   if not resume_training:
    for split in ("fit","dev"):
     run_jobs(phase,"qwen_"+split,shard_jobs("experiments.aster_cr.score",split))
    status("preparation",state="complete",data_version="repair-v2")
   downstream(phase,final_report=False,resume_training=resume_training)
   for model in MODELS:
    jobs=[["experiments.aster_cr.external_score","--model",model,"--dataset",dataset] for dataset in ("nli4ct","medpic")]
    if model=="medgemma":
     for args in jobs:phase("external_medgemma_"+args[-1],[(GPUS[-1],args)])
    else:run_jobs(phase,"external_"+model,jobs)
    summarize(model)
   status("external",state="complete",datasets=["nli4ct","medpic"],included_in_mega=False)
   status("pipeline",state="complete",seconds=time.time()-started)
   report(final=True)
   with (ROOT/"REPORT.zh.md").open("a") as f:
    f.write("\n## 独立外部评价\n\nNLI4CT 与 MedPIC 保持原任务标签和评分；不参与 MEGA 训练、选参及宏平均。\n")
    for model in MODELS:
     for dataset in ("nli4ct","medpic"):
      r=read(ROOT/model/"external"/dataset/"metrics.json")
      f.write(f"\n{model} / {dataset}：{r['items']} 题，准确率 {r['exact_accuracy']:.4f}，缺失或无效 {r['missing_or_invalid']}。\n")
   write(ROOT/"runtime.json",dict(state="complete",seconds=time.time()-started))
  except BaseException as exc:
   status("pipeline",state="failed",error=str(exc),traceback=traceback.format_exc())
   try:report(final=False)
   except Exception:pass
   raise
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--prepare-only",action="store_true");p.add_argument("--resume-training",action="store_true");a=p.parse_args()
 prepare() if a.prepare_only else run(a.resume_training)
