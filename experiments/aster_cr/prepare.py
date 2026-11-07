"""Reuse existing source correspondences; freeze original cases before model inference."""
import copy,random,re
from collections import defaultdict,Counter
from .common import *
from experiments.aster_representation.tree import tree_guideline
from experiments.aster_closed_form.data import ancestors
DATA=Path("/home/gcx/TRIAGE/data/MEGA-CDP")
CLUSTERS=Path("/home/gcx/aster_manuscript_20261030/clinical01/source_clusters.json")
def inventory():
 meta=read(CLUSTERS); rec=meta["records"]
 byfile=defaultdict(set)
 for key,r in rec.items():byfile[r["filename"]].add(r["cluster"])
 used=defaultdict(set);files=[]
 for key in ("pr40_clusters","previous_mechanism_clusters","official_test_clusters"):
  for c in meta.get(key,[]):used[c].add(str(CLUSTERS)+":"+key)
 roots=[p for p in Path("/home/gcx").iterdir() if p.is_dir() and (p.name.startswith(("aster_","gemma3_27b_","semantic_masking_","m1_diagnostics","medgemma_m1")))]
 names=("groups","nodes","inputs","views","cases","development","confirmation","train","accepted","triplet","patient")
 for root in roots:
  if root==ROOT:continue
  # Actual experiment input records only, not raw dataset copies or candidate catalogues.
  candidates=set()
  for folder in [root/"data",root/"clinical01"/"data",root/"run03"/"data"]:
   if folder.exists():
    candidates.update(p for p in folder.glob("*.jsonl") if any(t in p.name for t in names))
  for p in sorted(candidates):
   n=0
   with p.open() as f:
    for line in f:
     if not line.strip():continue
     r=json.loads(line);matched=set()
     for key in ("filename","source_filename","source_id","source_file"):
      val=str(r.get(key,""))
      matched.update(byfile.get(val,set()))
      for part,ident in re.findall(r"(train_val|test):(\d+)",val):
       if part+":"+ident in rec:matched.add(rec[part+":"+ident]["cluster"])
     for key in ("case_id","original_case_id","node_id","sample_id"):
      for part,ident in re.findall(r"(train_val|test):(\d+)",str(r.get(key,""))):
       if part+":"+ident in rec:matched.add(rec[part+":"+ident]["cluster"])
     if not matched and r.get("guideline_id") is not None:
      part="test" if "official_test" in str(p) else "train_val"
      key=part+":"+str(r["guideline_id"])
      if key in rec:matched.add(rec[key]["cluster"])
     for c in matched:used[c].add(str(p))
     n+=bool(matched)
   files.append(dict(path=str(p),matched_records=n))
 # Saved experimental selection: values in sources may contain the entire mapping,
 # so only IDs referenced by the actual split lists establish historical use.
 splitfile=REPO/"experiments/aster_representation/splits.json"; d=read(splitfile)
 for name,values in d.items():
  if name=="sources":continue
  for val in values:
   if isinstance(val,dict):val=val.get("original","")
   for part,ident in re.findall(r"(train_val|test):(\d+)",str(val)):
    key=part+":"+ident
    if key in rec:used[rec[key]["cluster"]].add(str(splitfile)+":"+name)
 return rec,used,files
def prepare():
 if (ROOT/"data/frozen.json").exists():return read(ROOT/"data/frozen.json")
 rec,used,files=inventory()
 trees=rows(DATA/"train_val/decision_trees.jsonl")
 raw={int(r["guideline_id"]):r for r in trees}
 paths={int(r["guideline_id"]):r["paths"] for r in rows(DATA/"train_val/case_paths.jsonl")}
 cases=defaultdict(list)
 for c in rows(DATA/"train_val/vignette_cases.jsonl"):cases[int(c["guideline_id"])].append(c)
 pool=defaultdict(list)
 for gid,t in raw.items():
  key="train_val:"+str(gid)
  if key not in rec:raise ValueError("Missing source correspondence "+key)
  source=rec[key]["cluster"]
  if source not in used and gid in paths and cases[gid]:pool[source].append(gid)
 rng=random.Random(SEED);sources=sorted(pool);rng.shuffle(sources)
 if len(sources)<140:raise ValueError("Only "+str(len(sources))+" eligible sources; cannot freeze 80/20/40.")
 splitmap={s:("fit" if i<80 else "dev" if i<100 else "test") for i,s in enumerate(sources[:140])}
 nodes=[];allcases=[];source_records=[]
 for source,split in splitmap.items():
  gids=pool[source];rng.shuffle(gids);candidates=[]
  for gid in gids:
   tr=raw[gid];tree=json.loads(tr["text"])["decision_tree"]
   try:guide,mapping=tree_guideline(tree)
   except (ValueError,KeyError) as exc:
    append(ROOT/"data/preparation_errors.jsonl",dict(guideline_id=gid,error=str(exc)));continue
   md=DATA/"train_val/guideline_md"/tr["filename"]/(tr["filename"]+".md")
   source_text=md.read_text(errors="replace") if md.exists() else ""
   # Keep all original source definitions in prompts. No semantic stripping at freeze time.
   defs=[]
   nodecases=defaultdict(list)
   for c in sorted(cases[gid],key=lambda x:int(x["path_id"])):
    p=paths[gid][int(c["path_id"])-1]
    gold=[dict(gid="G"+str(mapping[str(x["index"])]["node"]),label=int(x["decision_outcome"]=="yes")) for x in p if "decision_outcome" in x]
    case=dict(case_id=f"mega:train_val:{gid}:{c['path_id']}",source_id=source,split=split,
       guideline_id=gid,source_filename=tr["filename"],patient=c["text"],guideline=guide,definitions=defs,
       gold_path=gold,gold_leaf="L"+str(mapping[str(p[-1]["index"])]["node"]))
    allcases.append(case)
    for x in gold:nodecases[x["gid"]].append((case,x["label"]))
   for target,options in nodecases.items():
    # One original case per independent node; balanced labels within source by deterministic sampling.
    case,label=rng.choice(options)
    group=dict(group_id=f"new-{gid}-{target}",source_id=source,split=split,cohort="main",
       guideline_id=gid,source_filename=tr["filename"],language=tr["language"],
       gid=target,guideline=guide,definitions=defs,patient=case["patient"],case_id=case["case_id"],
       original_label=label)
    group["ancestors"]=ancestors(group)
    candidates.append(group)
  rng.shuffle(candidates)
  nodes.extend(candidates[:10])
  source_records.append(dict(source_id=source,split=split,guideline_ids=sorted(gids),available_nodes=len(candidates),selected_nodes=min(10,len(candidates))))
 lines(ROOT/"data/nodes.jsonl",nodes);lines(ROOT/"data/path_cases.jsonl",allcases)
 lines(ROOT/"data/sources.jsonl",source_records)
 write(ROOT/"data/history_usage.json",dict(correspondence=str(CLUSTERS),files=files,
      excluded={k:sorted(v) for k,v in sorted(used.items())},available_sources=len(pool)))
 cfg=dict(seed=SEED,gpus=GPUS,models=MODELS,source_targets=dict(fit=80,dev=20,test=40),
  node_target_per_source=10,bootstrap=2000,ranks=[4,8,16,"full"],regularization=[.001,.01,.1],
  loss_weights=[.1,1.,10.],h2=dict(require_r2_above=0.,maximum_leakage_rms_ratio=.1),
  dev_path_max_drop=.005,test_path_noninferiority_margin=.005,
  score_protocol="BF16 backbone, native immutable prefix cache, FP32 output label projection; no legacy/full-forward score mixing",
  grid=[-2,-1,-.5,0,.5,1,2],test_policy="No test model scoring before development configuration is frozen",
  history_policy="Reuse evidence and checkpoints; recompute only new input/template/model combinations")
 write(ROOT/"configuration.json",cfg)
 summary=dict(frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"),available_sources=len(pool),
  excluded_sources=len(used),sources=dict(Counter(splitmap.values())),nodes=dict(Counter(n["split"] for n in nodes)),
  original_paths=dict(Counter(c["split"] for c in allcases)),
  paired_targets=dict(test=dict(A=100,P=100,C=100,F=100),dev=dict(C=50)),
  source_correspondence=str(CLUSTERS),historical_originals="MEGA benchmark vignettes; not clinical prevalence",
  test_scored=False)
 write(ROOT/"data/frozen.json",summary)
 write(ROOT/"history_reuse.json",dict(no_rerun=[
  "/home/gcx/aster_existing_data_20261106","/home/gcx/aster_soft_angle_20261106",
  "/home/gcx/aster_reference_anchor_20261106","/home/gcx/aster_representation_20261101",
  "/home/gcx/m1_diagnostics_20261106","/home/gcx/medgemma_m1_repair_20261106"],
  learned_m3_report="/home/gcx/TRIAGE-worktrees/aster-m3-p0-p3-20261107/results/aster_m3_followup_20261107/summary.json"))
 status("prepare",state="complete",**summary)
 return summary
if __name__=="__main__":print(json.dumps(prepare(),ensure_ascii=False))
