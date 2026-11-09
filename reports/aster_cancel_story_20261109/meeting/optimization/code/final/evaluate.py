
"""Evaluate frozen score arrays on the existing held-out Qwen test set.

No fitting or model selection takes place here. Importing this module does not
read evaluation labels or compute metrics. Run only after configurations freeze.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
os.environ.setdefault("OMP_NUM_THREADS","2")
import argparse,csv,json,sys,time
from pathlib import Path
from collections import defaultdict
import numpy as np

OLD=Path("/home/gcx/aster_cancel_story_20261109/meeting/real_completion")
ROOT=Path("/home/gcx/aster_cr_repair_v2_run_20261108")
SOURCE_REPO=Path("/home/gcx/TRIAGE-worktrees/aster-cr-20261108")
DEFAULT_BASELINES=("raw","no_w","linear_output","closed_form","leace","iti",
 "ordinary_r8","ordinary_matched","last_layer","loreft","lora","lofit",
 "mlp_output_seed11","mlp_output_seed22","mlp_output_seed33",
 "linear_representation_seed11","mlp_representation_seed11")
GROUPS=(("A",("A",)),("P",("P",)),("AP",("A","P")),("C",("C",)),
 ("F",("F",)),("CF",("C","F")),("F0",("F0",)))

def rows(path):
 path=Path(path)
 return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []

def dump(path,value):
 Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")

def finite_float(x):
 x=float(x)
 return x if np.isfinite(x) else None

def percentile_ci(values):
 values=np.asarray(values,dtype=float)
 values=values[np.isfinite(values)]
 return np.quantile(values,[.025,.975]).tolist() if len(values) else None

def source_aggregate(values,source_index,n_sources):
 values=np.asarray(values,dtype=float)
 if values.ndim==1:values=values[:,None]
 out=np.zeros((n_sources,values.shape[1]),dtype=float)
 np.add.at(out,np.asarray(source_index,dtype=int),values)
 return out

def cluster_mean(values,source_index,draws):
 values=np.asarray(values,dtype=float)
 if not len(values):return None,np.full(len(draws),np.nan)
 ag=source_aggregate(np.column_stack([np.ones(len(values)),values]),source_index,draws.shape[1])
 z=draws@ag
 with np.errstate(divide="ignore",invalid="ignore"):rep=z[:,1]/z[:,0]
 return float(np.mean(values)),rep

def pair_statistics(score,raw,descriptor,draws,raw_ap_rms,raw_ap_boot,selected=None):
 """Return exact point metrics and aligned source-bootstrap replicates."""
 li,ri,yl,yr,source_index=descriptor
 valid=np.isfinite(score[li])&np.isfinite(score[ri])
 if selected is not None:valid&=selected
 expected=len(li)
 li=li[valid];ri=ri[valid];yl=yl[valid];yr=yr[valid];ss=source_index[valid]
 if not len(li):return dict(n=0,expected=expected,missing=expected,sources=0),{},valid
 dy0=raw[ri]-raw[li];dy=score[ri]-score[li];dc=dy-dy0;direction=yr-yl
 signed0=direction*dy0;signed=direction*dy
 cor=((score[li]>0)==yl)&((score[ri]>0)==yr)
 degraded=(signed0>0)&(signed<=0)
 improved=(signed0<=0)&(signed>0)
 values=np.column_stack([np.ones(len(li)),dy0,dy0**2,dy,dy**2,dc**2,
  signed0,signed,cor.astype(float),degraded.astype(float)])
 z=draws@source_aggregate(values,ss,draws.shape[1])
 n=z[:,0]
 with np.errstate(divide="ignore",invalid="ignore"):
  brms=np.sqrt(z[:,4]/n);bcor=np.sqrt(z[:,5]/n)
  bvr=1-(z[:,4]/n-(z[:,3]/n)**2)/(z[:,2]/n-(z[:,1]/n)**2)
  ber=z[:,7]/z[:,6]
  bpa=z[:,8]/n;bdeg=z[:,9]/n
  brat=bcor/raw_ap_boot
 reps=dict(difference_rms=brms,correction_rms=bcor,
  correction_rms_over_ap_rms=brat,paired_accuracy=bpa,
  response_direction_degraded_fraction=bdeg)
 point=dict(n=len(li),expected=expected,missing=expected-len(li),sources=len(np.unique(ss)),
  raw_difference_rms=float(np.sqrt(np.mean(dy0**2))),
  difference_rms=float(np.sqrt(np.mean(dy**2))),difference_mean=float(np.mean(dy)),
  correction_rms=float(np.sqrt(np.mean(dc**2))),
  correction_rms_over_ap_rms=float(np.sqrt(np.mean(dc**2))/raw_ap_rms),
  correction_abs_max=float(np.max(np.abs(dc))),correction_abs_p95=float(np.quantile(np.abs(dc),.95)),
  paired_correct=int(cor.sum()),paired_accuracy=float(cor.mean()),
  response_direction_degraded=int(degraded.sum()),
  response_direction_degraded_fraction=float(degraded.mean()),
  response_direction_improved=int(improved.sum()))
 if np.any(direction):
  point["mean_evidence_retention"]=finite_float(np.mean(signed)/np.mean(signed0))
  reps["mean_evidence_retention"]=ber
 else:
  point["variance_reduction"]=finite_float(1-np.var(dy)/np.var(dy0))
  reps["variance_reduction"]=bvr
 for key,rep in reps.items():point[key+"_ci"]=percentile_ci(rep)
 return point,reps,valid

def accuracy_statistics(score,indices,labels,source_index,draws,selected=None):
 valid=np.isfinite(score[indices])
 if selected is not None:valid&=selected
 ii=indices[valid];yy=labels[valid];ss=source_index[valid]
 correct=((score[ii]>0)==yy).astype(float)
 value,rep=cluster_mean(correct,ss,draws)
 return dict(n=len(ii),expected=len(indices),missing=len(indices)-len(ii),
  sources=len(np.unique(ss)),correct=int(correct.sum()),accuracy=value,accuracy_ci=percentile_ci(rep)),dict(accuracy=rep),valid

def metric_differences(point,reps,reference_point,reference_reps):
 out={}
 for key,rep in reps.items():
  if key not in reference_reps:continue
  left=point.get(key);right=reference_point.get(key)
  out[key]=dict(difference=finite_float(left-right) if left is not None and right is not None else None,
   ci=percentile_ci(rep-reference_reps[key]))
 return out

def load_score_arrays(scores_path,include_baselines=True):
 with np.load(scores_path,allow_pickle=False) as a:
  if "ids" not in a:raise ValueError("scores npz requires ids")
  ids=a["ids"].tolist()
  scores={key:np.asarray(a[key],dtype=float) for key in a.files if key!="ids" and a[key].shape==(len(ids),)}
 if not scores:raise ValueError("scores npz contains no score arrays matching ids")
 with np.load(OLD/"coordinates.npz",allow_pickle=False) as a:
  expected=a["ids"].tolist();raw=a["ell"].astype(float)
 if ids!=expected:raise ValueError("ids must exactly match the existing 28186-row coordinate cache order")
 provenance={name:"provided frozen score array" for name in scores}
 fallback=set()
 with np.load(OLD/"evaluated_scores.npz",allow_pickle=False) as legacy:
  if legacy["ids"].tolist()!=ids:raise ValueError("legacy baseline ids do not align")
  if include_baselines:
   for name in DEFAULT_BASELINES:
    if name not in scores and name in legacy:
     scores[name]=legacy[name].astype(float)
     provenance[name]="existing frozen baseline scores"
  for name,value in scores.items():
   if name in legacy and np.array_equal(value,legacy[name],equal_nan=True):
    fallback.add(name)
 # Reference names are always present; do not use a newly supplied method with an
 # ambiguous reference name to redefine the prior raw/linear comparison.
 for required in ("raw","linear_output"):
  if required not in scores:
   with np.load(OLD/"evaluated_scores.npz",allow_pickle=False) as legacy:scores[required]=legacy[required].astype(float)
   provenance[required]="existing frozen reference scores";fallback.add(required)
  else:
   with np.load(OLD/"evaluated_scores.npz",allow_pickle=False) as legacy:
    if not np.array_equal(scores[required],legacy[required],equal_nan=True):
     raise ValueError("reference name "+required+" must retain its existing frozen scores")
 return ids,scores,raw,provenance,fallback

def evaluate_frozen_scores(scores_path,out_dir,include_baselines=True,replicates=2000,seed=11,selection_path=None):
 started=time.time();out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
 ids,scores,raw,provenance,legacy_fallback=load_score_arrays(scores_path,include_baselines)
 ix={v:i for i,v in enumerate(ids)}
 meta=json.loads((OLD/"metadata.json").read_text())
 test_views=[v for v in meta["views"] if v["split"]=="test"]
 test_pairs=[p for p in meta["pairs"] if p["split"]=="test"]
 sources=sorted({v["source_id"] for v in test_views});sx={s:i for i,s in enumerate(sources)}
 rng=np.random.default_rng(seed)
 draws=rng.multinomial(len(sources),np.full(len(sources),1/len(sources)),size=replicates).astype(float)
 pp_ap=[p for p in test_pairs if p["kind"] in ("A","P")]
 aps=np.array([raw[ix[p["right"]]]-raw[ix[p["left"]]] for p in pp_ap])
 ap_source=np.array([sx[p["source_id"]] for p in pp_ap])
 ag=source_aggregate(np.column_stack([np.ones(len(aps)),aps**2]),ap_source,len(sources))
 ab=draws@ag
 with np.errstate(divide="ignore",invalid="ignore"):raw_ap_boot=np.sqrt(ab[:,1]/ab[:,0])
 raw_ap_rms=float(np.sqrt(np.mean(aps**2)))
 descriptors={}
 group_pairs={}
 for group,kinds in GROUPS:
  pp=[p for p in test_pairs if p["kind"] in kinds];group_pairs[group]=pp
  descriptors[group]=(np.array([ix[p["left"]] for p in pp],dtype=int),
   np.array([ix[p["right"]] for p in pp],dtype=int),
   np.array([p["left_label"] for p in pp],dtype=int),
   np.array([p["right_label"] for p in pp],dtype=int),
   np.array([sx[p["source_id"]] for p in pp],dtype=int))
 natural=[v for v in test_views if v["variant"]=="full" and v["original"]]
 ni=np.array([ix[v["view_id"]] for v in natural],dtype=int)
 ny=np.array([v["label"] for v in natural],dtype=int)
 ns=np.array([sx[v["source_id"]] for v in natural],dtype=int)
 metrics={};rep_cache={};valid_cache={}
 pair_file=out/"test_pair_results.csv"
 pair_fields=("method","pair_id","kind","source_id","left_label","right_label",
  "left_score","right_score","original_difference","corrected_difference","correction_difference","paired_correct")
 with pair_file.open("w",newline="",encoding="utf-8") as handle:
  writer=csv.DictWriter(handle,fieldnames=pair_fields);writer.writeheader()
  for name,score in scores.items():
   groups={}
   for group,descriptor in descriptors.items():
    point,reps,valid=pair_statistics(score,raw,descriptor,draws,raw_ap_rms,raw_ap_boot)
    groups[group]=point;rep_cache[name,group]=reps;valid_cache[name,group]=valid
    if group in ("AP","CF"):continue
    li,ri,yl,yr,ss=descriptor
    for j,p in enumerate(group_pairs[group]):
     if not valid[j]:continue
     l=int(li[j]);r=int(ri[j]);dy=float(score[r]-score[l]);dy0=float(raw[r]-raw[l])
     writer.writerow(dict(method=name,pair_id=p["pair_id"],kind=p["kind"],source_id=p["source_id"],
      left_label=int(yl[j]),right_label=int(yr[j]),left_score=float(score[l]),right_score=float(score[r]),
      original_difference=dy0,corrected_difference=dy,correction_difference=dy-dy0,
      paired_correct=bool((score[l]>0)==yl[j] and (score[r]>0)==yr[j])))
   point,reps,valid=accuracy_statistics(score,ni,ny,ns,draws)
   groups["natural"]=point;rep_cache[name,"natural"]=reps;valid_cache[name,"natural"]=valid
   metrics[name]=dict(provenance=provenance[name],groups=groups)
 # Autonomous path execution uses the same canonical implementation as the old
 # cache evaluation. Missing score entries remain missing; no raw substitution.
 sys.path.insert(0,str(SOURCE_REPO))
 from experiments.aster_cr.paths import execute
 cases=[c for c in rows(ROOT/"data/path_cases.jsonl") if c["split"]=="test"]
 legacy_paths=defaultdict(dict)
 for r in rows(OLD/"path_results.jsonl"):legacy_paths[r["method"]][r["case_id"]]=r
 path_indices=defaultdict(dict)
 for r in meta["paths"]:
  key="path|"+r["id"]
  if key in ix:path_indices[r["case_id"]][r["gid"]]=ix[key]
 path_vectors={};new_raw=[];path_methods={}
 with (out/"path_results.jsonl").open("w",encoding="utf-8") as handle:
  for name,score in scores.items():
   result=[];reused=0
   for case in cases:
    indexes=path_indices[case["case_id"]]
    complete=bool(indexes) and all(np.isfinite(score[i]) for i in indexes.values())
    if complete:
     item=dict(case_id=case["case_id"],source_id=case["source_id"],
      **execute(case,{gid:float(score[i]) for gid,i in indexes.items()}))
    elif name in legacy_fallback and case["case_id"] in legacy_paths[name]:
     item={key:value for key,value in legacy_paths[name][case["case_id"]].items() if key!="method"}
     item["reused_existing_frozen_path"]=True;reused+=1
    else:
     node_scores={gid:float(score[i]) for gid,i in indexes.items() if np.isfinite(score[i])}
     item=dict(case_id=case["case_id"],source_id=case["source_id"],**execute(case,node_scores))
    result.append(item);handle.write(json.dumps(dict(method=name,**item),ensure_ascii=False,allow_nan=False)+"\n")
   correct=np.array([int(r["correct"]) for r in result],dtype=float)
   ss=np.array([sx[r["source_id"]] for r in result],dtype=int)
   point,rep=cluster_mean(correct,ss,draws)
   metrics[name]["path"]=dict(n=len(result),expected=len(cases),missing=sum(bool(r.get("error")) for r in result),
    correct=int(correct.sum()),accuracy=point,accuracy_ci=percentile_ci(rep),
    sources=len(np.unique(ss)),unavailable_scores=sum(r.get("error")=="unavailable_score" for r in result),
    reused_existing_frozen_paths=reused)
   rep_cache[name,"path"]=dict(accuracy=rep);path_vectors[name]=correct
   path_methods[name]=result
   if name=="raw":new_raw=result
 comparisons={}
 for name,score in scores.items():
  comparison={}
  for reference in ("raw","linear_output"):
   reference_comparison={}
   for group,descriptor in descriptors.items():
    common=valid_cache[name,group]&valid_cache[reference,group]
    if np.array_equal(common,valid_cache[name,group]) and np.array_equal(common,valid_cache[reference,group]):
     p1=metrics[name]["groups"][group];r1=rep_cache[name,group]
     p0=metrics[reference]["groups"][group];r0=rep_cache[reference,group]
    else:
     p1,r1,_=pair_statistics(score,raw,descriptor,draws,raw_ap_rms,raw_ap_boot,common)
     p0,r0,_=pair_statistics(scores[reference],raw,descriptor,draws,raw_ap_rms,raw_ap_boot,common)
    difference=metric_differences(p1,r1,p0,r0)
    reference_comparison[group]=dict(common_pairs=int(common.sum()),metrics=difference)
   common=valid_cache[name,"natural"]&valid_cache[reference,"natural"]
   p1,r1,_=accuracy_statistics(score,ni,ny,ns,draws,common)
   p0,r0,_=accuracy_statistics(scores[reference],ni,ny,ns,draws,common)
   reference_comparison["natural"]=dict(common_nodes=int(common.sum()),metrics=metric_differences(p1,r1,p0,r0))
   reference_comparison["path"]=dict(common_paths=len(cases),
    metrics=metric_differences(metrics[name]["path"],rep_cache[name,"path"],metrics[reference]["path"],rep_cache[reference,"path"]))
   comparison[reference]=reference_comparison
   # Convenient compatibility fields for common reports.
   for group in descriptors:
    diff=reference_comparison[group]["metrics"].get("paired_accuracy")
    if diff is not None:
     metrics[name]["groups"][group]["paired_accuracy_difference_vs_"+reference]=diff["difference"]
     metrics[name]["groups"][group]["paired_accuracy_difference_vs_"+reference+"_ci"]=diff["ci"]
   for group in ("natural","path"):
    target=metrics[name]["path"] if group=="path" else metrics[name]["groups"]["natural"]
    diff=reference_comparison[group]["metrics"].get("accuracy")
    if diff is not None:
     target["difference_vs_"+reference]=diff["difference"];target["difference_vs_"+reference+"_ci"]=diff["ci"]
  comparisons[name]=comparison
 old_raw=legacy_paths["raw"]
 raw_mismatches=sum(r["case_id"] not in old_raw or r["correct"]!=old_raw[r["case_id"]]["correct"] or
  r["path"]!=old_raw[r["case_id"]]["path"] or r["leaf"]!=old_raw[r["case_id"]]["leaf"] for r in new_raw)
 result=dict(model="Qwen3-8B",protocol="Frozen configurations evaluated on the held-out test set previously used in legacy evaluations; no fitting or test selection",
  score_file=str(Path(scores_path).resolve()),frozen_selection_file=str(selection_path) if selection_path else None,
  test_sources=len(sources),source_ids=sources,test_views=len(test_views),test_natural_nodes=len(natural),
  test_path_cases=len(cases),test_AP_raw_rms=raw_ap_rms,
  uncertainty=dict(replicates=replicates,seed=seed,grouping="source_id",method="paired cluster percentile bootstrap",
   interpretation="Sources resampled jointly across all fixed trained methods; sparse-group undefined draws omitted; not training-seed uncertainty"),
  path_score_protocol="Each method follows its own decisions from the root; the full decision sequence and terminal leaf must equal gold",
  raw_path_comparison=dict(existing_cases=len(old_raw),new_cases=len(new_raw),mismatches=raw_mismatches),
  methods=metrics,seconds=time.time()-started)
 dump(out/"metrics.json",result)
 dump(out/"comparisons.json",dict(references=["raw","linear_output"],replicates=replicates,seed=seed,methods=comparisons))
 summary=[]
 for name,result_m in metrics.items():
  for group,values in result_m["groups"].items():summary.append(dict(method=name,group=group,**values))
  summary.append(dict(method=name,group="path",**result_m["path"]))
 fields=list(dict.fromkeys(key for r in summary for key in r))
 with (out/"summary.csv").open("w",newline="",encoding="utf-8") as handle:
  writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
  for row in summary:writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})
 verification=dict(input_rows=len(ids),expected_rows=28186,raw_paths=result["raw_path_comparison"],
  input_contains_all_test_views=all(v["view_id"] in ix for v in test_views),
  pair_coverage={group:len(pp) for group,pp in group_pairs.items()},
  missing={name:dict(groups={g:r.get("missing",0) for g,r in mm["groups"].items()},paths=mm["path"]["missing"]) for name,mm in metrics.items()},
  baselines_reused_without_training=True,no_test_fitting_or_selection=True)
 dump(out/"verification.json",verification)
 dump(out/"evaluation_done.json",dict(state="complete",methods=len(scores),seconds=time.time()-started))
 return result

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument("--scores",required=True,type=Path)
 parser.add_argument("--out",required=True,type=Path)
 parser.add_argument("--selection",type=Path)
 parser.add_argument("--no-baselines",action="store_true")
 parser.add_argument("--bootstrap",type=int,default=2000)
 parser.add_argument("--seed",type=int,default=11)
 args=parser.parse_args()
 result=evaluate_frozen_scores(args.scores,args.out,not args.no_baselines,args.bootstrap,args.seed,args.selection)
 print(json.dumps(dict(state="complete",methods=len(result["methods"]),seconds=result["seconds"])))

if __name__=="__main__":main()

