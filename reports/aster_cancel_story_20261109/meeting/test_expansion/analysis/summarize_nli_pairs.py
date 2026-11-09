
"""CPU-only native NLI4CT pair analysis of already cached frozen-method nodes.

This module performs no model forward through the Qwen backbone and never fits,
calibrates, selects, or changes any frozen method. Importing it reads no scores.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
os.environ.setdefault("OMP_NUM_THREADS","2")
from pathlib import Path
import argparse,csv,json,sys,time
from collections import defaultdict,Counter
import numpy as np
BASE=Path("/home/gcx/aster_next_20261109")
DEFAULT_CASES=BASE/"inventory/nli4ct_all_cases.jsonl"
DEFAULT_PAIRS=BASE/"inventory/nli4ct_native_pairs.jsonl"
DEFAULT_MANIFEST=BASE/"frozen_methods.json"

def read_rows(path):
 path=Path(path)
 if not path.exists():return []
 return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def dump(path,value):
 Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")

def finite_float(value):
 value=float(value)
 return value if np.isfinite(value) else None

def ci(values):
 values=np.asarray(values,dtype=float);values=values[np.isfinite(values)]
 return np.quantile(values,[.025,.975]).tolist() if len(values) else None

def write_csv(path,values):
 if not values:return
 fields=list(dict.fromkeys(key for row in values for key in row))
 with Path(path).open("w",newline="",encoding="utf-8") as handle:
  writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
  for row in values:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})

def source_sums(values,index,count):
 values=np.asarray(values,dtype=float)
 if values.ndim==1:values=values[:,None]
 out=np.zeros((count,values.shape[1]))
 np.add.at(out,np.asarray(index,dtype=int),values)
 return out

def score_cached_nodes(cases,node_paths,manifest,method_names):
 # Import only the small frozen readout adapter; no Engine/model loader is used.
 sys.path.insert(0,str(BASE/"inference"))
 from frozen import FrozenMethods
 adapter=FrozenMethods(manifest)
 names=method_names or adapter.names
 if set(names)-set(adapter.names):raise ValueError("Unsupported frozen methods: "+str(sorted(set(names)-set(adapter.names))))
 bycase={case["case_id"]:case for case in cases}
 nodes={};duplicate_records=0;record_failures=[]
 for path in node_paths:
  for record in read_rows(path):
   cid=record.get("case_id")
   if cid not in bycase:continue
   gid=record.get("gid","G0")
   if gid!=bycase[cid].get("gid","G0"):continue
   if cid in nodes:duplicate_records+=1
   existing=nodes.get(cid)
   usable=record.get("status")=="ok" and record.get("feature") and Path(record["feature"]).is_file()
   if usable or existing is None:nodes[cid]=record
 scores={name:np.full(len(cases),np.nan,dtype=float) for name in names}
 for j,case in enumerate(cases):
  record=nodes.get(case["case_id"])
  if not record:
   record_failures.append(dict(case_id=case["case_id"],reason="missing_cached_node"));continue
  if record.get("status")!="ok" or not record.get("feature") or not Path(record["feature"]).is_file():
   record_failures.append(dict(case_id=case["case_id"],reason=record.get("error","missing_or_unavailable_feature")));continue
  values=adapter.predict_file(record)
  for name in names:scores[name][j]=float(values[name])
 return scores,names,dict(node_files=[str(p) for p in node_paths],cached_cases=len(nodes),duplicate_records=duplicate_records,unavailable=record_failures)

def pair_stats(score,raw,pp,ix,draws,family_index):
 expected=len(pp)
 valid=[p for p in pp if p["left"] in ix and p["right"] in ix and
  np.isfinite(score[ix[p["left"]]]) and np.isfinite(score[ix[p["right"]]]) and
  np.isfinite(raw[ix[p["left"]]]) and np.isfinite(raw[ix[p["right"]]])]
 if not valid:return dict(n=0,expected=expected,missing=expected),{}
 li=np.array([ix[p["left"]] for p in valid]);ri=np.array([ix[p["right"]] for p in valid])
 yl=np.array([p["left_label"] for p in valid]);yr=np.array([p["right_label"] for p in valid])
 source=np.array([family_index[p["family"]] for p in valid])
 dy0=raw[ri]-raw[li];dy=score[ri]-score[li];dc=dy-dy0
 cor=((score[li]>0)==yl)&((score[ri]>0)==yr)
 base_cor=((raw[li]>0)==yl)&((raw[ri]>0)==yr)
 direction=yr-yl;rawsigned=direction*dy0;signed=direction*dy
 changing=bool(np.any(direction))
 point=dict(n=len(valid),expected=expected,missing=expected-len(valid),
  families=len({p["family"] for p in valid}),
  raw_difference_rms=float(np.sqrt(np.mean(dy0**2))),
  difference_rms=float(np.sqrt(np.mean(dy**2))),
  difference_rms_ratio=finite_float(np.sqrt(np.mean(dy**2))/np.sqrt(np.mean(dy0**2))),
  difference_mean=float(np.mean(dy)),correction_rms=float(np.sqrt(np.mean(dc**2))),
  correction_rms_over_raw_difference_rms=finite_float(np.sqrt(np.mean(dc**2))/np.sqrt(np.mean(dy0**2))),
  paired_correct=int(cor.sum()),paired_accuracy=float(cor.mean()),
  left_accuracy=float(np.mean((score[li]>0)==yl)),right_accuracy=float(np.mean((score[ri]>0)==yr)),
  paired_accuracy_difference_vs_raw=float(np.mean(cor.astype(float)-base_cor.astype(float))),
  corrected_pairs_vs_raw=int(np.sum(cor&~base_cor)),new_error_pairs_vs_raw=int(np.sum(~cor&base_cor)),
  raw_difference_sign_changed_or_zero_count=int(np.sum((dy0!=0)&(np.sign(dy0)*dy<=0))),
  raw_nonzero_difference_count=int(np.sum(dy0!=0)))
 if changing:
  point.update(raw_signed_response_mean=float(np.mean(rawsigned)),signed_response_mean=float(np.mean(signed)),
   mean_signed_response_retention=finite_float(np.mean(signed)/np.mean(rawsigned)),
   raw_gold_direction_positive=int(np.sum(rawsigned>0)),gold_direction_positive=int(np.sum(signed>0)),
   gold_direction_degraded=int(np.sum((rawsigned>0)&(signed<=0))),
   gold_direction_improved=int(np.sum((rawsigned<=0)&(signed>0))))
 # Same-label pairs are descriptive; reducing their differences is not assumed
 # to be the same objective as MEGA A/P invariance.
 if not draws.size:return point,{}
 values=np.column_stack([np.ones(len(valid)),dy0**2,dy**2,dc**2,
  cor.astype(float),cor.astype(float)-base_cor.astype(float),rawsigned,signed])
 z=draws@source_sums(values,source,len(family_index))
 with np.errstate(divide="ignore",invalid="ignore"):
  drms=np.sqrt(z[:,2]/z[:,0]);crms=np.sqrt(z[:,3]/z[:,0])
  ratio=np.sqrt(z[:,2]/z[:,1]);cratio=np.sqrt(z[:,3]/z[:,1])
  bpa=z[:,4]/z[:,0];bpd=z[:,5]/z[:,0];ber=z[:,7]/z[:,6]
 reps=dict(difference_rms=drms,correction_rms=crms,difference_rms_ratio=ratio,
  correction_rms_over_raw_difference_rms=cratio,paired_accuracy=bpa,paired_accuracy_difference_vs_raw=bpd)
 if changing:reps["mean_signed_response_retention"]=ber
 for key,value in reps.items():point[key+"_ci"]=ci(value)
 return point,reps

def case_accuracy(cases,scores,selected,draws,family_index):
 out={}
 indices=np.array(selected,dtype=int)
 for name,values in scores.items():
  valid=indices[np.isfinite(values[indices])]
  if not len(valid):
   out[name]=dict(n=0,expected=len(indices),missing=len(indices),correct=0,accuracy=None);continue
  labels=np.array([cases[j]["label"] for j in valid])
  cor=((values[valid]>0)==labels).astype(float)
  result=dict(n=len(valid),expected=len(indices),missing=len(indices)-len(valid),correct=int(cor.sum()),accuracy=float(np.mean(cor)))
  if draws.size:
   fs=np.array([family_index[cases[j]["family"]] for j in valid])
   z=draws@source_sums(np.column_stack([np.ones(len(valid)),cor]),fs,len(family_index))
   with np.errstate(divide="ignore",invalid="ignore"):rep=z[:,1]/z[:,0]
   result["accuracy_ci"]=ci(rep)
  out[name]=result
 return out

def summarize(cases_path,pairs_path,node_paths,manifest,out_dir,methods=None,iterations=2000,seed=11):
 started=time.time();out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
 cases=[c for c in read_rows(cases_path) if c.get("dataset")=="NLI4CT-2024"]
 if not cases:raise ValueError("No NLI4CT cases in manifest")
 pairs=read_rows(pairs_path)
 case_ids={c["case_id"] for c in cases};ix={c["case_id"]:j for j,c in enumerate(cases)}
 scores,names,cache_coverage=score_cached_nodes(cases,node_paths,manifest,methods)
 if "raw" not in scores:raise ValueError("raw frozen method is required as the difference reference")
 raw=scores["raw"]
 # Every pair must already reference a released base and supplied labels.
 accepted=[];unmatched=[]
 for pair in pairs:
  if pair["left"] not in case_ids or pair["right"] not in case_ids:
   unmatched.append(dict(pair_id=pair["pair_id"],reason="base_or_variant_not_in_supplied_cases"));continue
  l=cases[ix[pair["left"]]];r=cases[ix[pair["right"]]]
  if l["label"]!=pair["left_label"] or r["label"]!=pair["right_label"]:
   unmatched.append(dict(pair_id=pair["pair_id"],reason="pair_labels_do_not_match_case_metadata"));continue
  if l["dataset_split"]!=r["dataset_split"]:
   unmatched.append(dict(pair_id=pair["pair_id"],reason="native_pair_crosses_official_split"));continue
  relation="preserving" if l["label"]==r["label"] else "changing"
  accepted.append(dict(pair,relation=relation))
 case_subsets={};groups={};flat=[];native_inventory={};rep_cache={}
 split_order=["test","practice","dev","train"]
 for split in split_order:
  cc=[j for j,c in enumerate(cases) if c["dataset_split"]==split]
  pp=[p for p in accepted if p["dataset_split"]==split]
  family_ids=sorted({cases[j]["family"] for j in cc});family_index={f:i for i,f in enumerate(family_ids)}
  draws=np.random.default_rng(seed).multinomial(len(family_ids),np.full(len(family_ids),1/len(family_ids)),size=iterations) if iterations and family_ids else np.empty((0,len(family_ids)))
  by_variant={p["right"]:p["relation"] for p in pp}
  for subset in ("all","original","preserving","changing"):
   if subset=="all":selected=cc
   elif subset=="original":selected=[j for j in cc if not cases[j].get("native_causal_type")]
   else:selected=[j for j in cc if by_variant.get(cases[j]["case_id"])==subset]
   case_subsets[split+":"+subset]=dict(split=split,subset=subset,expected=len(selected),
    methods=case_accuracy(cases,scores,selected,draws,family_index))
  native_inventory[split]=dict(records=len(cc),native_originals=sum(not cases[j].get("native_causal_type") for j in cc),
   native_variants=len(pp),relations=dict(Counter(p["relation"] for p in pp)),
   by_intervention={kind:dict(Counter(p["relation"] for p in pp if p["intervention"]==kind)) for kind in sorted({p["intervention"] for p in pp})})
  interventions=["all"]+sorted({p["intervention"] for p in pp})
  for intervention in interventions:
   for relation in ("preserving","changing"):
    selected=[p for p in pp if p["relation"]==relation and (intervention=="all" or p["intervention"]==intervention)]
    if not selected:continue
    key=split+":"+intervention+":"+relation
    item=dict(split=split,intervention=intervention,relation=relation,expected_pairs=len(selected),methods={})
    for name,values in scores.items():
     point,reps=pair_stats(values,raw,selected,ix,draws,family_index)
     item["methods"][name]=point;rep_cache[name,key]=reps
     flat.append(dict(group=key,split=split,intervention=intervention,relation=relation,method=name,**point))
    groups[key]=item
 comparisons={}
 if "old_linear" in scores:
  for name in names:
   if name in ("raw","old_linear"):continue
   comparisons[name]={}
   for key,item in groups.items():
    ours=item["methods"][name];ref=item["methods"]["old_linear"]
    # Paired common examples are identical when coverage is complete.
    if ours.get("n")!=ref.get("n"):continue
    comparison={}
    for metric,rep in rep_cache.get((name,key),{}).items():
     refrep=rep_cache.get(("old_linear",key),{}).get(metric)
     if refrep is not None and metric in ours and metric in ref:
      left=ours[metric];right=ref[metric]
      comparison[metric]=dict(difference=finite_float(left-right) if left is not None and right is not None else None,
       ci=ci(rep-refrep))
    comparisons[name][key]=comparison
 # Save each native pair once per frozen method, without aggregate duplicates.
 pair_rows=[]
 for pair in accepted:
  l=ix[pair["left"]];r=ix[pair["right"]]
  for name,values in scores.items():
   if not(np.isfinite(values[l]) and np.isfinite(values[r]) and np.isfinite(raw[l]) and np.isfinite(raw[r])):continue
   dy=float(values[r]-values[l]);dy0=float(raw[r]-raw[l]);direction=pair["right_label"]-pair["left_label"]
   pair_rows.append(dict(method=name,pair_id=pair["pair_id"],split=pair["dataset_split"],family=pair["family"],
    intervention=pair["intervention"],relation=pair["relation"],left=pair["left"],right=pair["right"],
    left_label=pair["left_label"],right_label=pair["right_label"],
    left_score=float(values[l]),right_score=float(values[r]),raw_difference=dy0,difference=dy,correction_difference=dy-dy0,
    gold_signed_response=float(direction*dy) if direction else None,
    paired_correct=bool(int(values[l]>0)==pair["left_label"] and int(values[r]>0)==pair["right_label"])))
 method_coverage={name:int(np.isfinite(v).sum()) for name,v in scores.items()}
 metrics=dict(dataset="NLI4CT-2024",protocol="CPU postprocessing of immutable cached nodes with frozen MEGA-trained readouts; zero-shot transfer without NLI fitting or selection",
  cases_manifest=str(cases_path),native_pairs_manifest=str(pairs_path),frozen_manifest=str(manifest),
  methods=names,case_subsets=case_subsets,groups=groups,native_label_inventory=native_inventory,
  coverage=dict(expected_records=len(cases),case_scores=method_coverage,expected_native_pairs=len(pairs),
   accepted_native_pairs=len(accepted),unmatched_native_pairs=unmatched,cache=cache_coverage),
  uncertainty=dict(replicates=iterations,seed=seed,grouping="native base family within each official split",
   qualification="Family cluster intervals account for derived variants sharing a base. Different families can still share trial material; these are not asserted independent trial sources."),
  interpretation=dict(preserving="Released base/variant labels are equal; score differences are descriptive native NLI variation, not automatically a MEGA A/P invariance target.",
   changing="Released base/variant labels differ; signed responses use right_label-left_label, without inferring a label flip from intervention names.",
   ratio="difference_rms_ratio is method pair RMS / raw pair RMS. mean_signed_response_retention is reported only for true label-changing pairs, with its raw signed denominator.",
   warning="Native Contradiction and Numerical_contradiction names include both same-label and flipping-label records."),
  no_additional_backbone_forward=True,no_training_or_calibration=True,seconds=time.time()-started)
 dump(out/"metrics.json",metrics);dump(out/"comparisons_vs_old_linear.json",comparisons)
 write_csv(out/"summary.csv",flat);write_csv(out/"pair_results.csv",pair_rows)
 np.savez_compressed(out/"case_scores.npz",ids=np.array([c["case_id"] for c in cases]),**scores)
 dump(out/"done.json",dict(state="complete",records=len(cases),native_pairs=len(accepted),methods=names,seconds=time.time()-started))
 return metrics

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument("--cases",type=Path,default=DEFAULT_CASES)
 parser.add_argument("--pairs",type=Path,default=DEFAULT_PAIRS)
 parser.add_argument("--nodes",type=Path,nargs="+",required=True)
 parser.add_argument("--manifest",type=Path,default=DEFAULT_MANIFEST)
 parser.add_argument("--methods",nargs="+")
 parser.add_argument("--out",type=Path,required=True)
 parser.add_argument("--iterations",type=int,default=2000)
 parser.add_argument("--seed",type=int,default=11)
 args=parser.parse_args();methods=[name for token in args.methods for name in token.split(",")] if args.methods else None
 result=summarize(args.cases,args.pairs,args.nodes,args.manifest,args.out,methods,args.iterations,args.seed)
 print(json.dumps(dict(state="complete",records=result["coverage"]["expected_records"],pairs=result["coverage"]["accepted_native_pairs"],methods=len(result["methods"]),seconds=result["seconds"])))
if __name__=="__main__":main()

