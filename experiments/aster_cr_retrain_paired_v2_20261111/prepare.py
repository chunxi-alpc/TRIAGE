"""Build exact-input tables once; reuse canonical cached inference without changing old files."""
import argparse, collections, json, os, time, zipfile
from pathlib import Path
import numpy as np
from .common import *
from experiments.aster_closed_form.data import canonical_guide, prompt
TRACE=Path("/home/gcx/aster_trace_repair_20261110")
CROSS=Path("/home/gcx/aster_cross_task_transfer_20261109")
KINDS=("A","P","C","F")
DEFAULT_LABELS=[chr(26159),chr(21542)]
class Registry:
    def __init__(self):
        self.inputs=[];self.bykey={};self.reused={};self.origin_counts=collections.Counter()
    def add(self,text,labels,stage,oldid,dataset,meta=None,grad=False):
        labels=list(labels or DEFAULT_LABELS)
        key=(text,tuple(labels))
        if key not in self.bykey:
            uid="u%08d"%len(self.inputs)
            m=meta or {}
            rec=dict(uid=uid,view_id=uid,prompt=text,labels=labels,split=m.get("split",stage),
                     stages=[],aliases=[],required_grads=False,
                     source_id=m.get("source_id"),group_id=m.get("group_id"),case_id=m.get("case_id"),
                     kind=m.get("kind"),label=m.get("label"),task=dataset)
            self.inputs.append(rec);self.bykey[key]=rec
        rec=self.bykey[key]
        if stage not in rec["stages"]:rec["stages"].append(stage)
        alias=dict(dataset=dataset,id=oldid,stage=stage)
        if alias not in rec["aliases"]:rec["aliases"].append(alias)
        rec["required_grads"]|=bool(grad)
        return rec["uid"]
    def reuse(self,uid,record,origin,heads="all",has_grads=False):
        if record.get("status","ok")!="ok" or "ell" not in record:return
        feature=record.get("feature")
        if not feature or not Path(feature).is_file():return
        r=dict(record,uid=uid,view_id=uid,feature=feature,origin=origin,status="ok",
               available_heads=heads,has_grads=bool(record.get("has_grads",has_grads)),reused=True)
        old=self.reused.get(uid)
        quality=lambda x:(x.get("available_heads")=="all",x.get("has_grads",False))
        if old is None or quality(r)>quality(old):self.reused[uid]=r
        self.origin_counts[origin]+=1
    def save(self):
        lines(DATA/"inputs.jsonl",self.inputs)
        lines(DATA/"reused_scores.jsonl",list(self.reused.values()))
def slimview(v,uid,oldid):
    return dict({k:val for k,val in v.items() if k not in ("prompt","prompt_text")},
                view_id=uid,uid=uid,original_view_id=oldid)
def remap_pair(p,mapping):
    return dict(p,left=mapping[p["left"]],right=mapping[p["right"]],
                original_left=p["left"],original_right=p["right"])
def score_files(root):
    return [p for p in sorted(root.glob("*.jsonl")) if p.stat().st_size]
def grad_flag(r,default):
    if "has_grads" in r:return bool(r["has_grads"])
    if not default:return False
    try:
        with zipfile.ZipFile(r["feature"]) as z:return "grads.npy" in z.namelist()
    except (FileNotFoundError,zipfile.BadZipFile):return False
def run(interference=False):
    if (OUT/"data_inventory.json").exists() and (DATA/"inputs.jsonl").exists():
        print("prepared data already exist; no repeated frozen-data checks")
        return read(OUT/"data_inventory.json")
    DATA.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    started=time.time();reg=Registry();manifest=read(FROZEN/"dataset_manifest.json")
    allowed=set(manifest["fit_source_ids"])
    frozen={}
    inventory={"status":"prepared","frozen_commit":"5a329fe68eaa84fb5f7a791c50d15a47181ac4f2",
        "data_root":str(FROZEN),"historical_root":str(HIST),"checks":"one file/count/field/source check; no review, generation, hashing or human sample audit",
        "datasets":{},"source_ids":sorted(allowed),"prior_evaluation_use":"40 historical test and 60 newer evaluation sources were previously used"}
    for group,name,expected in [("E1","balanced_train.jsonl",dict(A=635,P=635,C=635,F=635)),
                                ("E2","full_pool_train.jsonl",dict(A=4368,P=635,C=676,F=666))]:
        path=FROZEN/name
        if not path.is_file():raise FileNotFoundError(path)
        rr=rows(path);counts=collections.Counter(p["kind"] for p in rr)
        if dict(counts)!=expected:raise ValueError("Frozen count mismatch: "+name)
        for r in rr:
            for k in ("pair_id","kind","source_id","split","original","modified"):
                if k not in r:raise ValueError("Missing frozen field "+k)
            if r["split"]!="fit" or r["source_id"] not in allowed:raise ValueError("Frozen source split mismatch")
            for role in ("original","modified"):
                v=r[role]
                if not isinstance(v.get("prompt"),str) or not v["prompt"]:raise ValueError("Missing frozen prompt")
                if int(v["label"]) not in (0,1):raise ValueError("Invalid frozen label")
        frozen[group]=rr
        inventory["datasets"][group]=dict(file=str(path),pairs=len(rr),kind_counts=dict(counts),
           sources=len({r["source_id"] for r in rr}),
           sources_by_kind={k:len({r["source_id"] for r in rr if r["kind"]==k}) for k in KINDS})
    # Original historical inputs, unchanged. Their exact prompt strings identify cache candidates.
    hv=rows(HIST/"data/views.jsonl");hp=rows(HIST/"data/pairs.jsonl")
    historical_map={};historical_views=[]
    for v in hv:
        uid=reg.add(v["prompt"],v.get("labels"),v["split"],v["view_id"],"historical",v,grad=v["split"]=="fit")
        historical_map[v["view_id"]]=uid;historical_views.append(slimview(v,uid,v["view_id"]))
    historical_pairs=[remap_pair(p,historical_map) for p in hp]
    lines(DATA/"historical_views.jsonl",historical_views)
    lines(DATA/"historical_pairs.jsonl",historical_pairs)
    for p in score_files(HIST/"qwen/scores"):
        for r in rows(p):
            if r.get("view_id") in historical_map:
                reg.reuse(historical_map[r["view_id"]],r,str(p),has_grads=grad_flag(r,r.get("split")=="fit"))
    originals={n["group_id"] for n in rows(HIST/"data/original_nodes.jsonl") if n["split"]=="fit"}
    natural=[v for v in historical_views if v["split"]=="fit" and v.get("variant")=="full" and v["group_id"] in originals]
    if len(natural)!=761:raise ValueError("Historical natural calibration node count is not 761")
    lines(DATA/"natural_views.jsonl",natural)
    e0=[p for p in historical_pairs if p["split"]=="fit" and p["kind"] in KINDS]
    lines(DATA/"E0_pairs.jsonl",e0)
    inventory["datasets"]["E0"]=dict(file=str(HIST/"data/pairs.jsonl"),pairs=len(e0),
        kind_counts=dict(collections.Counter(p["kind"] for p in e0)),sources=len({p["source_id"] for p in e0}),
        full_fit_policy="Existing historical fit is referenced; only new five-fold fits are computed")
    for group,rr in frozen.items():
        pp=[];vv=[]
        for r in rr:
            ends={}
            for role in ("original","modified"):
                v=r[role];oldid=v.get("view_id",r["pair_id"]+":"+role)
                meta=dict(source_id=r["source_id"],group_id=r.get("group_id"),case_id=r.get("case_id"),split="fit",
                          label=int(v["label"]),kind="original" if role=="original" else r["kind"],
                          variant="full" if role=="original" else r["kind"],original=role=="original")
                uid=reg.add(v["prompt"],v.get("labels"),"fit",oldid,group,meta,grad=True)
                ends[role]=uid;vv.append(dict(meta,view_id=uid,uid=uid,original_view_id=oldid))
            pp.append(dict(pair_id=r["pair_id"],kind=r["kind"],source_id=r["source_id"],split="fit",
                case_id=r.get("case_id"),group_id=r.get("group_id"),left=ends["original"],right=ends["modified"],
                left_label=int(r["original"]["label"]),right_label=int(r["modified"]["label"]),origin=r.get("origin")))
        lines(DATA/(group+"_pairs.jsonl"),pp);lines(DATA/(group+"_views.jsonl"),vv)
    lines(DATA/"E3_pairs.jsonl",rows(DATA/"E2_pairs.jsonl"))
    lines(DATA/"E3_views.jsonl",rows(DATA/"E2_views.jsonl"))
    inventory["datasets"]["E3"]=dict(inventory["datasets"]["E2"],weights="natural sample weights")
    # Historical all-reachable-node inference inputs, including new fit-fold path evaluation.
    pathcases=rows(HIST/"data/path_cases.jsonl")
    oldpath={}
    for p in score_files(HIST/"qwen/paths"):
        if p.name.startswith("features_"):
            for r in rows(p):
                if r.get("status")=="ok":oldpath[r["id"]]=(r,str(p))
    paths=[]
    for c in pathcases:
        gold={n["gid"]:n["label"] for n in c["gold_path"]}
        for n in canonical_guide(c["guideline"])["nodes"]:
            oldid=c["case_id"]+":"+n["id"]
            text=prompt(c["guideline"],c["definitions"],c["patient"],n["id"])
            meta=dict(case_id=c["case_id"],source_id=c["source_id"],gid=n["id"],split=c["split"],variant="original",
                      kind="original",original=True,label=gold.get(n["id"]))
            uid=reg.add(text,None,"path_"+c["split"],oldid,"historical_path",meta)
            paths.append(dict(meta,view_id=uid,uid=uid,original_view_id=oldid))
            if oldid in oldpath:
                r,origin=oldpath[oldid];reg.reuse(uid,r,origin)
    lines(DATA/"path_features.jsonl",paths)
    # Existing 60-source evaluation inputs; no construction and no audit are repeated.
    nv=[];newmap={}
    for filename in ("tasks_final.jsonl","tasks_diagnostic_final.jsonl","tasks_diagnostic_program_final.jsonl"):
        for v in rows(TRACE/"data"/filename):
            uid=reg.add(v["prompt"],v.get("labels"),"new60",v["id"],"new60",v)
            newmap[v["id"]]=uid;nv.append(slimview(v,uid,v["id"]))
    npairs=[]
    for filename in ("pairs_final.jsonl","pairs_diagnostic_final.jsonl","pairs_diagnostic_program_final.jsonl"):
        for p in rows(TRACE/"data"/filename):
            if p["left"] in newmap and p["right"] in newmap:npairs.append(remap_pair(p,newmap))
    lines(DATA/"new60_views.jsonl",nv);lines(DATA/"new60_pairs.jsonl",npairs)
    for p in sorted((TRACE/"scoring/final").glob("**/*scores.jsonl")):
        done=read(p.parent/"done.json")
        head_ids=done.get("head_ids",read(TRACE/"scoring/final/done.json").get("workers",[{}])[0].get("head_ids",[]))
        for r in rows(p):
            oldid=r.get("task_id",r.get("id"))
            if oldid in newmap:
                # Some reused records point to full historical npz files.
                full=not str(r.get("feature","")).startswith(str(TRACE))
                reg.reuse(newmap[oldid],r,str(p),heads="all" if full else head_ids)
    for task in ("nli4ct","medpic"):
        vv=[];mapping={}
        for v in rows(CROSS/task/"views.jsonl"):
            oldid=v["view_id"];uid=reg.add(v["prompt"],v.get("labels"),task,oldid,task,v)
            mapping[oldid]=uid;vv.append(slimview(v,uid,oldid))
        pp=[remap_pair(p,mapping) for p in rows(CROSS/task/"pairs.jsonl")]
        lines(DATA/(task+"_views.jsonl"),vv);lines(DATA/(task+"_pairs.jsonl"),pp)
        for r in rows(CROSS/task/"scores.jsonl"):
            if r.get("view_id") in mapping:reg.reuse(mapping[r["view_id"]],r,str(CROSS/task/"scores.jsonl"),has_grads=r.get("has_grads",False))
    reg.save()
    stages=collections.Counter(s for i in reg.inputs for s in i["stages"])
    inventory.update(unique_inputs=len(reg.inputs),natural_fit_nodes=len(natural),stages=dict(stages),
        old_cache_reused=len(reg.reused),fit_unique_inputs=sum(i["required_grads"] for i in reg.inputs),
        reused_full_heads=sum(r["available_heads"]=="all" for r in reg.reused.values()),
        prepared_seconds=time.time()-started)
    write(OUT/"data_inventory.json",inventory)
    protocol=dict(model="/home/gcx/TRIAGE/models/modelscope/Qwen3-8B",model_version="same local frozen checkpoint directory",
        prompt="exact raw prompt strings, no normalization; original canonical SYSTEM + official tokenizer chat template",
        tokenizer="same model tokenizer; enable_thinking=False",
        scoring="native immutable prefix cache replay; final prompt token; FP32 YES-NO label projection",
        label_tokens=[20412,32376],labels=DEFAULT_LABELS,
        evidence=[str(HIST/"configuration.json"),str(HIST/"verification/qwen_forward.json"),
          "/home/gcx/aster_five_models_20261109/engine.py",
          "/home/gcx/aster_cross_task_transfer_20261109/medpic/model_metadata.json",
          "/home/gcx/aster-final-two-20261111/reports/aster_cancel_story_20261109/meeting/trace_repair_followup/score_worker.py"],
        reuse_rule="identical canonical model/prompt/tokenizer/final-token score protocol plus exact prompt and label tuple",
        feature_policy="full historical and cross-task heads; 60-source cache provides recorded head subset and will be supplemented only when required",
        no_hash_checks=True,unique_input_table=str(DATA/"inputs.jsonl"),
        cache_records=str(DATA/"reused_scores.jsonl"),reused=len(reg.reused),stages=dict(stages))
    write(OUT/"input_cache_manifest.json",protocol)
    if interference:add_interference()
    print(json.dumps({k:inventory[k] for k in ("unique_inputs","natural_fit_nodes","old_cache_reused","stages")}))
    return inventory
def add_interference():
    # Reuse only the three action edits already defined by the historical experiment.
    from experiments.aster_cr.repair.build import action_guides
    inputs=rows(DATA/"inputs.jsonl");reg=Registry();reg.inputs=inputs
    reg.bykey={(v["prompt"],tuple(v["labels"])):v for v in inputs}
    reg.reused={r["view_id"]:r for r in rows(DATA/"reused_scores.jsonl")}
    node_lang={n["source_id"]:n.get("language","zh") for n in rows(HIST/"data/nodes.jsonl")}
    vv=[]
    rename={"presence":"action_presence","neutral":"action_neutral","swap":"action_swap"}
    for c in rows(HIST/"data/path_cases.jsonl"):
        for mode,guide,protected in action_guides(dict(guideline=c["guideline"],language=node_lang.get(c["source_id"],"zh"))):
            if mode not in rename:continue
            variant=rename[mode];gold={n["gid"]:n["label"] for n in c["gold_path"]}
            for n in canonical_guide(c["guideline"])["nodes"]:
                text=prompt(guide,c["definitions"],c["patient"],n["id"])
                oldid=c["case_id"]+":"+n["id"]+":"+variant
                meta=dict(case_id=c["case_id"],source_id=c["source_id"],gid=n["id"],split=c["split"],
                    variant=variant,kind="A",original=False,label=gold.get(n["id"]))
                uid=reg.add(text,None,"path_interference_"+c["split"],oldid,"historical_path_interference",meta)
                vv.append(dict(meta,view_id=uid,uid=uid,original_view_id=oldid))
    reg.save();lines(DATA/"path_interference_views.jsonl",vv)
    inv=read(OUT/"data_inventory.json");inv["unique_inputs"]=len(reg.inputs)
    inv["stages"]=dict(collections.Counter(s for i in reg.inputs for s in i["stages"]))
    inv["historical_interference_path_views"]=len(vv);write(OUT/"data_inventory.json",inv)
    man=read(OUT/"input_cache_manifest.json");man.update(unique_inputs=len(reg.inputs),stages=inv["stages"]);write(OUT/"input_cache_manifest.json",man)
    print(json.dumps(dict(interference_views=len(vv),unique_inputs=len(reg.inputs))))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--interference",action="store_true");p.add_argument("--interference-only",action="store_true")
    a=p.parse_args();add_interference() if a.interference_only else run(a.interference)
