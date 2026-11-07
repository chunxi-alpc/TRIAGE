"""Freeze follow-up data, model variants, and source assignments before evaluation."""
import copy,re
from collections import Counter
from .common import *
def same_label_pairs(pairs,views):
    from experiments.aster_closed_form.common import evaluate
    annotations={r["group_id"]:r["value"] for r in rows("/home/gcx/aster_closed_form_20261105/data/constructions.jsonl") if r.get("value")}
    additions=[];new_views=[];seen=set()
    def numeric_fields(expr):
        fields=set()
        if expr.get("op") in ["<",">","<=",">=","==","!="] and type(expr.get("value")) in (int,float):
            fields.add(expr["field"])
        for child in expr.get("args",[]):fields.update(numeric_fields(child))
        if isinstance(expr.get("arg"),dict):fields.update(numeric_fields(expr["arg"]))
        return fields
    for p in pairs:
        if p["kind"]!="F" or p["group_id"] not in annotations:continue
        ann=annotations[p["group_id"]]
        for side in ["left","right"]:
            v=views[p[side]];key=(p["group_id"],v["label"])
            if key in seen:continue
            seen.add(key)
            facts=ann["facts"][int(v["label"])]
            if int(evaluate(ann["predicate"],facts))!=v["label"]:continue
            made=False
            for field,old in facts.items():
                if type(old) not in (int,float) or field not in numeric_fields(ann["predicate"]):continue
                pat=r"(?<![\d.])"+re.escape(str(int(old)) if float(old).is_integer() else str(old))+r"(?![\d.])"
                matches=list(re.finditer(pat,v["patient"]))
                if len(matches)!=1:continue
                for step in [1,-1,.1,-.1]:
                    new=old+step
                    updated=dict(facts,**{field:new})
                    if int(evaluate(ann["predicate"],updated))!=v["label"]:continue
                    if any(evaluate(a["predicate"],updated)!=evaluate(a["predicate"],facts) for a in ann.get("ancestor_predicates",[])):continue
                    text=str(int(new)) if float(new).is_integer() else str(new)
                    patient=re.sub(pat,text,v["patient"])
                    if v["patient"] not in v["prompt"]:continue
                    nv=dict(v,view_id=v["view_id"]+":same_numeric",patient=patient,
                            prompt=v["prompt"].replace(v["patient"],patient),original=False)
                    new_views.append(nv)
                    additions.append(dict(pair_id=nv["view_id"]+":pair",group_id=p["group_id"],source_id=p["source_id"],
                        split=p["split"],kind="F_same",subtype="same_side_numeric",left=v["view_id"],right=nv["view_id"],
                        left_label=v["label"],right_label=v["label"],truth_relation="preserve",
                        edit=dict(field=field,old=old,new=new),annotation_origin="existing executable predicate; numeric fact changed without truth or ancestor change"))
                    made=True;break
                if made:break
    return additions,new_views
def medguide_data():
    import pyarrow.parquet as pq
    native=pq.read_table("/home/gcx/TRIAGE/data/MedGUIDE/data/train-00000-of-00001.parquet").to_pylist()
    pilot=rows(DIAG/"data/medguide_200.jsonl");ids={r["row_index"] for r in pilot}
    def family(s):return re.sub(r"_?\d+$","",s)
    families=sorted({family(r["disease"]) for r in native})
    rng=np.random.default_rng(SEED);rng.shuffle(families)
    calibration_families=set(families[:8]);test_families=set(families[8:])
    pool=[dict(sample_id=f"medguide:{i}",row_index=i,source_id=r["disease"],family=family(r["disease"]),native=r)
        for i,r in enumerate(native) if i not in ids]
    calibration=[r for r in pool if r["family"] in calibration_families]
    rng.shuffle(calibration);calibration=calibration[:200]
    confirm=[r for r in pool if r["family"] in test_families]
    for name,values in [("pilot",pilot),("calibration",calibration),("confirm",confirm)]:
        write_rows(ROOT/"data"/f"medguide_{name}.jsonl",values)
    return dict(pilot=len(pilot),calibration=len(calibration),confirm=len(confirm),
        calibration_families=sorted(calibration_families),confirmation_families=sorted(test_families),
        policy="Scalar answer-position bias fitted on calibration only; M3 never trained on MedGUIDE. Pilot is historically observed and spans all 53 pages. Confirmation is family-disjoint from calibration and excludes pilot IDs; its families have prior pilot exposure.")
def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    pairs=rows(LEGACY/"data/pairs.jsonl");vs=rows(LEGACY/"data/views.jsonl");views={r["view_id"]:r for r in vs}
    extra,new_views=same_label_pairs(pairs,views);pairs+=extra;vs+=new_views
    write_rows(ROOT/"data/pairs.jsonl",pairs);write_rows(ROOT/"data/views.jsonl",vs)
    mg=medguide_data()
    variants=[
        dict(name=f"random_{s}",heads="same_layer_random",seed=s,support="N",reference="reference",paired=False) for s in [13,37,73]]
    variants += [
        dict(name="candidates",heads="candidates",seed=SEED,support="N",reference="reference",paired=False),
        dict(name="all_heads",heads="all",seed=SEED,support="N",reference="reference",paired=False)]
    variants += [dict(name="paired_"+s.lower(),heads="selected",seed=SEED,support=s,reference="online_self",paired=True) for s in ["N","S","NS"]]
    variants.append(dict(name="online_n",heads="selected",seed=SEED,support="N",reference="online_self",paired=False))
    config=dict(training_seed=20261105,primary_new_method="paired_ns",seed=SEED,gpus=[0,1],model="/home/gcx/TRIAGE/models/modelscope/Qwen3-8B",
        epochs=3,lr=.001,weight_decay=.01,accumulation=8,kl=.1,edit=.001,pair_weight=1.,
        loss="original-node CE + 0.1 clean-correct KL + 0.001 edit; paired variants additionally use both-side CE and A/P/F_same JS or C/F label-oriented margin with margin=1",
        pairs=dict(Counter(f"{p['split']}:{p['kind']}" for p in pairs)),medguide=mg,variants=variants,
        selection="100 original development cases, path accuracy then smaller relative edit then earlier epoch",
        precision="bf16 backbone and label projection, float32 trainable editors; preserves frozen M3 scoring",
        old_results=str(OLD),resources=dict(gpu_memory_gb_per_card=[18,26],estimated_wall_hours=[10,20]),
        confirmation_trigger="pilot learned_reference or any frozen editor has more correct cases than full; no pilot-based checkpoint or hyperparameter tuning",
        publication="Results and revised docs are pushed to codex/aster-m3-p0-p3-20261107 after completion; no merge or force push.")
    write(ROOT/"configuration.json",config)
    status("prepare",state="complete",variants=len(variants),pairs=config["pairs"],medguide=mg)
    return config
if __name__=="__main__":
    c=prepare();print(json.dumps(dict(variants=len(c["variants"]),pairs=c["pairs"],medguide=c["medguide"]),ensure_ascii=False))
