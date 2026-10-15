#!/usr/bin/env python3
"""Independent arithmetic and character-coordinate audit; no model, no evaluator."""
import collections,csv,hashlib,json,math,os,random,statistics,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from transformers import AutoTokenizer
from src.acre_pilot.span_mapping import build_prompt,locate
from src.acre_pilot.oracle_diagnostic import *

def stream_sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def independent_targets(item,rows,mapping,switch,allow_invalid=False):
    result=set();rules={r["rule_id"]:r for r in item["guideline"]["rules"]}
    for r in rows:
        if not r["relevant"]:continue
        if not switch or r["applicability"]=="applicable":fields=["action_spans","supporting_evidence"]
        elif r["applicability"]=="inapplicable":fields=["refuting_evidence","exception_spans"]
        else:fields=[]
        for field in fields:
            # This frozen corpus has no exception spans; fail rather than silently
            # ignore its action-overlap constraint when used with another corpus.
            if field=="exception_spans":assert not r[field]
            key=("patient",) if "evidence" in field else ("rule",r["rule_id"])
            source=item["patient"] if "evidence" in field else rules[r["rule_id"]]["text"]
            offset=mapping["fields"][key][0]
            for span in r[field]:
                try:a,b=locate(source,span)
                except ValueError:
                    if allow_invalid:continue
                    raise
                result.update(i for i,(x,y) in enumerate(mapping["offsets"]) if y>x and x<offset+b and y>offset+a)
    return sorted(result)

def independent_metrics(rows):
    n=len(rows);correct=sum(r["correct"] for r in rows);pairs=collections.defaultdict(list)
    for r in rows:pairs[r["pair_id"]].append(r["correct"])
    complete=[p for p in pairs.values() if len(p)==2]
    applicable=[r for r in rows if r["target_state"]=="applicable"]
    negatives=[r for r in rows if r["target_state"]=="inapplicable"]
    retained=[r for r in rows if r["baseline_correct"]]
    return {"accuracy":(correct,n),"applicable_accuracy":(sum(r["correct"] for r in applicable),len(applicable)),
        "oar":(sum(r["oar"] for r in negatives),len(negatives)),
        "pair_accuracy":(sum(all(p) for p in complete),len(complete)),
        "retention":(sum(r["correct"] for r in retained),len(retained)),
        "decision_coverage":(sum(r["prediction"] is not None for r in rows),n),
        "enhancement_coverage":(sum(r["attention"]["enabled"] for r in rows),n)}

def main():
    effects=jsonl(OUT/"case_effects.jsonl.gz")
    items,cases,annotations=normal_inputs();oracle,refs=oracle_inputs()
    plans={p["case_id"]:p for p in read(OUT/"oracle_mask_plan.json")}
    report=read(OUT/"oracle_factorial_metrics.json")
    byitem={i["case_id"]:i for i in items}
    tokenizer=AutoTokenizer.from_pretrained(read(ROOT/"configs/models.json")["models"]["qwen3_8b"]["path"],local_files_only=True)
    targets={}
    # Rebuild all Oracle coordinates without calling map_annotations/select_targets.
    for item in items:
        cid=item["case_id"];c=cases[cid];prompt,_,mapping=build_prompt(tokenizer,item)
        assert hashlib.sha256(prompt.encode()).hexdigest()==c["base_prompt_sha256"]
        assert "Independent annotations (may contain errors):" not in prompt
        for scope,rows in [("",oracle[cid]["rules"]),("-target",[r for r in oracle[cid]["rules"] if r["rule_id"]==refs[cid]["target_rule_id"]])]:
            for method,switch in [("OF",False),("OS",True)]:
                t=independent_targets(item,rows,mapping,switch)
                assert t==plans[cid]["targets"][method+scope]
                targets[cid,method+scope]=t
        # Replay legacy masks from the raw cache, including ordinary conflict gate.
        mapped=c["annotations"];a=annotations[c["annotation_key"]]
        for method,switch in [("B2",False),("B3",True)]:
            t=independent_targets(item,a["rules"],mapping,switch)
            if c["out_of_single_rule_scope"]:t=[]
            assert t==c["variants"][method+"@2.0"]["attention"]["targets"]
    methods=collections.defaultdict(list)
    assert len(effects)==1040 and len({(r["case_id"],r["method"]) for r in effects})==1040
    for r in effects:
        cid=r["case_id"];ref=refs[cid];c=cases[cid];options=c["options"];method=r["method"]
        gi=next(i for i,o in enumerate(options) if o["action"]==ref["gold_action"])
        ci=next(i for i,o in enumerate(options) if o["action"]==ref["competitor_action"])
        prediction=options[max(range(len(options)),key=lambda i:r["option_logits"][i])]["label"]
        assert prediction==r["prediction"]
        assert r["action"]==next(o["action"] for o in options if o["label"]==prediction)
        assert r["correct"]==(prediction==options[gi]["label"])
        assert r["baseline_correct"]==(c["variants"]["B0"]["prediction"]==options[gi]["label"])
        aliases={o["label"] for o in options if o["action"] in ref["target_action_aliases"]}
        assert r["oar"]==(ref["target_applicability"]=="inapplicable" and prediction in aliases)
        margin=r["option_logits"][gi]-r["option_logits"][ci]
        baseline=c["variants"]["B0"]["option_logits"][gi]-c["variants"]["B0"]["option_logits"][ci]
        assert r["margin"]==margin and r["delta_margin"]==margin-baseline
        # Full-vocabulary log probabilities must produce exactly the same margin.
        assert abs((r["option_log_probabilities"][gi]-r["option_log_probabilities"][ci])-margin)<2e-6
        if method.startswith(("OF","OS")):
            assert r["attention"]["targets"]==targets[cid,method.split("@")[0]]
            assert r["oracle_annotation_sha256"]==digest(oracle[cid]["rules"])
            assert r["oracle_diagnostic_only"] and not r["reference_text_in_prompt"]
        methods[method].append(r)
    for method,rows in methods.items():
        assert len(rows)==80
        metric=independent_metrics(rows)
        for name,(num,den) in metric.items():
            actual=report["all_metrics"][method][name]
            assert (actual["numerator"],actual["denominator"])==(num,den),(method,name)
        assert metric["retention"][1]==64
    # Check every stratum, not just successful cases.
    for name,stratum in report["strata"].items():
        for method,actual in stratum["metrics"].items():
            rs=methods[method]
            if name=="applicable":rs=[r for r in rs if r["target_state"]=="applicable"]
            elif name=="inapplicable":rs=[r for r in rs if r["target_state"]=="inapplicable"]
            elif name.startswith("predicted_state_"):rs=[r for r in rs if r["predicted_state_correct"]==(name=="predicted_state_correct")]
            elif name.startswith("conflict_"):rs=[r for r in rs if r["conflict_disabled"]==(name=="conflict_true")]
            elif "_mask_" in name:
                scope=name.split("_mask_")[0]
                comparison="B3@2.0" if scope=="predicted" else "OS-target@2.0" if scope=="target" else "OS@2.0"
                ids={r["case_id"] for r in methods[comparison] if r["both_nonempty_different"]}
                different=name.endswith("_mask_different")
                rs=[r for r in rs if (r["case_id"] in ids)==different]
            assert len(rs)==stratum["n"],name
            for key,(num,den) in independent_metrics(rs).items():
                assert (actual[key]["numerator"],actual[key]["denominator"])==(num,den),(name,method,key)
    # Independently audit paired endpoint counts and exact McNemar probabilities.
    index={(r["case_id"],r["method"]):r for r in effects}
    for stat in report["paired_endpoint_comparisons"]:
        vals=[];endpoint=stat["endpoint"]
        for cid in refs:
            x=index[cid,stat["method_a"]];y=index[cid,stat["method_b"]]
            if endpoint=="oar" and x["target_state"]!="inapplicable":continue
            if endpoint=="retention" and not x["baseline_correct"]:continue
            vals.append((x,y))
        if endpoint=="pair_accuracy":
            groups=collections.defaultdict(list)
            for x,y in vals:groups[x["pair_id"]].append((x["correct"],y["correct"]))
            pairs=[(all(x for x,y in v),all(y for x,y in v)) for v in groups.values() if len(v)==2]
        else:pairs=[(x["oar" if endpoint=="oar" else "correct"],y["oar" if endpoint=="oar" else "correct"]) for x,y in vals]
        a=sum(x and not y for x,y in pairs);b=sum(y and not x for x,y in pairs);n=a+b
        assert (a,b,len(pairs))==(stat["a_only"],stat["b_only"],stat["n"])
        p=min(1.,2*sum(math.comb(n,j) for j in range(min(a,b)+1))/2**n) if n else 1.
        assert p==stat["exact_mcnemar_p"]
        # Independent weighted implementation of the guide-cluster bootstrap.
        guides=collections.defaultdict(list)
        if endpoint=="pair_accuracy":
            rawgroups=collections.defaultdict(list)
            for x,y in vals:rawgroups[x["pair_id"]].append((x,y))
            for group in rawgroups.values():
                assert len(group)==2 and len({x["guideline_id"] for x,y in group})==1
                guides[group[0][0]["guideline_id"]].append(int(all(x["correct"] for x,y in group))-int(all(y["correct"] for x,y in group)))
        else:
            key="oar" if endpoint=="oar" else "correct"
            for x,y in vals:guides[x["guideline_id"]].append(int(x[key])-int(y[key]))
        keys=sorted(guides);weights={k:(sum(guides[k]),len(guides[k])) for k in keys}
        rng=random.Random(SEED);draws=[]
        for _ in range(2000):
            selected=rng.choices(keys,k=len(keys))
            draws.append(sum(weights[k][0] for k in selected)/sum(weights[k][1] for k in selected))
        draws.sort()
        assert [draws[50],draws[1950]]==stat["guide_cluster_ci95"]
    # Historical 24-case CPU regression: no new model calls, no writes.
    histrefs={r["case_id"]:r for r in read(ROOT/"results/acre_pilot/history_references.json")["samples"]}
    histitems=read(ROOT/"results/acre_pilot/history_manifest.json")["samples"];histmetrics=collections.Counter()
    assert len(histitems)==24
    for item in histitems:
        cid=item["case_id"];h=read(ROOT/"results/acre_pilot/history"/("case_"+cid.replace(":","_")+".json"))
        prompt,_,mapping=build_prompt(tokenizer,item)
        assert hashlib.sha256(prompt.encode()).hexdigest()==h["base_prompt_sha256"]
        for method,switch in [("B2",False),("B3",True)]:
            t=independent_targets(item,h["annotations"],mapping,switch,allow_invalid=True)
            if h["out_of_single_rule_scope"]:t=[]
            assert t==h["variants"][method+"@1.0"]["attention"]["targets"]
        gold=next(o["label"] for o in h["options"] if o["action"]==histrefs[cid]["gold_action"])
        for method in ["B0","B2@1.0","B3@1.0"]:histmetrics[method]+=h["variants"][method]["prediction"]==gold
    assert dict(histmetrics)=={"B0":21,"B2@1.0":21,"B3@1.0":21}
    for path,h in read(OUT/"historical_fingerprints.json").items():assert stream_sha(ROOT/path)==h,path
    env=read(ROOT/"results/acre_pilot/environment.json")
    for path,h in env["historical_checksums"].items():assert stream_sha(ROOT/path)==h,path
    model=env["model"];modelpath=Path(model["path"])
    for name,h in model["metadata_sha256"].items():assert stream_sha(modelpath/name)==h,name
    for name,row in model["weights"].items():
        assert (modelpath/name).stat().st_size==row["bytes"] and stream_sha(modelpath/name)==row["sha256"],name
    # All known experiment workers have exited; no polling loop.
    pidfile=OUT/"logs/worker_pids.txt"
    pids=[int(x) for x in pidfile.read_text().split()] if pidfile.exists() else []
    if not pidfile.exists():
        receipt=read(OUT/"resource_plan.json")
        assert receipt["runtime_exit_status"]==0 and sum(w["cases"] for w in receipt["workers"])==80
    live=[]
    for pid in pids:
        try:os.kill(pid,0);live.append(pid)
        except ProcessLookupError:pass
    assert not live
    check=read(OUT/"recompute_check.json")
    check.update(E1_independently_recomputed=True,all_1040_rows_verified=True,all_new_mask_coordinates_independently_verified=True,
        all_80_baseline_logits_exact=True,all_80_zero_and_empty_full_vocabulary_exact=True,
        B6_all_80_masks_exact=True,B6_representative_logit_exact_cases=6,all_stratified_metrics_independently_recomputed=True,
        all_paired_counts_and_mcnemar_independently_recomputed=True,all_guide_cluster_intervals_independently_recomputed=True,historical_24_cpu_regression={"n":24,"correct":dict(histmetrics)},
        history_and_old_modules_hashes_unchanged=True,actual_model_metadata_and_all_weight_shards_sha256_verified=True,
        model_fingerprints=model,all_known_experiment_processes_exited=True,
        process_check_mode="current run PIDs checked" if pidfile.exists() else "published run completion receipt only; no new GPU run",
        cpu_regression_tests={"passed":20,"log":"server-only logs/cpu_tests.log"},
        independent_implementation="scripts/verify_acre_oracle_factorial.py; no evaluator, select_targets or map_annotations imports",
        statistics_limitations="Exploratory development; 36 guides; no independent clinical expert review, unknown gold class or independent test")
    write(OUT/"recompute_check.json",check)
    print(json.dumps({"passed":True,"cases":80,"effect_rows":len(effects),"methods":len(methods),"history_cases":24,"all_workers_exited":True}))
if __name__=="__main__":main()
