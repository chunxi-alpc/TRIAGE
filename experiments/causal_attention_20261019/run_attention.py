"""Run complete fixed-config real-model arms; errors remain explicit rows."""
import argparse,json,sys,time,traceback
from collections import defaultdict
from pathlib import Path
from engine import QwenEngine,read_cases

def summarize(cases,rows,baseline=None):
    by={r["case_id"]:r for r in rows}; base={r["case_id"]:r for r in (baseline or [])}
    correct={c["case_id"]:by.get(c["case_id"],{}).get("prediction")==c["gold_answer"] for c in cases}
    pairs=defaultdict(list); primary=defaultdict(list)
    for c in cases:
        if c.get("pair_eligible"):
            pairs[c["pair_id"]].append(c["case_id"])
            if c["variant"]=="canonical": primary[c["pair_id"]].append(c["case_id"])
    out=dict(cases=len(cases),completed=sum(r.get("status")=="completed" for r in rows),correct=sum(correct.values()),
        accuracy=sum(correct.values())/len(cases),pairs=len(pairs),pair_correct=sum(all(correct[c] for c in p) for p in pairs.values()),
        primary_pairs=len(primary),primary_pair_correct=sum(all(correct[c] for c in p) for p in primary.values()),
        failures=sum(r.get("status")!="completed" for r in rows),noop=sum(r.get("noop",False) for r in rows),
        attention_changed_cases=sum(any(d["attention_absolute_change_mean"]>1e-9 for d in r.get("attention_diagnostics",[])) for r in rows),
        mean_margin=sum(r["margin"] for r in rows if "margin" in r)/max(1,sum("margin" in r for r in rows)))
    if base:
        eligible=[c for c in cases if c["case_id"] in base]
        corrected=sum(not (base[c["case_id"]].get("prediction")==c["gold_answer"]) and correct[c["case_id"]] for c in eligible)
        damaged=sum((base[c["case_id"]].get("prediction")==c["gold_answer"]) and not correct[c["case_id"]] for c in eligible)
        out.update(corrected=corrected,damaged=damaged,net_corrected=corrected-damaged,
            prediction_changes=sum(by[c["case_id"]].get("prediction")!=base[c["case_id"]].get("prediction") for c in eligible))
        margins=[by[c["case_id"]]["margin"]-base[c["case_id"]]["margin"] for c in eligible if "margin" in by[c["case_id"]] and "margin" in base[c["case_id"]]]
        out["mean_margin_delta"]=sum(margins)/len(margins) if margins else None
    out["reference_state_accuracy"]={s:dict(n=sum(c["reference_state"]==s for c in cases),
        correct=sum(correct[c["case_id"]] for c in cases if c["reference_state"]==s)) for s in "TFU"}
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--split",default="dev");p.add_argument("--configs",required=True);p.add_argument("--output",required=True)
    p.add_argument("--structures");p.add_argument("--source",default="reference");p.add_argument("--baseline")
    p.add_argument("--limit",type=int);p.add_argument("--check",action="store_true")
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    configs=json.loads(Path(args.configs).read_text());cases=read_cases(args.split)
    if args.limit: cases=cases[:args.limit]
    structures={}
    if args.structures:
        for r in map(json.loads,Path(args.structures).read_text().splitlines()): structures[r["case_id"]]=r.get("structure",r)
    baseline=list(map(json.loads,Path(args.baseline).read_text().splitlines())) if args.baseline else None
    started_run=time.monotonic()
    model=QwenEngine()
    (out/"provenance.json").write_text(json.dumps(model.provenance|dict(argv=sys.argv,split=args.split,configs=configs),indent=2))
    if args.check:
        c=cases[0]; cfg=dict(layers=[0],strength=0.,variant="C",query_window=1)
        original=model.score(c,dict(layers=[]));zero=model.score(c,cfg);positive=model.score(c,cfg|dict(strength=1.))
        check=dict(case_id=c["case_id"],zero_max_candidate_logit_difference=max(abs(a-b) for a,b in zip(original["candidate_logits"],zero["candidate_logits"])),
            zero_prediction_same=original["prediction"]==zero["prediction"], target_text=[dict(text=positive["prompt"][t["span"][0]:t["span"][1]],**t) for t in positive["targets"]],
            actual_nonzero_attention=positive["attention_diagnostics"])
        (out/"necessary_check.json").write_text(json.dumps(check,indent=2))
        assert check["zero_max_candidate_logit_difference"]<.2
        assert all(t["text"]==next(r["action"] for r in c["rules"] if r["rule_id"]==t["rule_id"]) for t in check["target_text"])
        assert any(d.get("positive",{}).get("score_delta_mean",0)>.9 and d.get("negative",{}).get("score_delta_mean",0)<-.9 for d in positive["attention_diagnostics"])
    summaries={}
    for cfg in configs:
        rows=[];name=cfg["name"];start=time.monotonic()
        with (out/(name+".jsonl")).open("x") as stream:
            for c in cases:
                try: row=model.score(c,cfg,structures.get(c["case_id"], {"status":"missing_structure", "rules":[], "technical_failure":True}) if args.structures else None)
                except Exception as e:
                    row=dict(status="technical_failure",is_model_prediction=False,error=type(e).__name__+": "+str(e),traceback=traceback.format_exc())
                row.update(case_id=c["case_id"],arm=name,source=args.source,split=args.split,config=cfg,gold_answer=c["gold_answer"],
                    reference_state=c["reference_state"],family_id=c["family_id"],variant=c["variant"])
                rows.append(row);stream.write(json.dumps(row,ensure_ascii=False)+"\n");stream.flush()
        if cfg.get("baseline"): baseline=rows
        summaries[name]=summarize(cases,rows,baseline)
        summaries[name]["elapsed_seconds"]=time.monotonic()-start
        (out/"summary.json").write_text(json.dumps(summaries,indent=2))
        print(json.dumps(dict(arm=name,**summaries[name])),flush=True)
    (out/"completion.json").write_text(json.dumps(dict(status="completed",forward_calls=model.forward_calls,arms=len(configs),cases=len(cases),elapsed_seconds=time.monotonic()-started_run,peak_allocated_bytes=__import__("torch").cuda.max_memory_allocated(),peak_reserved_bytes=__import__("torch").cuda.max_memory_reserved()),indent=2))
if __name__=="__main__":main()

