#!/usr/bin/env python3
"""Summarize ASTER predictions without dropping malformed or missing answers."""
import argparse, json, os
from collections import defaultdict

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output",required=True); a=p.parse_args()
    rows=[json.loads(s) for s in open(a.input,encoding="utf-8") if s.strip()]
    arms=defaultdict(list)
    for r in rows: arms[(r.get("variant"),r.get("condition"))].append(r)
    summary={"n_predictions":len(rows),"arms":{}}
    for (variant,condition),rs in sorted(arms.items()):
        n=len(rs); valid=[r for r in rs if r.get("prediction") is not None]
        correct=sum(r.get("prediction")==r.get("gold_answer") for r in rs)
        stubborn=sum(r.get("prediction")==r.get("original_answer") and r.get("original_answer")!=r.get("gold_answer") for r in rs)
        misuse=sum(r.get("prediction")==r.get("sibling_action_label") and r.get("sibling_action_label")!=r.get("gold_answer") for r in rs)
        summary["arms"][f"{variant}/{condition}"]={"n":n,"valid_answers":len(valid),"accuracy":correct/n if n else None,
          "invalid_answer_rate":1-len(valid)/n if n else None,"stubborn_errors":stubborn,"stubborn_rate":stubborn/n if n else None,
          "sibling_branch_misuse":misuse,"sibling_branch_misuse_rate":misuse/n if n else None}
    byid=defaultdict(dict)
    for r in rows: byid[r["id"]][(r["variant"],r["condition"])]=r
    paired={}
    for variant in sorted({r["variant"] for r in rows}):
        ids=set(i for i,d in byid.items() if (variant,"baseline") in d)
        for condition in ("applicable_only","unsigned","signed"):
            common=[i for i in ids if (variant,condition) in byid[i]]
            if not common: continue
            fixed=sum(byid[i][(variant,condition)]["prediction"]==byid[i][(variant,condition)]["gold_answer"] and
                      byid[i][(variant,"baseline")]["prediction"]!=byid[i][(variant,"baseline")]["gold_answer"] for i in common)
            broken=sum(byid[i][(variant,condition)]["prediction"]!=byid[i][(variant,condition)]["gold_answer"] and
                       byid[i][(variant,"baseline")]["prediction"]==byid[i][(variant,"baseline")]["gold_answer"] for i in common)
            paired[f"{variant}/{condition}"]={"n_paired":len(common),"corrected":fixed,"broken":broken}
    summary["paired_changes"]=paired
    os.makedirs(os.path.dirname(os.path.abspath(a.output)),exist_ok=True)
    with open(a.output,"w",encoding="utf-8") as f: json.dump(summary,f,ensure_ascii=False,indent=2)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
