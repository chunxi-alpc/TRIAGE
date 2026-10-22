#!/usr/bin/env python3
"""Build ASTER A/B/C examples from MedGUIDE and a normalized tree export."""
import argparse, json, random, re
from collections import defaultdict

def norm(x):
    return re.sub(r"\\s+", " ", str(x or "")).strip().casefold()

def path_ids(x):
    if isinstance(x, str):
        x = json.loads(x)
    if not isinstance(x, list):
        raise ValueError("path must be a JSON list of node IDs")
    out=[]
    for n in x:
        if isinstance(n, dict): n=n.get("node_id", n.get("id"))
        if n is None: raise ValueError("path node lacks node_id/id")
        out.append(str(n))
    return out

def options_map(x):
    if isinstance(x, dict): return {str(k).upper():str(v).strip() for k,v in x.items()}
    if isinstance(x, list):
        return {str(v.get("label", v.get("key", chr(65+i))).upper()):str(v.get("text", v.get("value", v.get("option","")))).strip()
                if isinstance(v,dict) else str(v).strip() for i,v in enumerate(x)}
    raise ValueError("options must be a list or object")

def label_for(opts, action):
    found=[k for k,v in opts.items() if norm(v)==norm(action)]
    return found[0] if len(found)==1 else None

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",default="MedGUIDE/MedGUIDE-MCQA-8K")
    p.add_argument("--split",default="train")
    p.add_argument("--tree-json",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--limit",type=int,default=300)
    p.add_argument("--seed",type=int,default=20261022)
    p.add_argument("--cache-dir")
    a=p.parse_args()
    from datasets import load_dataset
    tree=json.load(open(a.tree_json,encoding="utf-8"))
    nodes=tree.get("nodes",[]) if isinstance(tree,dict) else []
    byid={str(n["node_id"]):n for n in nodes if n.get("node_id") is not None}
    children=defaultdict(list)
    for n in nodes:
        if n.get("node_id") is not None: children[str(n.get("parent_id","root"))].append(n)
    ds=load_dataset(a.dataset,split=a.split,cache_dir=a.cache_dir)
    rng=random.Random(a.seed); candidates=[]; counts=defaultdict(int)
    for idx,row in enumerate(ds):
        try:
            ids=path_ids(row.get("path"))
            if not ids or any(i not in byid for i in ids): counts["path_tree_mismatch"]+=1; continue
            selected=None
            for nid in reversed(ids):
                node=byid[nid]
                sibs=[s for s in children[str(node.get("parent_id","root"))]
                      if str(s.get("node_id"))!=nid and norm(s.get("action")) and norm(s.get("action"))!=norm(node.get("action"))]
                if sibs: selected=(node,sibs[0]); break
            if not selected: counts["no_sibling"]+=1; continue
            app,sib=selected; opts=options_map(row.get("options"))
            old=str(app.get("action","")).strip(); new=str(sib.get("action","")).strip()
            old_label,new_label=label_for(opts,old),label_for(opts,new)
            gold=str(row.get("answer","")).strip().upper()
            if old_label is None or new_label is None or old_label==new_label: counts["action_not_in_options"]+=1; continue
            if gold!=old_label and norm(row.get("answer_text"))!=norm(old): counts["gold_action_mismatch"]+=1; continue
            profile=str(row.get("profile","")).strip(); prompt=str(row.get("prompt","")).strip()
            if not profile or not prompt: counts["missing_text"]+=1; continue
            rule=f"IF {app.get('condition','')}, THEN {old}."
            sibling_rule=f"IF {sib.get('condition','')}, THEN {new}."
            common={"id":str(row.get("id",row.get("example_id",idx))),"profile":profile,"prompt":prompt,
                    "options":opts,"path":ids,"applicable_rule_id":str(app["node_id"]),
                    "sibling_rule_id":str(sib["node_id"]),"original_action":old,"sibling_action":new,
                    "original_answer":gold,"original_answer_text":str(row.get("answer_text",old)),
                    "closed_book_prompt":profile+"\n\n"+prompt,
                    "rule_text":rule+"\n"+sibling_rule}
            variants=[]
            for variant in ("A_original","B_applicable_rule_changed","C_sibling_rule_changed"):
                r1,r2=rule,sibling_rule; ans=gold
                if variant=="B_applicable_rule_changed":
                    r1=f"IF {app.get('condition','')}, THEN {new}."; ans=new_label
                elif variant=="C_sibling_rule_changed":
                    r2=f"IF {sib.get('condition','')}, THEN {old}."
                guide="Relevant decision rules:\n"+r1+"\n"+r2+"\n\n"
                full=guide+common["closed_book_prompt"]
                # Exact character spans are retained for tokenizer offset mapping.
                app_text=r1; sib_text=r2
                common_variant={**common,"variant":variant,"prompt_text":full,"gold_answer":ans,
                                "applicable_rule_text":app_text,"sibling_rule_text":sib_text,
                                "applicable_char_span":[full.index(app_text),full.index(app_text)+len(app_text)],
                                "sibling_char_span":[full.index(sib_text),full.index(sib_text)+len(sib_text)]}
                variants.append(common_variant)
            candidates.extend(variants); counts["eligible_examples"]+=1
        except (ValueError,KeyError,TypeError,json.JSONDecodeError):
            counts["invalid_record"]+=1
    # Sample at example-group level, preserving all A/B/C variants.
    groups={}
    for r in candidates: groups.setdefault(r["id"],[]).append(r)
    keys=sorted(groups); rng.shuffle(keys); keys=keys[:a.limit]
    import os
    os.makedirs(os.path.dirname(os.path.abspath(a.output)),exist_ok=True)
    with open(a.output,"w",encoding="utf-8") as f:
        for k in keys:
            for r in groups[k]: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    print(json.dumps({"output":a.output,"examples":len(keys),"rows":sum(len(groups[k]) for k in keys),"counts":dict(counts)},ensure_ascii=False,indent=2))
if __name__=="__main__": main()
