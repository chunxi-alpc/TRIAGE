"""Freeze identifiers first; preserve released content exactly."""
from .common import *
from difflib import SequenceMatcher
from collections import Counter

def render(tokenizer,prompt,task="binary"):
    messages=([dict(role="system",content=SYSTEM)] if task=="binary" else [])+[dict(role="user",content=prompt)]
    return tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)

def alignment(left,right):
    n=min(len(left),len(right));prefix=0
    while prefix<n and left[prefix]==right[prefix]:prefix+=1
    suffix=0
    while suffix<n-prefix and left[-1-suffix]==right[-1-suffix]:suffix+=1
    positions=dict(before=[prefix-1,prefix-1] if prefix else None,
        after=[len(left)-suffix,len(right)-suffix] if suffix else None,
        judgment=[len(left)-1,len(right)-1])
    changed=[op for op in SequenceMatcher(None,left,right,autojunk=False).get_opcodes() if op[0]!="equal"]
    distances=[]
    for _,a,b,c,d in changed:
        for length,lo,hi in [(len(left),a,b),(len(right),c,d)]:
            if hi>lo:distances.extend([length-hi,length-1-lo])
            else:distances.append(length-1-min(lo,length-1))
    return dict(positions=positions,prefix=prefix,suffix=suffix,
        nearest_change=min(distances) if distances else None,
        farthest_change=max(distances) if distances else None,
        lengths=[len(left),len(right)])

def build_freeze():
    path=ROOT/"data/frozen_heads.json"
    if path.exists():return read(path)
    values={}
    for model,selpath,total in [("qwen",QWEN/"m1/selection.json",1152),
                                ("medgemma",REPAIR/"selection/selection.json",1984)]:
        sel=read(selpath)
        values[model]={k:sel[k] for k in ["H_A","H_P","H","T_h"]}
        values[model].update(total_heads=total,candidates={})
        for kind in ["A","P"]:
            ranking=read(LEGACY[model]/f"m1/attribution_{kind}.json")["ranking"]
            values[model]["candidates"][kind]=[x["head"] for x in ranking[:128]]
    values["medgemma_history"]=read(MED/"m1/selection.json")
    assert [len(values[m][k]) for m in ["qwen","medgemma"] for k in ["H_A","H_P","H"]]==[22,39,51,125,49,141]
    write(path,values);return values

def prepare_diagnostics():
    from transformers import AutoTokenizer
    target=ROOT/"data/diagnostic_pairs.jsonl"
    if target.exists():return rows(target)
    views={x["view_id"]:x for x in rows(QWEN/"data/views.jsonl")}
    pairs=[p for p in rows(QWEN/"data/pairs.jsonl") if p["kind"] in ["A","P"] and p["split"] in ["fit","dev"]]
    tokenizers={m:AutoTokenizer.from_pretrained(p,local_files_only=True) for m,p in MODELS.items()}
    allrows=[]
    for p in pairs:
        x=dict(p,alignment={})
        for m,tok in tokenizers.items():
            ids=[tok.encode(render(tok,views[p[s]]["prompt"]),add_special_tokens=False) for s in ["left","right"]]
            x["alignment"][m]=alignment(*ids)
        far=[x["alignment"][m]["farthest_change"] or 0 for m in MODELS]
        x["distance_group"]="within" if max(far)<1024 else "across" if min(far)>=1024 else "mixed"
        allrows.append(x)
    write_rows(ROOT/"data/pair_distances.jsonl",allrows)
    selected=[]
    for split,num in [("fit",24),("dev",8)]:
        a=[p for p in allrows if p["kind"]=="A" and p["split"]==split]
        chosen=[]
        for group in ["within","across"]:
            chosen+=balanced([p for p in a if p["distance_group"]==group],num//2)
        chosen+=balanced([p for p in a if p["pair_id"] not in {x["pair_id"] for x in chosen}],num-len(chosen))
        selected+=chosen
        selected+=balanced([p for p in allrows if p["kind"]=="P" and p["split"]==split],num)
    assert len(selected)==64
    residual=[]
    for kind,group in [("A","within"),("A","across"),("P",None)]:
        pool=[p for p in selected if p["kind"]==kind and (group is None or p["distance_group"]==group)]
        residual+=balanced(pool,4)
    assert len(residual)==12,"The existing pool cannot supply the fixed residual sample."
    residual_ids={p["pair_id"] for p in residual}
    for p in selected:p["residual"]=p["pair_id"] in residual_ids
    write_rows(target,selected)
    ids={p[s] for p in selected for s in ["left","right"]}
    write_rows(ROOT/"data/diagnostic_views.jsonl",[views[i] for i in sorted(ids)])
    write(ROOT/"data/diagnostic_sampling.json",dict(seed=SEED,
      counts={str(k):v for k,v in Counter((p["kind"],p["split"],p["distance_group"]) for p in selected).items()},
      source_counts=dict(Counter(p["source_id"] for p in selected)),selection_uses_effects=False,
      pair_policy="仅复用现有拟合/开发构造配对做机制回顾；不生成新文本；不新增测试修补"))
    return selected

def parse_tree(text):
    if isinstance(text,dict):return text.get("decision_tree",text)
    obj=json.loads(text)
    return obj.get("decision_tree",obj)

def flatten_tree(tree):
    nodes={}
    def walk(node):
        nodes[str(node["index"])]=node
        for side in ["yes","no"]:
            if isinstance(node.get(side),dict):walk(node[side])
    walk(tree);return nodes

def mega_prompt(case,node):
    return ("完整决策树（原文）：\n"+case["tree_text"]+"\n\n病历（原文）：\n"+case["text"]+
      "\n\n当前节点："+str(node["index"])+"\n当前条件："+node["description"]+
      "\n请判断当前条件。只输出“是”或“否”。\n答案：")

def prepare_baselines():
    import pyarrow.parquet as pq
    mgpath=ROOT/"data/medguide_200.jsonl";megapath=ROOT/"data/mega_200.jsonl"
    if not mgpath.exists():
        native=pq.read_table(REPO/"data/MedGUIDE/data/train-00000-of-00001.parquet").to_pylist()
        pool=[dict(sample_id=f"medguide:{i}",row_index=i,source_id=r["disease"],native=r) for i,r in enumerate(native)]
        chosen=balanced(pool,200);write_rows(mgpath,chosen)
    if not megapath.exists():
        base=REPO/"data/MEGA-CDP/test"
        trees={r["filename"]:r for r in rows(base/"decision_trees.jsonl")}
        paths={r["filename"]:r for r in rows(base/"case_paths.jsonl")}
        pool=[]
        for i,r in enumerate(rows(base/"vignette_cases.jsonl")):
            tree=trees.get(r["filename"]);pathrow=paths.get(r["filename"])
            # Only original published matching records; do not fix or relabel missing data.
            annotated=pathrow["paths"][int(r["path_id"])-1] if pathrow and 0<int(r["path_id"])<=len(pathrow["paths"]) else None
            pool.append(dict(sample_id=f"mega:{r['guideline_id']}:{r['path_id']}",row_index=i,
                source_id=r["filename"],native=r,text=r["text"],
                tree_text=tree["text"] if tree else None,annotated_path=annotated))
        chosen=balanced(pool,200);write_rows(megapath,chosen)
    mg,mega=rows(mgpath),rows(megapath)
    assert len(mg)==len(mega)==200
    write(ROOT/"data/baseline_sampling.json",dict(seed=SEED,
      medguide=dict(n=200,sources=len({r["source_id"] for r in mg}),options=dict(Counter(r["native"]["option_count"] for r in mg))),
      mega=dict(n=200,sources=len({r["source_id"] for r in mega}),official_test_cases=7458),
      scope="来源均衡探索样本；官方7458例属于已有基准，不是新的确认集。原文超长单列，不截断或改写。"))

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    build_freeze();prepare_diagnostics();prepare_baselines()
    write(ROOT/"configuration.json",dict(seed=SEED,bootstrap=BOOTSTRAPS,epsilon=EPSILON,rho=RHO,
      models={k:str(v) for k,v in MODELS.items()},gpus=[0,1],precision=PRECISION,
      diagnostic_pairs=64,residual_pairs=12,medguide=200,mega=200,
      new_selection=False,M1_v2=False,M3=False,legacy_read_only=True))
    status("prepare",state="complete")
