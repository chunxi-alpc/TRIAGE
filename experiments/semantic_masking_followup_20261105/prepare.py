"""Sample guideline sources and retain original text, mappings and case labels."""
import collections
import json
import random
import re
import shutil
from pathlib import Path
from common import CONFIG, DATA, ROOT, blocks_for, compact, read_rows, write_json, write_rows

def source_identity(text):
    head = text[:700]
    if "NCCN Clinical Practice Guidelines in Oncology" in head:
        after = text[text.index("NCCN Clinical Practice Guidelines in Oncology"):]
        before_version = re.split(r"(?:Version|版本)\s*\d", after[:1100])[0]
        titles = re.findall(r"^#+\s+(.+)$", before_version, re.M)
        title = next((t for t in titles if "NCCN" not in t), None)
        if title:
            return "NCCN", title, "NCCN Clinical Practice Guidelines in Oncology"
    # Organization in the author area, not a citation somewhere in the article.
    author_area = re.split(r"【摘要】|\[摘要\]|#.*摘要|\bAbstract\b", text[:2000])[0]
    lines = author_area.splitlines()
    author = next((line for line in lines if re.match(
        r"^[#\s]*中华医学会[^。]{0,160}(?:分会|学组|委员会)", line)), None)
    if author:
        titles = re.findall(r"^#+\s+(.+)$", author_area, re.M)
        title = next((t for t in titles if "中华医学会" not in t and len(t)>5), None)
        if title:
            return "CMA", title, author
    return None

def flatten(node, ancestors=()):
    if not isinstance(node, dict):
        return []
    if node.get("leaf") or not node.get("description"):
        return []
    current = dict(index=node["index"], condition=node["description"],
                   ancestors=list(ancestors))
    out = [(current, node)]
    for branch in ("yes", "no"):
        history = tuple(ancestors) + (dict(index=node["index"],
                    condition=node["description"], answer=branch),)
        out += flatten(node.get(branch), history)
    return out

def terminal_nodes(node):
    if not isinstance(node, dict):
        return []
    if node.get("leaf") or "conclusion" in node:
        return [node]
    return terminal_nodes(node.get("yes")) + terminal_nodes(node.get("no"))

def choose_context(blocks, queries, condition):
    """Source mappings first, then neighboring and definition candidate paragraphs."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    nonempty = [b for b in blocks if len(b["text"]) > 10]
    normal = [compact(b["text"]) for b in nonempty]
    chosen, mapped = set(), []
    for query in queries:
        q = compact(query)
        matches = [b["block_id"] for b,n in zip(nonempty,normal)
                   if q and (q in n or (len(n)>60 and n in q))]
        chosen.update(matches)
        mapped.append(dict(quote=query, block_ids=matches, match="normalized_exact" if matches else "unlocated"))
    corpus = [b["text"] for b in nonempty]
    if corpus:
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2,4), max_features=45000)
        matrix = vectorizer.fit_transform(corpus)
        score = (matrix @ vectorizer.transform([" ".join(queries) + " " + condition]).T).toarray().ravel()
        ranked = [nonempty[i]["block_id"] for i in score.argsort()[::-1]]
    else:
        ranked = []
    prioritized = list(sorted(chosen))
    for bid in list(prioritized):
        for offset in (-1,1):
            if 0 <= bid+offset < len(blocks) and bid+offset not in prioritized:
                prioritized.append(bid+offset)
    prioritized += [bid for bid in ranked[:18] if bid not in prioritized]
    selected, size = [], 0
    for bid in prioritized:
        block = blocks[bid]
        if size + len(block["text"]) > 24000 and selected:
            continue
        selected.append(block)
        size += len(block["text"])
    return sorted(selected,key=lambda b:b["start"]), mapped

def prepare():
    rng = random.Random(CONFIG["seed"])
    base = Path(CONFIG["mega_root"])
    trees, paths, cases = {}, {}, collections.defaultdict(list)
    parse_failures = []
    for split in ("train_val","test"):
        for row in read_rows(base/split/"decision_trees.jsonl"):
            try:
                tree = json.loads(row["text"]) if isinstance(row["text"],str) else row["text"]
                trees[row["filename"]] = (split,row,tree)
            except (ValueError,TypeError) as exc:
                parse_failures.append(dict(filename=row["filename"],reason=str(exc)))
        for row in read_rows(base/split/"case_paths.jsonl"):
            paths[row["filename"]] = row["paths"]
        for row in read_rows(base/split/"vignette_cases.jsonl"):
            cases[row["filename"]].append(row)
    eligible = collections.defaultdict(list)
    source_inventory = []
    for p in sorted(base.glob("*/guideline_md/*/*.md")):
        text = p.read_text()
        identity = source_identity(text)
        if not identity:
            continue
        publisher,title,evidence = identity
        record = dict(source_id=p.stem,publisher=publisher,title=title,
                      original_path=str(p),publisher_evidence=evidence,
                      tree_available=p.stem in trees,case_paths_available=p.stem in paths)
        source_inventory.append(record)
        eligible[publisher].append(record)
    sources = []
    for publisher in ("NCCN","CMA"):
        pool = eligible[publisher]
        rng.shuffle(pool)
        titles = set()
        for item in pool:
            # A single edition per title prevents the same guideline leaking across splits.
            key = compact(item["title"])
            if key in titles:
                continue
            titles.add(key)
            sources.append(item)
            if sum(s["publisher"]==publisher for s in sources)==CONFIG["sources_per_publisher"]:
                break
        subset = [s for s in sources if s["publisher"]==publisher]
        if len(subset) != CONFIG["sources_per_publisher"]:
            raise RuntimeError(f"Only {len(subset)} usable distinct {publisher} source titles")
        for i,item in enumerate(subset):
            item["split"] = "development" if i<CONFIG["development_per_publisher"] else "test"
    (DATA/"sources").mkdir(exist_ok=True)
    nodes, extra, missing = [], [], []
    for source in sources:
        sid=source["source_id"]
        text=Path(source["original_path"]).read_text()
        (DATA/"sources"/(sid+".md")).write_text(text)
        if sid not in trees:
            missing.append(dict(source_id=sid,reason="decision_tree_missing"))
            continue
        original_split,raw,tree=trees[sid]
        flat=flatten(tree["decision_tree"])
        rng.shuffle(flat)
        sampled=flat[:CONFIG["nodes_per_source"]]
        candidates=[pair for pair in flat[CONFIG["nodes_per_source"]:]
                    if re.search(r"\d",pair[0]["condition"])]
        rng.shuffle(candidates)
        chosen= [(pair,"statistical") for pair in sampled]+[(pair,"construction") for pair in candidates[:2]]
        block_list=blocks_for(text)
        mapping=tree.get("node_guideline_mapping",{})
        for (node,subtree),role in chosen:
            index=node["index"]
            matching=[]
            for case in cases[sid]:
                pid=case["path_id"]
                source_paths=paths.get(sid,[])
                if not isinstance(pid,int) or not 1<=pid<=len(source_paths):
                    continue
                path=source_paths[pid-1]
                step=next((x for x in path if x.get("index")==index and "decision_outcome" in x),None)
                if step:
                    matching.append((case,step,path))
            rng.shuffle(matching)
            case,step,path=matching[0] if matching else (None,None,None)
            rule_mapping=mapping.get(str(index),mapping.get(index,[]))
            action_map=[]
            for branch in ("yes","no"):
                for terminal in terminal_nodes(subtree.get(branch))[:2]:
                    for quote in mapping.get(str(terminal["index"]),[]):
                        if quote not in action_map:
                            action_map.append(quote)
            queries=list(dict.fromkeys(rule_mapping+action_map[:6]))
            context,matches=choose_context(block_list,queries,node["condition"])
            record=dict(node_id=f"{sid}:{index}",source_id=sid,split=source["split"],
                        publisher=source["publisher"],title=source["title"],role=role,
                        dataset_split=original_split,language=raw["language"],**node,
                        source_file=str(DATA/"sources"/(sid+".md")),blocks=context,
                        node_mapping=rule_mapping,action_mapping=action_map,
                        mapping_locations=matches,original_case=case["text"] if case else None,
                        original_case_id=f"{sid}:{case['path_id']}" if case else None,
                        original_gold=int(step["decision_outcome"]=="yes") if step else None,
                        original_path=path,case_origin="MEGA-CDP dataset-generated vignette",
                        source_original_available=True)
            (nodes if role=="statistical" else extra).append(record)
    # At most 80 construction nodes total; use statistical numeric nodes as a fallback pool.
    extras_needed = CONFIG["construct_max"]-len(extra)
    available=[dict(r,role="construction") for r in nodes if re.search(r"\d",r["condition"])]
    rng.shuffle(available)
    extra += available[:max(0,extras_needed)]
    construct_ids={r["node_id"] for r in extra[:CONFIG["construct_max"]]}
    unique={r["node_id"]:r for r in nodes+extra}
    # Reusing a statistical node for construction must not remove it from Q/A.
    for row in nodes:
        unique[row["node_id"]]=row
    for row in unique.values():
        row["try_construction"]=row["node_id"] in construct_ids
    pilot=[]
    for source in [s for s in sources if s["split"]=="development"]:
        rows=[r for r in unique.values() if r["source_id"]==source["source_id"]]
        candidates=[r for r in rows if r["try_construction"]]
        selected=candidates[:1]+[r for r in rows if not candidates or r["node_id"]!=candidates[0]["node_id"]][:1 if candidates else 2]
        pilot.extend(selected)
    pilot_ids={r["node_id"] for r in pilot[:CONFIG["pilot_nodes"]]}
    for row in unique.values():
        row["pilot"]=row["node_id"] in pilot_ids
    write_rows(DATA/"sources.jsonl",sources)
    write_rows(DATA/"source_inventory.jsonl",source_inventory)
    write_rows(DATA/"nodes.jsonl",unique.values())
    write_json(DATA/"sampling.json",dict(seed=CONFIG["seed"],sources=len(sources),
        development_sources=sum(s["split"]=="development" for s in sources),
        test_sources=sum(s["split"]=="test" for s in sources),statistical_nodes=len(nodes),
        construction_candidates=len(construct_ids),pilot_nodes=len(pilot_ids),
        missing_sources=missing,tree_parse_failures=parse_failures,
        selection="uniform fixed-seed source/title and node sampling; no prediction-based selection",
        originals="MEGA-CDP vignettes are dataset-generated, not natural clinical cases"))
    reference=Path(CONFIG["reference_report"])
    if reference.exists():
        shutil.copyfile(reference,ROOT/"results"/"existing_reference_swap.zh.md")
    print(json.dumps(dict(sources=len(sources),statistical_nodes=len(nodes),
                          construction_candidates=len(construct_ids),pilot_nodes=len(pilot_ids))))

if __name__=="__main__":
    prepare()
