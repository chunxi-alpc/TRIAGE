"""Reconstruct the registered inputs from the original MEGA-CDP JSONL files."""
import json
from pathlib import Path

from .tree import clean_anchor, tree_guideline

QUESTION = "请依据完整指南给出患者的临床决策路径及最终处置。最后输出 PATH: 经过的规则编号；FINAL: 终点处置的L编号。"
SPLITS = Path(__file__).with_name("splits.json")


def read_rows(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def prepare(dataset, output, split):
    index = json.loads(SPLITS.read_text())
    partition = "test" if split == "official_test" else "train_val"
    folder = dataset / partition
    wanted = index.get(split)
    selected = None if wanted is None else {
        item if isinstance(item, str) else item["original"] for item in wanted
    }
    trees = {int(r["guideline_id"]): r for r in read_rows(folder / "decision_trees.jsonl")}
    paths = {int(r["guideline_id"]): r["paths"] for r in read_rows(folder / "case_paths.jsonl")}
    guides = {}
    saved = {}

    def cases():
        for item in read_rows(folder / "vignette_cases.jsonl"):
            gid, pid = int(item["guideline_id"]), int(item["path_id"])
            source = f"mega:{partition}:{gid}"
            cid = f"{source}:{pid}"
            if selected is not None and cid not in selected:
                continue
            if gid not in guides:
                guides[gid] = tree_guideline(json.loads(trees[gid]["text"])["decision_tree"])
            guide, mapping = guides[gid]
            path = paths[gid][pid - 1]
            branches = {"yes": "Y", "no": "N"}
            gold = [
                f"R{mapping[str(node['index'])]['node']}{branches[node['decision_outcome']]}"
                for node in path[:-1]
            ]
            leaf = mapping[str(path[-1]["index"])]
            row = dict(case_id=cid, source_id=index["sources"][source], patient=item["text"],
                       question=QUESTION, guideline=guide, gold_rules=gold, gold_leaf=f"L{leaf['node']}")
            if selected is None:
                yield row
            else:
                saved[cid] = row
        for item in wanted or []:
            if isinstance(item, str):
                yield saved[item]
            else:
                row = saved[item["original"]].copy()
                _, body = clean_anchor(row["patient"])
                row.update(case_id=item["case_id"], patient=body + "\nAge at current encounter: " + item["age"] + " years.",
                           target_rule=item["target_rule"])
                del row["gold_rules"], row["gold_leaf"]
                yield row

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial")
    with temporary.open("w", encoding="utf8") as handle:
        for row in cases():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output)
