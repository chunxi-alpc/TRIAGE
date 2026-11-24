"""Prepare private offline forms and compare confirmed clinical returns with archived labels."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import io
import os
from pathlib import Path
import time
from zipfile import ZipFile, ZIP_DEFLATED

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_PRIVATE = Path.home() / "aster-v3-review-followup-private"
ARCHIVED = REPO / "reports/aster_v3_selective_20261124/s1/first_errors.csv"
BATCHES = ("disputes", "queue")
CAUSES = ("比较符或阈值", "测量对象或单位", "时间条件", "逻辑、否定或例外", "患者事实或跨句定位",
          "事实缺失或矛盾", "定义或适用范围", "技术或数据问题", "其他", "无法判定")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save_private(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("x", encoding="utf-8", newline="") as f:
        os.chmod(tmp, 0o600)
        f.write(text)
    os.replace(tmp, path)


def cards_and_key(folder):
    file = folder / "reviewer/cards.jsonl"
    with file.open(encoding="utf-8") as f:
        cards = [json.loads(line) for line in f if line.strip()]
    keys = read_csv(folder / "coordinator/key.csv")
    ids = [card["blind_id"] for card in cards]
    if len(ids)!=len(set(ids)) or len(keys)!=len(ids) or set(ids)!={r["blind_id"] for r in keys}:
        raise ValueError("盲卡和协调人对照表未一一对应")
    return cards, keys


def context(card, folder):
    def display(value):
        return value if isinstance(value,str) else json.dumps(value, ensure_ascii=False)
    conditions = card.get("guideline_conditions",[])
    definitions = card.get("guideline_definitions",[])
    blocks = ["指南节点条件\n" + "\n".join(display(v) for v in conditions),
              "术语定义\n" + ("\n".join(display(v) for v in definitions) or "归档未单列定义。")]
    relative = card.get("guide_path") or card.get("guideline_document")
    if isinstance(relative,str) and relative:
        file = (folder/relative).resolve()
        if file.is_relative_to(folder.resolve()) and file.is_file() and file.suffix.lower() in (".md",".txt"):
            blocks.append("原指南文档\n"+file.read_text(encoding="utf-8-sig"))
    return "\n\n".join(blocks)


def make_form(cards, folder, destination):
    contexts, index, safe = [], {}, []
    for card in cards:
        body = context(card, folder)
        if body not in index:
            index[body] = len(contexts)
            contexts.append(body)
        safe.append(dict(blind_id=card["blind_id"], condition=card["condition"],
                         patient=card["patient"], context_index=index[body]))
    payload = json.dumps(dict(cards=safe, contexts=contexts, causes=CAUSES), ensure_ascii=False)
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&","\\u0026")
    template = (HERE/"form.html").read_text()
    if template.count("__BLIND_FORM_PAYLOAD__")!=1:
        raise ValueError("缺少盲审表入口")
    save_private(destination,template.replace("__BLIND_FORM_PAYLOAD__",payload))


def review_batches(cards, file):
    """Use the issued batch assignments. Each blind ID occurs once."""
    assignments = read_csv(file)
    ids = [r['blind_id'] for r in assignments]
    if len(set(ids))!=len(ids) or set(ids)!={c['blind_id'] for c in cards}:
        raise ValueError("分批记录与盲卡未一一对应")
    number = {r['blind_id']:r['batch'] for r in assignments}
    groups=defaultdict(list)
    for card in cards:
        groups[number[card['blind_id']]].append(card)
    return [groups[n] for n in sorted(groups)]


def normalize(row, patient):
    decision = str(row.get("decision","")).strip()
    reviewer = str(row.get("reviewer_id","")).strip()
    explanation = str(row.get("explanation","")).strip()
    if not (decision or explanation or str(row.get("cause","")).strip()):
        return None
    cause = str(row.get("cause","")).strip()
    if cause not in CAUSES:
        cause = "其他" if cause else "无法判定"
    coordinates = [row.get(k,"") for k in ("evidence_start","evidence_end")]
    span = ""
    if any(str(x).strip() for x in coordinates):
        try:
            start,end = map(int,coordinates)
        except (ValueError,TypeError):
            return {"state":"incomplete"}
        if not 0<=start<end<=len(patient):
            return {"state":"incomplete"}
        span = f"{start}-{end}"
    if not reviewer or not explanation or decision not in ("yes","no","indeterminate"):
        return {"state":"incomplete"}
    if decision != "indeterminate" and not span:
        return {"state":"incomplete"}
    return dict(state="filled",reviewer_id=reviewer,decision=decision,cause=cause,span=span)


def reconcile(records, cards, keys, gold, confirmed):
    filled = defaultdict(lambda:defaultdict(set))
    partial, unconfirmed = set(), set()
    for r in records:
        uid = r["blind_id"]
        if uid not in cards:
            raise ValueError("回填盲号不属于本批材料")
        record = normalize(r,cards[uid]["patient"])
        if record is None:
            continue
        if record["state"]!="filled":
            partial.add(uid)
        elif record["reviewer_id"] not in confirmed:
            unconfirmed.add(uid)
        else:
            filled[uid][record["reviewer_id"]].add((record["decision"],record["cause"],record["span"]))
    output = []
    for uid,k in keys.items():
        old = gold[k["case_id"],k["gid"]]
        reports = filled.get(uid,{})
        decisions = {d for variants in reports.values() for d,_,_ in variants}
        state = "pending"
        answer, cause, same = "", "", ""
        if len(decisions)>1:
            state="disagreement"
        elif decisions:
            answer = next(iter(decisions))
            state="indeterminate" if answer=="indeterminate" else "reviewed"
            causes={c for variants in reports.values() for _,c,_ in variants}
            cause=next(iter(causes)) if len(causes)==1 else "归因待协调"
            if state=="reviewed":
                same=int((answer=="yes")==bool(old))
        elif uid in unconfirmed:
            state="reviewer_unconfirmed"
        elif uid in partial:
            state="incomplete"
        output.append(dict(blind_id=uid,source_id=k["source_id"],case_id=k["case_id"],gid=k["gid"],
                           batch=k["batch"],archived_label=old,decision=answer,state=state,
                           matches_archive=same,cause=cause,confirmed_readers=len(reports)))
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE)
    parser.add_argument("--confirmed-reviewer", action="append", default=[],
                        help="负责人已核实临床资格并授权参与本批次的审核者代号")
    parser.add_argument("--summary", type=Path, help="可选的去标识汇总 JSON")
    args=parser.parse_args()
    os.umask(0o077)
    started=time.process_time()
    root=args.private_root
    gold={}
    for r in read_csv(ARCHIVED):
        key=r["case_id"],r["gid"]
        if gold.setdefault(key,int(r["gold"]))!=int(r["gold"]):
            raise ValueError("同一病例节点存在不同的归档标签")
    cards,keys,records={},{},[]
    handoff = root / "clinical_review_forms"
    sizes = []
    for batch in BATCHES:
        folder=root/batch
        batch_cards,batch_keys=cards_and_key(folder)
        if set(cards)&{c["blind_id"] for c in batch_cards}:
            raise ValueError("两个批次使用了同一盲号")
        cards.update({c["blind_id"]:c for c in batch_cards})
        keys.update({k["blind_id"]:dict(k,batch=batch) for k in batch_keys})
        parts = [batch_cards] if batch=="disputes" else review_batches(batch_cards,folder/"reviewer/batches.csv")
        for part in parts:
            number = len(sizes)+1
            make_form(part, folder/"reviewer", handoff/f"batch_{number:02}.html")
            sizes.append(len(part))
        records.extend(read_csv(folder/"reviewer/responses.csv"))
    instructions = ("本包按批次提供离线盲审表。请在浏览器中打开 batch_01.html，再打开获分配的后续批次。\n"
                    "填写审核者代号，阅读当前条件、原指南和患者文本，给出判断并选取决定性事实，填写理由。"
                    "无法判定时请写明缺失的事实。页面可随时导出或重新载入保存的记录。\n"
                    "每批导出的 JSON 文件请使用不同文件名，通过项目已授权渠道退交负责人。文件内包含审核记录，请妥善保存。\n")
    save_private(handoff/"使用说明.txt", instructions)
    archive = root / "clinical_review_forms.zip"
    temp = archive.with_name(f".clinical_review_forms.{os.getpid()}.zip")
    with ZipFile(temp, "x", compression=ZIP_DEFLATED) as z:
        z.write(handoff/"使用说明.txt","使用说明.txt")
        for i in range(1,len(sizes)+1):
            name=f"batch_{i:02}.html"
            z.write(handoff/name,name)
    os.replace(temp,archive)
    returns=root/"returns"
    returns.mkdir(exist_ok=True)
    for file in sorted(returns.iterdir()):
        if file.suffix==".json":
            records.extend(json.loads(file.read_text(encoding="utf-8-sig")))
        elif file.suffix==".csv":
            records.extend(read_csv(file))
    result=reconcile(records,cards,keys,gold,set(args.confirmed_reviewer))
    dest=root/"coordinator"
    dest.mkdir(exist_ok=True)
    buffer=io.StringIO(newline="")
    writer=csv.DictWriter(buffer,fieldnames=result[0],lineterminator="\n")
    writer.writeheader()
    writer.writerows(result)
    save_private(dest/"reconciled.csv", buffer.getvalue())
    valid=[r for r in result if r["state"] in ("reviewed","indeterminate")]
    cause_sources=defaultdict(set)
    for r in valid:
        if r["cause"] in CAUSES[:-2]:
            cause_sources[r["cause"]].add(r["source_id"])
    summary=dict(batches={b:dict(Counter(r["state"] for r in result if r["batch"]==b)) for b in BATCHES},
                 total_nodes=len(result), sources=len({r["source_id"] for r in result}),completed=len(valid),
                 determinate=sum(r["state"]=="reviewed" for r in valid),
                 determinate_matches_archive=sum(r["matches_archive"]==1 for r in valid),
                 determinate_differs_from_archive=sum(r["matches_archive"]==0 for r in valid),
                 indeterminate=sum(r["state"]=="indeterminate" for r in valid),
                 cause_source_counts={k:len(v) for k,v in cause_sources.items()},
                 label_changes=0,body_or_guide_text_exported_to_summary=False,cpu_seconds=time.process_time()-started)
    summary["private_form_batch_sizes"] = sizes
    if args.summary:
        args.summary.parent.mkdir(parents=True,exist_ok=True)
        args.summary.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary,ensure_ascii=False))


if __name__=="__main__":
    main()
