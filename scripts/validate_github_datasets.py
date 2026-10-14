#!/usr/bin/env python3
import json, sys
from pathlib import Path
import pandas as pd
from resume_requested_downloads import valid, atomic, register, ROOT, RESULTS
specs={
 "ConFiQA":("source_byronBBL__Context-DPO.json",["README.md","ConFiQA/ConFiQA-MC.json","ConFiQA/ConFiQA-MR.json","ConFiQA/ConFiQA-QA.json"]),
 "MedChangeQA":("source_jvladika__MedChange.json",["README.md","Datasets/MedChangeQA.csv"]),
 "ClashEval":("source_ClashEval_history.json",["README.md","data/dataset/croissant.json","data/dataset/dataset.pqt"])
}
report={}
for name,(meta,files) in specs.items():
 d=json.loads((RESULTS/meta).read_text());items={i["path"]:i for i in d["tree"] if i["type"]=="blob"};dest=ROOT/"data"/name
 records=[]
 for fn in files:
  p=dest/fn;entry={"file":fn,"exists":p.is_file(),"official_git_blob_sha":items[fn]["sha"],"expected_bytes":items[fn]["size"],"verified":valid(p,items[fn])}
  if entry["verified"] and fn.endswith(".json") and "ConFiQA" in fn:
   a=json.loads(p.read_text());assert isinstance(a,list);entry.update(rows=len(a),fields=sorted(set().union(*(x.keys() for x in a))),field_types={k:type(v).__name__ for k,v in a[0].items()},first_record_question=a[0].get("question"))
  elif entry["verified"] and (fn.endswith(".csv") or fn.endswith(".pqt")):
   a=pd.read_csv(p) if fn.endswith(".csv") else pd.read_parquet(p)
   entry.update(rows=len(a),fields=a.columns.tolist(),dtypes={k:str(v) for k,v in a.dtypes.items()},null_counts={k:int(v) for k,v in a.isna().sum().items()})
   if name=="MedChangeQA":entry.update(question_count=int(a["Question"].nunique()),label_values={k:sorted(a[k].dropna().unique().tolist()) for k in ["Newest Label","Outdated Label"]},first_record_structure={k:type(v).__name__ for k,v in a.iloc[0].to_dict().items()})
   if name=="ClashEval":entry["first_record_structure"]={k:type(v).__name__ for k,v in a.iloc[0].to_dict().items()}
  records.append(entry)
 complete=all(i["verified"] for i in records)
 report[name]={"status":"passed" if complete else "pending","source_tree_sha":d["sha"],"files":records}
 if complete:
  extra={"validation_file":str(RESULTS/"github_datasets_validation.json"),"transport":"official raw.githubusercontent.com with Git blob SHA verification"}
  if name=="ConFiQA":
   extra.update(data_files=files[1:],split_files={s:[f"ConFiQA/ConFiQA-{s}.json"] for s in ["QA","MR","MC"]},rows_by_split={i["file"].split("-")[-1].split(".")[0]:i["rows"] for i in records if "rows" in i},download_scope="all three official benchmark JSON files and README; images excluded")
  elif name=="MedChangeQA":extra.update(records=records[-1]["rows"],download_scope="official MedChangeQA CSV and README; auxiliary datasets excluded")
  elif name=="ClashEval":extra.update(records=records[-1]["rows"])
  register(name,extra)
atomic(RESULTS/"github_datasets_validation.json",report)
print(json.dumps({k:v["status"] for k,v in report.items()}))
