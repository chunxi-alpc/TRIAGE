#!/usr/bin/env python3
import csv, json, time, traceback
from pathlib import Path
from resume_requested_downloads import ROOT, RESULTS, atomic, metadata, hf_paths, valid, hf_download, register
import resume_requested_downloads as downloader
original_fetch=downloader.fetch
def cached_text_fetch(url,path,item,refresh=False,headers=None):
    if not item.get("lfs"):
        url=url.replace("/resolve/","/raw/")
    return original_fetch(url,path,item,refresh=refresh and bool(item.get("lfs")),headers=headers)
downloader.fetch=cached_text_fetch
REPORT=RESULTS/"text_datasets_validation.json"
report={"status":"running","started":time.time(),"device":"CPU","images_requested":False,"datasets":{}}

def inspect(name,repo):
    meta=metadata(repo); dest,_=hf_paths(repo,meta)
    items=[i for i in meta["siblings"] if "/imgs/" not in i["rfilename"]]
    checks=[{"file":i["rfilename"],"bytes":i["size"],"digest_type":"sha256" if i.get("lfs") else "git_blob_sha1","expected_digest":i.get("lfs",{}).get("sha256",i.get("blobId")),"valid":valid(dest/i["rfilename"],i)} for i in items]
    result={"path":str(dest),"source":f"https://huggingface.co/datasets/{repo}","revision":meta["sha"],"files":checks,"all_selected_official_digests_valid":all(c["valid"] for c in checks),"selected_files":len(items),"selected_bytes":sum(i["size"] for i in items),"status":"validated"}
    if not result["all_selected_official_digests_valid"]:raise ValueError(name+": invalid official file digests")
    if name=="MedCalc-Bench":
        result["splits"]={}
        for split in ["train","test"]:
            file=dest/(split+"_data_11_18_final.csv")
            with file.open(newline="",encoding="utf-8-sig") as handle:
                reader=csv.DictReader(handle);rows=list(reader)
                needed=["Patient Note","Question","Ground Truth Answer"]
                bad=[i+1 for i,row in enumerate(rows) if any(not row.get(k," ").strip() for k in needed) or None in row]
                result["splits"][split]={"file":file.name,"rows":len(rows),"columns":reader.fieldnames,"missing_required_values_or_extra_fields":bad}
                if bad:raise ValueError(name+": malformed or incomplete required rows")
    elif name=="FaithEval-unanswerable":
        import pyarrow.parquet as pq
        file=dest/"data/test-00000-of-00001.parquet";table=pq.read_table(file);rows=table.to_pylist()
        bad=[i+1 for i,row in enumerate(rows) if any(row.get(k) is None or len(row[k])==0 for k in ["qid","question","context","answers"])]
        ids=[row["qid"] for row in rows]
        result["splits"]={"test":{"file":str(file.relative_to(dest)),"rows":table.num_rows,"columns":table.column_names,"null_counts":{n:table[n].null_count for n in table.column_names},"missing_required_values":bad,"unique_qids":len(set(ids)),"subset_counts":{s:sum(r["subset"]==s for r in rows) for s in sorted(set(r["subset"] for r in rows))}}}
        if bad:result.setdefault("source_record_warnings",[]).append("Official parquet contains empty required values at rows: "+str(bad)+"; digests match, originals preserved")
    else:
        result["splits"]={};total_md=0
        for split in ["train_val","test"]:
            records={};summaries={}
            for filename in ["case_paths.jsonl","decision_trees.jsonl","gt_paths.jsonl","vignette_cases.jsonl"]:
                file=dest/split/filename;rows=[]
                with file.open(encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():rows.append(json.loads(line))
                required={"guideline_id","filename","language"}
                required.add("paths" if filename=="case_paths.jsonl" else "output_text" if filename=="gt_paths.jsonl" else "text")
                if filename in ["gt_paths.jsonl","vignette_cases.jsonl"]:required.add("path_id")
                bad=[i+1 for i,r in enumerate(rows) if any(k not in r or r[k] is None for k in required)]
                keys=[(r["guideline_id"],r["filename"],r.get("path_id")) for r in rows]
                records[filename]=rows
                summaries[filename]={"rows":len(rows),"unique_guidelines":len(set(r["guideline_id"] for r in rows)),"duplicate_keys":len(keys)-len(set(keys)),"missing_required_fields":bad}
                if bad:raise ValueError(name+": incomplete JSONL record")
            md=list((dest/split/"guideline_md").glob("*/*.md"));total_md+=len(md)
            for file in md:
                if not file.read_text(encoding="utf-8").strip():raise ValueError("empty markdown: "+str(file))
            mdnames={f.stem for f in md}
            missing_md=sorted({r["filename"] for rs in records.values() for r in rs}-mdnames)
            gt={(r["guideline_id"],r["filename"],r["path_id"]) for r in records["gt_paths.jsonl"]}
            vignette={(r["guideline_id"],r["filename"],r["path_id"]) for r in records["vignette_cases.jsonl"]}
            tree={(r["guideline_id"],r["filename"]) for r in records["decision_trees.jsonl"]}
            missing_tree=sorted({(r["guideline_id"],r["filename"]) for rs in records.values() for r in rs}-tree)
            result["splits"][split]={"files":summaries,"guideline_markdown_files":len(md),"records_without_markdown":missing_md,"gt_and_vignette_key_sets_match":gt==vignette,"gt_only_keys":sorted(gt-vignette),"vignette_only_keys":sorted(vignette-gt),"records_without_decision_tree":missing_tree,"case_path_count":sum(len(r["paths"]) for r in records["case_paths.jsonl"])}
            if missing_md or missing_tree or gt!=vignette:result.setdefault("source_record_warnings",[]).append(split+": official source record alignment differs; originals preserved")
        if total_md!=2274:raise ValueError("MEGA-CDP: expected 2274 guideline Markdown files plus root README")
        result["guideline_markdown_files"]=total_md
        result["total_markdown_files_including_readme"]=total_md+1
        result["images_deliberately_excluded"]=True
        register(name,{"status":"downloaded","complete":True,"contains_all_original_files":False,"images_requested":False,"image_policy":"images deliberately excluded by user request","download_scope":"all original train_val/test JSONL and 2274 Markdown guidelines plus README; official digests verified; images deliberately excluded","markdown_files_downloaded":total_md+1,"markdown_files_expected":2275,"guideline_markdown_files_downloaded":total_md,"guideline_markdown_files_expected":2274,"available_splits":["test","train_val"]})
    report["datasets"][name]=result
    atomic(REPORT,report)
    register(name,{"validation_file":str(REPORT),"verification":"passed","record_counts":{s: x.get("rows",{f:d["rows"] for f,d in x.get("files",{}).items()}) for s,x in result["splits"].items()}})
    return result

try:
    for name,repo in [("MedCalc-Bench","ncbi/MedCalc-Bench"),("FaithEval-unanswerable","Salesforce/FaithEval-unanswerable-v1.0")]:inspect(name,repo)
    hf_download("MEGA-CDP","nchennnn/MEGA-CDP",lambda i:"/imgs/" not in i["rfilename"],workers=2)
    inspect("MEGA-CDP","nchennnn/MEGA-CDP")
    report["status"]="passed"
except Exception as error:
    report["status"]="failed";report["error"]=str(error);traceback.print_exc()
finally:
    report["finished"]=time.time();atomic(REPORT,report)
    print(json.dumps({"status":report["status"],"datasets":list(report["datasets"]),"report":str(REPORT)}),flush=True)
