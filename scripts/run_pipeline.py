#!/usr/bin/env python3
"""JSON file input with guideline, patient, question; shared offline TRIAGE."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.pipeline import TriagePipeline

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--components",type=Path,default=ROOT/"results/module2/components.json")
    p.add_argument("--projections",type=Path,default=ROOT/"results/module3/projections.pt")
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--max-new-tokens",type=int,default=1284)
    p.add_argument("--top-k",type=int,default=12)
    args=p.parse_args()
    item=json.loads(args.input.read_text())
    missing={"guideline","patient","question"}-set(item)
    if missing: p.error("Missing input keys: "+str(sorted(missing)))
    pipe=TriagePipeline(args.components,args.projections,args.device,args.top_k,args.max_new_tokens)
    result=pipe.run(item["guideline"],item["patient"],item["question"],input_source=item.get("input_source"))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    if not result["baseline"]["json_valid"] or not result["intervened"]["json_valid"]:
        raise AssertionError("Pipeline decision JSON is incomplete; output text preserved in report")
    print("Saved "+str(args.output),flush=True)
if __name__=="__main__": main()
