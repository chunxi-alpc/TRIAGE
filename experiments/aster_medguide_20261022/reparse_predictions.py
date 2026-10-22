#!/usr/bin/env python3
"""Reparse saved completions consistently; never generate or invent answers."""
import argparse
import hashlib
import json
from pathlib import Path
from run_pilot import parse_answer

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--dataset',required=True);a=p.parse_args()
    path=Path(a.input);backup=path.with_suffix('.raw.jsonl')
    if not backup.exists():backup.write_bytes(path.read_bytes())
    data=[json.loads(s) for s in Path(a.dataset).read_text().splitlines() if s.strip()]
    labels={r['id']:r['options'] for r in data}
    rows=[json.loads(s) for s in backup.read_text().splitlines() if s.strip()]
    closed={r['id']:parse_answer(r['completion'],labels[r['id']]) for r in rows if r['condition']=='closed_book'}
    for r in rows:
        r['prediction']=parse_answer(r['completion'],labels[r['id']])
        r['closed_book_prediction']=closed[r['id']]
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    manifest_path=path.with_suffix('.manifest.json');manifest=json.loads(manifest_path.read_text())
    manifest.update(parser='leading answer, Markdown/Answer/option-text compatible; no prose search',
                    raw_inference_output_sha256=hashlib.sha256(backup.read_bytes()).hexdigest(),
                    reparse_only=True)
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(path.name,'reparsed',len(rows),'invalid',sum(r['prediction'] is None for r in rows))

if __name__=='__main__':main()
