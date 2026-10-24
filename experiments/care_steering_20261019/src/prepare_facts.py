import hashlib,json,random
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
p=Path("/home/gcx/TRIAGE/data/ClashEval/data/dataset/dataset.pqt")
df=pd.read_parquet(p).drop_duplicates("question")
rows=[r for r in df.to_dict("records") if isinstance(r["answer_original"],str) and isinstance(r["answer_mod"],str) and r["answer_original"].lower()!=r["answer_mod"].lower() and len(r["context_mod"])<3000]
random.Random(261019).shuffle(rows)
selected=[]
for i,r in enumerate(rows[:384]):
 r={k:str(v) for k,v in r.items()};r.update(item_id=f"fact_{i:04d}",split="train" if i<256 else "dev" if i<320 else "test",question_hash=hashlib.sha256(r["question"].encode()).hexdigest()); selected.append(r)
(ROOT/"data/facts.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in selected))
(ROOT/"sources/facts_provenance.json").write_text(json.dumps(dict(source=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),seed=261019,source_rows=len(df),selected_rows=len(selected),dedup="question",limitation="ClashEval source adaptation, not original NQ-Swap; all variants of same question remain in one split"),indent=2))
print({s:sum(r["split"]==s for r in selected) for s in ("train","dev","test")})
