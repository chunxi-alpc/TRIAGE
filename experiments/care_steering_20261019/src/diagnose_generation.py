from engine import *
import json
if __name__=="__main__":
 e=Engine();data=rows(ROOT/"data/dev.jsonl")
 for r in data[:4]:
  out=e.generate(r["prompt"],max_new_tokens=96)
  append(ROOT/"results/generation_diagnostic.jsonl",dict(item_id=r["item_id"],gold=r["gold"],**out))
 print("complete")
