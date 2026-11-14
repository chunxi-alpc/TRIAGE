from pathlib import Path
import json,collections
import numpy as np
import torch
torch.set_num_threads(2)
root=Path(__file__).resolve().parents[1]
d=torch.load(root/"results/direction.pt",map_location="cpu",weights_only=False)
c=d["clean_activations"].float();f=d["conflict_activations"].float();rng=np.random.default_rng(20261019)
facts={r["item_id"]:r for r in [json.loads(x) for x in (root/"data/facts.jsonl").read_text().splitlines()]}
eligible=d["eligible_ids"];ref=d["direction"].float();L=d["layer"]
cos=[];layer=[];norm=[]
for _ in range(1000):
 idx=rng.integers(0,len(c),len(c));cb=c[idx].mean(0);fb=f[idx].mean(0)
 v=cb[L]-fb[L];cos.append(float(torch.nn.functional.cosine_similarity(v,ref,dim=0)));norm.append(float(v.norm()))
 layer.append(int(torch.nn.functional.cosine_similarity(cb,fb,dim=1)[:-1].argmin()))
out=dict(kind="exploratory_train_only_empirical_bootstrap",replicates=1000,seed=20261019,n=len(c),reference_layer=L,
         selected_layer_counts=dict(collections.Counter(layer)),cosine_to_reference_quantiles_025_50_975=np.quantile(cos,[.025,.5,.975]).tolist(),
         norm_quantiles_025_50_975=np.quantile(norm,[.025,.5,.975]).tolist(),
         eligible_source_categories=dict(collections.Counter(facts[i]["dataset"] for i in eligible)),
         limitation="Resamples these same 11 pairs. Does not measure uncertainty over unseen eligible data, source distributions, or models. Does not retune any inference parameter.")
(root/"audit/training_direction_bootstrap.json").write_text(json.dumps(out,indent=2))
print(json.dumps(out))
