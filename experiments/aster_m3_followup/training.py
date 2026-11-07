import random,time
from collections import Counter
import torch
from torch.nn import functional as F
from .common import *
from .scoring import load_editor,predict_case,ref_heads
from experiments.aster_learnable_m3.data import target_prompt,reference_prompt
def js_loss(a,b):
    pa,pb=a.softmax(-1),b.softmax(-1);m=(pa+pb)*.5
    return .5*((pa*(pa.clamp_min(1e-12).log()-m.clamp_min(1e-12).log())).sum()
               +(pb*(pb.clamp_min(1e-12).log()-m.clamp_min(1e-12).log())).sum())
def relation_loss(a,b,ya,yb,kind):
    if kind in ["A","P","F_same"]:return js_loss(a,b)
    ea,eb=a[0]-a[1],b[0]-b[1]
    return .5*(F.relu(1-(2*ya-1)*ea)+F.relu(1-(2*yb-1)*eb))
def forward_loss(engine,editor,spec,prompt,label,donor=None):
    ref=ref_heads(engine,donor) if spec["reference"]=="reference" else None
    c=engine.prepare(prompt);base=engine.run(c,capture=spec["reference"]=="self")
    if spec["reference"]=="self":ref=base["heads"].flatten(0,1).to(engine.device)
    out=engine.run(c,m3=editor,reference=ref,training=True,capture=False)
    logits=out["logits_tensor"]
    ce=F.cross_entropy(logits[None],torch.tensor([0 if label else 1],device=engine.device))
    keep=ce*0
    if base["prediction"]==label:
        pb=torch.tensor(base["label_logits"],device=engine.device).softmax(-1)
        keep=(pb*(pb.clamp_min(1e-12).log()-logits.log_softmax(-1))).sum()
    return ce+.1*keep+.001*out["magnitude"],logits,out["magnitude"]
def train(engine,name):
    config=read(ROOT/"configuration.json");dest=ROOT/"models"/name;dest.mkdir(parents=True,exist_ok=True)
    if (dest/"complete.json").exists():return
    train_seed=config.get("training_seed",20261105)
    torch.manual_seed(train_seed);random.seed(train_seed)
    editor,spec=load_editor(engine,name,selected=False)
    opt=torch.optim.AdamW(editor.parameters(),lr=config["lr"],weight_decay=config["weight_decay"])
    raw=rows(OLD/"data/train.jsonl");dev=rows(OLD/"data/dev.jsonl")
    pairs=[p for p in rows(ROOT/"data/pairs.jsonl") if p["split"]=="fit"] if spec["paired"] else []
    views={v["view_id"]:v for v in rows(ROOT/"data/views.jsonl")} if pairs else {}
    kinds=Counter(p["kind"] for p in pairs)
    tasks=[("raw",r) for r in raw]+[("pair",r) for r in pairs]
    best=None;start=time.perf_counter();completed=0
    for epoch in range(1,config["epochs"]+1):
        metric=dest/f"dev_epoch_{epoch}.json"
        if metric.exists():
            m=read(metric);key=(m["accuracy"],-m["magnitude"],-epoch)
            best=key if best is None or key>best else best
            state=torch.load(dest/f"epoch_{epoch}.pt",map_location=engine.device,weights_only=False)
            editor.load_state_dict(state["state_dict"]);opt.load_state_dict(state["optimizer"]);continue
        order=list(tasks);random.Random(train_seed+epoch).shuffle(order)
        editor.train();opt.zero_grad(set_to_none=True)
        for i,(kind,item) in enumerate(order):
            group_size=min(8,len(order)-(i//8)*8)
            if kind=="raw":
                indices=sorted(set([0,len(item["gold_path"])//2,len(item["gold_path"])-1]))
                weight=(len(tasks)/len(raw)) if pairs else 1.
                total=0.
                for index in indices:
                    n=item["gold_path"][index]
                    loss,logits,mag=forward_loss(engine,editor,spec,target_prompt(item,n["gid"]),n["label"],reference_prompt(item,n["gid"]))
                    (weight*loss/len(indices)/group_size).backward();total+=float(loss.detach())/len(indices)
                    del loss,logits,mag
            else:
                a,b=[views[item[s]] for s in ["left","right"]]
                la,za,ma=forward_loss(engine,editor,spec,a["prompt"],a["label"])
                lb,zb,mb=forward_loss(engine,editor,spec,b["prompt"],b["label"])
                relation=relation_loss(za,zb,a["label"],b["label"],item["kind"])
                loss=.5*(la+lb)+relation
                # The paired objective averages each pair type separately.
                weight=len(tasks)/(len(kinds)*kinds[item["kind"]])
                (weight*loss/group_size).backward();total=float(loss.detach())
                del la,za,ma,lb,zb,mb,relation,loss
            if (i+1)%8==0 or i+1==len(order):
                opt.step();opt.zero_grad(set_to_none=True)
            append(dest/"training.jsonl",dict(epoch=epoch,step=i+1,kind=kind,loss=total))
            completed+=1
            if i==19 or (i+1)%20==0:
                status("train_"+name,state="running",epoch=epoch,done=i+1,total=len(order),
                    elapsed_seconds=time.perf_counter()-start,peak_memory_gb=torch.cuda.max_memory_allocated()/2**30)
        editor.eval()
        predictions=[predict_case(engine,c,name,editor,spec) for c in dev]
        write_rows(dest/f"dev_epoch_{epoch}.jsonl",predictions)
        accuracy=sum(r["correct"] for r in predictions)/len(predictions)
        magnitude=sum(r["magnitude"] for r in predictions)/len(predictions)
        metric=dict(epoch=epoch,accuracy=accuracy,magnitude=magnitude)
        write(dest/f"dev_epoch_{epoch}.json",metric)
        state=dict(state_dict=editor.state_dict(),optimizer=opt.state_dict(),epoch=epoch,spec=spec)
        torch.save(state,dest/f"epoch_{epoch}.pt")
        key=(accuracy,-magnitude,-epoch)
        if best is None or key>best:
            best=key;torch.save(dict(state_dict=editor.state_dict(),epoch=epoch,spec=spec),dest/"selected.pt")
            write(dest/"selected.json",metric)
    write(dest/"complete.json",dict(elapsed_seconds=time.perf_counter()-start,selected=read(dest/"selected.json"),
        parameters=sum(p.numel() for p in editor.parameters()),engine_counts=engine.counts,
        peak_memory_gb=torch.cuda.max_memory_allocated()/2**30))
    status("train_"+name,state="complete",selected=read(dest/"selected.json"))
