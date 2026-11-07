"""Controls use the same fitting pairs; selected M1/M2 tensors stay frozen."""
import torch
from .common import *
def orth(x):
    if not x.numel():return x
    u,s,_=torch.linalg.svd(x.double(),full_matrices=False)
    return u[:,s>max(1e-6,float(s[0])*1e-6)].float()
def prepare():
    torch.set_num_threads(4)
    out=ROOT/"bases";out.mkdir(parents=True,exist_ok=True)
    old=torch.load(OLD/"frozen_m1_m2.pt",map_location="cpu",weights_only=False)
    selection=read(LEGACY/"m1/selection.json")
    pairs=[p for p in rows(LEGACY/"data/pairs.jsonl") if p["split"]=="fit"]
    ids=sorted({p[s] for p in pairs for s in ["left","right"]})
    features={i:torch.load(LEGACY/"features"/(i.replace("/","_")+".pt"),map_location="cpu",weights_only=False) for i in ids}
    type_bases={};attrib={}
    for kind in ["A","P","C","F"]:
        ps=[p for p in pairs if p["kind"]==kind]
        if kind in ["A","P"]:ps=[p for p in ps if abs(features[p["right"]]["ell"]-features[p["left"]]["ell"])>=.1]
        delta=torch.stack([(features[p["right"]]["heads"]-features[p["left"]]["heads"]).reshape(1152,128) for p in ps]).float()
        moment=torch.einsum("nhd,nhe->hde",delta,delta)/len(ps)
        values,vectors=torch.linalg.eigh(moment)
        type_bases[kind]=vectors[:,:,-4:].flip(-1).contiguous()
        scores=[]
        for p in ps:
            a,b=features[p["left"]],features[p["right"]]
            d=(b["heads"]-a["heads"]).float()
            available=[x["grads"].float() for x in [a,b] if "grads" in x]
            if not available:continue
            s=torch.stack([(g*d).sum(-1).abs() for g in available]).mean(0)
            scores.append(s.reshape(-1))
        attrib[kind]=torch.stack(scores).mean(0).tolist()
        del delta,moment,values,vectors
    del features
    candidates={}
    for kind in ["A","P"]:
        candidates[kind]=[int(x["head"]) for x in read(LEGACY/"m1"/f"attribution_{kind}.json")["ranking"][:64]]
    specs=read(ROOT/"configuration.json")["variants"];summary={}
    for spec in specs:
        name=spec["name"];control={};mapping={}
        if spec["heads"]=="selected":
            for h,op in old.items():
                u=op["U_N"] if spec["support"]=="N" else op["S"] if spec["support"]=="S" else orth(torch.cat([op["U_N"],op["S"]],1))
                control[int(h)]=dict(U_N=u.clone(),S=op["S"].clone())
        else:
            if spec["heads"]=="same_layer_random":
                rng=np.random.default_rng(spec["seed"])
                for layer in sorted({int(h)//32 for h in old}):
                    originals=sorted(int(h) for h in old if int(h)//32==layer)
                    pool=list(range(layer*32,(layer+1)*32))
                    picked=rng.choice(pool,len(originals),replace=False).tolist()
                    mapping.update(dict(zip(picked,originals)))
                heads=sorted(mapping)
            elif spec["heads"]=="candidates":heads=sorted(set(candidates["A"]+candidates["P"]))
            else:heads=list(range(1152))
            for h in heads:
                if h in mapping:types=selection["T_h"][str(mapping[h])]
                elif spec["heads"]=="candidates":types=[k for k in ["A","P"] if h in candidates[k]]
                else:types=["A","P"]
                u=orth(torch.cat([type_bases[k][h] for k in types],1))
                if h in mapping:
                    rank=old[mapping[h]]["U_N"].shape[1]
                    if u.shape[1]<rank:raise ValueError("Insufficient control rank")
                    u=u[:,:rank]
                signal=orth(torch.cat([type_bases[k][h] for k in ["C","F"]],1))
                control[h]=dict(U_N=u,S=signal)
        torch.save(control,out/f"{name}.pt")
        params=sum(130*op["U_N"].shape[1]+1 for op in control.values())
        summary[name]=dict(heads=len(control),parameters=params,total_rank=sum(op["U_N"].shape[1] for op in control.values()),
            mapping_to_selected=mapping,reference=spec["reference"],support=spec["support"],
            capacity_matched=params==31771)
    overlap={}
    selected=set(map(int,old))
    for kind in ["C","F"]:
        ranking=sorted(range(1152),key=lambda h:attrib[kind][h],reverse=True)
        signal=set(ranking[:64])
        overlap[kind]=dict(top64=ranking[:64],selected_overlap=sorted(signal&selected),
            selected_A_overlap=sorted(signal&set(selection["H_A"])),selected_P_overlap=sorted(signal&set(selection["H_P"])),
            definition="fit-pair mean absolute first-order gradient times activation difference, using available endpoint gradients; descriptive ranking, not a newly validated causal head set")
    write(ROOT/"p2/head_controls.json",summary);write(ROOT/"p3/signal_head_overlap.json",overlap)
    status("bases",state="complete",variants=summary)
    return summary
if __name__=="__main__":print(json.dumps(prepare()))
