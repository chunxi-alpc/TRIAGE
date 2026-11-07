"""Small checks of changed training/inference behavior, not a new benchmark."""
import copy
import torch
from .common import *
from .engine import Engine
from .scoring import load_editor,mcq
from .training import relation_loss,forward_loss
from experiments.aster_learnable_m3.data import target_prompt
def run():
    torch.set_num_threads(4)
    a=torch.tensor([3.,-3.],requires_grad=True);b=torch.tensor([-3.,3.],requires_grad=True)
    assert float(relation_loss(a,a,1,1,"A").detach())<1e-6
    assert float(relation_loss(a,b,1,0,"F").detach())==0
    assert float(relation_loss(b,a,1,0,"C").detach())>0
    assert float(relation_loss(a,b,1,1,"F_same").detach())>0
    controls=read(ROOT/"p2/head_controls.json")
    assert all(controls[f"random_{s}"]["parameters"]==31771 for s in [13,37,73])
    engine=Engine();case=rows(OLD/"data/train.jsonl")[0];gid=case["gold_path"][0]["gid"]
    editor,spec=load_editor(engine,"paired_n",selected=False)
    c=engine.prepare(target_prompt(case,gid));base=engine.run(c,capture=False)
    zero=engine.run(c,m3=editor,reference=None,force_zero=True,capture=False)
    initial=engine.run(c,m3=editor,reference=None,capture=False)
    assert base["ell"]==zero["ell"]==initial["ell"]
    loss,_,_=forward_loss(engine,editor,spec,target_prompt(case,gid),case["gold_path"][0]["label"])
    loss.backward()
    assert all(p.grad is None for p in engine.model.parameters())
    assert any(p.grad is not None and bool((p.grad!=0).any()) for p in editor.parameters())
    vs={v["view_id"]:v for v in rows(ROOT/"data/views.jsonl")}
    for kind in ["C","F_same"]:
        p=next(p for p in rows(ROOT/"data/pairs.jsonl") if p["kind"]==kind and p["split"]=="fit")
        la,za,_=forward_loss(engine,editor,spec,vs[p["left"]]["prompt"],p["left_label"])
        lb,zb,_=forward_loss(engine,editor,spec,vs[p["right"]]["prompt"],p["right_label"])
        (la+lb+relation_loss(za,zb,p["left_label"],p["right_label"],kind)).backward()
    mg=rows(ROOT/"data/medguide_pilot.jsonl")[0]
    r=mcq(engine,mg,"full");changed=copy.deepcopy(mg);changed["native"]["answer"]="H"
    r2=mcq(engine,changed,"full");assert r["prediction"]==r2["prediction"]
    all_editor,all_spec=load_editor(engine,"all_heads",selected=False)
    ref=engine.run(c)["heads"].flatten(0,1).to(engine.device)
    out=engine.run(c,m3=all_editor,reference=ref,training=True,capture=False)
    out["logits_tensor"].sum().backward()
    result=dict(state="passed",zero_edit_ell_difference=0.,online_initial_ell_difference=0.,
        random_parameters=31771,pair_loss_directions=True,pair_backprop=True,
        only_editor_gradients=True,mcq_prediction_label_independent=True,all_heads_backprop=True,
        peak_memory_gb=torch.cuda.max_memory_allocated()/2**30)
    write(ROOT/"smoke.json",result)
if __name__=="__main__":run()
