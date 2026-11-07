import json
from pathlib import Path
import numpy as np
import pytest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM,Gemma3TextConfig,Gemma3ForCausalLM
from .common import *
from .statistics import estimates,summarize
from .data import alignment,parse_tree,flatten_tree,mega_prompt
from .engine import Engine

class Tokens:
    def encode(self,text,add_special_tokens=False):
        if text=="是":return [3]
        if text=="否":return [4]
        return [5+(ord(c)%20) for c in text]
    def apply_chat_template(self,messages,**kwargs):return messages[-1]["content"]
    def __call__(self,text,return_tensors="pt",add_special_tokens=False):return {"input_ids":torch.tensor([self.encode(text)])}

def tiny(name):
    torch.manual_seed(9)
    kwargs=dict(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=3,num_attention_heads=4,
        num_key_value_heads=2,head_dim=8,max_position_embeddings=256)
    if name=="qwen":model=Qwen3ForCausalLM(Qwen3Config(**kwargs))
    else:model=Gemma3ForCausalLM(Gemma3TextConfig(**kwargs,sliding_window=16,sliding_window_pattern=2,
        layer_types=["sliding_attention","full_attention","sliding_attention"]))
    return Engine(name,device="cpu",model=model,tokenizer=Tokens())

def test_estimates_zero_small_and_direction():
    t=np.array([2.,-4.,.01,0.]);e=np.array([1.,-1.,.2,.7])
    x=estimates(t,e)
    assert x["original_effective"]==.375
    assert np.isclose(x["weighted_all"],2.2/6.01)
    assert np.isclose(x["regression_all"],6.002/20.0001)
    assert estimates([0],[7])["original_effective"] is None
    assert estimates([0],[7])["weighted_all"] is None
    a=effects(1,3,2,2)
    assert a["ie_forward"]==1 and a["ie_reverse"]==-1 and a["e"]==1
    assert effects(1,1,2,0)["raw_ratio"] is None

def test_cluster_bootstrap_keeps_pairs():
    rs=[dict(pair_id=str(i),source_id="s",group_id=str(i),te=t,ie_forward=t*.5,ie_reverse=-t*.5) for i,t in enumerate([2.,-4.,.01])]
    out=summarize(rs)
    assert out["sources"]==1 and out["effective_pairs"]==2
    for name,ci in out["ci95"].items():
        assert np.isclose(ci["low"],.5) and np.isclose(ci["high"],.5)

@pytest.mark.parametrize("name",["qwen","medgemma"])
def test_replay_norm_heads_and_residual(name):
    e=tiny(name);e.prefill_chunk=12
    prompt="abcdefghijklmnopqrstuv"
    c=e.prepare(prompt);base=e.run(c,grad=True,capture_scales=True)
    assert base["heads"].shape==(3,4,8) and base["grads"].shape==(3,4,8)
    assert e.run(c,capture=False)["ell"]==base["ell"]
    assert abs(e.full_score(prompt)-base["ell"])<2e-6
    h=base["heads"].reshape(-1,8);allhs={i:h[i] for i in range(12)}
    assert e.run(c,replacements=allhs,capture=False)["ell"]==base["ell"]
    assert abs(e.run(c,capture=False,norm_scales=base["rms_scales"])["ell"]-base["ell"])<1e-7
    c2=e.prepare("abcdefZZZZopqrstuv");b=e.run(c2,capture_scales=True)
    donor={i:b["heads"].reshape(-1,8)[i] for i in range(12)}
    assert abs(e.run(c,replacements=donor,capture=False)["ell"]-b["ell"])<2e-6
    assert e.run(c,capture=False)["ell"]==base["ell"]
    pos={"before":4,"after":13,"judgment":len(prompt)-1}
    r=e.residual_run(prompt,pos,[0,1,2])
    assert abs(r["ell"]-base["ell"])<2e-6
    edited=e.residual_run(prompt,pos,[0,1,2],(1,"after",r["residuals"][1,"after"]))
    assert edited["ell"]==r["ell"]
    # Changing a residual must affect suffix computation and leave earlier anchors intact.
    edited=e.residual_run(prompt,pos,[0,1,2],(1,"after",r["residuals"][1,"after"]+torch.arange(32)*.03))
    assert abs(edited["ell"]-r["ell"])>1e-8
    assert torch.equal(edited["residuals"][1,"before"],r["residuals"][1,"before"])
    assert not any(p.requires_grad for p in e.model.parameters())

def test_alignment_insert_and_substitution():
    a=list("abcXXdef");b=list("abcYYYYdef")
    pos=alignment(a,b)["positions"]
    assert pos=={"before":[2,2],"after":[5,7],"judgment":[7,9]}
    for key in ["before","after"]:
        assert a[pos[key][0]]==b[pos[key][1]]
    assert alignment(list("abcdefgh"),list("abcZefgh"))["nearest_change"]==4

def test_frozen_original_content():
    import pyarrow.parquet as pq
    raw=pq.read_table(REPO/"data/MedGUIDE/data/train-00000-of-00001.parquet").to_pylist()
    selected=rows(ROOT/"data/medguide_200.jsonl")
    assert len(selected)==200 and any(r["native"]["option_count"]==8 for r in selected)
    for r in selected:assert r["native"]==raw[r["row_index"]]
    native=rows(REPO/"data/MEGA-CDP/test/vignette_cases.jsonl")
    trees={r["filename"]:r for r in rows(REPO/"data/MEGA-CDP/test/decision_trees.jsonl")}
    paths={r["filename"]:r for r in rows(REPO/"data/MEGA-CDP/test/case_paths.jsonl")}
    for r in rows(ROOT/"data/mega_200.jsonl"):
        assert r["native"]==native[r["row_index"]]
        assert r["text"]==r["native"]["text"]
        assert r["tree_text"]==trees[r["source_id"]]["text"]
        assert r["annotated_path"]==paths[r["source_id"]]["paths"][r["native"]["path_id"]-1]
        tree=parse_tree(r["tree_text"]);nodes=flatten_tree(tree)
        for step in r["annotated_path"]:
            if "decision_outcome" in step:
                assert step["condition"]==nodes[str(step["index"])]["description"]
                prompt=mega_prompt(r,nodes[str(step["index"])])
                assert r["text"] in prompt and r["tree_text"] in prompt and step["condition"] in prompt
    assert all(p["split"] in ["fit","dev"] for p in rows(ROOT/"data/diagnostic_pairs.jsonl"))

def test_old_results_and_double_wrong_retained():
    c=read(ROOT/"cpu/summary.json")
    assert all(r["passed"] for r in c["reproduction"])
    ps=rows(ROOT/"cpu/medgemma_test_A_pair_scores.jsonl")
    dw=[r for r in ps if r["both_wrong"]]
    assert len(dw)==1 and abs(dw[0]["te"])<.1
    rs=[r for r in rows(ROOT/"cpu/legacy_effects.jsonl") if r["model"]=="medgemma" and r["scheme"]=="frozen" and r["pair_id"]==dw[0]["pair_id"]]
    assert rs and all(r["recovery"] is None for r in rs)

def test_native_option_H_and_offpath_labels(tmp_path):
    from .baseline import medguide_case,mega_case
    class FakeEngine:
        model_name="qwen"
        tokenizer=Tokens()
        def prepare(self,prompt,task="binary"):
            from types import SimpleNamespace
            return SimpleNamespace(prompt=prompt,length=len(prompt))
        def run(self,context,capture=False,label_ids=None):
            if label_ids is not None:return {"label_logits":list(range(len(label_ids)))}
            return {"ell":-1. if "root question" in context.prompt.split("当前条件：")[-1] else 1.}
    case=dict(native=dict(prompt="ORIGINAL",options=list("abcdefgh"),answer="H"))
    assert medguide_case(FakeEngine(),case)["prediction"]=="H"
    tree={"index":1,"description":"root question","yes":{"index":2,"leaf":True,"conclusion":"yes leaf"},
       "no":{"index":3,"description":"offpath question","yes":{"index":4,"leaf":True,"conclusion":"offpath yes"},
             "no":{"index":5,"leaf":True,"conclusion":"offpath no"}}}
    case=dict(sample_id="test",source_id="source",text="ORIGINAL CASE",tree_text=json.dumps({"decision_tree":tree}),
        annotated_path=[dict(index=1,condition="root question",decision_outcome="yes"),dict(index=2,conclusion="yes leaf")])
    out=mega_case(FakeEngine(),case,tmp_path/"case.json")
    assert not out["path_correct"] and out["node_total"]==1 and out["node_correct"]==0
    off=[r for r in out["node_scores"].values() if r["index"]=="3"]
    assert len(off)==1 and off[0]["label"] is None

@pytest.mark.parametrize("name",["qwen","medgemma"])
def test_pair_checkpoint_resume_and_mode_separation(name,tmp_path,monkeypatch):
    from . import patching
    monkeypatch.setattr(patching,"ROOT",tmp_path)
    monkeypatch.setattr(patching,"cache_path",lambda model,mode,pid:tmp_path/model/mode/(pid+".json"))
    monkeypatch.setattr(patching,"plans",lambda model,kind:{"type_selected":[1,5],"selected_union":[1,5,9],
      "candidate128":[1,5,9],"single_1":[1],"single_5":[5]})
    write_rows(tmp_path/"data/diagnostic_views.jsonl",[
      dict(view_id="l",prompt="abcdefghijklmnopqrstuv"),dict(view_id="r",prompt="abcZZZZlmnopqrstuv")])
    pair=dict(pair_id="p",group_id="g",source_id="s",split="fit",kind="A",left="l",right="r")
    e=tiny(name);e.prefill_chunk=12
    result=patching.run_pair(e,pair,name)
    assert result["state"]=="complete"
    assert result["effects"]["native/candidate128"]["heads"]==[1,5,9]
    if name=="medgemma":
        assert "fixed_rms/type_selected" in result["effects"]
        assert "fixed_rms/single_1" in result["effects"]
    else:assert not any(k.startswith("fixed_rms") for k in result["effects"])
    before=e.counts["final_forwards"]
    again=patching.run_pair(e,pair,name)
    assert e.counts["final_forwards"]==before and again==result
