from pathlib import Path
import pathlib, sys
ROOT=pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from src.module1.care_v6 import CAREModule1V6, TRUE, UNKNOWN, map_trialmatch_label
from src.pipeline.runner import TriagePipeline

class TM:
    def judge(self, patient, condition, criterion_kind="inclusion"):
        return {"raw_output":"Met","raw_label":"Met","state":TRUE,"criterion_kind":criterion_kind}
class NLI:
    def predict(self, pairs, batch_size=32):
        return [{"relation":"entailment","class_id":2,"scores":{}} for _ in pairs]

def make_graph():
    rule=[{"rule_id":"R1","text":"Has disease","conditions":[{"condition_id":"c1","field":"distant","text":"distant metastasis is present"}],"logic":{"op":"AND","args":["c1"]},"exceptions":[]}]
    return CAREModule1V6(TM(),NLI()).run(rule,"Current distant metastasis is present.","Assess")

def test_runner_factory_is_v6(monkeypatch):
    created=[]
    def factory(cls, **kwargs):
        created.append(kwargs)
        return object()
    monkeypatch.setattr(CAREModule1V6, "from_pretrained", classmethod(factory))
    pipe=TriagePipeline(backend=object(), module1=None, module1_device="cuda:7")
    assert pipe._get_module1() is not None
    assert created and created[0]["device"]=="cuda:7"

def test_runner_rejects_legacy_graph():
    pipe=TriagePipeline(backend=object(), module1=object())
    pipe.load=lambda: pipe
    try:
        pipe.run([], "", "", module1_output={"schema_version":"module1.v1"})
    except ValueError as exc:
        assert "module1.v6" in str(exc)
    else:
        raise AssertionError("legacy graph was accepted")

def test_v6_graph_has_module2_module3_contract():
    graph=make_graph()
    assert graph["schema_version"]=="module1.v6"
    assert graph["applicability"][0]["decisive_condition_ids"]==["c1"]
    relation=graph["condition_evidence_relations"][0]
    assert relation["confidence"] >= .8
    assert graph["relation_weights"][0]["action_gate"] in (0.0,1.0)

if __name__=="__main__":
    test_v6_graph_has_module2_module3_contract()
    print("pipeline contract tests passed")

def test_unified_runner_computes_live_v6(monkeypatch, tmp_path):
    import torch
    import src.pipeline.runner as runner
    import src.module3 as module3
    graph=make_graph()
    class Engine:
        def run(self, **kwargs):
            return graph
    class Hook:
        def __init__(self, *args, **kwargs):
            self.stats={"context_changed_elements":0,"prior_changed_elements":0}
        def __enter__(self): return self
        def __exit__(self, *args): return False
    monkeypatch.setattr(module3, "GatedAttentionIntervention", Hook)
    class Encoded(dict):
        def to(self, device):
            return self
    monkeypatch.setattr(runner, "build_prompt_and_mapping", lambda *args: ("prompt", Encoded(input_ids=torch.ones(1,2,dtype=torch.long)), {"relations":[],"rules":{}}))
    class Backend:
        tokenizer=object()
        model=type("Model", (), {"device":"cpu"})()
        max_input_tokens=100
    pipe=TriagePipeline(backend=Backend(), module1=Engine(), components_path=tmp_path/"components.json", projection_path=tmp_path/"projection.pt")
    pipe.load=lambda: pipe
    pipe.components={}
    pipe.components_path.write_text("{}")
    pipe.projection_path.write_bytes(b"projection")
    pipe._logits=lambda encoded: torch.tensor([1.0,2.0])
    pipe._decode=lambda encoded: {"text":"{}","decision":{},"json_valid":True,"output_tokens":1}
    result=pipe.run(graph["candidate_rules"], "Current distant metastasis is present.", "Assess", check_zero_gate=False)
    assert result["module1_source"]=="computed_live_v6"
    assert result["module1_schema_version"]=="module1.v6"
