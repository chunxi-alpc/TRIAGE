import copy
from .annotate import validate
from .build import edited_patient,expr_text,ancestors_ok
from experiments.aster_closed_form.common import evaluate
from .statistics import interval,pair_metrics
def fixture():
 g=dict(patient="Age 65 years. Diagnosis confirmed.",original_label=1,ancestors=[])
 a=dict(available=True,facts={"age":65},descriptions={"age":"age in years"},
    predicate={"field":"age","op":">=","value":65},ancestor_predicates=[],
    evidence={"age":["Age 65 years."]},decisive_sentences=["Age 65 years."],
    edits=[{"field":"age","spans":[{"old":"Age 65 years.","template":"Age {value} years."}],"step":1}])
 return g,a
def test_strict_endpoint():
 g,a=fixture()
 assert validate(g,a)["state"]=="complete"
 a["predicate"]["op"]=">"
 assert validate(g,a)["state"]=="E4_candidate"
def test_evidence_not_invented():
 import pytest
 g,a=fixture();a["evidence"]["age"]=["Age 66 years."]
 with pytest.raises(ValueError):validate(g,a)
def test_numeric_edit_preserves_other_text():
 g,a=fixture()
 assert edited_patient(g,a["edits"][0],64.5)=="Age 64.5 years. Diagnosis confirmed."
def test_logic_rule_and_protected_ancestor():
 g,a=fixture();g["ancestors"]=[{"gid":"G0","label":1}]
 a["ancestor_predicates"]=[{"gid":"G0","predicate":{"field":"age","op":">=","value":18}}]
 assert ancestors_ok(g,a,{"age":64})
 assert not ancestors_ok(g,a,{"age":17})
def test_cluster_interval_and_missing_denominator():
 r=interval([1,0,1],["a","a","b"]);assert r["denominator"]==3 and r["sources"]==2
 p=[dict(kind="A",source_id="a",group_id="n",left="x",right="y",left_label=1,right_label=1)]
 r=pair_metrics([],p,{"x":2.0})
 assert r["A"]["denominator"]==1 and r["A"]["numerator"]==0

def test_development_path_gate(tmp_path,monkeypatch):
 from . import pipeline
 from .common import write,read
 monkeypatch.setattr(pipeline,"ROOT",tmp_path);root=tmp_path/"qwen"
 write(root/"readout_candidates.json",[
  dict(name="full",parameters=0,dev={"macro":.8}),
  dict(name="bias",parameters=1,dev={"macro":.82}),
  dict(name="cr_bad",parameters=9,rank=8,dev={"macro":.99}),
  dict(name="cr_good",parameters=5,rank=4,dev={"macro":.9}),
  dict(name="ordinary_r4",parameters=5,rank=4,dev={"macro":.89})])
 write(root/"path_results/dev.json",{n:{"accuracy":v} for n,v in dict(full=.95,bias=.95,cr_bad=.94,cr_good=.95,ordinary_r4=.95).items()})
 result=pipeline.choose("qwen")
 assert result["aster_cr"]=="cr_good"
 assert "cr_bad" in result["excluded_path"]
 assert not (root/"frozen.json").exists()
