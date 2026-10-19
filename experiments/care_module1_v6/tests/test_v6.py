from pathlib import Path
import pathlib, sys
ROOT=pathlib.Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
from src.module1.care_v6 import CAREModule1V6, TRUE, FALSE, UNKNOWN, map_trialmatch_label, eval_logic
class TM:
 def __init__(self,label): self.label=label
 def judge(self,patient,condition,criterion_kind="inclusion"):
  return {"raw_output":self.label,"raw_label":self.label,"state":map_trialmatch_label(self.label,criterion_kind),"criterion_kind":criterion_kind}
class NLI:
 def __init__(self,rel): self.rel=rel
 def predict(self,pairs,batch_size=32): return [{"relation":self.rel,"class_id":2 if self.rel=="entailment" else 1,"scores":{}} for _ in pairs]
def rule(): return [{"rule_id":"R1","text":"Has disease","conditions":[{"condition_id":"c1","field":"distant","text":"distant metastasis is present"}],"logic":{"op":"AND","args":["c1"]},"exceptions":[]}]
def test_v6_keeps_trialmatch_on_nli_conflict():
 out=CAREModule1V6(TM("Met"),NLI("contradiction")).run(rule(),"Current distant metastasis is present.","Assess")
 assert out["rules"][0]["applicability"]==TRUE
 assert out["rules"][0]["evidence_status"]=="supported"
 assert out["rules"][0]["allow_attention_intervention"] is False
 assert out["schema_version"] == "module1.v6"
 assert out["applicability"][0]["state"] == TRUE
 assert out["condition_evidence_relations"][0]["evidence_status"] == "supported"
 for span in out["condition_evidence_relations"][0]["evidence"]:
  assert "start" in span and "end" in span

def test_v6_does_not_upgrade_unknown_from_nli():
 out=CAREModule1V6(TM("Unclear"),NLI("entailment")).run(rule(),"Current distant metastasis is present.","Assess")
 assert out["rules"][0]["applicability"]==UNKNOWN

def test_v6_numeric_overrides_only_when_observed():
 r=[{"rule_id":"R1","text":"size <= 5","conditions":[{"condition_id":"c1","field":"tumor_cm","text":"tumor diameter at most 5 cm","numeric":{"field":"tumor_cm","operator":"<=","value":5}}],"logic":{"op":"AND","args":["c1"]},"exceptions":[]}]
 out=CAREModule1V6(TM("Not Met"),NLI("neutral")).run(r,"Verified tumor diameter was 3 cm.","Assess")
 assert out["rules"][0]["applicability"]==TRUE
if __name__ == "__main__":
 test_v6_keeps_trialmatch_on_nli_conflict(); test_v6_does_not_upgrade_unknown_from_nli(); test_v6_numeric_overrides_only_when_observed(); print("v6 basic tests passed")

def test_v6_default_package_entry_uses_injected_v6_instance():
 import src.module1 as module1
 fake = CAREModule1V6(TM("Met"), NLI("entailment"))
 module1.set_default_v6(fake)
 result = module1.run(rule(), "Current distant metastasis is present.", "Assess")
 assert result["schema_version"] == "module1.v6"

def test_care_v6_run_factory_does_not_reference_compat_v5(monkeypatch):
 import src.module1.care_v6 as care
 care.set_default_v6(None)
 created=[]
 def factory(cls, **kwargs):
  created.append(kwargs)
  return CAREModule1V6(TM("Met"), NLI("entailment"))
 monkeypatch.setattr(CAREModule1V6, "from_pretrained", classmethod(factory))
 result=care.run(rule(), "Current distant metastasis is present.", "Assess")
 assert created and result["schema_version"]=="module1.v6"
 assert "compat_v5" not in Path(care.__file__).read_text()
