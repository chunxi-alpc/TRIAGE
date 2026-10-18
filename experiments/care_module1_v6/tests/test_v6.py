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

def test_v6_does_not_upgrade_unknown_from_nli():
 out=CAREModule1V6(TM("Unclear"),NLI("entailment")).run(rule(),"Current distant metastasis is present.","Assess")
 assert out["rules"][0]["applicability"]==UNKNOWN

def test_v6_numeric_overrides_only_when_observed():
 r=[{"rule_id":"R1","text":"size <= 5","conditions":[{"condition_id":"c1","field":"tumor_cm","text":"tumor diameter at most 5 cm","numeric":{"field":"tumor_cm","operator":"<=","value":5}}],"logic":{"op":"AND","args":["c1"]},"exceptions":[]}]
 out=CAREModule1V6(TM("Not Met"),NLI("neutral")).run(r,"Verified tumor diameter was 3 cm.","Assess")
 assert out["rules"][0]["applicability"]==TRUE
if __name__ == "__main__":
 test_v6_keeps_trialmatch_on_nli_conflict(); test_v6_does_not_upgrade_unknown_from_nli(); test_v6_numeric_overrides_only_when_observed(); print("v6 basic tests passed")
