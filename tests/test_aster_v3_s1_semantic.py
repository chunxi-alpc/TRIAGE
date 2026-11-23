from experiments.aster_v3_selective.s1_verifier import HB, SBP, verify
from experiments.aster_v3_selective.s1_semantic_eval import aggregate


def test_same_unit_distinct_measurements():
    text = "示例甲：平均动脉压为72 mmHg，收缩压为108 mmHg，肺动脉收缩压为38 mmHg。"
    decision, _, span = verify(SBP,text)
    assert (decision, text[slice(*span)]) == (1,"108")
    assert verify(SBP,text,simple=True)[0] == 0


def test_ambiguous_sbp_abstains():
    assert verify(SBP, "收缩压为102 mmHg，收缩压为125 mmHg。")[0] is None
    assert verify(SBP, "收缩压为110 mmHg，未处于90–120 mmHg范围。")[0] is None
    assert verify(SBP, "肺动脉收缩压为95 mmHg。")[0] is None


def test_preoperative_timing_and_units():
    assert verify(HB,"Preoperative Hb was 13.0 g/dL.")[0] == 0
    assert verify(HB,"Postoperative Hb was 11.0 g/dL.")[0] is None
    assert verify(HB,"Preoperative Hb was 11.0 mmol/L.")[0] is None
    assert verify(HB,"Preoperative Hb was 11.0 g/dL, not below 13 g/dL.")[0] is None
    assert verify(HB,"Preoperative Hb was 11.0 g/dL. Preoperative Hb was 14.0 g/dL.")[0] is None


def test_fallback_metrics_keep_uncovered_in_denominator():
    rows=[dict(rule="other",gold=1,R0=0,verifier=""),dict(rule="sbp",gold=1,R0=0,verifier=1)]
    metrics=aggregate(rows,"verifier")
    assert (metrics["full_accuracy"],metrics["covered_accuracy"],metrics["B"],metrics["H"]) == (.5,1,1,0)
    assert verify("其他条件","收缩压为100 mmHg。")[0] is None
