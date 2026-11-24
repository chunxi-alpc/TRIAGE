from experiments.aster_v3_selective.s2_events import changed_scores, classification, scope, source_quantile, summary
from collections import Counter


def test_source_balanced_quantile():
    events = [dict(source_id="a", s=-1.0)] * 10 + [dict(source_id="b", s=2.0)]
    assert source_quantile(events) == 2.0
    assert source_quantile([]) is None


def test_four_outcomes_and_oracle():
    assert [classification(a, b) for a,b in [(False,True),(True,False),(True,True),(False,False)]] == [
        "benefit","harm","kept_correct","kept_wrong"]
    row = summary(Counter(inputs=10, benefit=2,harm=1, kept_correct=5,kept_wrong=1, unlabeled=1))
    assert row["oracle_correct"] == 8
    assert row["labeled"] == 9
    assert row["harm_rate"] == 1 / 6
    assert row["benefit_rate"] == 2 / 3


def test_training_unit_and_path_variants():
    assert scope(dict(natural=True)) == "natural_fit"
    assert scope(dict(natural=False, in_evaluation_views=True)) == "other_evaluation_views"
    assert scope(dict(natural=False, in_evaluation_views=False, variant="original")) == "path_original_only"
    assert scope(dict(natural=False, in_evaluation_views=False, variant="correlation")) == "path_interference_only"


def test_quantile_selects_changed_scores_without_labels():
    events = changed_scores([dict(s=0, u=1, source_id="a"), dict(s=-3, u=-2, source_id="b")])
    assert len(events) == 1
    assert source_quantile(events) == 0
