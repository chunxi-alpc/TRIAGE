from experiments.aster_v3_selective.preservation import budget_scale, count_transitions, protected_score


def test_protection_boundaries():
    assert protected_score(-3, 1, None) == 1
    assert protected_score(0, 1, 2) == 0
    assert protected_score(2, -1, 2) == 0.2
    assert protected_score(-2, 1, 2) == -0.2
    assert protected_score(1, -1, 2) == -1
    assert budget_scale(0, 1) == 1 and budget_scale(4, 1) == 0.25


def test_conditional_denominators():
    result = count_transitions([(False, True), (True, False), (True, True)])
    assert result['B_over_wrong'] == 1 and result['H_over_correct'] == 0.5
