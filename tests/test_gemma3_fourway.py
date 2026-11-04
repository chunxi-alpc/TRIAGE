"""Four-input metrics and checkpoint ordering; no model/GPU required."""
from experiments.gemma3_head_pilot.fourway import summarize, selection_key, paired_groups
from experiments.gemma3_head_pilot.fourway_data import PAIR_SPECS

def records(group="g", branch=None):
    result=[]
    for state in (0,1):
        for view in ("original","swapped"):
            result.append(dict(group_id=group,state=state,view=view,method="m",
                               status="complete",branch=branch or ("Y" if state else "N"),
                               scores=[-0.1,-2.] if state else [-2.,-0.1],magnitude=0.))
    return result

def test_four_inputs_must_all_be_correct():
    group=[dict(group_id="g",source_id="s")]
    values=records()
    assert summarize(group,values,"m")["all_four_correct"]==1
    values[0]["branch"]="Y"
    s=summarize(group,values,"m")
    assert s["correct_cells"]==3
    assert s["all_four_correct"]==0

def test_constant_branch_cannot_pass():
    s=summarize([dict(group_id="g",source_id="s")],records(branch="N"),"m")
    assert s["cell_accuracy"]==0.5
    assert s["all_four_correct"]==0

def test_failure_keeps_group_in_denominator():
    values=records("g1")+records("g2")
    values[-1]["status"]="failed"
    s=summarize([dict(group_id=g,source_id="s") for g in ("g1","g2")],values,"m")
    assert s["group_count"]==2
    assert s["all_four_accuracy"]==0.5
    assert s["failures"]==1
    assert s["missing"]==0

def test_checkpoint_order():
    def candidate(four,path,edit):
        return dict(all_four_correct=four,development_path_correct=path,
                    mean_magnitude=edit,layer=53,epoch=1)
    assert selection_key(candidate(12,79,1.))>selection_key(candidate(11,99,0.))
    assert selection_key(candidate(12,85,1.))>selection_key(candidate(12,79,0.))
    assert selection_key(candidate(12,85,0.01))>selection_key(candidate(12,85,1.))

def test_group_pairing():
    baseline=dict(by_group=[dict(group_id="a",all_correct=False),dict(group_id="b",all_correct=True)])
    other=dict(by_group=[dict(group_id="a",all_correct=True),dict(group_id="b",all_correct=False)])
    assert paired_groups(baseline,other)==dict(corrections=1,regressions=1,net=0)

def test_fixed_source_balancing():
    from collections import Counter
    counts=Counter(s[0] for s in PAIR_SPECS)
    assert len(PAIR_SPECS)==20
    assert len(counts)==10
    assert set(counts.values())=={2}
    assert len({(s[0],s[1]) for s in PAIR_SPECS})==20
