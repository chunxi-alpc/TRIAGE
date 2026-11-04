"""Source separation, frozen development selection and paired source bootstrap."""
import json
import pytest
from experiments.gemma3_head_pilot.expanded import frozen_choice, require_frozen, bootstrap_sources
from experiments.gemma3_head_pilot.expanded_specs import SOURCE_SPECS
from experiments.gemma3_head_pilot.expanded_data import validate

def test_fixed_40_sources_and_200_distinct_target_slots():
    assert len(SOURCE_SPECS)==40
    assert len({x[0] for x in SOURCE_SPECS})==40
    assert sum(len(x[2]) for x in SOURCE_SPECS)==200
    assert all(len(x[2])==5 for x in SOURCE_SPECS)

def test_100_groups_require_20_sources():
    guide={"groups":[{"group_id":"G0"}]}
    groups=[dict(group_id=str(i),source_id="s"+str(i//5),source_filename="f"+str(i//5),
                 guideline=guide,gid="G0",history=[],shared_record="fixed record",
                 facts=["negative fact","positive fact"],fact_type="measurement") for i in range(100)]
    assert len(validate(groups))==20
    for g in groups:
        if g["source_id"]=="s19":g["source_id"]="s18";g["source_filename"]="f18"
    with pytest.raises(ValueError,match="20"):validate(groups)

def test_confirmation_is_blocked_until_freeze(tmp_path):
    with pytest.raises(ValueError,match="frozen"):require_frozen(tmp_path,"confirmation")
    require_frozen(tmp_path,"development")
    (tmp_path/"frozen_selection.json").write_text("{}")
    require_frozen(tmp_path,"confirmation")

def test_development_selection_freeze_is_immutable(tmp_path):
    values={}
    for kind in ("whole","heads"):
        for index in range(9):
            name=kind+str(index)
            values[name]=dict(method=name,kind=kind,layer=index//3+53,epoch=index%3+1,groups=None,
                             missing=0,all_four_correct=50+index,development_path_correct=90-index,
                             mean_magnitude=.1,by_group=[])
    frozen=frozen_choice(tmp_path,values)
    assert frozen["selected"]["whole"]["method"]=="whole8"
    assert frozen["selected"]["heads"]["method"]=="heads8"
    assert frozen["confirmation_predictions_used"] is False
    values["whole0"]["all_four_correct"]=100
    assert frozen_choice(tmp_path,values)==frozen

def test_source_bootstrap_retains_paired_groups():
    groups=[dict(group_id=str(i),source_id="s"+str(i//5)) for i in range(100)]
    summaries={}
    for name,correct in (("zero",False),("blind",True),("edited",True)):
        summaries[name]=dict(by_group=[dict(**g,all_correct=correct) for g in groups])
    out=bootstrap_sources(summaries,samples=100)
    assert out["sources"]==20
    assert out["paired_difference_95_percentage_points"]["zero"]["edited"]==[100.,100.]
    assert out["paired_difference_95_percentage_points"]["blind"]["edited"]==[0.,0.]


def test_final_report_keeps_development_and_confirmation_columns(tmp_path):
    from experiments.gemma3_head_pilot.expanded import report, make_summaries
    from experiments.gemma3_head_pilot.core import write_json
    groups=[dict(group_id=str(i),source_id="source"+str(i//5)) for i in range(100)]
    specs=[]
    for name,kind,layer in (("zero","zero",None),("blind","blind",None),("w","whole",61),("h","heads",59)):
        specs.append(dict(method=name,kind=kind,layer=layer,epoch=None if layer is None else 1,
                          groups=[0,1,2,3] if kind=="heads" else None,
                          development_path_correct=79,development_path_cases=100))
    records=[]
    for g in groups:
        for state in (0,1):
            for view in ("original","swapped"):
                for spec in specs:
                    records.append(dict(method=spec["method"],group_id=g["group_id"],state=state,view=view,
                                        status="complete",branch="Y" if state else "N",scores=[-1.,-1.],magnitude=0.))
    summaries=make_summaries(groups,records,specs)
    (tmp_path/"data").mkdir()
    for split in ("development","confirmation"):
        (tmp_path/"data"/(split+".jsonl")).write_text("".join(json.dumps(g)+"\n" for g in groups))
    write_json(tmp_path/"candidate_manifest.json",specs)
    write_json(tmp_path/"frozen_selection.json",dict(selected={"whole":dict(method="w"),"heads":dict(method="h")}))
    write_json(tmp_path/"data_manifest.json",dict(test_fixture=True))
    out=report(tmp_path,summaries,summaries)
    assert out["complete"]
    assert len(out["confirmation"])==4
    assert out["confirmation"]["w"]["all_four_correct"]==100
    text=(tmp_path/"REPORT.zh.md").read_text()
    assert "开发四项全对" in text and "独立确认四项全对" in text
