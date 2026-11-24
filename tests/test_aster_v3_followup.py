import json
import re

from experiments.aster_v3_selective.clinical_followup.review import make_form, normalize, reconcile


def case():
    return dict(blind_id="opaque",patient="示例\r\n符号😀",condition="条件",guideline_conditions=[])


def response():
    return dict(blind_id="opaque",decision="yes",reviewer_id="reader",explanation="原文支持条件",
                evidence_start=3,evidence_end=6,cause="测量对象或单位")


def evaluate(records, approved=()):
    cards={"opaque":case()}
    key={"opaque":dict(source_id="s",case_id="c",gid="g",batch="queue")}
    return reconcile(records,cards,key,{("c","g"):0},approved)[0]


def test_empty_or_unconfirmed_forms_cannot_count_as_expert_review():
    assert evaluate([{"blind_id":"opaque"}])["state"]=="pending"
    assert evaluate([response()])["state"]=="reviewer_unconfirmed"


def test_confirmation_with_conflicting_archive_label_never_changes_the_original():
    row=evaluate([response(),response()],["reader"])
    assert (row["state"],row["confirmed_readers"],row["matches_archive"],row["archived_label"])==("reviewed",1,0,0)
    other=dict(response(),reviewer_id="reader2",decision="no")
    assert evaluate([other,response()],["reader","reader2"])["state"]=="disagreement"


def test_indeterminate_and_incomplete_keep_separate_status():
    r=dict(response(), decision="indeterminate",evidence_start="",evidence_end="")
    assert evaluate([r],["reader"])["state"]=="indeterminate"
    assert normalize(dict(response(),evidence_end=60),case()["patient"])["state"]=="incomplete"


def test_offline_form_only_exposes_blinded_inputs_and_escapes_script_delimiter(tmp_path):
    r=dict(case(),patient="原文</script>\n😀",gold_path=[{"label":1}],source_id="source_canary")
    file=tmp_path/"form.html"
    make_form([r],tmp_path,file)
    html=file.read_text()
    payload=json.loads(re.search(r'<script id="payload" type="application/json">(.*?)</script>',html,re.S)[1])
    assert payload["cards"][0]["patient"]==r["patient"]
    assert "source_canary" not in html and "gold_path" not in html
