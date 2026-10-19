from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.build_minimal_pairs import make_cases
from scripts.plot_observations import plot_obs1, plot_obs3


def test_groups_and_states_are_split_together():
    data = make_cases(20, 42)
    assert len(data) == 120
    assert len({x["group_id"] for x in data}) == 20
    assert {x["split"] for x in data} == {"train", "dev", "test"}
    for group in {d["group_id"] for d in data}:
        rows = [x for x in data if x["group_id"] == group]
        assert len(rows) == 6
        assert len({x["split"] for x in rows}) == 1
        assert {x["state"] for x in rows} == {"RA", "RN", "IA", "IN", "RA2", "RN2"}
        assert len({x["gold_answer"] for x in rows}) == 2
        assert len({x["option_map"][x["gold_answer"]] for x in rows}) == 2
        assert all(not x["prior_eligible"] for x in rows)


def test_span_alignment_and_pairing():
    data = make_cases(12, 99)
    for row in data:
        text = row["prompt"]
        for name, span in row["spans"].items():
            assert span[1] > span[0] and span[1] <= len(text)
        assert text[slice(*row["spans"]["rule"])] == row["modified_rule"]
        assert text[slice(*row["spans"]["action"])] == "select Route Cedar"
        assert row["applicable"] == (row["patient_value"] < row["threshold"])
        assert row["option_map"][row["gold_answer"]] == ("Route Cedar" if row["applicable"] else "Route Dune")
        assert row["spans"]["rule"][1] <= text.index("\nOptions:")


def test_no_conclusions_without_real_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        plot_obs1(tmp_path / "notrun", tmp_path / "figs")
    with pytest.raises(FileNotFoundError):
        plot_obs3(tmp_path / "notrun", tmp_path / "figs")


def test_obs1_refuses_fake_prior_data(tmp_path):
    r = tmp_path / "results" / "obs1"
    r.mkdir(parents=True)
    pd.DataFrame([{
        "group_id": "g0", "case_id": "c0", "method": "baseline", "subset": "app", "correct": 1,
        "misapplied": 0, "prior_reversion": 0, "prior_eligible": False,
        "split": "test", "source_type": "synthetic_logic_only"
    }]).to_csv(r / "predictions.csv", index=False)
    with pytest.raises(ValueError, match="0 clinically validated"):
        plot_obs1(tmp_path / "results", tmp_path / "figs")


def test_local_key_edit_logit_identity():
    rng=np.random.default_rng(42)
    d=64
    q=rng.normal(size=d); k=rng.normal(size=d)
    U=np.linalg.qr(rng.normal(size=(d,9)))[0]
    P=U@U.T
    omega=-.45; scale=.7
    z=(q @ (P@k))/np.sqrt(d)
    kprime=k+scale*omega*(P@k)
    da=(q@kprime-q@k)/np.sqrt(d)
    assert da == pytest.approx(scale*omega*z,rel=1e-10,abs=1e-10)
    # A negative steering weight reverses its intended logit change if z<0.
    assert np.sign(da) == np.sign(omega*z) if z!=0 else True


def test_o3_fails_bad_rope_numeric_data(tmp_path):
    r=tmp_path/"results"/"obs3"
    r.mkdir(parents=True)
    df=pd.DataFrame([{
        "group_id":"g0","case_id":"c0","layer":1,"kv_head":0,"query_head":0,"token_idx":30,
        "omega":-1,"z":0.7,"a_before":0.1,"a_after":0.5,"delta_a":0.4,
        "lambda":0.3,"split":"test","model":"unit-test"
    }])
    df.to_csv(r/"logits.csv",index=False)
    with pytest.raises(ValueError,match="identity failed"):
        plot_obs3(tmp_path/"results",tmp_path/"figs")


def test_negative_alignment_reverses_raw_weight_and_smooth_correction_fixes_it():
    rng=np.random.default_rng(7)
    d=32
    k=rng.normal(size=d)
    U=np.linalg.qr(rng.normal(size=(d,5)))[0]
    P=U@U.T
    u=P@k
    q=-u  # force negative alignment: same toy coordinate frame, no RoPE
    omega=+0.6; lam=0.8
    z=(q@u)/np.sqrt(d)
    assert z<0
    direct_delta=lam*omega*z
    assert direct_delta<0  # the intended positive weight *decreases* this logit
    tau=0.4
    eps=tau*tau*(q@q)*(u@u)/d
    gamma=z/(z*z+eps)
    smooth_delta=lam*omega*gamma*z
    assert 0<smooth_delta<=lam*abs(omega)+1e-12
    assert smooth_delta == pytest.approx(lam*omega*z*z/(z*z+eps),rel=1e-12)
