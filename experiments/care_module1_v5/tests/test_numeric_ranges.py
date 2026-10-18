"""Generic numeric parser regression; no patient/test-label-specific rules."""
from __future__ import annotations
import pathlib
import sys
import pytest
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.module1.care_v5 import _numeric_observation, sentences_with_offsets


def observe(field, text, threshold, op="<=", unit=None):
    spec = {"field": field, "operator": op, "value": threshold}
    if unit:
        spec["unit"] = unit
    return _numeric_observation({"field": field, "numeric": spec}, sentences_with_offsets(text))


@pytest.mark.parametrize("text,expected", [
    ("Invasive tumor diameter is 3–4mm.", [[0.3, 0.4]]),
    ("Invasive tumor diameter is 3 to 6cm.", [[3.0, 6.0]]),
    ("Invasive tumor diameter is 30mm to 6cm.", [[3.0, 6.0]]),
    ("Invasive tumor diameter is 3cm–60mm.", [[3.0, 6.0]]),
])
def test_length_ranges_convert_both_endpoints(text, expected):
    result = observe("tumor_cm", text, 0.35 if "3–4mm" in text else 5)
    assert result["status"] == "U"
    assert result["values"] == expected
    assert len(result["values"]) == 1


def test_range_wholly_below_or_above_threshold():
    assert observe("tumor_cm", "Tumor diameter is 30–40 mm.", 5)["status"] == "T"
    assert observe("margin_mm", "Closest margin is 0.1 to 0.4 cm.", 1, ">=")["status"] == "T"
    assert observe("margin_mm", "Closest margin is 0.1–0.4 cm.", 5, ">=")["status"] == "F"


def test_age_month_ranges_and_single_months():
    result = observe("age", "Patient age is 48–72 months.", 5, "<")
    assert result["status"] == "U" and result["values"] == [[4.0, 6.0]]
    assert observe("age", "Patient age is 48 months to 6 years.", 5, "<")["status"] == "U"
    assert observe("age", "The patient is 60 months old.", 5, "<")["status"] == "F"
    assert observe("age", "Patient age is 4–6.", 5, "<")["status"] == "U"


def test_relapse_year_month_ranges():
    result = observe("relapse_months", "Relapse interval is 1 to 2 years.", 18, "<")
    assert result["status"] == "U" and result["values"] == [[12.0, 24.0]]
    result = observe("relapse_months", "Relapse interval is 12 months to 2 years.", 18, "<")
    assert result["status"] == "U" and result["values"] == [[12.0, 24.0]]
    assert observe("relapse_months", "Relapse interval is 1 year.", 18, "<")["status"] == "T"


def test_stage_ranges_and_multiple_values_keep_unknown():
    assert observe("pt", "Original pathology is pT1/pT2.", 2)["status"] == "T"
    assert observe("pt", "Original pathology is pT1/pT3.", 2)["status"] == "U"
    assert observe("cn", "Clinical stage is cN+.", 0, ">")["status"] == "T"
    assert observe("tumor_cm", "Tumor diameter is 3cm. Tumor diameter is 6cm.", 5)["status"] == "U"


def test_equality_inside_range_is_unknown_and_requested_units_apply():
    assert observe("tumor_cm", "Tumor diameter is 3 to 6cm.", 4, "==")["status"] == "U"
    assert observe("tumor_cm", "Tumor diameter is 3 to 6cm.", 4, "!=")["status"] == "U"
    assert observe("tumor_cm", "Tumor diameter is 3 to 6cm.", 7, "!=")["status"] == "T"
    assert observe("age", "Patient age is 4–6 years.", 60, "<", "months")["status"] == "U"
