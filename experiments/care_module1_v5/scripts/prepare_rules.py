"""Create label-free V5 rule pack from the frozen V4 structured-rule manifest."""
from __future__ import annotations
import argparse, json
from pathlib import Path

FIELD_TEXT = {
    "ato_prior": "arsenic trioxide exposure before the first relapse",
    "relapse_months": "months from initial therapy completion to first relapse",
    "flt3": "diagnostic FLT3 mutation result", "pt": "original pathological T stage",
    "pn": "original pathological N stage", "ct": "preoperative clinical T stage",
    "cn": "preoperative clinical N stage", "preop_chemo": "chemotherapy received before breast surgery",
    "sln_count": "positive sentinel lymph node count", "wbrt": "whole breast radiotherapy planned for this patient",
    "ax_nodes": "positive axillary lymph node count", "tumor_cm": "invasive tumor diameter in centimeters",
    "margin_mm": "closest resection margin in millimeters", "ypn": "post-neoadjuvant pathological N stage",
    "distant": "current distant metastasis", "dmmr": "mismatch-repair deficiency / MSI-high",
    "pole": "POLE or POLD1 mutation", "tmb": "tumor mutational burden in mutations per megabase",
    "csf": "malignant cells in latest CSF cytology", "margin": "thyroid lobectomy margin involvement",
    "contra": "contralateral thyroid lesion on postoperative imaging",
    "susp_nodes": "suspicious cervical lymph nodes on postoperative imaging",
    "age": "patient age in years at the surgical decision", "ret634": "germline RET codon 634 mutation",
    "v804m": "germline RET V804M mutation", "strong_family": "strong family history of medullary thyroid carcinoma",
    "gross_margin": "gross positive thyroid lobectomy margin", "gross_ete": "gross extrathyroidal extension",
    "thyroid_nodes": "confirmed cervical nodal thyroid metastasis",
    "confirmed_contra": "confirmed contralateral thyroid carcinoma", "vascular": "vascular invasion",
    "resectable": "recurrent thyroid cancer is resectable", "iodine_avid": "recurrent thyroid cancer is radioiodine-avid",
    "progressive": "recurrent thyroid cancer is progressive", "symptoms": "current cancer-related symptoms",
    "indolent": "systemic metastatic thyroid disease is indolent", "brain_mets": "confirmed brain metastasis on latest brain imaging",
}
OP_TEXT = {"eq": "is", "lt": "is less than", "le": "is at most", "gt": "is greater than",
           "ge": "is at least", "ne": "is not"}
SYMBOL = {"eq": "==", "lt": "<", "le": "<=", "gt": ">", "ge": ">=", "ne": "!="}


def describe(atom):
    field = atom["field"]
    value = atom["value"]
    op = atom["op"]
    label = FIELD_TEXT.get(field, field.replace("_", " "))
    if isinstance(value, str) and value in {"positive", "negative"}:
        return f"{label} is {value}"
    unit = {"relapse_months": " months", "tumor_cm": " cm", "margin_mm": " mm",
            "tmb": " mutations/Mb", "age": " years"}.get(field, "")
    return f"{label} {OP_TEXT.get(op, op)} {value}{unit}"


def build(manifest: Path, output: Path):
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    result = []
    for source in rows:
        schema = source["schema_gold"]
        conditions = []
        for index, atom in enumerate(schema.get("conditions", []), 1):
            cond = {"condition_id": f"c{index}", "field": atom["field"], "text": describe(atom),
                    "criterion_kind": "inclusion"}
            if isinstance(atom.get("value"), (int, float)):
                cond["numeric"] = {"field": atom["field"], "operator": SYMBOL[atom["op"]],
                                   "value": atom["value"], "unit": {"tumor_cm": "cm", "margin_mm": "mm",
                                   "relapse_months": "months", "age": "years", "tmb": "mutations/Mb"}.get(atom["field"], "")}
            conditions.append(cond)
        exceptions = []
        for index, atom in enumerate(schema.get("exceptions", []), 1):
            cond = {"condition_id": f"e{index}", "field": atom["field"], "text": describe(atom),
                    "criterion_kind": "exclusion"}
            if isinstance(atom.get("value"), (int, float)):
                cond["numeric"] = {"field": atom["field"], "operator": SYMBOL[atom["op"]], "value": atom["value"]}
            exceptions.append(cond)
        normalized_text = source["text"].rstrip() + "\nPrepared conditions:\n" + "\n".join(
            f"[{c['condition_id']}] {c['text']}" for c in conditions + exceptions)
        for group in (conditions, exceptions):
            for condition in group:
                needle = f"[{condition['condition_id']}] {condition['text']}"
                start = normalized_text.find(needle)
                condition["source_span"] = {"start": start, "end": start + len(needle),
                                             "unit": "character", "source": "offline_normalized_rule_pack"}
        result.append({"rule_id": source["rule_id"], "text": normalized_text,
            "source_text": source["text"],
            "source_text_span": {"start": 0, "end": len(source["text"]), "unit": "character",
                                  "source": "V4_prepared_paraphrase_not_original_guideline"},
            "source_span": {"start": 0, "end": len(normalized_text), "unit": "character",
                             "source": "normalized_rule_pack_not_original_guideline"},
            "logic": {"op": "AND" if schema["mode"] == "all" else "OR",
                      "args": [c["condition_id"] for c in conditions]},
            "conditions": conditions, "exceptions": exceptions,
            "exception_logic": {"op": "OR", "args": [c["condition_id"] for c in exceptions]},
            "action": None, "action_span": None,
            "provenance": {"scope": source.get("scope"), "source_node_sha256": source.get("source_node_sha256")}})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in result) + "\n")
    print(f"wrote {len(result)} structured rules to {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.manifest, args.output)
