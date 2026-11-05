"""Check existing MedGUIDE conversion against its supplied ordered source paths.
This is a path-dependency check, never part of the original-guideline Q denominator.
"""
import json
from pathlib import Path
from common import DATA, RESULTS, read_rows, write_json, write_rows

def run():
    import pandas as pd
    source=Path("/home/gcx/aster_medguide_20261022/private")
    frame=pd.read_parquet(source/"train.parquet")
    all_paths=json.loads((source/"all_paths.json").read_text())
    converted=read_rows(source/"pilot20.jsonl")
    output=[]
    for row in converted:
        if row.get("variant")!="A_original":
            continue
        original=frame.iloc[row["source_row"]]
        path=[str(x) for x in original["path"]]
        disease=str(original["disease"])
        full_rule=row.get("applicable_rule_text","")
        prefix=path[:-1]
        positions=[full_rule.find(x) for x in prefix]
        known=path in all_paths.get(disease,[])
        preserved=all(p>=0 for p in positions) and positions==sorted(positions)
        output.append(dict(example_id=row["id"],source_row=row["source_row"],disease=disease,
            original_path=path,converted_rule=full_rule,
            path_found_in_supplied_graph=known,
            all_ancestor_text_preserved_in_order=preserved,
            missing_ancestor_text=[text for text,pos in zip(prefix,positions) if pos<0],
            explicit_reference_steps=[s for s in prefix if "see " in s.lower() or "参见" in s],
            semantic_definition_resolution="not established from path strings alone"))
    write_rows(DATA/"medguide_dependencies.jsonl",output)
    write_json(RESULTS/"medguide_dependencies.json",dict(n=len(output),
        paths_found=sum(r["path_found_in_supplied_graph"] for r in output),
        ordered_ancestors_preserved=sum(r["all_ancestor_text_preserved_in_order"] for r in output),
        explicit_external_references=sum(bool(r["explicit_reference_steps"]) for r in output),
        scope="Existing 20-case MedGUIDE conversion; not a prevalence estimate or validation of unavailable definitions",
        included_in_Q=False))
    print("MedGUIDE conversion paths checked:",len(output))

if __name__=="__main__":
    run()
