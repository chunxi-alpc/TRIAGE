#!/usr/bin/env python3
"""Separate functional coverage check of a fixed original guideline subtree."""
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_pipeline_smoke import load_inputs
from src.pipeline import TriagePipeline

def main():
    item=load_inputs()[0]
    all_rules=item["guideline"]["rules"]
    # This original subtree concerns food/energy intake and nutritional outcomes.
    # Keep every descendant, including alternative leaf actions and all ancestors.
    # Selection uses guideline structure only, never patient content/path labels.
    prefix={"mega_5_node_1_yes","mega_5_node_2_yes","mega_5_node_3_yes"}
    rules=[r for r in all_rules if prefix.issubset({c["condition_id"] for c in r["conditions"]})]
    assert len(rules)==3
    item["guideline"]={"rules":rules}
    item["question"]="For this patient, what management should directly improve food and energy intake and nutritional outcomes in dementia?"
    item["input_source"]["guideline"].update(all_tree_paths_used=False,full_guideline_rule_count=len(all_rules),
        rule_count=len(rules),fixed_subtree_node=4,official_node_id=4,
        selected_leaf_indices=[r["source_node_index"] for r in rules],
        selection_uses_patient=False,selection_uses_reference=False,
        token_budget_reason="A 15-rule conflict comparison risks the current 1024-token response budget; functional coverage retains every branch of the fixed nutrition subtree",
        all_subtree_descendants_retained=True,
        inherited_conditions_retained=True,
        selection_reason="Fixed question scope: food/energy intake and nutritional outcomes; use complete original node4 subtree")
    item["input_source"]["question"]["origin"]="Generic fixed subtree management question; not derived from patient/path labels or reference answer"
    out=ROOT/"results/pipeline/coverage_smoke.json"
    pipe=TriagePipeline(max_new_tokens=384)
    start=time.time()
    result=pipe.run(item["guideline"],item["patient"],item["question"],
                    input_source=item["input_source"],check_prior_control=True)
    report={"run_kind":"fixed_original_subtree_functional_coverage_smoke",
            "effectiveness_evaluation":False,"dataset":"MEGA-CDP","split":"train_val",
            "single_qwen_model_instance":True,"reference_answers_used":False,
            "initial_full_tree_smoke_report":str(ROOT/"results/pipeline/smoke.json"),
            "components_and_projection_frozen":True,"samples":[result],"elapsed_seconds":round(time.time()-start,3)}
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    assert result["baseline"]["json_valid"] and result["intervened"]["json_valid"],"Incomplete clinical decision JSON"
    assert result["intervention_stats"]["context_changed_elements"]>0,"No active context computation change"
    assert result["comparison"]["max_abs_next_token_logit_delta"]>0,"No active output logit change"
    assert result["zero_gate_control"]["all_equal"],"Zero gate changed logits"
    print("Saved "+str(out),flush=True)
if __name__=="__main__": main()
