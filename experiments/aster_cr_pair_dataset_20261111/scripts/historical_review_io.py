"""Pure packet and recorded-decision IO used by independent Sol reviewers. No model calls."""
import datetime,json,os,re
from pathlib import Path

CHECKS = ["factor_matches_class", "no_other_decisive_change", "labels_correct", "source_support",
          "medical_plausibility", "protected_content_preserved", "duplicate", "no_hidden_second_factor"]

SEMANTIC_CHECKS = [k for k in CHECKS if k != "duplicate"]

SCOPE = "fit_only;full_source_md;complete_baseline_tree;original_patient;lossless_actual_pair_differences;no_constructor_predicate_facts_reason_or_model_predictions"

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def read_rows(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def append_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())

def split_prompt(text):
    before, marker, tail = text.rpartition("\n\n病历：\n")
    if not marker: raise ValueError("patient delimiter absent")
    patient, marker, tail = tail.rpartition("\n\n当前节点：")
    if not marker: raise ValueError("node delimiter absent")
    gid, marker, tail = tail.partition("\n当前条件：")
    condition, marker, ending = tail.rpartition("\n请判断当前条件。")
    if not marker: raise ValueError("condition delimiter absent")
    tree = before.removeprefix("完整决策树：\n")
    return {"patient": patient, "condition": condition, "tree": tree, "gid": gid}

def leaf_actions(tree):
    content = tree.split("\n\n指南原文定义与说明：", 1)[0]
    matches = list(re.finditer(r"(?m)^(L\d+): ", content))
    return {m.group(1): content[m.end():matches[i+1].start() if i+1 < len(matches) else len(content)].rstrip("\n")
            for i, m in enumerate(matches)}

def edits_for_side(base, side, original_leaves):
    result = {}
    if side["patient"] != base["patient"]:
        result["patient"] = side["patient"]
    if side["condition"] != base["condition"]:
        result["current_condition"] = side["condition"]
    if side["tree"] != base["tree"]:
        old, new = leaf_actions(base["tree"]), leaf_actions(side["tree"])
        # A leaf-only changes are supplied losslessly; C supplies full exact changed condition.
        tree_old_without = re.sub(r"(?ms)^L\d+: .*?(?=^L\d+: |\n\n指南原文定义与说明：|\Z)", "", base["tree"])
        tree_new_without = re.sub(r"(?ms)^L\d+: .*?(?=^L\d+: |\n\n指南原文定义与说明：|\Z)", "", side["tree"])
        changes = []
        for leaf in sorted(set(old) | set(new)):
            if old.get(leaf) == new.get(leaf): continue
            item = {"leaf_id": leaf}
            identical = next((k for k, v in original_leaves.items() if v == new.get(leaf)), None)
            if identical is not None: item["new_action_from_original_leaf"] = identical
            else: item["new_action"] = new.get(leaf)
            changes.append(item)
        if changes: result["action_changes"] = changes
        if tree_old_without != tree_new_without:
            # Display all actual changed non-leaf spans, preserving exact content.
            import difflib
            result["tree_text_changes"] = [{"old": base["tree"][a:b], "new": side["tree"][c:d]}
                for tag,a,b,c,d in difflib.SequenceMatcher(None,base["tree"],side["tree"],autojunk=False).get_opcodes() if tag != "equal"]
    return result

def validated_result(item,pair,packet,model,tokens):
    required = ["status","checks","independent_left_label","independent_right_label","reason"]
    if any(k not in item for k in required): raise ValueError("required review field absent")
    checks = item["checks"]
    if not isinstance(checks,dict) or set(checks)!=set(CHECKS): raise ValueError("eight checks absent or inconsistent")
    if checks["duplicate"]!="DEFERRED_DETERMINISTIC": raise ValueError("duplicate falsely claimed reviewed")
    if any(checks[k] not in ["PASS","FAIL","UNCERTAIN"] for k in SEMANTIC_CHECKS): raise ValueError("invalid check value")
    ll,rl=item["independent_left_label"],item["independent_right_label"]
    if any(v is not None and (type(v) is not int or v not in [0,1]) for v in [ll,rl]): raise ValueError("invalid independent label")
    if not isinstance(item["reason"],str) or not item["reason"].strip(): raise ValueError("reason absent")
    status = "FAIL" if "FAIL" in [checks[k] for k in SEMANTIC_CHECKS] else "UNCERTAIN" if "UNCERTAIN" in [checks[k] for k in SEMANTIC_CHECKS] or ll is None or rl is None else "PASS"
    # This only combines AI semantic judgments; it does not supply missing AI judgments.
    if item["status"] != status: raise ValueError("status contradicts semantic checks")
    if ((ll is not None and ll != pair["declared_left_label"]) or
            (rl is not None and rl != pair["declared_right_label"])):
        checks = dict(checks, labels_correct="FAIL")
        status = "FAIL"
    return {"pair_id":pair["pair_id"],"group_id":packet["group_id"],"expected_kind":pair["expected_kind"],
            "status":status,"checks":checks,"independent_left_label":ll,"independent_right_label":rl,
            "reason":item["reason"],"review_model":model,"round":1,"input_scope":SCOPE,
            "input_tokens":tokens,"processing_status":"semantic_review_complete","reviewed_at":now()}
