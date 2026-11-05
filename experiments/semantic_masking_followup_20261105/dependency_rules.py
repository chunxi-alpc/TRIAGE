"""Bind each definition to the current condition and derive actual restoration spans."""
from common import compact, locate_quote

def usable_dependency(value, validation, unresolved=()):
    label = value.get("source_semantic_dependency")
    if (type(label) is bool and validation.get("source_traceable") and
            validation.get("dependency_evidence_valid") and not unresolved):
        return label
    return None

def row_dependency(row):
    return usable_dependency(row, {
        "source_traceable": row.get("source_traceable", True),
        "dependency_evidence_valid": row.get("dependency_evidence_valid", True),
    }, row.get("unresolved_references", []))

def reconcile_review(node, value):
    value = dict(value)
    blocks = {b["block_id"]: b for b in node["blocks"]}
    deleted = set(node["designated_delete_block_ids"])
    issues, accepted, definitions, removed, restores = [], [], [], [], []
    evidence = []
    for item in value.get("rule_evidence", []):
        try:
            locate_quote(blocks[item["block_id"]], item["quote"])
            evidence.append(item["quote"])
        except (KeyError, TypeError, ValueError):
            pass
    anchor = compact("\n".join(evidence))
    focus_text = compact(node.get("condition", ""))
    def source_span(item):
        bid, quote = item["block_id"], item["quote"]
        a, b = locate_quote(blocks[bid], quote)
        return dict(block_id=bid, quote=quote, start_in_block=a, end_in_block=b,
                    source_start=blocks[bid]["start"]+a, source_end=blocks[bid]["start"]+b)
    def add_unique(pool, item):
        if not any(x["block_id"] == item["block_id"] and x["quote"] == item["quote"] for x in pool):
            pool.append(item)
    if value.get("source_semantic_dependency") is True:
        for index, link in enumerate(value.get("definition_links", [])):
            try:
                focus, term = compact(link["focus_quote"]), compact(link["target_term"])
                if not focus or focus not in focus_text:
                    raise ValueError("focus_quote is not in the CURRENT condition")
                if not term or term not in anchor:
                    raise ValueError("target_term is not in the mapped ORIGINAL condition")
                if link.get("relation") != "needed_to_interpret_target":
                    raise ValueError("definition is not required by this target")
                if link.get("kind") not in ("definition", "eligibility_criteria", "referent_list", "abbreviation"):
                    raise ValueError("recommendation or bare terminology is not a definition")
                primary = source_span(link["definition_span"])
                if compact(primary["quote"]) == term:
                    raise ValueError("bare term repeats the term without explaining it")
                equivalents = []
                for item in link.get("equivalent_spans", []):
                    add_unique(equivalents, source_span(item))
                # Also recognize literal copies remaining in the supplied context.
                for block in node["blocks"]:
                    if block["block_id"] not in deleted and primary["quote"] in block["text"]:
                        add_unique(equivalents, source_span(dict(block_id=block["block_id"], quote=primary["quote"])))
                all_spans = [primary] + equivalents
                retained = [s for s in all_spans if s["block_id"] not in deleted]
                removed_spans = [s for s in all_spans if s["block_id"] in deleted]
                for span in all_spans:
                    add_unique(definitions, span)
                for span in removed_spans:
                    add_unique(removed, span)
                if removed_spans and not retained:
                    add_unique(restores, removed_spans[0])
                accepted.append(dict(link, definition_span=primary, equivalent_spans=equivalents,
                                     retained_after_deletion=retained,
                                     lost_after_deletion=bool(removed_spans and not retained)))
            except (KeyError, TypeError, ValueError) as exc:
                issues.append(f"definition_link[{index}]: {exc}")
        if not accepted:
            value["source_semantic_dependency"] = None
            value["dataset_definition_completed"] = None
            issues.append("No target-bound source definition; dependency is unknown")
    else:
        value["dataset_definition_completed"] = None
    value.update(definition_links=accepted, missing_definitions=definitions,
                 removed_definition_spans=removed, necessary_restores=restores,
                 definition_lost=bool(restores), review_issues=issues,
                 restoration_derivation="Confirmed target definitions inside deleted blocks with no retained equivalent",
                 semantic_review_version=3)
    return value
