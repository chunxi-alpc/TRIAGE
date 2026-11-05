"""Review affected semantic labels without requesting any patient answer."""
import argparse
import json
import os
import time
import traceback
from common import CONFIG, DATA, LOGS, RESULTS, append_row, read_rows, write_json
from revision import all_annotations, normalize_annotation
from source_annotations import rule_spans
from annotation_queue import queue_path, claim, finish, remaining, nodes_for
from annotate import parse_json

INSTRUCTION = """
Review ONLY the ORIGINAL decision condition corresponding to CURRENT_CONDITION.
MAPPED_ORIGINAL_EVIDENCE identifies that condition. Other source paragraphs and ancestor
conditions are context, not additional targets. Do not confuse an ancestor's performance
status definition with a current gene-fusion question, for example.
No patient case or answer is requested.

Does the original condition need an additional source definition to interpret its meaning?
A separate definition paragraph can be needed even when the rewritten CURRENT_CONDITION
already expands it. A fully explicit numerical threshold is self-contained. Merely using
medical terminology does not establish dependence on a particular passage.
A treatment recommendation, a list unrelated to the target, or repetition of an unexplained
term is NOT a definition. Use unknown when the necessary source evidence is unavailable.

Return JSON:
source_condition_action_mixed: "yes", "no", or "unknown" for the mapped original condition.
source_semantic_dependency: "yes", "no", or "unknown".
dataset_definition_completed: "yes", "no", or "unknown"; judge whether CURRENT_CONDITION
already includes the identified required definition; use unknown if no dependency is established.
dependency_reason: brief reasoning specifically naming the CURRENT target.
rule_evidence: optional block_id/quote source spans resolving the original condition if
the supplied mapped evidence is empty; use only literal source.
definition_links: an array, empty unless dependence is supported. Each link contains:
  focus_quote: an exact phrase from CURRENT_CONDITION identifying the target meaning;
  target_term: an exact phrase in MAPPED_ORIGINAL_EVIDENCE (or your rule_evidence) that needs explanation;
  relation: "needed_to_interpret_target";
  kind: one of "definition", "eligibility_criteria", "referent_list", "abbreviation";
  definition_span: {block_id, quote}, the MINIMAL verbatim source definition, excluding
    adjacent treatment advice, recommendations and patient answers;
  equivalent_spans: [{block_id, quote}] for OTHER supplied passages giving the same definition,
    including copies that remain after deletion; empty when none exist;
  explanation: how this definition resolves this target term.
missing_reference_queries: list of explicitly referenced definitions still unavailable, otherwise [].

Use real block IDs and literal text. Do not output necessary_restores: the program will derive
the deleted and retained positions from the verified definition spans. Do not prefer yes/no.
"""
def worker(shard):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    nodes = read_rows(DATA/"nodes.jsonl")
    previous = all_annotations()
    owner = dict(pid=os.getpid(), physical_gpu=os.environ["CUDA_VISIBLE_DEVICES"])
    first = claim(queue_path("full"), owner)
    status_path = RESULTS/f"annotation_full_{shard}_status.json"
    started = time.time()
    if first is None:
        write_json(status_path, dict(status="complete", done=0, successful=0))
        return
    write_json(status_path, dict(status="loading_model", pid=os.getpid(), **{"physical_gpu": owner["physical_gpu"]}))
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["medgemma"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(CONFIG["medgemma"], local_files_only=True,
        dtype=torch.bfloat16, device_map={"":"cuda:0"}, attn_implementation="sdpa")
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    processed = successful = failed = 0
    with (DATA/f"annotations_full_{shard}.jsonl").open("a") as output, \
         (LOGS/f"review_requests_{shard}.jsonl").open("a") as requests, \
         (LOGS/f"review_raw_{shard}.jsonl").open("a") as rawlog:
        for node in nodes_for("full", nodes, first):
            before = time.time()
            old = previous[node["node_id"]]
            payload = dict(CURRENT_CONDITION=node["condition"], node_id=node["node_id"],
                MAPPED_ORIGINAL_EVIDENCE=rule_spans(node), original_mapping=node["node_mapping"],
                designated_delete_block_ids=node["designated_delete_block_ids"],
                source_blocks=node["blocks"])
            prompt = INSTRUCTION + "\nDATA:\n" + json.dumps(payload, ensure_ascii=False)
            error = None
            try:
                for attempt in range(2):
                    text = tokenizer.apply_chat_template([dict(role="user", content=prompt+
                        ("\nCorrect the following response issue: "+error if error else ""))],
                        tokenize=False, add_generation_prompt=True)
                    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")
                    n_input = int(inputs["input_ids"].shape[1])
                    append_row(requests, dict(node_id=node["node_id"], attempt=attempt, prompt=text, input_tokens=n_input))
                    if n_input > CONFIG["annotation_max_input_tokens"]:
                        raise RuntimeError(f"{n_input} tokens exceed the unchanged input limit")
                    with torch.inference_mode():
                        generated = model.generate(**inputs, max_new_tokens=1600, do_sample=False,
                                                   pad_token_id=tokenizer.pad_token_id)
                    raw = tokenizer.decode(generated[0,n_input:], skip_special_tokens=True)
                    append_row(rawlog, dict(node_id=node["node_id"], attempt=attempt, text=raw))
                    try:
                        review = parse_json(raw)
                        for field in ("source_condition_action_mixed", "source_semantic_dependency", "dataset_definition_completed"):
                            if review.get(field) not in ("yes", "no", "unknown"):
                                raise ValueError(field+" must be yes/no/unknown")
                        if not isinstance(review.get("definition_links"), list):
                            raise ValueError("definition_links must be an array")
                        value = dict(old["annotation"])
                        value.update({k:review.get(k) for k in (
                            "source_condition_action_mixed", "source_semantic_dependency",
                            "dataset_definition_completed", "dependency_reason", "definition_links")})
                        value["missing_reference_queries"] = review.get("missing_reference_queries") or []
                        if review.get("rule_evidence"):
                            value["rule_evidence"] = review["rule_evidence"]
                        value["semantic_review_version"] = 3
                        record = normalize_annotation(node, value)
                        issues = record["annotation"].get("review_issues", [])
                        if review["source_semantic_dependency"] == "yes" and issues and attempt == 0:
                            error = "; ".join(issues)
                            continue
                        record.update(review_status="complete", semantic_schema=3,
                            semantic_review_version=3, input_tokens=n_input,
                            output_tokens=int(generated.shape[1]-n_input))
                        break
                    except (ValueError, KeyError, TypeError) as exc:
                        error = str(exc)
                        if attempt:
                            raise
                successful += 1
            except Exception as exc:
                traceback.print_exc()
                failed += 1
                value = dict(old["annotation"], source_semantic_dependency=None,
                    source_condition_action_mixed=None, dataset_definition_completed=None,
                    definition_links=[], semantic_review_version=3,
                    missing_reference_queries=["Automatic semantic review failed: "+str(exc)])
                record = normalize_annotation(node, value)
                record.update(review_status="failed", review_error=str(exc),
                              semantic_schema=3, semantic_review_version=3)
            record.update(management_extraction_completed=True, seconds=time.time()-before,
                previous_source_semantic_dependency=old["annotation"].get("source_semantic_dependency"))
            append_row(output, record)
            processed += 1
            finish(queue_path("full"), node["node_id"], record["review_status"])
            write_json(status_path, dict(status="running", done=processed, successful=successful,
                failed=failed, pending=remaining(queue_path("full")), pid=os.getpid(),
                elapsed_seconds=time.time()-started, mean_seconds_per_node=(time.time()-started)/processed,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
    write_json(status_path, dict(status="complete", done=processed, successful=successful, failed=failed,
        elapsed_seconds=time.time()-started, pid=os.getpid(),
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
    if not successful:
        raise RuntimeError("No semantic review completed")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, choices=[0,1], required=True)
    worker(p.parse_args().shard)
