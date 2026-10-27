"""Run current action-blind M1 using explicitly supplied policy parameters.

For the full system, instantiate HFLabelScorer with the already-loaded deployment
backbone instead of loading another copy through this standalone command. Model
files must already be local/cached. This command does not run M2 or edit attention.
"""

import argparse
import json
from pathlib import Path

from src.aster import (ActionBlindBranchClassifier, HFLabelScorer, build_blind_prompt,
                       summarize_logits, validate_guideline)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Deployment backbone ID in local cache, or local path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, required=True, help="Single temperature frozen on development data")
    parser.add_argument("--probability-threshold", type=float, required=True)
    parser.add_argument("--margin-threshold", type=float, required=True)
    parser.add_argument("--include-logits", action="store_true", help="Save raw label logits for development calibration/audit")
    args = parser.parse_args(argv)
    sample = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(sample, dict) or "guideline" not in sample or "patient" not in sample:
        parser.error("Input must contain guideline and patient")
    policy = {"temperature": args.temperature, "probability_threshold": args.probability_threshold,
              "margin_threshold": args.margin_threshold}
    try:
        summarize_logits({"R1": 0.0, "NONE": 0.0}, **policy)
        for group in validate_guideline(sample["guideline"]):
            build_blind_prompt(sample["patient"], group, task_context=sample.get("task_context"),
                               task_context_verified=sample.get("task_context_verified", False))
    except ValueError as exc:
        parser.error(str(exc))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                                               dtype="auto").to(args.device).eval()
    scorer = HFLabelScorer(model, tokenizer, max_input_tokens=args.max_input_tokens)
    engine = ActionBlindBranchClassifier(scorer, **policy)
    result = engine.run(sample["guideline"], sample["patient"], sample.get("question"), sample.get("options"),
                        task_context=sample.get("task_context"),
                        task_context_verified=sample.get("task_context_verified", False),
                        include_logits=args.include_logits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
