"""Run CARE V6 by default; --legacy explicitly selects the historical V1 modeler."""
import argparse
import json
from pathlib import Path

from . import create_module1_v6, validate_v6_input


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="JSON with guideline, patient, and question; V6 requires structured rules")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", help="Model device; V6 also reads CARE_MODULE1_V6_DEVICE")
    parser.add_argument("--cache-dir", help="V6 model download/cache directory")
    parser.add_argument("--base-model", help="V6 Phi-4 model ID or local path")
    parser.add_argument("--adapter-model", help="V6 TrialMatchAI LoRA ID or local path")
    parser.add_argument("--evidence-model", help="V6 BioLinkBERT-MedNLI ID or local path")
    parser.add_argument("--legacy", action="store_true", help="Run the historical Qwen/MedCPT V1 modeler")
    parser.add_argument("--top-k", type=int, help="Legacy rule retrieval count (default: 12); requires --legacy")
    args = parser.parse_args(argv)
    if args.top_k is not None and (not args.legacy or args.top_k < 1):
        parser.error("--top-k must be positive and is available only with --legacy")
    if args.legacy and any(value is not None for value in
                           (args.cache_dir, args.base_model, args.adapter_model, args.evidence_model)):
        parser.error("V6 model/cache options cannot be combined with --legacy")
    sample = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(sample, dict):
        parser.error("Input must be a JSON object")
    missing = {"guideline", "patient", "question"} - set(sample)
    if missing:
        parser.error("Missing input keys: " + ", ".join(sorted(missing)))
    if args.legacy:
        from . import LegacyModule1
        engine = LegacyModule1(top_k=args.top_k or 12, device=args.device or "cuda:0")
    else:
        try:
            validate_v6_input(sample["guideline"], sample["patient"], sample["question"])
        except ValueError as exc:
            parser.error(str(exc))
        engine = create_module1_v6(device=args.device, cache_dir=args.cache_dir,
            base_model=args.base_model, adapter_model=args.adapter_model,
            evidence_model=args.evidence_model)
    result = engine.run(guideline=sample["guideline"], patient=sample["patient"], question=sample["question"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
