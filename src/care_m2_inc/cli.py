"""Command line entry points used by the staged N1--N5 protocol."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from .generate import prepare
from .io import read_json, read_jsonl, write_json
from .leakcheck import audit


def _default_data(config: str | None, out: str | None) -> Path:
    if out:
        return Path(out)
    if config:
        path = Path(config)
        return path if path.is_dir() else path.parent / "data"
    return Path("experiments/care_module2_incremental_20261020/data")


def cmd_prepare(args: argparse.Namespace) -> int:
    manifest = prepare(_default_data(args.config, args.out))
    print(json.dumps({"status": "data_prepared", "manifest": manifest}, ensure_ascii=False, indent=2))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    data = _default_data(args.config, args.out)
    report = audit(data, data / "leakcheck.json")
    print(json.dumps({"status": "cpu_verified" if report["valid"] else "blocked", "valid": report["valid"],
                      "tasks": report["tasks"], "violations": report["violations"]}, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 2


def cmd_smoke(args: argparse.Namespace) -> int:
    """CPU contract smoke; model smoke is implemented in model.py when a path is supplied."""
    if args.model:
        from .gpu_smoke import run
        result = run(args.model, _default_data(args.config, None), args.out, args.device)
    else:
        from .smoke import cpu_contract_smoke
        result = cpu_contract_smoke(_default_data(args.config, None))
    if args.out:
        write_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def cmd_n5_join(args: argparse.Namespace) -> int:
    from .n5 import build
    result = build(args.cases, args.gold, args.predictions, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_environment(args: argparse.Namespace) -> int:
    result = {"python": sys.version, "platform": platform.platform(), "argv": sys.argv}
    try:
        import torch
        result["torch"] = torch.__version__
        result["cuda"] = torch.cuda.is_available()
        result["gpu_count"] = torch.cuda.device_count()
    except Exception as exc:
        result["torch_error"] = repr(exc)
    try:
        import transformers
        result["transformers"] = transformers.__version__
    except Exception as exc:
        result["transformers_error"] = repr(exc)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    from .runner import fit
    return fit(args)


def cmd_select(args: argparse.Namespace) -> int:
    from .runner import select_stage
    return select_stage(args)


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .runner import evaluate_stage
    return evaluate_stage(args)


def cmd_analyze(args: argparse.Namespace) -> int:
    from .io import read_jsonl, read_json, write_json
    from .runner import summarize
    predictions = read_jsonl(Path(args.run) / "test_predictions.jsonl")
    task = args.experiment
    data = Path(args.data)
    rows = []
    inputs = read_jsonl(data / task / "inputs.jsonl")
    labels = read_jsonl(data / task / "labels.jsonl")
    splits = read_jsonl(data / task / "splits.jsonl")
    rows = [{**i, **label, **split} for i, label, split in zip(inputs, labels, splits) if split["split"] == "test"]
    result = summarize(task, predictions, rows)
    write_json(Path(args.run) / "analysis.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="care_m2_inc", description="CARE module-two incremental N1--N5 runner")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func in (("prepare", cmd_prepare), ("verify", cmd_verify)):
        p = sub.add_parser(name)
        p.add_argument("--config")
        p.add_argument("--out")
        p.set_defaults(func=func)
    p = sub.add_parser("smoke")
    p.add_argument("--config")
    p.add_argument("--out")
    p.add_argument("--model")
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=cmd_smoke)
    p = sub.add_parser("n5-join")
    p.add_argument("--cases", required=True)
    p.add_argument("--gold", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_n5_join)
    p = sub.add_parser("environment")
    p.set_defaults(func=cmd_environment)
    p = sub.add_parser("fit")
    p.add_argument("--experiment", choices=("N1", "N2", "N3", "N5"), required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--layer", type=int)
    p.set_defaults(func=cmd_fit)
    p = sub.add_parser("select")
    p.add_argument("--experiment", choices=("N1", "N2", "N3", "N4", "N5"), required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--layer", type=int)
    p.set_defaults(func=cmd_select)
    p = sub.add_parser("evaluate")
    p.add_argument("--experiment", choices=("N1", "N2", "N3", "N4", "N5"), required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--vectors")
    p.add_argument("--selection")
    p.add_argument("--device", default="cuda")
    p.add_argument("--git-commit", default="unknown")
    p.set_defaults(func=cmd_evaluate)
    p = sub.add_parser("analyze")
    p.add_argument("--experiment", choices=("N1", "N2", "N3", "N4", "N5"), required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--run", required=True)
    p.set_defaults(func=cmd_analyze)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
