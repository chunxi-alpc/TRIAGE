"""主稿方法的离线拟合与逐病例推理；显式本地模型、显式校准参数、新输出目录。"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import traceback

from src.aster.manuscript_core import FitConfig, load_expert, save_expert
from src.aster.manuscript_runtime import PromptBuilder, ManuscriptBackbone, learn_expert, run_case


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} 不是对象")
            rows.append(row)
    if not rows:
        raise ValueError(f"输入为空：{path}")
    return rows


def write_json(path: Path, record: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["fit", "run"])
    parser.add_argument("--model", required=True, help="已下载的本地模型目录；不会联网下载")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True, help="真实权重修订标识；由运行者核验")
    parser.add_argument("--output", required=True, help="必须不存在的新目录")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--layers", nargs="+", type=int, help="默认全部层；限制层须在开发前固定")
    parser.add_argument("--raw-template", action="store_true", help="基础模型无聊天模板时显式使用")
    parser.add_argument("--fit-data")
    parser.add_argument("--selection-data")
    parser.add_argument("--forbidden-sources", help="包含锁定测试来源编号数组的 JSON，不读取测试答案")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--expert")
    parser.add_argument("--cases")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--probability-threshold", type=float)
    parser.add_argument("--margin-threshold", type=float)
    parser.add_argument("--g-plus", type=float)
    parser.add_argument("--g-minus", type=float)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--classifier-max-input-tokens", type=int, default=8192)
    parser.add_argument("--with-baseline", action="store_true")
    args = parser.parse_args()
    names = (["fit_data", "selection_data", "rank", "top_k"] if args.mode == "fit" else
             ["expert", "cases", "temperature", "probability_threshold", "margin_threshold", "g_plus", "g_minus"])
    for name in names:
        if getattr(args, name) is None:
            parser.error("缺少 --" + name.replace("_", "-"))
    if not Path(args.model).is_dir():
        parser.error("--model 必须指向真实的本地目录")
    if importlib.metadata.version("transformers") != "4.57.6":
        parser.error("请使用 transformers==4.57.6")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", dict(status="loading", configuration=vars(args)))
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                  dtype=getattr(torch, args.dtype), attn_implementation="eager").to(args.device).eval()
        kwargs = {"enable_thinking": False} if model.config.model_type == "qwen3" else {}
        builder = PromptBuilder(tokenizer, use_chat_template=not args.raw_template, chat_template_kwargs=kwargs)
        backbone = ManuscriptBackbone(model, model_id=args.model_id, model_revision=args.model_revision, layers=args.layers)
        if args.mode == "fit":
            forbidden = json.loads(Path(args.forbidden_sources).read_text()) if args.forbidden_sources else []
            if not isinstance(forbidden, list) or any(not isinstance(x, str) for x in forbidden):
                raise ValueError("禁止来源必须是字符串数组")
            expert, audit = learn_expert(backbone, builder, read_jsonl(args.fit_data), read_jsonl(args.selection_data),
                                        FitConfig(args.rank, args.top_k), forbidden_sources=forbidden)
            save_expert(expert, output / "expert.json")
            write_json(output / "audit.json", audit)
            write_json(output / "status.json", dict(status="complete", mode="fit", audit=audit))
        else:
            from src.aster import ActionBlindBranchClassifier, HFLabelScorer
            expert = load_expert(args.expert)
            classifier = ActionBlindBranchClassifier(HFLabelScorer(model, tokenizer, max_input_tokens=args.classifier_max_input_tokens, chat_template_kwargs=kwargs),
                         temperature=args.temperature, probability_threshold=args.probability_threshold,
                         margin_threshold=args.margin_threshold)
            if args.raw_template:
                raise ValueError("现有 ABAA 依赖聊天模板；原始文本模板目前只用于离线机制实验")
            cases = read_jsonl(args.cases)
            ids = [r.get("case_id") for r in cases]
            if any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
                raise ValueError("病例编号缺失或重复")
            failed = 0
            with (output / "predictions.jsonl").open("x", encoding="utf-8") as handle:
                for case in cases:
                    try:
                        record = run_case(backbone, builder, classifier, expert, case, g_plus=args.g_plus,
                                  g_minus=args.g_minus, max_new_tokens=args.max_new_tokens, with_baseline=args.with_baseline)
                        record["status"] = "complete"
                    except Exception as error:
                        failed += 1
                        record = dict(case_id=case["case_id"], status="failed", error_type=type(error).__name__, error=str(error))
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"); handle.flush()
            write_json(output / "status.json", dict(status="complete", mode="run", cases=len(cases), failures=failed,
                                                     identity=backbone.identity, configuration=vars(args)))
    except Exception as error:
        write_json(output / "status.json", dict(status="failed", error_type=type(error).__name__, error=str(error)))
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
