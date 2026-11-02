"""Train, run, or score the frozen no_js experiment from the repository root."""
import argparse
import json
import random
from pathlib import Path

from .data import prepare, read_rows
from .metrics import strict_score

DEFAULT_WEIGHTS = Path(__file__).with_name("no_js_layer18_epoch3.safetensors")
REVISION = "26028140be3ee69b82b1d1450179ab71bb1121b9"
SEED = 20261101


def make_engine(model_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .engine import Engine

    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda:0")
    return Engine(model, tok)


def train(args):
    import torch
    from safetensors.torch import save_file
    from .intervention import ConditionalIntervention
    from .native_memory import query_chunked_attention
    from .training import train_case

    rows = list(read_rows(args.cases)) + list(read_rows(args.age_cases))
    engine = make_engine(args.model)
    torch.manual_seed(SEED)
    module = ConditionalIntervention(engine.width, rank=4, variant="no_js").to(engine.device)
    opt = torch.optim.AdamW(module.parameters(), lr=0.001, weight_decay=0.01)
    args.output.mkdir(parents=True, exist_ok=False)
    with query_chunked_attention(engine.model, 256):
        for epoch in (1, 2, 3):
            order = list(rows)
            random.Random(SEED + epoch).shuffle(order)
            opt.zero_grad(set_to_none=True)
            for i, case in enumerate(order, 1):
                train_case(engine, case, module, 18)
                engine.blind_cache.clear()
                if i % 8 == 0 or i == len(order):
                    if i == len(order) and i % 8:
                        for param in module.parameters():
                            if param.grad is not None:
                                param.grad.mul_(8 / (i % 8))
                    torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0, error_if_nonfinite=True)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
            save_file({k: v.cpu().contiguous() for k, v in module.state_dict().items()},
                      args.output / f"epoch{epoch}.safetensors")


def predict(args):
    import torch
    from safetensors.torch import load_file
    from .inputs import deployment
    from .intervention import ConditionalIntervention
    from .native_memory import query_chunked_attention

    engine = make_engine(args.model)
    module = ConditionalIntervention(engine.width, rank=4, variant="no_js").to(engine.device)
    module.load_state_dict(load_file(args.weights))
    module.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle, torch.no_grad(), query_chunked_attention(engine.model, 256):
        for case in read_rows(args.cases):
            result = engine.generate(deployment(case), module, 18)
            row = dict(case_id=case["case_id"], kind="no_js", status="complete", result=result)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            engine.blind_cache.clear()


def score(cases, predictions):
    files = sorted(predictions.glob("*.jsonl")) if predictions.is_dir() else [predictions]
    texts = {}
    for file in files:
        for row in read_rows(file):
            if row["kind"] == "no_js":
                texts[row["case_id"]] = row["result"]["text"] if row["status"] == "complete" else ""
    total = correct = 0
    for case in read_rows(cases):
        correct += strict_score(case, texts[case["case_id"]])["exact_path"]
        total += 1
    print(json.dumps(dict(cases=total, exact_path=correct, PathAcc=correct / total)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--split", choices=["train", "age_train", "development", "confirmation", "official_test"], required=True)
    p.add_argument("--output", type=Path, required=True)
    for name in ("train", "predict", "score"):
        p = commands.add_parser(name)
        p.add_argument("--cases", type=Path, required=True)
        if name == "score":
            p.add_argument("--predictions", type=Path, required=True)
        else:
            p.add_argument("--model", type=Path, required=True)
            p.add_argument("--output", type=Path, required=True)
            if name == "train":
                p.add_argument("--age-cases", type=Path, required=True)
            else:
                p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.dataset, args.output, args.split)
    elif args.command == "train":
        train(args)
    elif args.command == "predict":
        predict(args)
    else:
        score(args.cases, args.predictions)


if __name__ == "__main__":
    main()
