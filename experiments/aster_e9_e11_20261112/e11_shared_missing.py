"""Measure only the two missing E11 rows; preserve the original six-method report."""
import argparse
import gc
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from experiments.aster_cr.comparators import Comparators
from experiments.aster_cr.engine import Engine
from .common import OUT, REPO, append, linefile, read, rows, write
from .e11 import Method, tensor_bytes

SUPPLEMENT = OUT / "e11_shared_missing"
MERGED = SUPPLEMENT / "merged"
ITI_ROOT = Path("/home/gcx/aster-e5-gaps-20261112/reports/aster_e5_gaps_20261112/representation_e1_plan/iti_style")
DORA_ROOT = Path("/home/gcx/aster-codex03-external-20261112/reports/aster_codex03_external_20261112/dora/full/seed11")
NAMES = ("iti", "dora")


class ITIMethod(Method):
    """Use the existing audited intervention, including its final-token-only scope."""
    def __init__(self, eng):
        super().__init__("iti", eng)
        self.comp = Comparators.__new__(Comparators)
        self.comp.eng = eng
        checkpoint = ITI_ROOT / "operator.pt"
        self.comp.ops = {"iti": torch.load(checkpoint, map_location=eng.device, weights_only=True)}
        operators = self.comp.ops["iti"]
        self.info.update(
            method_label="ITI-style task adaptation",
            checkpoint=str(checkpoint),
            frozen_configuration=str(ITI_ROOT / "selection.json"),
            trainable_parameters=0,
            fitted_offset_parameters=sum(t.numel() for t in operators.values()),
            deployment_tensor_bytes=tensor_bytes(operators)+len(operators)*8,
            storage_bytes=checkpoint.stat().st_size,
            parameters_note="Offsets fitted using the existing ITI artifact; no new training. Probe parameters are not deployed.",
            intervention="Existing Comparators.edited(context, 'iti'); unmodified prefix, last judgment token only",
            implementation_source=str(REPO / "experiments/aster_cr/comparators.py"),
        )

    def final(self, ctx):
        kv = []
        def record_cache(module, args, output):
            kv.append(sum(tensor_bytes(layer.keys) + tensor_bytes(layer.values)
                          for layer in output.past_key_values.layers))
        handle = self.eng.model.model.register_forward_hook(record_cache)
        try:
            score = self.comp.edited(ctx, "iti")
            return float(score), kv[-1]
        finally:
            handle.remove()


def make_method(name, eng):
    if name == "iti":
        return ITIMethod(eng)
    from peft import PeftModel, get_peft_model_state_dict
    checkpoint = DORA_ROOT / "adapter"
    frozen = read(DORA_ROOT / "frozen_configuration.json")
    wrapper = PeftModel.from_pretrained(eng.model, str(checkpoint), is_trainable=False)
    # The base Qwen model now contains the actual PEFT DoRA q/v modules.
    eng.model = wrapper.get_base_model()
    eng.layers = eng.model.model.layers
    eng.model.eval().requires_grad_(False)
    method = Method("dora", eng)
    method.peft_wrapper = wrapper
    method.info.update(
        method_label="DoRA-E1-SFT, seed 11",
        checkpoint=str(checkpoint),
        frozen_configuration=str(DORA_ROOT / "frozen_configuration.json"),
        trainable_parameters=frozen["parameter_count"],
        deployment_tensor_bytes=tensor_bytes(get_peft_model_state_dict(wrapper)),
        storage_bytes=sum(p.stat().st_size for p in checkpoint.rglob("*") if p.is_file()),
        seed=11,
        representative_selection="Fixed seed 11, independent of evaluation outcomes",
        adapter_merged=False,
        implementation="PEFT DoRA; adapted prefix and judgment, with no retraining",
        peft_version=frozen["peft_version"],
    )
    return method


def keys_for(name, cfg, examples):
    return {(name, length, repeat, ex["input_id"])
            for length in cfg["lengths"] for repeat in range(cfg["repeats"]) for ex in examples}


def record_key(row):
    return row["method"], row["length"], row["repeat"], row["input_id"]


def measure(name):
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    cfg = read(OUT / "configs/e11.json")
    examples = rows(OUT / "configs/e11_inputs.jsonl")
    root = SUPPLEMENT / name
    root.mkdir(parents=True, exist_ok=True)
    target = root / "measurements.jsonl"
    prior = rows(target) if target.exists() else []
    done = {record_key(r) for r in prior}
    expected = keys_for(name, cfg, examples)
    if done == expected and (root / "completion.json").exists():
        return
    write(root / "configuration.json", dict(
        **{k: v for k, v in cfg.items() if k not in ("pending_ITI", "new_methods", "gpu")},
        method=name, original_config=str(OUT / "configs/e11.json"),
        original_inputs=str(OUT / "configs/e11_inputs.jsonl"),
        original_gpu=cfg["gpu"], physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
        benchmark_only=True,
    ))
    environment = dict(
        torch=torch.__version__, python=platform.python_version(), cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        hardware=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version", "--format=csv"], text=True),
    )
    write(root / "environment.json", environment)
    started = time.time()
    eng = Engine("qwen")
    method = make_method(name, eng)
    method.info.update(physical_gpu=environment["visible_devices"], gpu_name=environment["gpu"])
    write(root / "method_info.json", {name: method.info})
    try:
        for length in cfg["lengths"]:
            if all((name, length, r, e["input_id"]) in done
                   for r in range(cfg["repeats"]) for e in examples):
                continue
            for w in range(cfg["warmup"]):
                method.measure(examples[w % len(examples)]["prompt"], length)
            torch.cuda.empty_cache()
            for repeat in range(cfg["repeats"]):
                for ix in np.random.default_rng(cfg["seed"] + repeat).permutation(len(examples)):
                    ex = examples[int(ix)]
                    key = name, length, repeat, ex["input_id"]
                    if key in done:
                        continue
                    append(target, dict(
                        method=name, length=length, repeat=repeat, input_id=ex["input_id"],
                        physical_gpu=environment["visible_devices"], gpu_name=environment["gpu"],
                        **method.measure(ex["prompt"], length),
                    ))
                    done.add(key)
        if done != expected:
            raise RuntimeError(f"Incomplete E11 measurements: {len(done)} / {len(expected)}")
        write(root / "completion.json", dict(
            complete=True, state="complete", method=name, measurements=len(done),
            seconds=time.time() - started, result_kind="actual_experiment",
            checkpoint=method.info["checkpoint"],
        ))
    finally:
        method.close()
        del method, eng
        gc.collect()
        torch.cuda.empty_cache()


def update_docs(complete, missing):
    doc = REPO / "docs/ASTER/experiments.zh.md"
    text = doc.read_text()
    text = text.replace("E11始终在GPU 1测量。", "E11原六方法在GPU 1测量，补测使用统一队列分配的同型号GPU。")
    text = text.replace("E11的审计通过ITI和新增代表方法按用户安排等待Codex00/01及后续方法完成，只补缺行。", "E11的ITI与DoRA缺失行使用冻结产物补测，状态与合并结果见下节。")
    status = "八种方法已完成" if complete else "六种方法已完成；ITI/DoRA补测已进入统一队列"
    evidence = ("原有240次测量加ITI/DoRA各40次，共320次；补测使用统一队列分配的实际GPU"
                if complete else "保留原有240次测量；补测ITI与固定种子11的DoRA，共80次")
    row = ("| E11 | 推理效率、参数量和显存 | **" + status + "** | " + evidence +
           " | 见[补充效率报告](../../reports/aster_e9_e11_20261112/e11_shared_missing/merged/REPORT.zh.md) |")
    text = "\n".join(row if line.startswith("| E11 |") else line for line in text.split("\n"))
    marker = "<!-- E11 shared missing rows -->"
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n"
    text += ("\n" + marker + "\n\n### E11缺失方法补测（2026-11-13）\n\n" +
             status + "。" + evidence +
             "。沿用原配置、4条开发输入、512/2048 token、每档3次预热和5轮重复；ITI复用已审计的任务适配检查点，DoRA固定种子11。原始六方法结果和E9报告保留。物理GPU编号与型号逐项记录；论文正文未修改。\n\n" +
             "[补充与合并结果](../../reports/aster_e9_e11_20261112/e11_shared_missing/merged/REPORT.zh.md)。\n")
    doc.write_text(text)


def merge():
    from . import summarize
    cfg = read(OUT / "configs/e11.json")
    examples = rows(OUT / "configs/e11_inputs.jsonl")
    MERGED.mkdir(parents=True, exist_ok=True)
    infos = read(OUT / "e11_method_info.json")
    records = rows(OUT / "e11_measurements.jsonl")
    missing = []
    added = []
    environments = {"original": read(OUT / "e11_environment.json")}
    for name in NAMES:
        root = SUPPLEMENT / name
        if not (root / "completion.json").exists():
            missing.append(name)
            continue
        new = rows(root / "measurements.jsonl")
        if len(new) != len(keys_for(name, cfg, examples)) or {record_key(r) for r in new} != keys_for(name, cfg, examples):
            raise RuntimeError(f"Incomplete or duplicate E11 measurements for {name}")
        records.extend(new)
        infos.update(read(root / "method_info.json"))
        environments[name] = read(root / "environment.json")
        added.append(name)
    linefile(MERGED / "e11_measurements.jsonl", records)
    write(MERGED / "e11_method_info.json", infos)
    write(MERGED / "environments.json", environments)
    write(MERGED / "configs/e11.json", dict(
        **{k: v for k, v in cfg.items() if k not in ("pending_ITI", "new_methods", "gpu")},
        original_gpu=cfg["gpu"], actual_hardware="environments.json",
        supplement_methods=list(NAMES), original_config=str(OUT / "configs/e11.json"),
    ))
    summarize.OUT = MERGED
    result = summarize.e11()
    complete = not missing
    result.update(
        state="complete" if complete else "partial", complete=complete, pending=missing,
        new_measurements=sum(1 for r in records if r["method"] in NAMES),
        preserved_original_measurements=len(rows(OUT / "e11_measurements.jsonl")),
        result_kind="actual_experiment", actual_supplement_methods=added,
        hardware_note="Original methods used GPU 1; supplements record their actual scheduler-assigned physical GPU and model.",
        source_directory=str(OUT), supplement_directory=str(SUPPLEMENT),
    )
    write(MERGED / "e11_statistics.json", result)
    lines = [
        "# E11补充与合并结果", "",
        f"已保存{len(records)}次实际测量，其中本次补充{result['new_measurements']}次。原始六方法的240次结果保留在原目录。",
        "ITI复用现有已审计的ITI-style任务适配检查点；DoRA使用固定种子11的已完成检查点。未重新训练。",
        "配置沿用原E11：批量1、4条固定开发输入、512/2048 token、每档3次预热与5轮重复。没有用评价结果选择代表种子。",
        "补测GPU由统一队列分配，编号和型号见 environments.json；原六方法使用GPU 1。跨卡延迟比较应结合实际硬件记录。",
        "DoRA保留未合并适配器，计入完整前缀和判断的开销；ITI只干预最后判断token，计入现有干预入口及KV测量开销。",
        "ITI的部署偏移参数另记 fitted_offset_parameters；其计时没有重新拟合探针。", "",
        "| 方法 | token上限 | 次数 | 延迟ms均值 | 吞吐样本/s | 显存峰值GiB | 梯度训练参数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["summary"]:
        lines.append(f"| {row['method']} | {row['length']} | {row['n']} | {row['latency_ms_mean']:.3f} | {row['throughput_samples_s']:.3f} | {row['peak_allocated_gib_max']:.3f} | {row['trainable_parameters']} |")
    lines += ["", "待完成：" + ("无。" if complete else "、".join(missing) + "。"),
              "", "逐例测量、均值/标准差、重复轮统计、KV、参数、存储及前向次数分别保存在本目录的 JSONL、CSV 和 JSON 文件中。"]
    (MERGED / "REPORT.zh.md").write_text("\n".join(lines) + "\n")
    from .e11_shared_report import render
    render(MERGED, OUT)
    update_docs(complete, missing)
    if complete:
        write(MERGED / "completion.json", dict(
            complete=True, state="complete", measurements=len(records),
            new_measurements=result["new_measurements"], result_kind="actual_experiment",
        ))
    write(MERGED / "merge_status.json", dict(
        merge_finished=True, methods_complete=complete, missing=missing, measurements=len(records),
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("measure", "merge"))
    parser.add_argument("--method", choices=NAMES)
    args = parser.parse_args()
    if args.action == "measure":
        if args.method is None:
            parser.error("--method is required for measure")
        measure(args.method)
    else:
        merge()


if __name__ == "__main__":
    main()
