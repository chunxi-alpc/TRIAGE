"""Create the GitHub delivery bundle from finished experiments, using CPU only."""
import argparse
import csv
import gzip
import importlib.metadata
import json
import shutil
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
REPORT_REL = Path("reports/aster_codex03_external_20261112")
CODE_REL = Path("experiments/aster_codex03_external_20261112")
REPORT = SOURCE / REPORT_REL
E11 = Path("/home/gcx/aster-e9-e11-20261112/reports/aster_e9_e11_20261112/e11_shared_missing")


def package(destination):
    destination = Path(destination).resolve()
    if destination == SOURCE:
        raise ValueError("Use a separate publication checkout")
    output = destination / REPORT_REL
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    def deliver(path, target=None):
        path = Path(path)
        target = Path(target) if target is not None else destination / path.relative_to(SOURCE)
        compress = path.suffix == ".jsonl" or path.name in ("path_predictions.csv", "pair_predictions.csv")
        if compress:
            target = target.with_suffix(target.suffix + ".gz")
        target.parent.mkdir(parents=True, exist_ok=True)
        if compress:
            with path.open("rb") as src, target.open("wb") as dest:
                with gzip.GzipFile(filename="", mode="wb", fileobj=dest, mtime=0) as zipped:
                    shutil.copyfileobj(src, zipped)
        else:
            shutil.copy2(path, target)
        entries.append(dict(path=str(target.relative_to(destination)),
                            bytes=target.stat().st_size, source_bytes=path.stat().st_size,
                            encoding="gzip" if compress else "original"))

    for path in (SOURCE / CODE_REL).iterdir():
        if path.is_file() and path.suffix in (".py", ".md"):
            deliver(path)
    for name in ("REPORT.zh.md", "STATUS.zh.md", "RUN_STATUS.zh.md", "status.json",
                 "section4_5_new_methods.csv", "section4_5_method_summary.csv",
                 "official_sources.json", "source_preparation.json"):
        path = REPORT / name
        if path.exists():
            deliver(path)
    for dirname in ("upstream_audit", "figures"):
        for path in (REPORT / dirname).rglob("*"):
            if path.is_file() and path.suffix not in (".pyc", ".log", ".lock"):
                deliver(path)
    for path in (REPORT / "prepared").glob("*.json"):
        deliver(path)
    for path in (REPORT / "evaluation").rglob("*"):
        if path.is_file() and path.suffix in (".csv", ".json", ".jsonl"):
            deliver(path)
    roots = [REPORT / "caa", REPORT / "inlp"]
    roots += list((REPORT / "dora/full").glob("seed*"))
    roots += list((REPORT / "reps").glob("seed[0-9]*"))
    roots += list((REPORT / "psr").glob("seed[0-9]*"))
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            parts = path.relative_to(root).parts
            if any(p in ("smoke", "answer_activations", "raw_prompt_residuals", "__pycache__") for p in parts):
                continue
            if path.suffix == ".json":
                deliver(path)
            elif path.suffix == ".jsonl" and len(parts) == 1:
                deliver(path)
            elif path.suffix in (".pt", ".npz") and path.stat().st_size < 2**20:
                deliver(path)
    for name in ("teacher_config.json", "summary.json"):
        deliver(REPORT / "psr/teacher_fit" / name)
    for name in ("evaluation_checks.json", "resume_check.json"):
        deliver(REPORT / "validation" / name)
    efficiency = output / "efficiency/dora_seed11"
    for path in (E11 / "dora").iterdir():
        if path.is_file() and path.suffix in (".json", ".jsonl"):
            deliver(path, efficiency / path.name)
    with (E11 / "merged/e11_summary.csv").open() as handle:
        summaries = [row for row in csv.DictReader(handle) if row["method"] == "dora"]
    with (efficiency / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (efficiency / "README.zh.md").write_text(
        "# DoRA固定种子11的E11推理计时\n\n"
        "复用已完成的DoRA检查点，512/2048 token各20次，批量1；4条固定开发输入、每档3次预热、5轮重复。"
        "实际GPU为4，型号NVIDIA A100-SXM4-80GB，见environment.json。完整逐次记录为measurements.jsonl.gz。\n\n"
        "计时包含分词、传入GPU、前缀计算、判断和分数回传，模型加载在计时外。未合并DoRA适配器。"
        "此处是固定输入长度的完整请求计时；主表的inference_seconds_per_input统计完整评价中的唯一UID。\n\n"
        "原始测量代码保存在服务器 /home/gcx/aster-e9-e11-20261112/experiments/aster_e9_e11_20261112/e11_shared_missing.py，"
        "完整E11交付由其对应报告维护。\n")
    packages = {}
    for name in ("torch", "transformers", "peft", "numpy", "scipy", "scikit-learn", "pandas", "matplotlib", "safetensors", "accelerate"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    (output / "environment_versions.json").write_text(json.dumps(packages, indent=2) + "\n")
    server = dict(
        host="gcx@10.29.128.115",
        experiment_root=str(SOURCE),
        environment="/home/gcx/gcx261014",
        official_sources="/home/gcx/aster_codex03_external_assets_20261112/official",
        optional_dependencies="/home/gcx/aster_codex03_external_assets_20261112/python",
        model="/home/gcx/TRIAGE/models/modelscope/Qwen3-8B",
        dataset="/home/gcx/aster_cr_pair_dataset_20261111/balanced_train.jsonl",
        prepared_source="/home/gcx/aster-cr-retrain-paired-v2-20261111/reports/aster_cr_retrain_paired_v2_20261111",
        retained_large_assets=[
            "dora/full/seed{11,22,33}/adapter/adapter_model.safetensors",
            "inlp/iterations_{1,4,8}/projection.npz",
            "psr/teacher_fit/", "caa/answer_activations/", "caa/raw_prompt_residuals/",
            "prepared/E1_pairs_full.jsonl", "prepared/fit/", "prepared/dev/", "prepared/test/",
        ],
        publication_contents="All completed test predictions/statistics, available development evaluations, development selections, configurations, training logs, small learned vectors, code and official version records",
    )
    (output / "SERVER_ARTIFACTS.json").write_text(json.dumps(server, ensure_ascii=False, indent=2) + "\n")
    (output / "DELIVERY.zh.md").write_text(
        "# GitHub交付说明\n\n"
        "本目录收录5种新增方法的11次实际冻结测试结果。入口为[结果与结论](REPORT.zh.md)，"
        "[逐运行主表](section4_5_new_methods.csv)和[方法均值与种子标准差](section4_5_method_summary.csv)。\n\n"
        "evaluation/test/保存11次完整测试；evaluation/dev/保存5次已执行的完整开发评价。"
        "RePS和PSR的开发选择分数、候选强度与冻结配置位于各自种子目录。\n\n"
        "逐例JSONL及逐对/路径CSV以gzip无损压缩保存，文件名追加.gz；可用Python gzip.open读取。"
        "其余CSV和JSON保持原格式。官方版本和实现差异见official_sources.json与各方法配置，"
        "早期准备记录中的pending/not_started表示记录当时状态，当前完成状态见status.json。\n\n"
        "大型权重、教师激活和完整输入缓存保留在服务器，位置见[SERVER_ARTIFACTS.json](SERVER_ARTIFACTS.json)。"
        "环境版本见[environment_versions.json](environment_versions.json)，运行入口见"
        "[代码说明](../../experiments/aster_codex03_external_20261112/README.md)。\n")
    (output / "publication_manifest.json").write_text(json.dumps(
        dict(source_root=str(SOURCE), result_kind="actual_experiment", methods=5,
             completed_test_runs=11, compression="lossless gzip for JSONL and detailed prediction CSV",
             uploaded_source_files=entries), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(source_files=len(entries), compressed_bytes=sum(x["bytes"] for x in entries))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination")
    args = parser.parse_args()
    package(args.destination)
