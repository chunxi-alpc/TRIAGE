"""Compare a matched Base/IT pair using separately fitted E1 artifacts."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .evaluate import read, rows, write


def effective_rank(singular_values):
    values = np.square(np.asarray(singular_values, dtype=float))
    values = values[values > 0]
    if not len(values):
        return 0.0
    probability = values / values.sum()
    return float(np.exp(-np.sum(probability * np.log(probability))))


def _orthonormal(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if not matrix.size:
        return np.empty((matrix.shape[0], 0))
    u, singulars, _ = np.linalg.svd(matrix, full_matrices=False)
    keep = singulars > max(1e-10, float(singulars.max(initial=0)) * 1e-7)
    return u[:, keep]


def principal_angles(left, right):
    """Angles within one checkpoint's common selected-head coordinate space."""
    left, right = _orthonormal(left), _orthonormal(right)
    if not left.shape[1] or not right.shape[1]:
        return []
    cosines = np.linalg.svd(left.T @ right, compute_uv=False)
    return np.degrees(np.arccos(np.clip(cosines, 0.0, 1.0))).tolist()


def _architecture(root, heads_per_layer=None):
    root = Path(root)
    candidates = [root / "model_metadata.json", root / "configuration.json",
                  root / "models/E1/full/model_metadata.json", root / "cache/model_metadata.json"]
    metadata = {}
    for path in candidates:
        value = read(path, {})
        if isinstance(value, dict):
            metadata.update(value)
    nested = [metadata]
    for key in ("model", "architecture", "metadata", "text_config", "config"):
        if isinstance(metadata.get(key), dict):
            nested.append(metadata[key])
    layers = None
    for value in nested:
        if heads_per_layer is None:
            heads_per_layer = value.get("num_attention_heads", value.get("n_heads", value.get("heads_per_layer")))
        layers = layers or value.get("num_layers", value.get("num_hidden_layers", value.get("n_layers")))
    if heads_per_layer is None:
        paths = [root / "data/reused_scores.jsonl", *sorted((root / "cache").glob("scores*.jsonl"))]
        for path in paths:
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    if "feature" not in record:
                        continue
                    feature = Path(record["feature"])
                    if not feature.is_absolute():
                        feature = root / feature
                    with np.load(feature, allow_pickle=False) as arrays:
                        shape = arrays["heads"].shape
                    if len(shape) == 3:
                        layers, heads_per_layer = shape[:2]
                    break
            if heads_per_layer is not None:
                break
    if heads_per_layer is None:
        raise ValueError("Provide heads_per_layer or model_metadata.num_attention_heads for " + str(root))
    return dict(heads_per_layer=int(heads_per_layer), layers=int(layers) if layers is not None else None)


def _correction_behavior(root):
    result = {}
    for metric_path in sorted((Path(root) / "evaluation").glob("*/metrics.json")):
        scope = metric_path.parent.name
        metrics = read(metric_path, {})
        predictions = rows(metric_path.parent / "predictions.jsonl")
        usable = [r for r in predictions if "E1" in r.get("scores", {}) and "raw" in r["scores"]]
        scopes = {}
        for population, records in (("all_views", usable), ("original_views", [r for r in usable if r.get("original")])):
            correction = np.asarray([r["scores"]["E1"] - r["scores"]["raw"] for r in records], float)
            labeled = [r for r in records if r.get("label") is not None]
            raw_correct = np.asarray([r["predictions"]["raw"] == r["label"] for r in labeled], bool)
            new_correct = np.asarray([r["predictions"]["E1"] == r["label"] for r in labeled], bool)
            scopes[population] = dict(
                count=len(records), labeled_count=len(labeled),
                mean_correction=float(correction.mean()) if len(correction) else None,
                correction_rms=float(np.sqrt(np.mean(correction ** 2))) if len(correction) else None,
                median_abs_correction=float(np.median(np.abs(correction))) if len(correction) else None,
                p95_abs_correction=float(np.quantile(np.abs(correction), 0.95)) if len(correction) else None,
                decision_changes=sum(r["predictions"]["E1"] != r["predictions"]["raw"] for r in records),
                corrected_errors=int(np.sum(~raw_correct & new_correct)),
                harmed_correct=int(np.sum(raw_correct & ~new_correct)),
            )
        result[scope] = dict(prediction_behavior=scopes, metrics=metrics.get("models", {}))
    return result


def checkpoint_summary(root, heads_per_layer=None):
    root = Path(root)
    directory = root / "models/E1/full"
    with np.load(directory / "m2.npz", allow_pickle=False) as archive:
        m2 = dict(archive)
    with np.load(directory / "m3.npz", allow_pickle=False) as archive:
        m3 = dict(archive)
    model = read(directory / "model.json", {})
    architecture = _architecture(root, heads_per_layer)
    heads = np.asarray(m2["heads"], dtype=int).reshape(-1)
    layer_counts = Counter(int(head) // architecture["heads_per_layer"] for head in heads)
    spectra = {}
    for key in ("S_singular_values", "U_singular_values", "N_orthogonalization_singular_values",
                "projected_feature_singular_values"):
        values = m2.get(key, model.get("M2", {}).get(key))
        if values is None:
            spectra[key] = dict(status="not_saved")
        else:
            values = np.asarray(values, dtype=float).reshape(-1)
            spectra[key] = dict(singular_values=values.tolist(), retained_effective_rank=effective_rank(values),
                                saved_components=len(values))
    geometry = dict(
        S_rank=int(m2["S"].shape[1]), U_rank=int(m2["U"].shape[1]), N_rank=int(m2["N"].shape[1]),
        S_U_principal_angles_degrees=principal_angles(m2["S"], m2["U"]),
        S_N_principal_angles_degrees=principal_angles(m2["S"], m2["N"]),
        coordinate_policy="Each checkpoint uses its own independently selected, standardized head coordinates.",
        spectra=spectra,
    )
    return dict(
        root=str(root), architecture=architecture, selected_heads=heads.tolist(),
        selected_head_count=len(heads), layer_counts={str(k): v for k, v in sorted(layer_counts.items())},
        geometry=geometry,
        correction_parameters=dict(w_norm=float(np.linalg.norm(m3["w"])),
                                   b=float(np.asarray(m3.get("b", 0.0)).reshape(-1)[0]),
                                   delta=float(np.asarray(m3.get("delta", model.get("delta", 0.0))).reshape(-1)[0])),
        evaluation=_correction_behavior(root),
    )


def compare_pair(base_root, it_root, out, base_name="Base", it_name="IT", heads_per_layer=None):
    base = checkpoint_summary(base_root, heads_per_layer)
    it = checkpoint_summary(it_root, heads_per_layer)
    if base["architecture"]["heads_per_layer"] != it["architecture"]["heads_per_layer"]:
        raise ValueError("Head-index overlap requires a matched architecture")
    left_layers, right_layers = base["architecture"]["layers"], it["architecture"]["layers"]
    if left_layers is not None and right_layers is not None and left_layers != right_layers:
        raise ValueError("Head-index overlap requires equal layer counts")
    a, b = set(base["selected_heads"]), set(it["selected_heads"])
    intersection = sorted(a & b)
    result = dict(
        base_name=base_name, it_name=it_name, base=base, instruction_tuned=it,
        head_selection=dict(shared_heads=intersection, intersection_count=len(intersection),
                            union_count=len(a | b), jaccard=len(intersection) / len(a | b) if a | b else None,
                            base_overlap_fraction=len(intersection) / len(a) if a else None,
                            it_overlap_fraction=len(intersection) / len(b) if b else None),
        interpretation="Descriptive checkpoint comparison. Base/IT differences are not attributed causally to a specific post-training stage.",
        cross_checkpoint_subspace_angles="Not computed: each model independently chooses heads and standardization; those feature coordinates are not aligned.",
    )
    out = Path(out)
    write(out / "comparison.json", result)
    report = [
        "# Base/IT 检查点比较",
        "",
        f"{base_name} 与 {it_name} 分别独立拟合 M1、M2、M3。",
        "",
        "| 指标 | Base | IT |",
        "|---|---:|---:|",
        f"| 选中头数 | {len(a)} | {len(b)} |",
        f"| 证据子空间秩 | {base['geometry']['S_rank']} | {it['geometry']['S_rank']} |",
        f"| 干扰子空间秩 | {base['geometry']['U_rank']} | {it['geometry']['U_rank']} |",
        f"| 正交化后子空间秩 | {base['geometry']['N_rank']} | {it['geometry']['N_rank']} |",
        "",
        f"共有 {len(intersection)} 个选中头；Jaccard 系数为 {result['head_selection']['jaccard']:.4f}。",
        "",
        "各检查点的子空间谱、有效秩、内部夹角和修正行为见 comparison.json。",
        "不同检查点独立选择的头坐标没有对齐，不直接计算跨检查点子空间夹角。",
        "上述差异是检查点比较，不能解释为某个特定后训练阶段的因果效应。",
        "",
    ]
    (out / "REPORT.zh.md").write_text("\n".join(report), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--it-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-name", default="Base")
    parser.add_argument("--it-name", default="IT")
    parser.add_argument("--heads-per-layer", type=int)
    args = parser.parse_args()
    result = compare_pair(args.base_root, args.it_root, args.out, args.base_name, args.it_name, args.heads_per_layer)
    print(json.dumps({"out": str(args.out), "shared_heads": result["head_selection"]["intersection_count"]}))


if __name__ == "__main__":
    main()

