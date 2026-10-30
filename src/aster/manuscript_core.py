"""ASTER 主稿公式的数值实现；不加载模型，不产生适用性伪标签。"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

Head = tuple[int, int]  # 层号、键值头号；均从零开始。
SCHEMA = "aster.manuscript.expert.v1"


def integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} 必须是大于等于 {minimum} 的整数")
    return value


def number(value: Any, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是有限数值")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} 超出有效范围")
    return result


def text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")
    return value


def array(value: Any, ndim: int, name: str) -> torch.Tensor:
    out = torch.as_tensor(value, device="cpu", dtype=torch.float64).detach()
    if out.ndim != ndim or not out.numel() or not torch.isfinite(out).all():
        raise ValueError(f"{name} 的形状、长度或有限性无效")
    return out


def head_name(head: Head) -> str:
    return f"{head[0]}:{head[1]}"


def parse_head(name: str) -> Head:
    parts = name.split(":")
    if len(parts) != 2 or any(not p.isdigit() for p in parts):
        raise ValueError("无效的层号:键值头号")
    return int(parts[0]), int(parts[1])


@dataclass(frozen=True)
class FitConfig:
    rank: int
    top_k: int
    epsilon: float = 1e-8
    min_norm: float = 1e-6
    eig_rtol: float = 1e-8

    def __post_init__(self) -> None:
        integer(self.rank, "rank", 1)
        integer(self.top_k, "top_k")
        for name in ("epsilon", "min_norm", "eig_rtol"):
            if number(getattr(self, name), name, 0) == 0:
                raise ValueError(f"{name} 必须大于零")
        if self.eig_rtol >= 1:
            raise ValueError("eig_rtol 必须小于一")


class ExcludedHead(ValueError):
    """数值秩不足或有向均值近零；记录排除原因，不补入其他无效头。"""


class PairMoments:
    """先逐词元求外积，再在最小对内平均，最后在最小对之间等权平均。"""
    def __init__(self) -> None:
        self.count = 0
        self.total_tokens = 0
        self.sum_mean: torch.Tensor | None = None
        self.sum_outer: torch.Tensor | None = None

    def update(self, positive: torch.Tensor, negative: torch.Tensor) -> None:
        p, n = array(positive, 2, "positive"), array(negative, 2, "negative")
        if p.shape != n.shape:
            raise ValueError("正负键必须逐词元对齐")
        d = p - n
        mean, outer = d.mean(0), d.T @ d / len(d)
        if not torch.isfinite(mean).all() or not torch.isfinite(outer).all():
            raise ValueError("差分或外积溢出")
        if self.sum_mean is None:
            self.sum_mean = torch.zeros_like(mean)
            self.sum_outer = torch.zeros_like(outer)
        if mean.shape != self.sum_mean.shape:
            raise ValueError("键维数发生变化")
        self.sum_mean += mean
        self.sum_outer += outer
        self.count += 1
        self.total_tokens += len(d)

    def estimate(self, config: FitConfig) -> dict[str, Any]:
        if not self.count:
            raise ExcludedHead("没有有效最小对")
        mean = self.sum_mean / self.count
        omega = self.sum_outer / self.count
        omega = (omega + omega.T) / 2
        if not torch.isfinite(omega).all() or not torch.isfinite(mean).all():
            raise ExcludedHead("累计统计量非有限")
        values, vectors = torch.linalg.eigh(omega)
        values, vectors = values.flip(0), vectors.flip(1)
        numerical_rank = int((values > max(float(values[0]), 0) * config.eig_rtol).sum())
        if config.rank > len(mean) or numerical_rank < config.rank:
            raise ExcludedHead("差分二阶矩数值秩不足")
        basis = vectors[:, :config.rank]
        projected = basis @ (basis.T @ mean)
        norm = float(projected.norm())
        if not math.isfinite(norm) or norm <= config.min_norm:
            raise ExcludedHead("投影后的有向均值近零")
        direction = projected / (norm + config.epsilon)
        energy = values.clamp_min(0)
        return {
            "basis": basis.tolist(), "direction": direction.tolist(),
            "mean": mean.tolist(), "projected_mean_norm": norm,
            "eigenvalues": values.tolist(), "numerical_rank": numerical_rank,
            "spectral_energy_retention": float(energy[:config.rank].sum() / energy.sum()),
            "mean_norm_retention": norm / float(mean.norm()),
            "fit_pairs": self.count, "fit_tokens": self.total_tokens,
        }


def rotate(vector: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """与 Qwen3/Llama 的半维旋转约定一致；输入必须已在同一设备。"""
    if vector.shape[-1] % 2 or cos.shape != sin.shape or cos.shape[-1] != vector.shape[-1]:
        raise ValueError("旋转维数不匹配")
    a, b = vector.chunk(2, dim=-1)
    return vector * cos + torch.cat((-b, a), dim=-1) * sin


def validation_observation(positive: torch.Tensor, negative: torch.Tensor,
                           unknown: torch.Tensor, direction: torch.Tensor,
                           query_positive: torch.Tensor, query_negative: torch.Tensor,
                           cos: torch.Tensor, sin: torch.Tensor) -> dict[str, Any]:
    """查询包含位置编码，形状为共享查询头数×键维度；旋转使用在线规则位置。"""
    p, n, u = (array(x, 2, name) for x, name in
               ((positive, "positive"), (negative, "negative"), (unknown, "unknown")))
    v = array(direction, 1, "direction")
    c, s = array(cos, 2, "cos"), array(sin, 2, "sin")
    if p.shape != n.shape or p.shape != u.shape or p.shape[-1] != len(v):
        raise ValueError("三状态键未对齐")
    if c.shape != s.shape or c.shape[-1] != len(v):
        raise ValueError("在线位置旋转形状无效")
    shifted = rotate(v[None, :], c, s)
    result = {"z": float(((p - n) @ v).mean()),
              "u": float((((p + n) / 2 - u) @ v).mean())}
    for name, query in (("plus", query_positive), ("minus", query_negative)):
        q = array(query, 2, name)
        if q.shape[-1] != len(v):
            raise ValueError("查询维数不匹配")
        response = q @ shifted.T / math.sqrt(len(v))
        result[f"read_{name}"] = float(response.mean())
        result[f"positive_fraction_{name}"] = float((response > 0).double().mean())
    return result


def select_expert(candidates: Mapping[Head, dict], observations: Mapping[Head, Sequence[dict]],
                  config: FitConfig, *, identity: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict:
    scores, eligible = {}, []
    for head, candidate in candidates.items():
        rows = observations.get(head, ())
        if len(rows) < 2:
            raise ValueError("选头至少需要两个完整验证最小对")
        z = array([r["z"] for r in rows], 1, "z")
        u = array([r["u"] for r in rows], 1, "u")
        rp = array([r["read_plus"] for r in rows], 1, "read_plus")
        rn = array([r["read_minus"] for r in rows], 1, "read_minus")
        numerator, deviation, missing = float(z.mean()), float(z.std(correction=0)), float(u.abs().mean())
        sep = numerator / (deviation + missing + config.epsilon)
        plus, minus = float(rp.mean()), float(rn.mean())
        valid = (candidate["projected_mean_norm"] > config.min_norm and
                 sep > 0 and plus > 0 and minus > 0)
        score = sep * (plus + minus) / 2
        if not math.isfinite(score):
            raise ValueError("非有限选头分数")
        scores[head_name(head)] = dict(separation=sep, numerator=numerator, std=deviation,
                                      missing_penalty=missing, read_plus=plus, read_minus=minus,
                                      score=score, valid=valid, validation_pairs=len(rows),
                                      positive_fraction_plus=sum(r["positive_fraction_plus"] for r in rows) / len(rows),
                                      positive_fraction_minus=sum(r["positive_fraction_minus"] for r in rows) / len(rows))
        if valid:
            eligible.append((score, head))
    selected = [h for _, h in sorted(eligible, key=lambda x: (-x[0], *x[1]))[:config.top_k]]
    return dict(schema_version=SCHEMA, config=asdict(config), identity=dict(identity),
                provenance=dict(provenance), selected_heads=[head_name(h) for h in selected],
                candidates={head_name(h): v for h, v in candidates.items()}, scores=scores)


def expert_directions(expert: Mapping[str, Any]) -> dict[Head, torch.Tensor]:
    if expert.get("schema_version") != SCHEMA:
        raise ValueError("专家格式不匹配，禁止使用查询谱投影产物替代")
    config = FitConfig(**expert["config"])
    names = expert["selected_heads"]
    if not isinstance(names, list) or len(names) != len(set(names)) or len(names) > config.top_k:
        raise ValueError("有效头集合无效")
    output = {}
    for name in names:
        head, record = parse_head(name), expert["candidates"][name]
        v = array(record["direction"], 1, "direction")
        basis, mean = array(record["basis"], 2, "basis"), array(record["mean"], 1, "mean")
        if basis.shape != (len(v), config.rank) or len(mean) != len(v):
            raise ValueError("专家维数不匹配")
        if not torch.allclose(basis.T @ basis, torch.eye(config.rank, dtype=torch.float64), atol=1e-7, rtol=1e-7):
            raise ValueError("子空间基不正交")
        projected = basis @ (basis.T @ mean)
        norm = float(projected.norm())
        if norm <= config.min_norm or not torch.allclose(v, projected / (norm + config.epsilon), atol=1e-8, rtol=1e-7):
            raise ValueError("专家方向不是归一化的投影均值")
        score = expert["scores"][name]
        if not score["valid"] or any(number(score[k], k) <= 0 for k in ("separation", "read_plus", "read_minus")):
            raise ValueError("所选头未通过双状态验证")
        output[head] = v
    return output


def save_expert(expert: Mapping[str, Any], path: str | Path) -> None:
    expert_directions(expert)
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(expert, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def load_expert(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        result = json.load(handle)
    expert_directions(result)
    return result


def subspace_overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = array(a, 2, "a"), array(b, 2, "b")
    if a.shape != b.shape:
        raise ValueError("只比较相同秩和原空间维数的子空间")
    for basis in (a, b):
        if not torch.allclose(basis.T @ basis, torch.eye(basis.shape[1], dtype=basis.dtype), atol=1e-7, rtol=1e-7):
            raise ValueError("子空间比较需要正交基")
    return float((a.T @ b).square().sum() / a.shape[1])


def check_splits(fit: Sequence[dict], selection: Sequence[dict], forbidden_sources: Sequence[str] = ()) -> None:
    """只按显式来源簇隔离；不声称自动识别不同名称下的同源指南。"""
    sources, seen = [], set()
    for rows, split in ((fit, "fit"), (selection, "selection")):
        if not rows:
            raise ValueError(f"{split} 分区为空")
        group = set()
        for row in rows:
            pid, source = text(row.get("pair_id"), "pair_id"), text(row.get("source_id"), "source_id")
            if pid in seen or row.get("split") != split:
                raise ValueError("配对标识重复或分区标记不匹配")
            seen.add(pid)
            group.add(source)
        sources.append(group)
    if sources[0] & sources[1] or (sources[0] | sources[1]) & set(forbidden_sources):
        raise ValueError("拟合、选头或禁止来源之间存在泄漏")


def guideline_groups(guideline: Mapping[str, Any]) -> dict[str, dict]:
    groups = guideline.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("guideline.groups 必须是非空列表")
    output, ids = {}, set()
    for group in groups:
        gid = text(group.get("group_id"), "group_id")
        rules = group.get("rules")
        if gid in output or group.get("mutually_exclusive") is not True or not isinstance(rules, list) or not 1 <= len(rules) <= 25:
            raise ValueError("规则组重复、未声明互斥或规则数量无效")
        ancestors = group.get("ancestor_conditions", [])
        if not isinstance(ancestors, list):
            raise ValueError("ancestor_conditions 必须为列表")
        for condition in ancestors:
            text(condition, "ancestor condition")
        for rule in rules:
            rid = text(rule.get("rule_id"), "rule_id")
            if rid == "NONE" or rid in ids:
                raise ValueError("规则编号保留或重复")
            ids.add(rid)
            text(rule.get("condition"), "condition")
            text(rule.get("action"), "action")
        output[gid] = group
    for gid, group in output.items():
        parent, via = group.get("parent_group_id"), group.get("parent_rule_id")
        if (parent is None) != (via is None):
            raise ValueError("父组和到达规则必须同时给出")
        if parent is not None and (parent not in output or via not in {r["rule_id"] for r in output[parent]["rules"]}):
            raise ValueError("父组到达边无效")
        if parent is not None:
            parent_rule = next(r for r in output[parent]["rules"] if r["rule_id"] == via)
            required = output[parent].get("ancestor_conditions", []) + [parent_rule["condition"]]
            if any(c not in group.get("ancestor_conditions", []) for c in required):
                raise ValueError("子组未包含完整祖先条件")
    for gid, group in output.items():
        seen, current = {gid}, group.get("parent_group_id")
        while current is not None:
            if current in seen:
                raise ValueError("规则组存在循环")
            seen.add(current)
            current = output[current].get("parent_group_id")
    return output


def plan_gains(guideline: dict, decisions: Mapping[str, dict | None], *, g_plus: float, g_minus: float) -> dict:
    """显式父组边控制可达性。None 表示分类接口失败，不冒充语义 NONE。"""
    plus, minus = number(g_plus, "g_plus", 0), number(g_minus, "g_minus", 0)
    groups = guideline_groups(guideline)
    if set(decisions) != set(groups):
        raise ValueError("分类结果必须恰好覆盖全部规则组")
    states, gains = {}, {}

    def visit(gid: str) -> dict:
        if gid in states:
            return states[gid]
        group = groups[gid]
        parent = group.get("parent_group_id")
        reachable = True
        if parent is not None:
            upstream = visit(parent)
            reachable = upstream["enabled"] and upstream["prediction"] == group["parent_rule_id"]
        decision = decisions[gid]
        prediction, enabled, reason = None, False, "classification_failure"
        if decision is not None:
            prediction = decision.get("prediction")
            if prediction not in {r["rule_id"] for r in group["rules"]} | {"NONE"} or not isinstance(decision.get("abstain"), bool):
                raise ValueError("分类结果的预测或弃权字段无效")
            reason = ("unreachable" if not reachable else "abstain" if decision["abstain"]
                      else "none" if prediction == "NONE" else "accepted")
            enabled = reason == "accepted"
        states[gid] = dict(reachable=reachable, enabled=enabled, prediction=prediction, reason=reason)
        for rule in group["rules"]:
            rid = rule["rule_id"]
            gains[rid] = (plus if rid == prediction else -minus) if enabled else 0.0
        return states[gid]

    for gid in groups:
        visit(gid)
    return {"gains": gains, "groups": states}


def edit_kv_layers(layers: Sequence[tuple[torch.Tensor, torch.Tensor]], directions: Mapping[Head, torch.Tensor],
                   positions: Mapping[str, Sequence[int]], gains: Mapping[str, float],
                   cos: torch.Tensor, sin: torch.Tensor) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], dict]:
    """主稿式(10)–(13)。只返回病例副本；不编辑值，不重复归一化，不投影当前键。"""
    if not layers or set(positions) != set(gains):
        raise ValueError("缓存为空或规则位置与增益集合不一致")
    length = layers[0][0].shape[-2]
    c, s = array(cos, 2, "cos"), array(sin, 2, "sin")
    if c.shape != s.shape or len(c) != length:
        raise ValueError("位置旋转必须覆盖整个指南前缀")
    occupied, checked = set(), {}
    for rid, indices in positions.items():
        if not indices:
            raise ValueError("每条规则必须有可编辑词元")
        p = [integer(i, "token position") for i in indices]
        if max(p) >= length or len(set(p)) != len(p) or occupied & set(p):
            raise ValueError("规则词元越界、重复或归属重叠")
        occupied.update(p)
        checked[rid] = (p, number(gains[rid], "gain"))
    clones = []
    for keys, values in layers:
        if keys.ndim != 4 or keys.shape[0] != 1 or keys.shape[-2] != length or values.shape != keys.shape:
            raise ValueError("仅支持单病例、全长度、非量化键值缓存")
        if not keys.is_floating_point() or not values.is_floating_point() or keys.shape[-1] != c.shape[-1]:
            raise ValueError("缓存类型或键维数无效")
        clones.append((keys.detach().clone(), values.detach().clone()))
    planned, actual, largest = 0, 0, 0.0
    for (layer, head), delta in directions.items():
        integer(layer, "layer"); integer(head, "kv_head")
        if layer >= len(clones) or head >= clones[layer][0].shape[1]:
            raise ValueError("专家头超出模型范围")
        v = array(delta, 1, "direction")
        if len(v) != c.shape[1] or float(v.norm()) > 1 + 1e-7:
            raise ValueError("方向维数或范数无效")
        keys = clones[layer][0]
        for indices, gain in checked.values():
            if gain == 0:
                continue
            index = torch.tensor(indices, device=keys.device)
            before = keys[0, head].index_select(0, index)
            precision = torch.float64 if keys.dtype == torch.float64 else torch.float32
            shift = rotate(v[None, :], c[indices], s[indices]).to(device=keys.device, dtype=precision) * gain
            after = (before.to(precision) + shift).to(keys.dtype)
            if not torch.isfinite(after).all():
                raise ValueError("键平移产生非有限值")
            keys[0, head].index_copy_(0, index, after)
            difference = after.to(precision) - before.to(precision)
            planned += len(indices)
            actual += int(difference.ne(0).any(-1).sum())
            largest = max(largest, float(difference.norm(dim=-1).max()))
    return clones, dict(planned_key_vectors=planned, changed_key_vectors=actual,
                        max_actual_shift_norm=largest, skipped=planned == 0)
