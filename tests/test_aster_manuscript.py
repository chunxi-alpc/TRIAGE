"""工程契约测试。随机微型权重只测试数值/缓存，不构成方法效果证据。"""
import copy
import importlib.util
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from src.aster.manuscript_core import (
    FitConfig, PairMoments, ExcludedHead, validation_observation, select_expert,
    expert_directions, save_expert, load_expert, subspace_overlap, check_splits,
    guideline_groups, plan_gains, edit_kv_layers, rotate,
)
from src.aster.manuscript_runtime import (
    PromptBuilder, PreparedPrompt, ManuscriptBackbone, verified_triplet,
    aligned_triplet, learn_expert, run_case,
)

torch.set_num_threads(1)


def guideline():
    return {"groups": [{"group_id": "g", "mutually_exclusive": True, "ancestor_conditions": [],
                       "rules": [{"rule_id": "R1", "condition": "x=1", "action": "Alpha"},
                                 {"rule_id": "R2", "condition": "x=0", "action": "Beta"}]}]}


def triplet(pid="p", source="f", split="fit"):
    return dict(pair_id=pid, source_id=source, split=split, guideline=guideline(), target_rule_id="R1",
                question="选择哪个规则？", states=dict(positive="x=1", negative="x=0", unknown="x=?"),
                verification=dict(verifier="工程测试逻辑核验器", reference="布尔测试夹具，不是临床标注",
                                  positive_rule_id="R1", negative_rule_id="R2", unknown_label="NONE",
                                  ancestors_reachable=True, independent_of_classifier=True))


def candidate(dim=4):
    moments = PairMoments()
    p = torch.zeros((2, dim), dtype=torch.float64); p[:, 0] = 2
    moments.update(p, torch.zeros_like(p))
    return moments.estimate(FitConfig(1, 4))


def observation(plus=1.0, minus=1.0):
    return dict(z=1.0, u=0.25, read_plus=plus, read_minus=minus,
                positive_fraction_plus=float(plus > 0), positive_fraction_minus=float(minus > 0))


def expert(identity=None, template="t"):
    c = candidate((identity or {}).get("head_dim", 4))
    return select_expert({(0, 0): c}, {(0, 0): [observation(), observation()]}, FitConfig(1, 1),
                         identity=identity or {}, provenance={"template_id": template})


class CharacterTokenizer:
    """字符级工程夹具，仅用于可审计的位置对齐测试。"""
    is_fast = True
    chat_template = None
    def __call__(self, value, **kwargs):
        return dict(input_ids=[ord(c) % 127 + 1 for c in value],
                    offset_mapping=[(i, i + 1) for i in range(len(value))])
    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))


class TinyCache:
    def __init__(self):
        self.layers = []
    def get_seq_length(self):
        return self.layers[0][0].shape[-2] if self.layers else 0
    def to_legacy_cache(self):
        return tuple(self.layers)
    @classmethod
    def from_legacy_cache(cls, layers):
        obj = cls(); obj.layers = list(layers); return obj


class RMS(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * self.weight


class TinyAttention(nn.Module):
    def __init__(self, cfg, layer):
        super().__init__(); self.cfg = cfg; self.layer_idx = layer
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.q_norm = RMS(cfg.head_dim) if cfg.model_type == "qwen3" else nn.Identity()
        self.k_norm = RMS(cfg.head_dim) if cfg.model_type == "qwen3" else nn.Identity()
    def forward(self, hidden_states, position_embeddings, past_key_values=None, use_cache=False):
        n, d = hidden_states.shape[1], self.cfg.head_dim
        q = self.q_norm(self.q_proj(hidden_states).view(1, n, self.cfg.num_attention_heads, d)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(1, n, self.cfg.num_key_value_heads, d)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(1, n, self.cfg.num_key_value_heads, d).transpose(1, 2)
        self.last_key_pre = k.detach().clone()
        cos, sin = position_embeddings
        q = rotate(q, cos[:, None], sin[:, None]); k = rotate(k, cos[:, None], sin[:, None])
        self.last_query_rotated = q.detach().clone()
        old = past_key_values.layers[self.layer_idx] if past_key_values and len(past_key_values.layers) > self.layer_idx else None
        start = 0 if old is None else old[0].shape[-2]
        if old is not None:
            k = torch.cat((old[0], k), dim=-2); v = torch.cat((old[1], v), dim=-2)
        if use_cache:
            if len(past_key_values.layers) <= self.layer_idx:
                past_key_values.layers.append((k, v))
            else:
                past_key_values.layers[self.layer_idx] = (k, v)
        repetitions = self.cfg.num_attention_heads // self.cfg.num_key_value_heads
        k, v = k.repeat_interleave(repetitions, dim=1), v.repeat_interleave(repetitions, dim=1)
        scores = q @ k.transpose(-1, -2) / math.sqrt(d)
        allowed = torch.arange(k.shape[-2], device=q.device)[None, :] <= torch.arange(start, start + n, device=q.device)[:, None]
        weights = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
        result = (weights @ v).transpose(1, 2).reshape(1, n, -1)
        return self.o_proj(result)


class TinyLayer(nn.Module):
    def __init__(self, cfg, index):
        super().__init__(); self.self_attn = TinyAttention(cfg, index)
        self.norm = nn.LayerNorm(cfg.hidden_size)
    def forward(self, x, cos, sin, cache, use_cache):
        return x + self.self_attn(hidden_states=self.norm(x), position_embeddings=(cos, sin),
                                  past_key_values=cache, use_cache=use_cache)


class TinyLM(nn.Module):
    """随机初始化的本地因果注意力夹具，不是预训练 Qwen/Llama。"""
    def __init__(self, kind="qwen3"):
        super().__init__(); torch.manual_seed(7)
        self.config = SimpleNamespace(model_type=kind, hidden_size=16, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=4, num_hidden_layers=2, max_position_embeddings=4096,
                         rope_scaling=None, use_sliding_window=False, layer_types=["full_attention"] * 2,
                         eos_token_id=None)
        self.generation_config = SimpleNamespace(eos_token_id=None)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(128, 16)
        self.model.layers = nn.ModuleList([TinyLayer(self.config, i) for i in range(2)])
        self.lm_head = nn.Linear(16, 128, bias=False)
        self.fail_next = False
    def forward(self, input_ids, position_ids, past_key_values=None, use_cache=False, logits_to_keep=1, **kwargs):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("预定的工程测试异常")
        x = self.model.embed_tokens(input_ids)
        frequencies = position_ids[..., None].float() * torch.tensor([1.0, 0.01], device=x.device)
        phase = torch.cat((frequencies, frequencies), dim=-1)
        cache = past_key_values if past_key_values is not None else TinyCache() if use_cache else None
        for layer in self.model.layers:
            x = layer(x, phase.cos(), phase.sin(), cache, use_cache)
        return SimpleNamespace(logits=self.lm_head(x[:, -logits_to_keep:]), past_key_values=cache)


class NumericalTests(unittest.TestCase):
    def test_pair_weighting_and_token_outer_product(self):
        m = PairMoments()
        m.update(torch.eye(2), torch.zeros(2, 2))
        m.update(torch.tensor([[3., 0.]]).repeat(10, 1), torch.zeros(10, 2))
        self.assertTrue(torch.allclose(m.sum_mean / m.count, torch.tensor([1.75, .25], dtype=torch.float64)))
        self.assertTrue(torch.allclose(m.sum_outer / m.count, torch.diag(torch.tensor([4.75, .25], dtype=torch.float64))))

    def test_mean_norm_is_not_total_spectral_energy(self):
        m = PairMoments(); m.update(torch.tensor([[2., 1.], [2., -1.]]), torch.zeros(2, 2))
        c = m.estimate(FitConfig(1, 1))
        self.assertAlmostEqual(c["mean_norm_retention"], 1.0)
        self.assertAlmostEqual(c["spectral_energy_retention"], .8)

    def test_near_zero_mean_excluded(self):
        m = PairMoments(); m.update(torch.tensor([[1., 0.], [-1., 0.]]), torch.zeros(2, 2))
        with self.assertRaises(ExcludedHead): m.estimate(FitConfig(1, 1))

    def test_rank_insufficiency_excluded(self):
        m = PairMoments(); m.update(torch.ones(2, 4), torch.zeros(2, 4))
        with self.assertRaises(ExcludedHead): m.estimate(FitConfig(2, 1))

    def test_invalid_values_do_not_mutate_moments(self):
        m = PairMoments()
        for p in (torch.tensor([[float("nan")]]), torch.tensor([[1e300]], dtype=torch.float64)):
            with self.assertRaises(ValueError): m.update(p, torch.zeros_like(p))
        self.assertEqual(m.count, 0)

    def test_flipping_labels_preserves_spectrum_reverses_direction(self):
        p = torch.tensor([[3., 1.], [2., -1.]], dtype=torch.float64)
        a, b = PairMoments(), PairMoments(); a.update(p, torch.zeros_like(p)); b.update(torch.zeros_like(p), p)
        ca, cb = a.estimate(FitConfig(1, 1)), b.estimate(FitConfig(1, 1))
        self.assertEqual(ca["eigenvalues"], cb["eigenvalues"])
        self.assertTrue(torch.allclose(torch.tensor(ca["direction"]), -torch.tensor(cb["direction"])))

    def test_unknown_penalty_after_span_average(self):
        p, n = torch.tensor([[2., 0.], [2., 0.]]), torch.zeros(2, 2)
        unknown = torch.tensor([[101., 0.], [-99., 0.]])
        obs = validation_observation(p, n, unknown, torch.tensor([1., 0.]), torch.ones(2, 2),
                                     torch.ones(2, 2), torch.ones(2, 2), torch.zeros(2, 2))
        self.assertEqual(obs["u"], 0)
        self.assertEqual(obs["z"], 2)

    def test_double_positive_response_required(self):
        out = select_expert({(0, 0): candidate()}, {(0, 0): [observation(1, -1)] * 2}, FitConfig(1, 4), identity={}, provenance={})
        self.assertEqual(out["selected_heads"], [])

    def test_population_std_and_abs_missing(self):
        rows = [observation(), observation()]
        rows[0].update(z=1, u=-2); rows[1].update(z=3, u=2)
        out = select_expert({(0, 0): candidate()}, {(0, 0): rows}, FitConfig(1, 4), identity={}, provenance={})
        s = out["scores"]["0:0"]
        self.assertEqual(s["std"], 1); self.assertEqual(s["missing_penalty"], 2)
        self.assertAlmostEqual(s["separation"], 2 / (3 + 1e-8))

    def test_top_k_ties_and_no_padding(self):
        cs = {(2, 1): candidate(), (0, 1): candidate(), (0, 0): candidate()}
        obs = {h: [observation()] * 2 for h in cs}
        out = select_expert(cs, obs, FitConfig(1, 2), identity={}, provenance={})
        self.assertEqual(out["selected_heads"], ["0:0", "0:1"])
        out = select_expert(cs, obs, FitConfig(1, 8), identity={}, provenance={})
        self.assertEqual(len(out["selected_heads"]), 3)

    def test_validation_needs_two_pairs(self):
        with self.assertRaises(ValueError):
            select_expert({(0, 0): candidate()}, {(0, 0): [observation()]}, FitConfig(1, 1), identity={}, provenance={})

    def test_artifact_roundtrip_and_exclusive_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expert.json"; obj = expert(); save_expert(obj, path)
            self.assertEqual(load_expert(path), obj)
            with self.assertRaises(FileExistsError): save_expert(obj, path)

    def test_artifact_rejects_wrong_direction_and_schema(self):
        obj = expert(); obj["candidates"]["0:0"]["direction"] = [0., 1., 0., 0.]
        with self.assertRaises(ValueError): expert_directions(obj)
        obj = expert(); obj["schema_version"] = "query_projection"
        with self.assertRaises(ValueError): expert_directions(obj)

    def test_subspace_sign_invariance(self):
        a = torch.eye(4, dtype=torch.float64)[:, :2]
        self.assertAlmostEqual(subspace_overlap(a, -a), 1)
        self.assertAlmostEqual(subspace_overlap(a, torch.eye(4)[:, 2:]), 0)

    def test_split_sources_and_duplicate_ids(self):
        for fit, sel in (([triplet()], [triplet("s", "f", "selection")]),
                         ([triplet()], [triplet("p", "s", "selection")])):
            with self.assertRaises(ValueError): check_splits(fit, sel)
        with self.assertRaises(ValueError): check_splits([triplet()], [triplet("s", "s", "selection")], ["s"])

    def test_configuration_rejects_nan_and_boolean(self):
        for kwargs in (dict(rank=True, top_k=1), dict(rank=1, top_k=-1), dict(rank=1, top_k=1, epsilon=float("nan"))):
            with self.assertRaises(ValueError): FitConfig(**kwargs)


class RoutingTests(unittest.TestCase):
    def hierarchy(self):
        g = guideline()
        g["groups"].append(dict(group_id="child", mutually_exclusive=True, parent_group_id="g", parent_rule_id="R1",
              ancestor_conditions=["x=1"], rules=[dict(rule_id="C1", condition="y=1", action="C"),
                                                  dict(rule_id="C2", condition="y=0", action="D")]))
        return g

    def test_reachable_children_and_opposite_siblings(self):
        g = self.hierarchy()
        out = plan_gains(g, {"g": dict(prediction="R1", abstain=False), "child": dict(prediction="C2", abstain=False)}, g_plus=2, g_minus=3)
        self.assertEqual(out["gains"], dict(R1=2, R2=-3, C1=-3, C2=2))

    def test_abstain_none_failure_and_unselected_parent_close_child(self):
        for parent in (dict(prediction="R1", abstain=True), dict(prediction="NONE", abstain=False),
                       dict(prediction="R2", abstain=False), None):
            out = plan_gains(self.hierarchy(), {"g": parent, "child": dict(prediction="C2", abstain=False)}, g_plus=1, g_minus=1)
            self.assertEqual(out["gains"]["C1"], 0); self.assertEqual(out["gains"]["C2"], 0)

    def test_ancestor_conditions_required(self):
        g = self.hierarchy(); g["groups"][1]["ancestor_conditions"] = []
        with self.assertRaises(ValueError): guideline_groups(g)

    def test_duplicate_rule_and_invalid_gain(self):
        g = guideline(); g["groups"][0]["rules"][1]["rule_id"] = "R1"
        with self.assertRaises(ValueError): guideline_groups(g)
        with self.assertRaises(ValueError): plan_gains(guideline(), {"g": dict(prediction="R1", abstain=False)}, g_plus=float("inf"), g_minus=1)


class CacheTests(unittest.TestCase):
    def inputs(self):
        torch.manual_seed(3)
        layers = [(torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4)) for _ in range(2)]
        angles = torch.arange(4).double()[:, None].expand(4, 4) * .2
        return layers, angles.cos(), angles.sin()

    def test_rotated_translation_only_target_key(self):
        layers, cos, sin = self.inputs(); original = copy.deepcopy(layers)
        delta = torch.tensor([1., 0., 0., 0.])
        changed, audit = edit_kv_layers(layers, {(0, 1): delta}, {"R1": [1], "R2": [2]}, {"R1": 2, "R2": -1}, cos, sin)
        expected = rotate(delta.double()[None], cos[[1]], sin[[1]]) * 2
        self.assertTrue(torch.allclose(changed[0][0][0, 1, 1].double() - layers[0][0][0, 1, 1].double(), expected[0], atol=2e-7))
        self.assertTrue(torch.equal(changed[0][0][0, 0], layers[0][0][0, 0]))
        self.assertTrue(torch.equal(changed[1][0], layers[1][0]))
        for before, after, raw in zip(original, changed, layers):
            self.assertTrue(torch.equal(before[0], raw[0])); self.assertTrue(torch.equal(before[1], after[1]))
        self.assertEqual(audit["changed_key_vectors"], 2)

    def test_fixed_query_score_equation(self):
        layers, cos, sin = self.inputs(); v = torch.tensor([1., 0., 0., 0.])
        changed, _ = edit_kv_layers(layers, {(0, 0): v}, {"R": [1]}, {"R": -2}, cos, sin)
        queries = torch.randn(2, 4, dtype=torch.float64)
        observed = queries @ (changed[0][0][0, 0, 1] - layers[0][0][0, 0, 1]).double() / 2
        expected = -2 * (queries @ rotate(v.double(), cos[1], sin[1])) / 2
        self.assertTrue(torch.allclose(observed, expected, atol=3e-7))

    def test_zero_and_empty_expert_exact_but_not_aliased(self):
        layers, cos, sin = self.inputs()
        for directions, gains in (({(0, 0): torch.tensor([1., 0., 0., 0.])}, {"R": 0}), ({}, {"R": 5})):
            changed, audit = edit_kv_layers(layers, directions, {"R": [0]}, gains, cos, sin)
            self.assertTrue(audit["skipped"])
            self.assertTrue(torch.equal(changed[0][0], layers[0][0]))
            self.assertNotEqual(changed[0][0].data_ptr(), layers[0][0].data_ptr())

    def test_overlapping_duplicate_and_out_of_prefix_positions_rejected(self):
        layers, cos, sin = self.inputs()
        for positions in ({"R": [4]}, {"R": [1, 1]}, {"R": [1], "S": [1]}):
            with self.assertRaises(ValueError):
                edit_kv_layers(layers, {}, positions, {r: 0 for r in positions}, cos, sin)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyLM().eval()
        self.backend = ManuscriptBackbone(self.model, model_id="工程夹具", model_revision="随机初始化")
        self.builder = PromptBuilder(CharacterTokenizer(), use_chat_template=False)
        self.prompt = self.builder.prepare(guideline(), "x=1", "选择规则", online=True)

    def test_triplet_alignment_and_external_verifier(self):
        aligned_triplet(self.builder, triplet())
        row = triplet(); row["states"]["unknown"] += "额外文字"
        with self.assertRaises(ValueError): aligned_triplet(self.builder, row)
        row = triplet(); row["verification"]["independent_of_classifier"] = False
        with self.assertRaises(ValueError): verified_triplet(row)

    def test_patient_occurs_once_after_guideline(self):
        prompt = self.builder.prepare(guideline(), "唯一患者事实", "唯一问题", online=True)
        self.assertEqual(prompt.text.count("唯一患者事实"), 1)
        self.assertLess(prompt.text.index("指南"), prompt.text.index("唯一患者事实"))
        for indices in prompt.positions.values(): self.assertLess(max(indices), prompt.prefix_length)

    def test_key_capture_after_norm_before_rope_and_gqa_query(self):
        snap = self.backend.extract(self.prompt, "R1", keys=True, queries=True)
        positions = list(self.prompt.positions["R1"])
        attn = self.model.model.layers[0].self_attn
        self.assertTrue(torch.equal(snap.keys[0, 0], attn.last_key_pre[0, 0, positions].double()))
        self.assertTrue(torch.allclose(snap.queries[0, 0], attn.last_query_rotated[0, :2, -1].double(), atol=2e-7))

    def test_hooks_removed_on_failure(self):
        self.model.fail_next = True
        with self.assertRaises(RuntimeError): self.backend.extract(self.prompt, "R1", keys=True, queries=True)
        for layer in self.model.model.layers:
            self.assertFalse(layer.self_attn._forward_pre_hooks)
            self.assertFalse(layer.self_attn.q_norm._forward_hooks)
            self.assertFalse(layer.self_attn.k_norm._forward_hooks)

    def test_zero_generation_matches_segmented_baseline(self):
        obj = expert(self.backend.identity, self.builder.template_id)
        prefix = self.backend.prefill(self.prompt)
        before = copy.deepcopy(prefix.cache.to_legacy_cache())
        raw = self.backend.generate(self.prompt, None, {"R1": 0, "R2": 0}, max_new_tokens=3, prefix=prefix)
        zero = self.backend.generate(self.prompt, obj, {"R1": 0, "R2": 0}, max_new_tokens=3, prefix=prefix)
        self.assertEqual(raw["token_ids"], zero["token_ids"])
        for x, y in zip(before, prefix.cache.to_legacy_cache()):
            self.assertTrue(torch.equal(x[0], y[0])); self.assertTrue(torch.equal(x[1], y[1]))

    def test_nonzero_edit_does_not_mutate_prefix_or_parameters(self):
        obj = expert(self.backend.identity, self.builder.template_id)
        prefix = self.backend.prefill(self.prompt)
        before = copy.deepcopy(prefix.cache.to_legacy_cache()); parameters = copy.deepcopy(self.model.state_dict())
        output = self.backend.generate(self.prompt, obj, {"R1": 2, "R2": -2}, max_new_tokens=3, prefix=prefix)
        self.assertGreater(output["edit"]["changed_key_vectors"], 0)
        self.assertEqual(prefix.cache.get_seq_length(), self.prompt.prefix_length)
        for x, y in zip(before, prefix.cache.to_legacy_cache()): self.assertTrue(torch.equal(x[0], y[0]))
        for k, v in parameters.items(): self.assertTrue(torch.equal(v, self.model.state_dict()[k]))

    def test_incompatible_identity_and_template_rejected(self):
        obj = expert(self.backend.identity, "different")
        with self.assertRaises(ValueError): self.backend.generate(self.prompt, obj, {"R1": 1, "R2": -1}, max_new_tokens=1)

    def test_context_budget_and_training_mode_fail_closed(self):
        with self.assertRaises(ValueError): self.backend.generate(self.prompt, None, {"R1": 0, "R2": 0}, max_new_tokens=5000)
        self.model.train()
        with self.assertRaises(ValueError): self.backend.extract(self.prompt, "R1", keys=True, queries=False)

    def test_llama_without_key_norm(self):
        model = TinyLM("llama").eval(); backend = ManuscriptBackbone(model, model_id="fixture", model_revision="random")
        snap = backend.extract(self.prompt, "R2", keys=True, queries=True)
        self.assertEqual(snap.keys[0, 0].shape[-1], 4)

    def test_learn_complete_workflow(self):
        fitted, audit = learn_expert(self.backend, self.builder, [triplet("a"), triplet("b")],
                    [triplet("c", "s", "selection"), triplet("d", "s", "selection")], FitConfig(1, 2, min_norm=1e-10))
        self.assertEqual(audit["accepted_fit"], 2)
        self.assertEqual(audit["accepted_selection"], 2)
        expert_directions(fitted)

    def test_run_case_uses_same_unedited_backbone(self):
        class Classifier:
            def __init__(self, model): self.scorer = SimpleNamespace(model=model)
            def run(self, *args, **kwargs): return {"groups": [dict(prediction="R1", abstain=False)]}
        case = dict(case_id="engineering-fixture", guideline=guideline(), patient="x=1", question="选择规则")
        result = run_case(self.backend, self.builder, Classifier(self.model), expert(self.backend.identity, self.builder.template_id),
                          case, g_plus=1, g_minus=1, max_new_tokens=2, with_baseline=True)
        self.assertEqual(result["gain_plan"]["gains"], {"R1": 1, "R2": -1})
        self.assertIn("baseline", result)
        with self.assertRaises(ValueError):
            run_case(self.backend, self.builder, Classifier(TinyLM()), expert(), case, g_plus=1, g_minus=1, max_new_tokens=1)


@unittest.skipUnless(importlib.util.find_spec("transformers"), "当前环境未安装 Transformers；真实架构测试未运行")
class ActualHFArchitectureTests(unittest.TestCase):
    def check_architecture(self, kind):
        from transformers import Qwen3Config, Qwen3ForCausalLM, LlamaConfig, LlamaForCausalLM
        cls, config_cls = (Qwen3ForCausalLM, Qwen3Config) if kind == "qwen3" else (LlamaForCausalLM, LlamaConfig)
        cfg = config_cls(vocab_size=128, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=4, max_position_embeddings=128)
        cfg._attn_implementation = "eager"
        model = cls(cfg).eval()
        backend = ManuscriptBackbone(model, model_id="随机微型真实架构", model_revision="不是预训练权重")
        prompt = PreparedPrompt("", torch.tensor([[1, 4, 7, 3, 8, 9, 6]]), {"R1": (2,), "R2": (3,)}, 4, "fixture")
        snap = backend.extract(prompt, "R1", keys=True, queries=True)
        self.assertEqual(snap.keys[0, 0].shape, (1, 4))
        prefix = backend.prefill(prompt)
        before = copy.deepcopy(prefix.cache.to_legacy_cache())
        obj = expert(backend.identity, "fixture")
        a = backend.generate(prompt, None, {"R1": 0, "R2": 0}, max_new_tokens=2, prefix=prefix)
        b = backend.generate(prompt, obj, {"R1": 0, "R2": 0}, max_new_tokens=2, prefix=prefix)
        self.assertEqual(a["token_ids"], b["token_ids"])
        backend.generate(prompt, obj, {"R1": 1, "R2": -1}, max_new_tokens=2, prefix=prefix)
        for x, y in zip(before, prefix.cache.to_legacy_cache()): self.assertTrue(torch.equal(x[0], y[0]))
    def test_qwen3(self): self.check_architecture("qwen3")
    def test_llama(self): self.check_architecture("llama")


if __name__ == "__main__":
    unittest.main()
