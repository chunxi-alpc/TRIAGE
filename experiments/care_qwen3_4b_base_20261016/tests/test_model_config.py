"""Guard against stale Qwen3 small-model GQA/head/layer configuration."""
from pathlib import Path
import yaml


def test_preliminary_model_fixed_to_qwen3_4b_base():
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs" / "pilot.yaml").read_text())
    model = cfg["model"]
    assert model["id"] == "Qwen/Qwen3-4B-Base"
    assert model["fallback_id"] is None
    assert model["expected_num_layers"] == 36
    assert model["expected_num_attention_heads"] == 32
    assert model["expected_num_key_value_heads"] == 8
    assert model["expected_head_dim"] == 128
    assert model["expected_num_attention_heads"] % model["expected_num_key_value_heads"] == 0
    assert model["expected_num_attention_heads"] // model["expected_num_key_value_heads"] == 4


def test_no_stale_small_model_config_in_task_instructions():
    root = Path(__file__).resolve().parents[1]
    for filename in ["AGENTS.md", "CODEX_TASK.md", "EXPERIMENT_PROTOCOL.md", "README.md", "CODEX_PROMPT.md"]:
        s=(root/filename).read_text()
        assert "Qwen/Qwen3-0.6B-Base" not in s
        assert "Qwen/Qwen3-1.7B-Base" not in s
