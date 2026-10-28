"""Integration contracts for token weighting, sibling orientation and resume.

Model/tokenizer I/O is replaced with analytically known arrays. These fixtures
are software checks only and are never written to research result directories.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import run_module2_mechanism as runner


class _Tokenizer:
    def encode(self, value, add_special_tokens=False):
        return [{"A": 32, "B": 33}[value]]


class _Model:
    def eval(self):
        return self


class MechanismRunnerTests(unittest.TestCase):
    def test_token_weighting_sibling_orientation_and_resume_are_identical(self):
        source = runner.ROOT / "experiments/module2_mechanism_20261028/inputs/triplets.jsonl"
        all_rows = [json.loads(line) for line in source.read_text().splitlines()]
        # One complete real construction block per partition; only activations
        # are mocked. The actual independent construction audit still runs.
        chosen = {split: next(r["group_id"] for r in all_rows if r["split"] == split)
                  for split in ("fit", "select", "confirm")}
        rows = [r for r in all_rows if r["group_id"] == chosen[r["split"]]]
        prompt_states = {r["prompts"][state]: state for r in rows if r["target_rule_id"] == "A"
                         for state in runner.VARIANTS}
        prompt_splits = {r["prompts"][state]: r["split"] for r in rows if r["target_rule_id"] == "A"
                         for state in runner.VARIANTS}
        partition_scale = {"fit": 1., "select": 3., "confirm": 7.}
        state_indices = {state: i for i, state in enumerate(runner.VARIANTS)}
        positions = {"A": [1, 2], "B": [3, 4, 5]}
        geometry = {"layers": 1, "kv_heads": 1, "query_heads": 1,
                    "head_dim": 2, "queries_per_kv_head": 1}

        def encoded(state, online=False, prompt="fixture"):
            ids = torch.arange(8)[None].long() + 50
            ids[0, 6 if online else 0] = 70 + state_indices[state]
            return {"encoded": {"input_ids": ids}, "length": 8,
                    "rule_positions": positions, "state": state,
                    "online": online, "prefix_length": 6, "prompt": prompt}

        def encode(_tokenizer, prompt, _spans, _roles):
            return encoded(prompt_states[prompt], prompt=prompt)

        def online(_tokenizer, row, state):
            return encoded(state, online=True, prompt=row["prompts"][state])

        def capture(_model, inputs, _candidate_ids, directions=None):
            sign = {"positive": 1., "negative": -1., "neutral": 0.}[inputs["state"]]
            state_index = state_indices[inputs["state"]]
            split = prompt_splits[inputs["prompt"]]
            if inputs["online"] and split == "confirm":
                self.assertTrue((output / "selection.json").exists(),
                                "Head selection must be saved before confirmation queries")
            result = {}
            for rule, base in (("A", np.array([[1., 2.], [1., -2.]])),
                               ("B", np.array([[-2., 1.], [-2., 0.], [-2., -1.]]))):
                tokens = (np.zeros_like(base) if inputs["online"] else
                          sign * base * partition_scale[split])[None, :, None]
                query = np.array([[[2. + 2 * (rule == "B") + state_index, 1.]]])
                scores = {name: np.einsum("lhd,lhd->lh", query, delta) / np.sqrt(2)
                          for name, delta in (directions or {}).items()}
                result[rule] = {"token_keys": tokens, "effective_query": query,
                    "attention_mass": np.array([[.2]]),
                    "unit_rms": {name: np.abs(value) for name, value in scores.items()},
                    "per_query": {name: value[..., None] for name, value in scores.items()},
                    "token_positive": {name: (value > 0).astype(float) for name, value in scores.items()}}
            return {"rules": result, "candidate_logits": np.array([sign, -sign]), "geometry": geometry}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_file, model_dir, output = root / "pairs.jsonl", root / "model", root / "result"
            input_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            model_dir.mkdir()
            args = argparse.Namespace(triplets=str(input_file), model=str(model_dir),
                                      output=str(output), threads=1)
            with (patch("transformers.AutoTokenizer.from_pretrained", return_value=_Tokenizer()),
                  patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=_Model()),
                  patch.object(runner, "validate_qwen3", return_value=geometry),
                  patch.object(runner, "encode_prompt", side_effect=encode),
                  patch.object(runner, "tokenize_online", side_effect=online),
                  patch.object(runner, "capture_tokens", side_effect=capture) as measured,
                  contextlib.redirect_stdout(io.StringIO())):
                runner.extract(args)
                first_calls = measured.call_count
                with np.load(output / "measurements.npz") as pack:
                    first = {key: pack[key].copy() for key in pack.files}
                selection = json.loads((output / "selection.json").read_text())
                # A has 2 tokens and B has 3. Each target pair must still get 1/2
                # the weight: E d_A d_A^T=diag(4,16), E d_B d_B^T=diag(16,8/3).
                np.testing.assert_allclose(first["fit_token_second_moment"][0, 0],
                                           np.diag([10., 28. / 3.]))
                for i, row in enumerate(rows):
                    expected = np.array([2., 0.] if row["target_rule_id"] == "A" else [4., 0.])
                    expected *= partition_scale[row["split"]]
                    np.testing.assert_allclose(first["offline_keys"][i, 0, 0, 0]
                        - first["offline_keys"][i, 1, 0, 0], expected)
                    if row["split"] != "fit" and row["target_rule_id"] == "B":
                        # Positive for B is the canonical A-negative patient.
                        np.testing.assert_allclose(first["online_effective_queries"][i, 0, 0, 0], [5., 1.])
                        np.testing.assert_allclose(first["online_effective_queries"][i, 1, 0, 0], [4., 1.])
                self.assertEqual(selection["selected_heads"], [[0, 0]])
                self.assertFalse(selection["confirmation_used"])
                self.assertEqual(first_calls, 52)  # 12 contexts*3 + 8 held-out contexts*2
                runner.extract(args)
                self.assertEqual(measured.call_count, first_calls)
                with np.load(output / "measurements.npz") as pack:
                    for name, values in first.items():
                        np.testing.assert_array_equal(pack[name], values)
                self.assertEqual(json.loads((output / "selection.json").read_text()), selection)
                final_status = json.loads((output / "status.json").read_text())
                self.assertEqual(final_status["status"], "complete")


if __name__ == "__main__":
    unittest.main()
