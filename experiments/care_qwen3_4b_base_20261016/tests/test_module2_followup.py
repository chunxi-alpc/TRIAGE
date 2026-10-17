"""Server tests for M2 algebra and controls. These are NOT recorded as executed."""
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from module2_subspaces import (
    bias_support, contrastive_basis, fit_basis, key_edit, normalized_delta,
    paired_rows, training_matrices,
)

def test_key_edit_budget_and_restore():
    rng = torch.Generator().manual_seed(7)
    x = torch.randn(1, 7, 8, 128, generator=rng)
    op = torch.diag(torch.tensor([1.] * 4 + [0.] * 124))
    mod = torch.nn.Identity()
    model = SimpleNamespace(model=SimpleNamespace(layers=[
        SimpleNamespace(self_attn=SimpleNamespace(k_norm=mod))]))
    idx = [2, 3]
    before = x.clone()
    with key_edit(model, idx, (0, 3), op, .05) as report:
        after = mod(x)
    assert torch.equal(x, before), "hook must not mutate its input"
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[0, idx, 3] = True
    assert torch.equal(after[~mask], before[~mask])
    assert report["relative_norm"] == pytest.approx(.05, abs=1e-6)
    assert torch.equal(mod(x), before), "hook leaked after context exit"
    assert torch.count_nonzero(normalized_delta(x[0, idx, 3], op, 0)) == 0

def test_patch_changes_only_projected_difference():
    mod = torch.nn.Identity()
    model = SimpleNamespace(model=SimpleNamespace(layers=[
        SimpleNamespace(self_attn=SimpleNamespace(k_norm=mod))]))
    x = torch.zeros(1, 3, 8, 128)
    donor = torch.ones(1, 128)
    op = torch.diag(torch.tensor([1.] + [0.] * 127))
    with key_edit(model, [1], (0, 0), op, 0, "patch", donor):
        y = mod(x)
    assert y.sum() == 1 and y[0, 1, 0, 0] == 1

def test_rank_and_contrastive_background():
    signal = np.zeros((20, 8))
    signal[:10, 0] = 2
    signal[10:, 1] = 1
    background = np.zeros((20, 8))
    background[:, 0] = 20
    q = contrastive_basis(signal, background, 1)
    assert abs(q[1, 0]) > .99, "contrastive direction should suppress background variation"
    q = fit_basis(signal, 2)
    np.testing.assert_allclose(q.T @ q, np.eye(2), atol=1e-10)
    with pytest.raises(ValueError):
        fit_basis(signal, 3)
    with pytest.raises(ValueError):
        contrastive_basis(signal, background, 3)

def test_numeric_controls_must_match_same_rule_and_magnitude():
    def row(cid, rule, pair, app, value):
        return dict(case_id=cid, group_id="family", rule_id=rule, pair_id=pair,
                    applicable=app, value=value, gold_label="A" if app else "B",
                    prompt="Rule: action", spans=dict(action=[6,12]), option_map={"A":"a","B":"b"})
    a = row("a","rule1","pair",True,0)
    b = row("b","rule1","pair",False,4)
    bad = row("c","rule2","other",True,0)
    wrong_rule = row("d","rule2","other",True,4)
    x = row("e","rule1","other",True,-1)
    matched = row("f","rule1","other",True,-4)
    rows = [a,b]
    pool = rows+[bad,wrong_rule,x,matched]
    keys = {r["case_id"]:{(0,0):np.ones((1,8))*i} for i,r in enumerate(pool)}
    _,_,background,audit = training_matrices(rows,keys,(0,0),control_rows=pool)
    assert background is not None and audit[0]["control"] == ["a","f"]
    _,_,background,audit = training_matrices(rows,keys,(0,0),control_rows=rows+[bad,wrong_rule,x])
    assert background is None and not audit

def test_bias_support_respects_causality_and_gqa():
    mask = bias_support(8,[2,4],(0,3),"cpu")
    assert mask.shape == (32,8,8)
    assert not mask[:12].any() and not mask[16:].any()
    assert not torch.triu(mask,diagonal=1).any()
    assert mask[12:16,4:,4].all()
    assert not mask[:,:,0].any()

def test_pairs_cannot_silently_drop_missing_members():
    row = dict(case_id="a",pair_id="pair",group_id="g",applicable=True)
    with pytest.raises(AssertionError):
        paired_rows([row])


def test_multigpu_selection_skips_occupied_and_small_cards(monkeypatch):
    import launch_module2_multigpu as launcher
    def fake_output(command, **kwargs):
        if "--query-compute-apps=gpu_uuid,pid" in command:
            return "GPU-1, 12345\n"
        return "\n".join([
            "0, GPU-0, 22000, 0", "1, GPU-1, 22000, 0",
            "2, GPU-2, 22000, 0", "3, GPU-3, 12000, 0",
            "6, GPU-6, 22000, 0", "7, GPU-7, 22000, 0"])
    monkeypatch.setattr(launcher.subprocess, "check_output", fake_output)
    selected, _ = launcher.available_gpus([0,1,2,3,6,7])
    assert selected == [0,2,6,7]
    with pytest.raises(RuntimeError):
        launcher.available_gpus([0,1,2,3])


def test_o2_cache_loads_token_array_once_and_preserves_location(tmp_path, monkeypatch):
    import json
    import module2_subspaces as subspaces
    folder = tmp_path/"results"
    folder.mkdir()
    (folder/"manifest.json").write_text(json.dumps(dict(artifact_hashes={
        "results/obs2/key_embeddings.npz":"verified",
        "data/derived/pilot.jsonl":"verified"})))
    tokens = np.zeros((1,6,36,1,8,128),dtype=np.float32)
    tokens[0,1,20,0,5,0] = 9.5
    class Archive(dict):
        reads = 0
        def __getitem__(self, key):
            if key == "token_keys":
                self.reads += 1
            return super().__getitem__(key)
    archive = Archive(token_keys=tokens, states=["RA","RN","IA","IN"],
                      group_ids=["g0"], split=["train"])
    monkeypatch.setattr(subspaces, "sha256", lambda path:"verified")
    monkeypatch.setattr(subspaces.np, "load", lambda *args,**kwargs:archive)
    rows = [dict(case_id="g0-"+s,split="train") for s in ["RA","RN","IA","IN","RA2","RN2"]]
    cached = subspaces.o2_cache(tmp_path, rows, [(2,0),(20,5)])
    assert archive.reads == 1, "compressed archive must not be decompressed for every head/case"
    assert cached["g0-RN"][20,5][0,0] == 9.5
    rows[0]["split"] = "test"
    with pytest.raises(AssertionError):
        subspaces.o2_cache(tmp_path, rows, [(2,0)])
