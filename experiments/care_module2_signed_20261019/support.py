"""Frozen PR3 training functions and independent executable clinical oracle."""

from collections import defaultdict

import numpy as np

import torch

def paired_rows(rows):
    pairs = defaultdict(list)
    for row in rows:
        pairs[row["pair_id"]].append(row)
    out = []
    for pid, values in sorted(pairs.items()):
        assert len(values) == 2, (pid, len(values))
        assert {r["applicable"] for r in values} == {True, False}, pid
        a = next(r for r in values if r["applicable"])
        b = next(r for r in values if not r["applicable"])
        assert a["group_id"] == b["group_id"]
        assert a["prompt"][slice(*a["spans"]["action"])] == b["prompt"][slice(*b["spans"]["action"])]
        assert a["option_map"] == b["option_map"]
        out.append((a, b))
    return out

def contrastive_basis(signal, background, rank, ridge=.1):
    """Generalized signal/background second moment; QR in original key coordinates."""
    a = np.asarray(signal, dtype=np.float64)
    b = np.asarray(background, dtype=np.float64)
    assert len(a) and len(b)
    ca = a.T @ a / len(a)
    cb = b.T @ b / len(b)
    reg = max(np.trace(cb) / cb.shape[0] * ridge, 1e-10)
    w, v = np.linalg.eigh(cb + reg * np.eye(cb.shape[0]))
    whitening = (v * (1 / np.sqrt(w))) @ v.T
    e, u = np.linalg.eigh(whitening @ ca @ whitening)
    available = int(np.sum(e > max(e[-1] * 1e-8, 1e-10)))
    if rank > available:
        raise ValueError(f"B4 rank {rank} exceeds signal rank {available}")
    directions = whitening @ u[:, np.argsort(e)[::-1][:rank]]
    q, _ = np.linalg.qr(directions)
    return q[:, :rank]

def training_matrices(rows, keys, location, tolerance=.15, control_rows=None):
    """Only same-rule, same-label numeric controls matched to flip magnitude."""
    L, h = location
    k = lambda r: np.asarray(keys[r["case_id"]][L, h], dtype=np.float64)
    raw = np.stack([k(r).mean(0) for r in rows])
    flips, controls, audit = [], [], []
    pairs = paired_rows(rows)
    pool = rows if control_rows is None else control_rows
    for a, b in pairs:
        assert k(a).shape == k(b).shape
        flips.append(k(a).mean(0) - k(b).mean(0))
        distance = abs(float(a["value"]) - float(b["value"]))
        candidates = []
        for i, x in enumerate(pool):
            if (x["group_id"], x["rule_id"]) != (a["group_id"], a["rule_id"]):
                continue
            for y in pool[i + 1:]:
                if (x["group_id"], x["rule_id"], x["gold_label"]) != (y["group_id"], y["rule_id"], y["gold_label"]):
                    continue
                if x["applicable"] != y["applicable"]:
                    continue
                delta = abs(float(x["value"]) - float(y["value"]))
                if delta and distance and abs(delta / distance - 1) <= tolerance:
                    candidates.append((abs(delta / distance - 1), x["case_id"], y["case_id"], x, y))
        if candidates:
            _, _, _, x, y = min(candidates, key=lambda z: z[:3])
            # Both vectors compare the exact same action tokens.
            assert k(x).shape == k(y).shape
            controls.append(k(x).mean(0) - k(y).mean(0))
            audit.append(dict(pair_id=a["pair_id"], control=[x["case_id"], y["case_id"]],
                              flip_distance=distance, control_distance=abs(float(x["value"]) - float(y["value"]))))
    return raw, np.stack(flips), np.stack(controls) if controls else None, audit

def classify(kind,x):
    # Independent executable oracle; never infer gold from the model or option position.
    if kind=='a1c': return 'Normal glycemia' if x<5.7 else 'Prediabetes' if x<6.5 else 'Diabetes range'
    if kind=='fpg': return 'Normal glycemia' if x<100 else 'Prediabetes' if x<126 else 'Diabetes range'
    if kind=='bmi': return 'Underweight' if x<18.5 else 'Healthy weight' if x<25 else 'Overweight' if x<30 else 'Obesity'
    if kind=='gfr': return 'G1' if x>=90 else 'G2' if x>=60 else 'G3a' if x>=45 else 'G3b' if x>=30 else 'G4' if x>=15 else 'G5'
    if kind=='bp_clinic': return 'Below 140/90 mmHg' if x<80 else 'Below 150/90 mmHg'
    if kind=='bp_home': return 'Below 135/85 mmHg' if x<80 else 'Below 145/85 mmHg'
    if kind=='pe': return 'PE likely' if x>4 else 'PE unlikely'
    if kind=='dvt': return 'DVT likely' if x>=2 else 'DVT unlikely'
    if kind=='crb': return 'Low risk' if x==0 else 'Intermediate risk' if x<=2 else 'High risk'
    if kind=='curb': return 'Low risk' if x<=1 else 'Intermediate risk' if x==2 else 'High risk'
    if kind=='centor': return 'Do not offer an antibiotic prescription' if x<=2 else 'Consider an immediate or a back-up prescription'
    if kind=='feverpain': return 'Do not offer an antibiotic prescription' if x<=1 else 'Consider no antibiotic or a back-up prescription' if x<=3 else 'Consider an immediate or a back-up prescription'
    if kind=='af': return 'Do not offer stroke-prevention anticoagulation' if x==0 else 'Consider a direct-acting oral anticoagulant' if x==1 else 'Offer a direct-acting oral anticoagulant'
    if kind=='bone': return 'Normal bone density' if x>=-1 else 'Low bone mass' if x> -2.5 else 'Osteoporosis range'
    if kind=='anaemia': return 'No anaemia' if x>=120 else 'Mild anaemia' if x>=110 else 'Moderate anaemia' if x>=80 else 'Severe anaemia'
    if kind=='copd': return 'Stage 1 (mild)' if x>=80 else 'Stage 2 (moderate)' if x>=50 else 'Stage 3 (severe)' if x>=30 else 'Stage 4 (very severe)'
    if kind=='hypo': return 'No biochemical hypoglycemia' if x>=70 else 'Level 1 hypoglycemia' if x>=54 else 'Level 2 hypoglycemia'
    raise ValueError(kind)
