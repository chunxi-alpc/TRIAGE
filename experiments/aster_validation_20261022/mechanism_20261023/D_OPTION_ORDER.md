# D-only option-order extension

Fixed selection.json 200 cases, Qwen3-8B, original K76/U124. Original D hidden_prune/visible_prune outputs and branch predictions are reused. Only the final treatment options rotate using the existing left/right inverse maps; patient, conditions, branch order, action assignment, generation (greedy, 8 tokens) and model remain unchanged. There is no new branch inference, CAD, Llama or additional case evaluation.

Primary tests, declared before new outputs: hidden minus visible on all 200 for each of original/left/right, exact paired McNemar with Holm across these three comparisons; 2000-replicate disease-cluster bootstrap. Six K/U comparisons form a separate exploratory family; four method-specific rotation versus original comparisons another. Parse failures count as wrong. Agreement uses original action labels after inverse mapping. Frozen branch predictions mean this extension addresses treatment choice order, not branch candidate order.

Run from the repository in the existing server environment:

```bash
python experiments/aster_validation_20261022/mechanism_20261023/d_option_order.py prepare
CUDA_VISIBLE_DEVICES=0 python experiments/aster_validation_20261022/mechanism_20261023/d_option_order.py left
CUDA_VISIBLE_DEVICES=1 python experiments/aster_validation_20261022/mechanism_20261023/d_option_order.py right
python experiments/aster_validation_20261022/mechanism_20261023/d_option_order.py analyze
python -m unittest discover -s experiments/aster_validation_20261022/mechanism_20261023 -p test_d_option_order.py
```

Run left/right concurrently in separate processes. Outputs live in private/run500/mechanism/priority200/d_option_order; public statistics and per-case predictions in results/d_option_order. Workers append and resume by (id,method), cache identical prompt hashes, and validate completed outputs with manifests. Do not launch two workers for the same rotation. Existing raw outputs are preserved. Preparation verifies exact fixed IDs, bijective mappings, unchanged rules/patient, and both branch prompts' invariance.
