# ASTER-CR paired-data retraining: Qwen3-8B

- [Chinese result summary](RESULTS_SUMMARY.zh.md)
- [Full Chinese report](REPORT.zh.md)
- [Next-stage plan](NEXT_STEPS.zh.md)
- [Frozen configuration](configuration.json)
- [Data inventory](data_inventory.json)
- [Input/cache manifest](input_cache_manifest.json)

E0 reuses the completed historical baseline. E1, E2 and E3 independently estimate M1, M2 and M3. E0/E1/E2 have five source-separated folds. The development-selected configuration is frozen before the historical held-out and cross-task evaluations.

## Results and parameters

- `fold_results.csv`: all 15 validation folds.
- `pair_metrics.csv`, `path_metrics.csv`: paired, node and complete-path results.
- `source_bootstrap.csv`: 2,000 paired source bootstrap draws, estimates and 95% intervals.
- `data_group_comparison.csv`: all data-group comparisons.
- `source_metrics.csv`: source-level paired errors and metrics.
- `model_comparison.csv`, `feature_rank_and_heads.json`: head overlap, layers, principal angles and effective ranks.
- `models/<group>/<full-or-fold>/`: M1 selections/scores, M2 parameters and M3 coefficients/intercepts.
- `evaluation/<scope>/metrics.json`: scope-specific complete metrics.
- `predictions/<scope>/*.jsonl.gz`: every published input, pair, question and path prediction. Use Python gzip or gunzip to read these without any model inference.
- `publication_manifest.json`: published prediction inventory and server locations.

Raw prediction JSONL files, unique-input tables, activation caches and runtime memory maps remain on the server at `/home/gcx/aster-cr-retrain-paired-v2-20261111/reports/aster_cr_retrain_paired_v2_20261111/`. The gzip prediction files contain the complete JSONL content, without filtering.

The original frozen data version is commit `5a329fe68eaa84fb5f7a791c50d15a47181ac4f2`. Historical 40-source and previously used 60-source results are comparisons on existing evaluation sets. They are not new independent confirmation. Native NLI4CT/MedPIC label relations are reported separately from A/P/C/F semantics.

Executable workflow: [experiment code](../../experiments/aster_cr_retrain_paired_v2_20261111/). Commands run on the experiment server using `source /home/gcx/gcx261014/bin/activate`. No additional full baseline grid is started by publication.

For the existing 60-source evaluation, 10,911 off-gold-path natural nodes have no released truth label. Their predictions are retained. Natural-node accuracy uses 4,928 labeled nodes; all 47,599 pairs remain in score metrics, and full paths use all available nodes. Unknown labels remain null.
