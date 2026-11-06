> Source snapshot: /home/gcx/aster_representation_20261101/official_test_20261102/semantic/REPORT.zh.md
> Dates and execution states below refer to that original report. Relative artifact paths refer to the source run.

# Frozen predictions: local Qwen semantic evaluation

Five sampled judgments per pathway and outcome. OT fixed implementation reg=.05, tau=1; paper cost formula. LNDS threshold >=.9.

| Method | Recorded / registered | Scored | Failed | DTW-PCC (%) | Acc (%) | OT-PCC (%) | LNDS-PCS (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_js | 0/7458 | 0 | 0 | — | — | — | — |
| ordinary | 0/7458 | 0 | 0 | — | — | — | — |
| context | 0/7458 | 0 | 0 | — | — | — | — |
| blind_readout | 0/7458 | 0 | 0 | — | — | — | — |
| full | 0/7458 | 0 | 0 | — | — | — | — |
| zero | 0/7458 | 0 | 0 | — | — | — | — |
| fusion | 0/7458 | 0 | 0 | — | — | — | — |

Only fully covered methods receive table scores. Scoring failures use the registered conservative penalties; valid-only values are in summary.json. These are structured-path outputs, not the original paper free-text input setting.
