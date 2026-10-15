# CARE observations — Qwen3-4B-Base, 2026-10-16

Completed CARE pilot and clinical O1 follow-up, separate from the repository's Qwen3-8B TRIAGE pipeline. Every measurement came from real pretrained Qwen3-4B-Base on server GPU 0.

## Results

Clinical O1 contains 24 source-grounded rules in 12 families and 384 simulated patient cases (192 pairs). Eligibility requires source verification, stable maximum-probability choice under two closed-book formulations, and disagreement with authentic rule gold. There is no probability or margin cutoff.

**153 cases qualify: 78 applicable, 75 nonapplicable. Both members qualify in 25 pairs (50 cases, eight families).** Three-fold family-held-out evaluation excludes the test family from projection building and strength selection.

| Method | Applicable accuracy | Nonapplicable accuracy | Inapplicable-rule misuse |
| --- | --- | --- | --- |
| Baseline | 33.3% [19.5, 50.7] | 46.7% [23.1, 68.8] | 36.0% [22.2, 53.1] |
| Official SEKA | 33.3% [19.5, 50.7] | 48.0% [25.0, 69.6] | 33.3% [20.0, 49.0] |
| Official PASTA | 33.3% [19.5, 50.7] | 45.3% [21.3, 68.6] | 37.3% [23.0, 54.1] |

Intervals are 95% from 10,000 clinical-rule-family bootstrap draws. **The proposed faithfulness–misuse tradeoff was not observed.** Both-eligible paired cases and unscreened cases are reported separately.

O2/O3 retain the fictional-rule scope: development-selected Key applicability AUROC 0.738 [0.660, 0.832]; actual fixed-query BF16 logit direction mismatch 8.91% [8.70, 9.14]. These mechanism results are not clinical validation.

## Files

- [Updated report](results/observations_summary.md), [clinical detailed report](results/obs1_clinical/clinical_report.md)
- [Source review](data/clinical/source_review.md), [rule cards](data/clinical/rule_cards.json), [per-case review table](data/clinical/case_review.csv)
- [Frozen screen](results/obs1_clinical/prior_screen.csv), [actual predictions](results/obs1_clinical/predictions.csv), [metrics](results/obs1_clinical/metrics.csv), [paired changes](results/obs1_clinical/contrasts.csv)
- [Ten paper figure groups in PNG/PDF/SVG](figures/), [captions](figures/CAPTIONS.md)
- [Run manifest](results/obs1_clinical/manifest.json), [fold protocol](data/clinical/protocol.json), [dependency versions](configs/environment.lock.txt)
- [Clinical execution](scripts/run_clinical_o1.py), [clinical analysis](scripts/analyze_clinical_o1.py), [initial execution](scripts/run_experiments.py), [initial analysis](scripts/analyze_results.py)

## Full scientific artifacts

[Download the complete snapshot from the Release](https://github.com/chunxi-alpc/TRIAGE/releases/tag/care-qwen3-4b-base-20261016). The ~502 MiB ZIP contains original Key/Query tensors, all projections and builder inputs, token-level attention records, full logs, the third-party SEKA snapshot and earlier stages. SHA256SUMS.txt is a Release asset. Pretrained model weights and environments are not included.

Large tensors, full data and third-party checkouts stay outside Git. RAW_ARTIFACTS.json lists their locations; UPLOAD_MANIFEST.json fingerprints the committed experiment files. Original run manifests also reference raw files available only in the Release snapshot.

## Reproduce on the server

Authoritative full directory: /home/gcx/care_20261016/CARE_Codex_Pilot. Activate /home/gcx/gcx261014/bin/activate, then run REPRODUCE_CLINICAL.sh there. All experiments run on the server.

To reconstruct elsewhere on the server, extract the full Release ZIP and reuse/download the exact model revision. Full-data tests require raw Release artifacts and the model/tokenizer; they are not repository-only unit tests.

Model revision: 906bfd4b4dc7f14ee4320094d8b41684abff8539.
Unmodified official [SEKA/PASTA source](https://github.com/waylonli/SEKA) commit: aab9cfed80dc012fb9e83f8bdefb8675c33f40e7.
Operators and traditional SVD are reused; adaptations assemble clinical prompts and stream keys.

The complete server snapshot passed **65 checks** and **156 scientific artifact hashes**. Clinical model execution took 2.14 minutes and peaked at 7.58 GiB allocated BF16 memory.

## Limits

Rules are authentic references; patients are simulated. Primary-reference and executable-threshold checks were performed by the assistant without independent clinician adjudication. Stable closed-book error is an operational conflict definition and may include option preference. Only twelve rule families are represented. Confidence intervals condition on fitted folds; they do not establish clinical safety or prove a memorized medical belief.
