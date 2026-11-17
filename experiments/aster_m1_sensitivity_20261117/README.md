# M1 statistical sensitivity

This directory is a separate M1-only experiment based on PR #81. The user confirmed the new statistical selector on 2026-11-17. No M2/M3 fitting or M4 modification occurs.

Read PROTOCOL.zh.md and configuration.json first. Historical artifacts and private caches are read only; all new artifacts are written to reports/aster_m1_sensitivity_20261117 and the new private runtime directory. Do not rerun preparation over existing outputs.

Server environment: /home/gcx/gcx261014. Working tree: /home/gcx/aster-m1-sensitivity-20261117.

1. Run audit.py with the explicit prior repository, old adaptive repository, dataset root, and a new audit output directory. The committed audit log records the inputs.
2. Run python -m experiments.aster_m1_sensitivity_20261117.prepare once. This computes source-level attributions from cached gradients, all fixed candidate families, donor assignments, and deduplicated missing-score requests.
3. Run the unit tests listed in statistics_tests.log and report_tests.log.
4. Submit the committed queue manifest through /home/gcx/aster-gpu-queue-20261113/scheduler.py. Worker modules require a direct scheduler parent and GPUs 0/1/5/6. Never invoke GPU workers outside that scheduler.
5. The CPU analysis job depends on all four workers and writes the complete numeric tables, four figures in PNG/PDF/SVG, a Chinese report and a paper subsection. Failures remain recorded. No automatic threshold replacement occurs.

Runtime requests and SQLite caches are kept on the server. Published worker score files contain anonymous target/donor IDs, local head IDs and scalar scores. Existing paired scores remain referenced by their original artifact paths. The matched-pair/source tables permit independent CPU recomputation.

Statistics tests use known exact probabilities and multiple-testing examples. Plot tests use synthetic arrays in temporary directories; they are not model results. A completion.json is written only after real matched controls, statistics and reports are complete.
