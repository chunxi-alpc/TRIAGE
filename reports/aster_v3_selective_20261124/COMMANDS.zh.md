# 服务器运行命令

本次实验在 `gcx@10.29.128.115` 的 `gcx261014` 环境使用 CPU。服务器各历史工作树及数据目录见 [ARTIFACT_MANIFEST.json](ARTIFACT_MANIFEST.json)。下列入口读取对应档案，顺序反映数据依赖。

```bash
source ~/gcx261014/bin/activate
cd ~/aster-v3-selective-20261124
export CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1
python -m experiments.aster_v3_selective.audit
python -m experiments.aster_v3_selective.coverage
python -m experiments.aster_v3_selective.s1_attribution
python -m experiments.aster_v3_selective.s1_review_index
python -m experiments.aster_v3_selective.s2_events
python -m experiments.aster_v3_selective.preservation
python -m experiments.aster_v3_selective.frozen_pairs
python -m experiments.aster_v3_selective.s1_semantic_eval
python -m experiments.aster_v3_selective.s1_dedup
python reports/aster_v3_selective_20261124/s4/archive_capacity.py
python reports/aster_v3_selective_20261124/s4/source_inventory.py
python -m experiments.aster_v3_selective.plot_results
python -m experiments.aster_v3_selective.report
python -m pytest -q tests/test_aster_v3_*.py
```

S0 的 PR 分支状态快照位于 `configs/aster_v3_selective/archive_prs.json`。S2 的参数文件记录去重总体、来源数、阈值以及监督训练资格。S3 使用首次计分前记录的规则及回退条件。[S2 参数](s2_events/e1_train80_parameters.json) · [S3 配置](s1/s3_protocol.json)
