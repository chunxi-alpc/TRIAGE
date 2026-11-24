# 临床审核回填

服务器已生成一份离线审核包，位于 `/home/gcx/aster-v3-review-followup-private/clinical_review_forms.zip`。其中有五张可填写的 HTML 表和一份使用说明。病例、指南、盲号映射及个人回填保存在服务器私有目录。公开仓库只收录代码、去标识的证据位置与状态。

**审核者使用。** 在获准处理这些资料的设备上解压，先打开 `batch_01.html`，再打开获分配的后续批次。第一批 2 个节点，后四批分别有 58、57、57、57 个节点；后四批按整份来源分配。每例可直接查看当前条件、完整患者文本和原指南。在患者文本中选取决定判断的事实，填写判断与理由。材料不足时可选择“无法确定”。点击“保存回填文件”可导出本批 JSON，之后可载入继续填写。每份最终回填使用不同文件名。

**协调人接收。** 将通过项目授权渠道收到的最终 JSON 或 CSV 放入 `/home/gcx/aster-v3-review-followup-private/returns/`。同一审核者更新同一批回填时，只在该目录保留其最终版本。负责人核实本次审核者身份与资格后，使用其实际代号运行下列命令。无回填时可直接运行最后一行并省略 `--confirmed-reviewer`，此时读取程序只登记待审状态。

```bash
source ~/gcx261014/bin/activate
cd ~/aster-v3-selective-20261124
python -m experiments.aster_v3_selective.clinical_followup.review --confirmed-reviewer '实际代号' --summary reports/aster_v3_selective_20261124/followup/clinical_status.json
```

每个经确认的代号传一次 `--confirmed-reviewer`。同一盲号的重复判断按节点合并。不同判断、无法确定、材料未填全及身份待核对分别计数。判断与归档标签的逐例对照写入私有 `coordinator/reconciled.csv`，仓库汇总只输出分母、状态和原因覆盖的来源数。标签修订须另作决定。
