# MEGA–MedGUIDE 指南来源重合审计

2026-10-26。代码已准备，等待协调器在115服务器执行；本地没有运行或读取数据题文。此项仅补充来源信息，不增加实验门槛，不删改任何测试样本。

## 已有事实

- 协调器已实际验证：MEGA `case_paths.jsonl` 的 train_val/test 分别1834/440行，顶层字段只有 `guideline_id:int`、`filename:str`、`language:str`、`primary_clinical_focus:str`、`paths:list`。无显式外部来源 URL、DOI、来源命名空间。这里的 `guideline_id` 只能先认作MEGA本地标识，不能拿同值与其他数据集直接匹配。
- MedGUIDE parquet 有 `generation_model,disease,path,profile,options,option_count,answer_text,prompt,answer` 和评分字段；已有来源字段搜索结果为 `[]`。旧审计未检查 `path` 的类型和含义，因此此处仅补查该列是否为整个 URL 或结构化对象；不因列名叫 path 就认为它是指南文件路径。
- MedGUIDE公开数据说明把来源机构描述为NCCN。这是数据集层级来源说明，不能自动给每条题目建立文档/版本映射。现有元数据不能证明MEGA与MedGUIDE来源独立。
- 官方MEGA代码固定版本的README与完整文件树没有单独的来源文档映射文件；代码仓库只提供生成/评价入口。这不是对其他未检查资料是否存在映射的否定结论。
- MEGA科研使用已由用户确认；此脚本不重新提出许可问题。

## 最小执行范围

`audit_guideline_sources.py` 从既有 `configs/datasets.json` 找数据。MedGUIDE只投影 `path` 和明确命名的来源字段，绝不投影 prompt/profile/options/answer。MEGA只流式扫描 `case_paths.jsonl`；只解码顶层来源标识与 filename，跳过参考路径的全部嵌套值，不读病例、gt_paths或指南正文。它仍属于程序级元数据访问测试容器，不表述为“从未访问测试文件”。

仅输出文件SHA、字段类型、行数、path类型统计和来源标识的哈希。非URL path不输出原文、值哈希或任意嵌套键。普通编号只统计数量；有明确相同命名空间和相同字段含义才比较编号。来源URL仅规范化协议/主机大小写、默认端口和片段；保留查询参数、路径大小写和末尾斜杠，避免把不同版本错误合并。

`path`/`filename` 即使恰为URL，也先作为“待确认语义的URL候选”，与明确 `source_url/guideline_url/document_url` 分开。不扫描题文中的链接，也不把引用同一网页认定为同一版本指南。匹配为来源位置证据；不匹配和缺字段都不证明来源独立。所有结果固定保留 `source_independence:unknown`，全部样本保留。

服务器执行（将本目录复制到服务器项目同相对路径）：

```bash
python -m unittest discover -s work/source_overlap -p 'test_*.py' -v
python work/source_overlap/audit_guideline_sources.py --config configs/datasets.json --output results/source_overlap/guideline_sources.json
```

依赖仅Python标准库和现有PyArrow，无GPU、无模型推理、无网络访问。5项合成测试覆盖元数据投影、URL区别、候选与明确来源区分、命名空间及完整保留分母。未声称这些测试已运行。

如果 `path` 只是决策内容，当前可核实的结论就是缺少文档级映射；保留unknown完成本次审计，不需要新增筛选步骤。若未来作者公开映射，再按映射补充来源信息，不倒推或删题。

## 来源

- [MEGA固定数据版本](https://huggingface.co/datasets/nchennnn/MEGA-CDP/tree/8f058c88ce1ed43693785b435ad0ff99907a2890)。
- [MedGUIDE固定数据版本](https://huggingface.co/datasets/MedGUIDE/MedGUIDE-MCQA-8K/tree/1460ec478280a7295040b92f7eb985327f77dcbe)；NCCN来源说明由已有 `work/data_audit/source_registry.json` 及数据审计文档记录。
- [MEGA官方README](https://github.com/nchennnn/MEGA-CDP/blob/15a3a12e73a9169a870bff2ac431ec2e9968b8c7/README.md)，blob `eb6c7b73230479eeb00dfa1edd3ab92a06162fc4`；[完整代码文件树](https://api.github.com/repos/nchennnn/MEGA-CDP/git/trees/15a3a12e73a9169a870bff2ac431ec2e9968b8c7?recursive=1)。
- 实际字段/行数来源：协调器115服务器元数据扫描回传，2026-10-26；完整原始审计产物由协调器归档。本代理没有把README或下载清单当作实际文件验证。
