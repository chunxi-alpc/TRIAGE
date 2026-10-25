# FaithEval 固定版本与下载来源

本次只读官方版本页面、数据卡 schema 和 LFS 指针；未下载原始数据到本地，未打开题目/答案实例。不声称这些提交是最新版本。完整来源与下载路径位于 `source_manifest.json`。

| 集合 | 固定 revision | 文件字节数 | 数据卡题数 |
|---|---|---:|---:|
| Counterfactual | `e655f7c8750aabe431eeeda63fa5c019a2567b18` | 1,637,656 | 1,000 |
| Inconsistent | `e29508db3955db2c82a80a1f3c3aadc11f93fc3f` | 2,650,076 | 1,500 |
| Unanswerable | `4a14e0e9860e270a8c9ff57c511821f4f50dce57` | 3,178,294 | 2,492 |

三者相对路径均为 `data/test-00000-of-00001.parquet`。SHA256 已逐项从固定 revision 的官方 raw LFS 指针核对；Counterfactual 与此前登记值完全一致。LFS 指针中的 oid 才是完整文件 SHA256；页面显示的 Xet hash 是另一种对象标识，不能拿它替代下载文件校验值。

已从固定版本数据卡确认：Inconsistent/Unanswerable 为 `qid, question, context, answers(list[string]), subset, justification`；Counterfactual 还包含 `choices`（label/text 字符串列表）、`answerKey` 和选项数。方法可以读取真实公开输入，但答案、answerKey、justification 不得进入方法。CF 的公开 choices 说明先前只按 README 二种 QA 示例准备的 `context/question` 接口仍需核对原作者选项拼接方式；这不是已经完成 CF 原生复现的证据。卡字段类型已经核实，真实文件 footer 尚待服务器下载后验证。

三份固定官方数据卡均未列出标准 `license:` 字段，仅写科研用途发布。GitHub 代码的 Apache 2.0 不能自动移植为数据许可。搜索找到同名 Salesforce ModelScope CF 页面，页面标注 Apache License 2.0，但其二进制及对应 revision 本轮未验证，也不能用该标签覆盖官方卡说明。IC/UA 未获得可独立确认的 ModelScope 文件指纹。

协调器已在服务器证实：使用 `requests.Session().trust_env=False`，hf-mirror 的 CF 数据集 API 返回 HTTP 200，revision 与上述 CF 提交一致；ModelScope API 也可达。因此不再把“可尝试的传输路径”记为完全不可访问。原 Hugging Face 直连 ConnectionError 仍如实保留。manifest 提供三个固定 revision 的 hf-mirror resolve 路径，只有服务器下载后的字节数与官方 SHA256 完全一致，才能把镜像文件当作该官方对象；目前仅登记候选路径，不预先声称下载成功。

可选的服务器核验入口只做字节 hash、Parquet footer 的行数和类型，不读取任何行值：

```bash
python "$ACCESS/verify_server_files.py" --manifest "$ACCESS/source_manifest.json" --counterfactual "$CF_PARQUET" --inconsistent "$IC_PARQUET" --unanswerable "$UA_PARQUET" --output "$FAITHEVAL_BYTE_AND_SCHEMA_AUDIT"
```

结果应单独保存，不覆盖本次公开来源 manifest。官方来源：[CF 指针](https://huggingface.co/datasets/Salesforce/FaithEval-counterfactual-v1.0/raw/e655f7c8750aabe431eeeda63fa5c019a2567b18/data/test-00000-of-00001.parquet)、[IC 指针](https://huggingface.co/datasets/Salesforce/FaithEval-inconsistent-v1.0/raw/e29508db3955db2c82a80a1f3c3aadc11f93fc3f/data/test-00000-of-00001.parquet)、[UA 指针](https://huggingface.co/datasets/Salesforce/FaithEval-unanswerable-v1.0/raw/4a14e0e9860e270a8c9ff57c511821f4f50dce57/data/test-00000-of-00001.parquet)。三份固定数据卡链接已逐项保存在 manifest。镜像仅用于传输，不接受改写、翻译或重打包数据替代。
