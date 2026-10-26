# ASTER 数据与公开来源审计

审计日期：2026-10-25，执行状态更新于2026-10-26（Asia/Shanghai）。角色：Agent A/D。协调器已在115服务器实际执行本目录元数据审计脚本及6项评价器合成测试，均通过；此代理未自行SSH或启动模型，未执行本地实验。服务器结果由协调器回传并归档。

## 结论与可继续工作

| 数据 | 已核实 | 尚未核实或阻碍 |
|---|---|---|
| SEKA 标准数据 | 作者发布包存在，固定版本 `91bf3f67fb2044c7e81514e103a962d6ef76c230`；实际文件在 `pasta_bench/` 和 `synthetic/`，并非 README 示例布局 | 服务器文件、数据哈希、官方筛选/拆分由协调器及基线代理核验 |
| MEGA-CDP | 115服务器元数据审计 complete_verified；原生指南+病例→路径+最终结论；原作者生成与评价代码存在 | 数据卡许可仍为 TBD；与MedGUIDE来源重合unknown；元数据完成不等于授权和独立性通过 |
| MedPIC | 115服务器实查467题，284 GF/183 CF；答案集合；CC BY 4.0 | 实际pair evidence_fields=[]，pair_count=0，Pair blocked；不应凭ID或题目相似性猜配 |
| MedGUIDE | 115服务器元数据审计 complete_verified；7,747题，单一 train split；NCCN来源单选任务 | 实际parquet的source_fields=[]；跨数据集来源独立性未确认；数据卡许可仍为 TBD；旧数据仅回归 |

MEGA 和 MedGUIDE 数据卡的原文均为 “TBD. Intended for academic and research use.”。这支持作者的科研使用意图，但不等于已发布正式许可证，也不等于源指南的再分发授权。MEGA官方README还明确给出下载公开数据并运行评测的步骤，提供了作者邀请科研复现的进一步依据；没有找到明确授予复制、修改、再分发等权利的许可条款。应分别登记“科研使用有发布说明支撑”“正式许可证未发布”“源指南再分发授权未确认”，不能合并写成“许可已通过”，也不能仅凭缺标准许可证推断作者禁止科研。

**实际使用决定（协调器，2026-10-26）**：依据用户明确要求科研复现、作者数据卡的科研用途说明以及官方README的下载和评测指令，继续MEGA内部科研开发准备；不再分发病例或指南全文。正式许可证状态仍为TBD，源指南再分发授权仍未确认。没有把额外作者询问设置为本次内部研究的前置条件，也没有宣称授权问题已经解决。开发仅用train_val；测试仍按原冻结与独立评价安排封存。决定及依据登记于`usage_decision.json`。没有向作者发信或创建公开issue。

2026-10-26再次核查MEGA官方代码仓库的完整文件树：固定commit下无LICENSE/COPYING/NOTICE，直接读取LICENSE返回404，GitHub仓库元数据license=null，公开issue列表为空。论文只显示arXiv论文分发许可，没有单独的数据使用授予条款；论文分发许可不能替代数据或第三方指南许可。未查到比数据卡与官方运行说明更明确的科研授权。

SEKA 发布包顶层标记 MIT，分项卡声明 CounterFact 为 MIT、BiasBios 为 CC BY 4.0、synthetic QA 为 CC BY 4.0。原始 `microsoft/biosbias/LICENSE` 为 MIT，覆盖仓库软件；其 README 介绍从 Common Crawl 重建传记。不能据此把网页原文统一改标 MIT，也没有证据足以断言科研评测被禁止。运行使用作者发布数据并保留来源，结果提交仅包含匿名输出和统计，不再分发传记全文。

## 原生评价器

MEGA 官方代码仓库为 <https://github.com/nchennnn/MEGA-CDP>，本次取得 commit `15a3a12e73a9169a870bff2ac431ec2e9968b8c7`（2026-10-20）。单轮 reference 为 `gt_paths.jsonl`。原脚本提供生成、解析、路径一致性、最终结论准确率和 Qwen3-8B judge。主论文表使用 GPT-5.4 Mini judge；Qwen judge 是作者另列设置，可以在不使用付费 API 的条件下实现，但应单列，不和主表混排。参考 `mega_cdp/scripts/evaluate/singleturn_evaluate_qwen3_8b_sglang.sh`、`pathway_metrics.py` 及 `summarize_eval_metrics.py`。README 示例要求5次 judge 结果，失败按最差/错误处理，不得悄悄缩小分母。尚未运行该代码。

MedPIC 论文采用零样本、每题独立输入、简短理由和指定答案字段，按选项集合精确匹配；每对两个病例都对才算 Pair 对。官方 HF 目录只有 `questions.json`、README 和 `.gitattributes`；服务器实际字段也未提供显式配对或来源规则字段。2026-10-26核查作者HF主页所链接的GitHub账号ZhitianHou、个人主页及公开仓库列表，未找到MedPIC补充材料或配对文件；HF讨论区只有Parquet自动转换条目。arXiv v1有9页PDF，正文提及补充材料，但当前公开页面未给出对应独立下载链接或89对ID映射。此为“未找到”的范围性结论，不声称作者从未发布。原生评测与增加外部规则后的开卷评测必须分别登记。缺失 Pair 映射不阻止冻结后开展全部467题的集合精确匹配，但 Pair 不能编造。

## 历史曝光与本次读取

PR #36 的 `configs/datasets.json` 与 `results/text_datasets_validation.json` 登记过 MEGA train_val/test 文件下载与结构检查。对应 `scripts/validate_requested_text_datasets.py` 会在 CPU 上解析所有 `gt_paths.jsonl`、病例和指南原文以核对键、必填字段和文件哈希。已存在程序级测试文件访问，不能写“从未读取测试文件”；该记录本身没有证明使用测试标签调方法。仍需审查服务器现有预测/配置与方法开发历史，才能判定确认资格。

本次 HF 数据卡网页自动包含 viewer 示例，Agent A 被动看到部分 MedPIC 和 MedGUIDE 示例正文与答案；协调器读取实际 README 时也被动看到了其中一条示例题答案。未按这些例子设计提示、挑选规则或参数，未向方法代理传递答案。此次为数据审计曝光，单独记录；后续仅读取字段、标识、文件哈希和来源信息。Agent A 没有读取 MEGA 测试正文/答案。

MedGUIDE 与 MEGA 来源指南重合经服务器检查仍为unknown：MedGUIDE parquet没有审计脚本识别到的原始文档来源字段。不能从癌种名称或不同病例文本推断无重合。后续若取得原始指南映射，应比较来源文档标识、标准化标题/年份/发布机构及文档哈希；按源指南划分MEGA train_val校准/开发。实际患者/指南正文不进入PR。元数据审计完成状态不能用于宣称来源独立性通过。

## 一手来源

- [MEGA 数据卡与许可证状态](https://huggingface.co/datasets/nchennnn/MEGA-CDP)，数据 revision `8f058c88ce1ed43693785b435ad0ff99907a2890`。
- [MEGA 代码 README](https://github.com/nchennnn/MEGA-CDP/blob/15a3a12e73a9169a870bff2ac431ec2e9968b8c7/README.md)；[论文](https://arxiv.org/html/2608.26592v1)。
- [MedPIC 数据卡](https://huggingface.co/datasets/TIM0927/MedPIC-Bench)，revision `9ef6db4f13865b14fc2e6be3f94dcfaf3a0cf983`；[论文第4节](https://arxiv.org/html/2608.03028v1)。
- [MedGUIDE 数据卡](https://huggingface.co/datasets/MedGUIDE/MedGUIDE-MCQA-8K)，revision `1460ec478280a7295040b92f7eb985327f77dcbe`。
- [SEKA 数据卡](https://huggingface.co/datasets/waylonli/SEKA-datasets)；[实际 pasta_bench 目录](https://huggingface.co/datasets/waylonli/SEKA-datasets/tree/main/pasta_bench)。
- [BiasBios 原始说明](https://github.com/microsoft/biosbias/blob/master/README.md)；[代码许可证](https://github.com/microsoft/biosbias/blob/master/LICENSE)。
- [CounterFact 上游 ROME 许可证](https://github.com/kmeng01/rome/blob/main/LICENSE)。
- [已有数据验证程序](https://github.com/chunxi-alpc/TRIAGE/blob/f37c5f8b6339216439e5fc257419ebe78ddf5763/scripts/validate_requested_text_datasets.py)。
- [MedPIC作者HF主页](https://huggingface.co/TIM0927)；[链接的作者GitHub](https://github.com/ZhitianHou)；[作者个人主页](https://zhitianhou.github.io/)；[HF讨论区](https://huggingface.co/datasets/TIM0927/MedPIC-Bench/discussions)；[arXiv下载入口](https://arxiv.org/abs/2608.03028)。
