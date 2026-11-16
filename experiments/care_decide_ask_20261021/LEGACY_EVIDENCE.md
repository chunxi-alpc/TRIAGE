# 历史 GPU 结果如何处理

**全部只读引用，不重新跑旧网格。** 原始代码、数据和 journal 保留在原 PR：

- [PR #3](https://github.com/chunxi-alpc/TRIAGE/pull/3)：B1–B4、SEKA 和随机基线，192 对：baseline 72，B4 74；没有稳定的学习空间优势。
- [PR #8](https://github.com/chunxi-alpc/TRIAGE/pull/8)：有符号 B4 75/192，直接注意力偏置 77/192；没有证据学习方向优于简单偏置。
- [PR #11](https://github.com/chunxi-alpc/TRIAGE/pull/11)：结构提示主导收益。旧 192 例中 baseline 71，结构提示 160，动作注意力 73，结构＋注意力 161；直接规则执行更高。
- [PR #12](https://github.com/chunxi-alpc/TRIAGE/pull/12)：冻结 A1 24 例 Qwen 接入，10（baseline）/15（结构）/10（单独注意力）/15（组合）。
- [PR #17](https://github.com/chunxi-alpc/TRIAGE/pull/17)：候选 token 修复后的 N3，baseline 21/32、二维 route 20/32、一维 mean 25/32、给定正确 D/Q 的文本 27/32。N1 未进入正式测试，N4 未正式运行。N3 mean 较 baseline 多纠正 4 例且没有退化，但已看过原测试，不能叫独立确认。N3 二维均值基底补修只在开发集 11/16（文本 12/16），未另跑正式测试。

负结果提供论文动机：单纯让模型把注意力投向规则片段，不能保证它正确组合条件、区分关键缺失/无关缺失。中心化配对差主成分可能丢失均值方向；绝对类质心替换也可能引入无关信息擦除。旧结果不否定所有 steering 机制，但**本论文不再以二维 steering 为主要贡献**。

全部历史性能来自不同样本/模型/统计协议，不能直接拼成一个统一新方法排行榜。旧 600 条合成样例已被使用，新论文确认数据必须重新生成、事前冻结。特别不能把旧 N3 oracle D/Q 成功解释为模型已经能自动判定 D/Q。
