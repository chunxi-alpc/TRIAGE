# 对消，而非擦除：论文故事与工作汇报

## 汇报入口

- [老师汇报版：方法依据、图表和实验结果](meeting/meeting_brief.zh.md)
- [完整论文故事](paper_story_latest.zh.md)
- [多模型、多任务主实验与消融：模拟研究](simulated_study/REPORT.zh.md)
- [五个模型的选择与分组说明](model_selection.zh.md)

汇报材料把 repair-v2 缓存上的实测结果与简化模型模拟分别列出。新增的多模型、多任务主实验与消融为预设分数关系的模拟，未运行所列模型或 benchmark。模拟图和表均标明来源，用于解释方法原理。

最新方案分为主实验与补充实验：主实验为 Qwen3-8B、Llama-3.1-8B-Instruct、Gemma-2-9B-IT；补充为 MedGemma-27B 与 Gemma-3-27B-IT。保留三个任务和五个模拟种子，主表与补充表分开。27B 组提供同系列、同规模的医学/通用比较，后训练差异仍需结合解释。

## 配套材料

- [同批配对的传导与预测汇总](trace/REPORT.zh.md)
- [证据响应保留率](trace/signed_response.csv)
- [配对回归与独立校准的开发集实验](meeting/paired_fit/REPORT.zh.md)
- [临床样本核验说明](clinical/README.zh.md)
- [医生核验表](clinical/review_sheet.csv)

临床核验材料已备齐，审核状态为待完成。独立测试集和完整路径评价按论文故事第八节继续开展。
