# 500 道测试题实验

检查时间：2026-10-23T08:01:13.470734+08:00
范围：原固定留出划分的前500题，排除40道开发题；不是根据模型准确率筛题。原3645题任务的已有结果保留，500题对应主输出原样复用。
CAD/prune_cad全部重跑128 token；两种选项重排覆盖同一500题；SEKA独立正对照500题。

llama: {"status": "cad128_500_cases", "positive_status": "not_started", "cad_fix_status": "waiting_for_existing_experiments", "main_complete": false, "option_order_complete": false, "positive_complete": false, "cad_fix_complete": false, "complete": false}
qwen: {"status": "cad128_500_cases", "positive_status": "not_started", "cad_fix_status": "waiting_for_existing_experiments", "main_complete": false, "option_order_complete": false, "positive_complete": false, "cad_fix_complete": false, "complete": false}

原始输出、病例及详细日志保留服务器 private/run500；本目录仅保存可发布数值和配置。
原始选项顺序和重排的动作一致率按动作身份比较。K/U仅代表一次闭卷原动作选择，不能证明参数记忆。
原先未具备条件的预测适用性、Context-DPO/CANOE、模型自选关键句不属于已完成结果。
