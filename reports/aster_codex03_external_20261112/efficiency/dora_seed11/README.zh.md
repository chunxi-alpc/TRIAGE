# DoRA固定种子11的E11推理计时

复用已完成的DoRA检查点，512/2048 token各20次，批量1；4条固定开发输入、每档3次预热、5轮重复。实际GPU为4，型号NVIDIA A100-SXM4-80GB，见environment.json。完整逐次记录为measurements.jsonl.gz。

计时包含分词、传入GPU、前缀计算、判断和分数回传，模型加载在计时外。未合并DoRA适配器。此处是固定输入长度的完整请求计时；主表的inference_seconds_per_input统计完整评价中的唯一UID。

原始测量代码保存在服务器 /home/gcx/aster-e9-e11-20261112/experiments/aster_e9_e11_20261112/e11_shared_missing.py，完整E11交付由其对应报告维护。
