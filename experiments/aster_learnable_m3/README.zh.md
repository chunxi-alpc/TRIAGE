# ASTER：仅训练 M3

保持既有 51 个头、未筛选 U_N、S 和各头秩。新的映射和门控共 31,771 个参数；Qwen 全部冻结。原固定算子接口仍通过 Engine.run(operators=...) 使用；新接口为 Engine.run(m3=..., reference=..., training=True)。

代码基于已部署的 ASTER 最后输入位置 o_proj 前向钩子扩展。Engine.run 的 training=True 返回带计算图的两个判断 logits，由训练程序计算损失并反向传播；原 grad=True 诊断接口保持原行为。

参考由原病历、当前条件和树结构中的祖先条件组成，删除祖先条件末尾的“判断为是／否”后缀，不提供分支答案或标注路径。自身参考版本使用原完整输入的无干预头激活。两种参考均停止梯度。信号基只用于记录编辑在其方向上的平方改变量。

从原 MEGA-CDP 按旧 200／100 例身份读取训练和开发病例，测试使用全部 7,458 例。全部方法按当前 ASTER 是／否 logits 差值执行自主路径。监督标签仅用于训练损失、开发选轮及路径预测结束后的评价。

运行目录：/home/gcx/TRIAGE/results/aster_learnable_m3_20261106。

在服务器激活 /home/gcx/gcx261014 后，从 TRIAGE 根目录运行 python -m experiments.aster_learnable_m3.run。stage 为 prepare、verify、train 或 evaluate；训练 mode 为 reference 或 self；评价分片使用 shard 和 shards。

controller 模块依次调度两卡训练、开发选轮、两卡测试分片与 report 模块。运行详情写入结果目录 logs；pipeline_status.json 保存总状态。训练和测试各自首批 20 例后将计时写入 first20 文件，稳定后无需人工持续监控。
