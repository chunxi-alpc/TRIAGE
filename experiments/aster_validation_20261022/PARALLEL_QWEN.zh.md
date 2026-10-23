# Qwen剩余任务双卡并行

用户于2026-10-23要求把尚未启动的Qwen任务放到GPU0，保持GPU1当前任务、不重复计算、Llama不恢复。

GPU0：原顺序候选过滤CAD → 右移500题（52记录/题）→ 右移无符号补充 → 右移候选过滤CAD。GPU1：原左移推理PID124995及其无符号补充继续；其后在原父队列等待右移输出期间，另一个任务执行左移候选过滤CAD。模型、增益、8/128词元上限、beta=0.1过滤、题目和选项数据不变。原始500题输出与已完成200题机制结果保持不变。

每个新任务在私有parallel_qwen/plan.json中只有一个执行方，按输出路径、输入路径和模型核对。run_cad128.py、supplement_unsigned.py、cad_filtered.py新增的入口检查仅作用于之后的调用：原父队列调用已分配任务时等待其完成文件，核对输入哈希与记录数后返回，不加载模型、不再次推理。正在运行的GPU1进程已载入代码，未被停止或重启。未登记的任务仍按原方式执行。

GPU1原父队列在进入右移阶段时，还会等待GPU1左移过滤任务结束，避免在同卡与补充/阳性对照阶段争抢资源。输出由原父队列合并，现有CAD跟随任务随后复用三份过滤输出并作原定分析。最终联合汇总仍由finish_joint200.sh等待这些自然完成。

服务器任务：parallel_qwen_gpu0.sh PID202171；parallel_qwen_gpu1.sh PID202175。私有状态、日志、任务表在/home/gcx/aster_validation_20261022/private/run500/parallel_qwen。原GPU1父队列86139、当前推理124995、CAD跟随87628均保留。Llama不恢复。

CPU测试覆盖唯一执行方启动、完成输出复用、失败时不重复启动、部分主输出保护、输入/模型校验、未分配任务不受影响，6项通过。复现命令：

```bash
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/TRIAGE-worktrees/aster-validation
python -m unittest discover -s experiments/aster_validation_20261022 -p test_delegation.py
python experiments/aster_validation_20261022/collect_parallel_qwen.py
```

两个后台任务已启动，不应重复启动。新增GPU0任务约占20–30GB显存；链式运行粗估8–12小时，过滤版进度需按实际速度更新。只要任务稳定，不持续监控。结果快照见results/run500/qwen/parallel，最终同批200题报告由mechanism_20261023/finish_joint200.sh生成。
