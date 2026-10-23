# Qwen优先200题

按原500题的固定顺序选取金标条件分支只有一个动作的前200题，不依据模型表现筛选。K=76、U=124，CS=2。原500题与CAD/重排任务保持运行；两模型原机制500题等待任务已撤下，先执行Qwen200。现有ABC结果全部复用，新分支/D/CS与改进方法仍等待修改版CAD自然完成。

|方法|A|B|C|
|---|---:|---:|---:|
|baseline|65.5%|41.5%|67.0%|
|annotate|77.0%|44.0%|72.5%|
|prune|99.0%|83.0%|99.0%|
|signed|75.0%|48.5%|74.5%|
|unsigned|73.0%|44.5%|75.0%|

复现：固定选择见selection.json；服务器数据在private/run500/mechanism/priority200。正式命令使用MECHANISM_PRIVATE与MECHANISM_PUBLIC设置独立目录，worker.sh保持原增益和解码设置，CUDA_VISIBLE_DEVICES=1，等待CAD PID87628。
正式机制结果完成后见qwen/full/REPORT.zh.md、summary.json与scores.jsonl；existing目录仅为已完成ABC复用结果。CS两例同病种，四版本K组调和平均仅作描述，不作为总体排名。


更新：Qwen200已拆成GPU0与GPU1各100题，各自等待该卡既有CAD自然完成；GPU0等待Llama CAD PID87624，GPU1等待Qwen CAD PID87628。两份输出分别写入私有priority200/gpu0与gpu1目录，合并后仍按同一200题和固定K/U统计，不取两份均值冒充整体指标。仅修改尚未开始的机制等待任务；现有CAD与重排进程保持运行。合并时检验200题覆盖、逐题键唯一与各100题manifest完成，注意力门槛在完整200题上计算，不在分片上决定。复现脚本priority200_shard.sh、merge_priority200.py、finish_priority200.sh。


最新执行安排：用户要求保存Llama并只先做Qwen200，Llama于2026-10-23 13:32保存后停止（左移重排367完整病例，见llama_saved_manifest.json）。尚未启动的双卡分片等待已撤下。Qwen机制200直接在释放的GPU0运行，GPU1原Qwen CAD与重排继续，不中断、不重启、不改设置。复现脚本priority200_gpu0.sh。机制结果不等待CAD；CAD自然完成后再补只读审计，不能把机制完成写成CAD完成。
