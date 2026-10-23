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
