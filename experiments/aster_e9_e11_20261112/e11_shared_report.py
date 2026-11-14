"""CPU-only delivery rendering for the scheduled E11 supplement."""
from pathlib import Path
import shutil
from .common import read,write
NAMES={'raw':'原始Qwen3-8B','aster':'ASTER-CR E1 16维线性','leace':'LEACE历史检查点','lofit':'LoFiT历史检查点','lora':'LoRA历史检查点','cad':'CAD双前向','iti':'ITI-style E1任务适配','dora':'DoRA E1，seed 11'}
def render(merged,original):
 merged=Path(merged);original=Path(original)
 result=read(merged/'e11_statistics.json');cfg=read(merged/'configs/e11.json');infos=read(merged/'e11_method_info.json')
 stats=result['summary'];complete=result['complete']
 envs=read(merged/'environments.json')
 hardware='；'.join(('原六方法' if name=='original' else NAMES.get(name,name))+'：GPU '+str(env['visible_devices'])+'，'+env['gpu'] for name,env in envs.items())
 text=f"""# E11：统一模型推理效率

{len(infos)}种方法、{result['measurements']}次实际测量已保存。原六方法240次测量直接复用，本次补充ITI和DoRA各40次。完成状态：{'已完成' if complete else '补测排队中，尚缺'+ '、'.join(result['pending'])}。

ASTER-CR使用一次逻辑模型前向和17个线性修正参数。统一计时同时记录预填充、端到端延迟、吞吐、显存、KV、前向次数和存储，为部署成本提供完整比较。

## 配置与计时

基础模型为Qwen3-8B，BF16、SDPA、批量1。各方法使用同样的4条开发输入与末尾512/2048 token；每档预热3次，再进行5轮、每轮4条，共20次计时。端到端计时包含模板、分词、输入传输、预填充、最终判断、方法修正和分数返回，计时边界使用CUDA同步。输出为最终“是/否”判断分数。

实际硬件：{hardware}。补测由统一队列分配，硬件型号一致，物理卡不同。环境记录见environments.json；时间保留各卡原始测量值。比较范围为当前固定输入、长度与实现。

ITI复用现有审计标注的任务适配实现和已完成E1补拟合检查点，强度15、51个头，干预最后判断token。DoRA采用官方PEFT实现，固定seed 11，保留未合并适配器，前缀及最终判断均计入；seed 11在效率测量前固定。两者均使用已冻结参数。

## 延迟、吞吐与显存

时间为均值±样本标准差；吞吐为每轮均值±轮间标准差。峰值显存统计allocated；reserved及分次显存保存在原始记录。

| 方法 | token | 预填充ms | 端到端ms | 吞吐 样本/s | 峰值显存GiB |
|---|---:|---:|---:|---:|---:|
"""
 for r in stats:
  text+=f"| {NAMES.get(r['method'],r['method'])} | {r['length']} | {r['prefill_ms_mean']:.2f} ± {r['prefill_ms_std']:.2f} | {r['latency_ms_mean']:.2f} ± {r['latency_ms_std']:.2f} | {r['throughput_repeat_mean']:.2f} ± {r['throughput_repeat_std']:.2f} | {r['peak_allocated_gib_max']:.3f} |\n"
 text+="""
## 前向、参数与存储

每个逻辑前向由预填充和最终判断两次底层调用组成。CAD的额外参考前向及其分词、预填充全部计入。其余方法均为一次逻辑前向。

| 方法 | 拟合时梯度训练参数 | 部署数值KiB | 检查点KiB | 逻辑前向/底层调用 |
|---|---:|---:|---:|---:|
"""
 for name,info in infos.items():
  text+=f"| {NAMES.get(name,name)} | {info['trainable_parameters']:,} | {info['deployment_tensor_bytes']/1024:.3f} | {info['storage_bytes']/1024:.3f} | {info['logical_forwards']}/{2*info['logical_forwards']} |\n"
 text+="""
各方法推理时全部冻结。ASTER为16个权重与1个截距，M1/M2数值存储另计。LoFiT拟合第一阶段另有147,456个选择参数，最终保留128个偏移参数。LEACE使用闭式拟合；ITI部署51×128=6,528个拟合偏移数值，头选择探针不进入部署。检查点存储包含文件元数据。

## KV Cache

逻辑KV按tensor实际字节计量，前缀回放副本和临时张量包含在实测峰值显存中。CAD顺序处理主输入和参考输入，参考长度与缓存分别记录。

| 方法 | 主输入token | 主输入KV MiB | 参考平均token | 参考KV MiB |
|---|---:|---:|---:|---:|
"""
 for r in stats:
  text+=f"| {NAMES.get(r['method'],r['method'])} | {r['length']} | {r['kv_cache_bytes_mean']/2**20:.3f} | {r['reference_tokens_mean']:.1f} | {r['reference_kv_cache_bytes_mean']/2**20:.3f} |\n"
 text+="""
## 主要结果

"""
 for length in cfg['lengths']:
  a=next(r for r in stats if r['method']=='aster' and r['length']==length);b=next(r for r in stats if r['method']=='raw' and r['length']==length)
  text+=f"- {length} token：ASTER平均延迟{a['latency_ms_mean']:.2f} ms，原模型{b['latency_ms_mean']:.2f} ms，相对开销{(a['latency_ms_mean']/b['latency_ms_mean']-1)*100:.2f}%。\n"
 text+="""
主方法保持原模型的逻辑前向次数，通过选中头收集和线性修正完成推理。DoRA与ITI的具体时间、缓存及存储代价由上表给出，保留各轮波动供比较。

历史特征提取性能直接复用：单进程3.8228输入/s、双进程合计3.7932输入/s。CPU后处理性能直接复用：配对统计加速3.5917倍、路径统计加速10.5983倍。两份原文件保存在原交付目录reused/，没有重新计时。

## 产物与复现

当前目录包含e11_measurements.jsonl、e11_summary.csv、e11_repeat_summary.csv、e11_statistics.json、e11_method_info.json及environments.json。补测逐例记录位于上级iti/、dora/，参数快照和上游审计/冻结配置位于上级checkpoints/、upstream/。原六方法数据继续保存在原交付目录。

统一队列文件：/home/gcx/aster-gpu-queue-20261113/codex06_jobs.json。两个单卡任务分别运行e11_shared_missing模块的measure --method iti和measure --method dora；二者完成后，CPU任务运行merge。每项预估显存24 GiB、时长0.12小时，CPU汇总0.01小时。实际GPU、用时和完成标记随任务保存。完整执行命令及队列记录见executed_codex06_jobs.json，运行日志见../../logs/codex06-e11-*.log。CPU统计复现入口为../../reproduce.sh。

E9结果与原六方法测量保持原样；实验文档同步更新，论文正文保持原样。
"""
 (merged/'REPORT.zh.md').write_text(text,encoding='utf-8')
 if complete:
  prior=original/'E11_SIX_METHODS_20261112.zh.md'
  if not prior.exists():shutil.copyfile(original/'E11_REPORT.zh.md',prior)
  roottext=text.replace('当前目录包含','[完整合并统计](e11_shared_missing/merged/REPORT.zh.md)所在目录包含').replace('见executed_codex06_jobs.json','见e11_shared_missing/merged/executed_codex06_jobs.json').replace('../../logs/','logs/').replace('../../reproduce.sh','reproduce.sh')
  (original/'E11_REPORT.zh.md').write_text(roottext,encoding='utf-8')
  status=read(original/'status.json')
  status.update(E11='complete',pending=[],E11_methods=list(infos),E11_measurements=result['measurements'],
                E11_report='e11_shared_missing/merged/REPORT.zh.md')
  write(original/'status.json',status)
  p=original/'README.zh.md';s=p.read_text()
  s=s.replace('六种可运行方法；ITI和新增代表方法等待外部审计或完成产物。','八种方法共320次测量，含ITI与DoRA补测；[合并统计](e11_shared_missing/merged/REPORT.zh.md)。')
  s=s.replace('六种可运行方法；ITI和DoRA补测已进入统一队列。','八种方法共320次测量，含ITI与DoRA补测；[合并统计](e11_shared_missing/merged/REPORT.zh.md)。')
  p.write_text(s,encoding='utf-8')
